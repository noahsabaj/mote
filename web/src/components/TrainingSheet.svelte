<script lang="ts">
  // Runs are read from the log files on disk. One run is the "primary" (details, probe sample);
  // any number can be selected to overlay their curves and compare final numbers.
  import { untrack } from 'svelte';
  import { api, ApiError } from '../lib/api';
  import {
    isEvalRecord,
    isTrainRecord,
    type EvalRecord,
    type LogRecord,
    type TrainRecord,
    type TrainingRun
  } from '../lib/types';
  import type { JobsStatus, TrainingJob } from '../lib/types';
  import Curve from './Curve.svelte';
  import type { Series } from '../lib/chart';
  import Icon from './Icon.svelte';
  import { count, minutes, num, pct, when } from '../lib/format';

  const POLL_MS = 3000;
  const PALETTE = ['var(--accent)', '#4f86e0', '#3aa37a', '#b35cc9', '#c9a227', '#7c8590'];

  let runs = $state<TrainingRun[]>([]);
  // the daemon's job queue (docs/shape.md): training runs the studio owns
  let jobs = $state<JobsStatus | null>(null);
  let jobArgs = $state('--preset local --data data/local_mix --out runs/studio_job --optimizer muon --max-minutes 30');
  let jobBusy = $state(false);
  let jobError = $state<string | null>(null);

  async function loadJobs() {
    try {
      jobs = await api.trainingQueue();
      jobError = null;
    } catch (e) {
      jobError = e instanceof ApiError ? e.message : String(e);
    }
  }

  async function startJob() {
    const argv = jobArgs.trim().split(/\s+/);
    if (!argv.length || jobBusy) return;
    jobBusy = true;
    try {
      jobs = await api.trainingStart(argv);
      jobError = null;
      void loadRuns();
    } catch (e) {
      jobError = e instanceof ApiError ? e.message : String(e);
    } finally {
      jobBusy = false;
    }
  }

  async function stopJob(id: string | null = null) {
    if (jobBusy) return;
    jobBusy = true;
    try {
      jobs = await api.trainingStop(id);
      jobError = null;
    } catch (e) {
      jobError = e instanceof ApiError ? e.message : String(e);
    } finally {
      jobBusy = false;
    }
  }

  const argline = (j: TrainingJob) => j.argv.join(' ');
  let runsError = $state<string | null>(null);
  let selected = $state<string[]>([]);
  let logs = $state<Record<string, LogRecord[]>>({});
  let logError = $state<string | null>(null);
  const cursors: Record<string, number> = {};
  const pulling = new Set<string>();

  async function loadRuns() {
    try {
      runs = await api.runs();
      runsError = null;
      if (!selected.length && runs.length) selected = [runs[0].id];
    } catch (e) {
      runsError = e instanceof ApiError ? e.message : String(e);
    }
  }

  async function pull(id: string) {
    if (pulling.has(id)) return;
    pulling.add(id);
    try {
      const page = await api.runLog(id, cursors[id] ?? 0);
      if (page.records.length) logs = { ...logs, [id]: [...(logs[id] ?? []), ...page.records] };
      cursors[id] = page.next;
      logError = null;
    } catch (e) {
      logError = e instanceof ApiError ? e.message : String(e);
    } finally {
      pulling.delete(id);
    }
  }

  function toggle(id: string) {
    if (selected.includes(id)) {
      if (selected.length > 1) selected = selected.filter((x) => x !== id);
    } else {
      selected = [...selected, id];
    }
  }

  $effect(() => {
    void loadRuns();
    void loadJobs();
  });

  $effect(() => {
    const timer = setInterval(() => {
      void loadJobs();
      if (jobs?.current) void loadRuns();
    }, POLL_MS * 2);
    return () => clearInterval(timer);
  });

  // First selection of a run starts its log; later pulls only ask for what is new.
  $effect(() => {
    const ids = selected;
    untrack(() => {
      for (const id of ids) {
        if (cursors[id] === undefined) {
          cursors[id] = 0;
          void pull(id);
        }
      }
    });
  });

  $effect(() => {
    const live = selected.filter((id) => runs.find((r) => r.id === id)?.running);
    if (!live.length) return;
    const timer = setInterval(() => {
      for (const id of live) void pull(id);
      void loadRuns();
    }, POLL_MS);
    return () => clearInterval(timer);
  });

  const single = $derived(selected.length === 1);
  const primary = $derived(selected[0] ?? null);
  const primaryRun = $derived(runs.find((r) => r.id === primary) ?? null);
  const primaryRecords = $derived(primary ? (logs[primary] ?? []) : []);
  const primaryRunning = $derived(primaryRun?.running === true);

  const colorOf = (i: number) => PALETTE[i % PALETTE.length];
  const evals = (recs: LogRecord[]): EvalRecord[] => recs.filter(isEvalRecord);
  const trains = (recs: LogRecord[]): TrainRecord[] => recs.filter(isTrainRecord);

  const bpbSeries = $derived.by<Series[]>(() => {
    const out: Series[] = [];
    selected.forEach((id, i) => {
      const recs = logs[id] ?? [];
      if (single) {
        out.push({ points: trains(recs).map((r) => ({ x: r.step, y: r.train_bpb })), label: 'train', weight: 'faint' });
      }
      out.push({
        points: evals(recs).map((r) => ({ x: r.step, y: r.eval.val_bpb })),
        label: single ? 'val' : id,
        weight: 'solid',
        dots: true,
        color: single ? undefined : colorOf(i)
      });
    });
    return out;
  });

  const bpicSeries = $derived.by<Series[]>(() => {
    const out: Series[] = [];
    selected.forEach((id, i) => {
      const recs = logs[id] ?? [];
      const t = trains(recs);
      if (single) {
        out.push({ points: t.map((r) => ({ x: r.step, y: r.bpic })), label: 'measured', weight: 'faint' });
        out.push({ points: t.map((r) => ({ x: r.step, y: r.target_ratio })), label: 'target', weight: 'solid' });
      } else {
        out.push({ points: t.map((r) => ({ x: r.step, y: r.bpic })), label: id, weight: 'solid', color: colorOf(i) });
      }
    });
    return out;
  });

  const table = $derived(
    selected.map((id, i) => {
      const recs = logs[id] ?? [];
      const ev = evals(recs);
      const tr = trains(recs);
      const last = ev.length ? ev[ev.length - 1] : null;
      const lastT = tr.length ? tr[tr.length - 1] : null;
      return {
        id,
        color: colorOf(i),
        steps: runs.find((r) => r.id === id)?.steps ?? 0,
        minutes: lastT?.elapsed_min,
        bps: lastT?.bytes_per_sec,
        valBpb: last?.eval.val_bpb,
        bpic: last?.eval.val_bpic,
        sep: last?.eval.boundary_on_separator_frac,
        mbp: last?.eval.mbp_top1_acc
      };
    })
  );

  const latest = $derived.by(() => {
    const ev = evals(primaryRecords);
    return ev.length ? ev[ev.length - 1] : null;
  });
  const lastTrain = $derived.by(() => {
    const t = trains(primaryRecords);
    return t.length ? t[t.length - 1] : null;
  });
  const anyRecords = $derived(selected.some((id) => (logs[id] ?? []).length > 0));
</script>

<section class="jobs" aria-label="Training jobs">
  <div class="jobrow head">
    <h3>Jobs</h3>
    {#if jobs?.current}
      <button class="btn" disabled={jobBusy} onclick={() => stopJob()}>Stop</button>
    {/if}
  </div>
  {#if jobError}<p class="fail"><Icon name="alert" size={13} />{jobError}</p>{/if}
  {#if jobs?.current}
    <p class="jobline"><span class="live">running</span> <span class="mono">{argline(jobs.current)}</span></p>
  {:else}
    <p class="meta">Nothing running. The queue is sequential; a chat makes a run yield for the length of the reply.</p>
  {/if}
  {#each jobs?.queued ?? [] as j (j.id)}
    <p class="jobline">
      <span class="meta">queued{j.resumed ? ' (resume)' : ''}</span>
      <span class="mono">{argline(j)}</span>
      <button class="quiet" disabled={jobBusy} onclick={() => stopJob(j.id)}>Cancel</button>
    </p>
  {/each}
  {#each (jobs?.recent ?? []).slice(0, 3) as j (j.id)}
    <p class="jobline meta"><span class="state {j.state}">{j.state}</span> <span class="mono">{argline(j)}</span></p>
  {/each}
  <form
    class="jobrow"
    onsubmit={(e) => {
      e.preventDefault();
      void startJob();
    }}
  >
    <input class="mono" bind:value={jobArgs} aria-label="Training arguments" />
    <button class="btn accent" disabled={jobBusy}>Start</button>
  </form>
  <p class="meta">Args exactly as <span class="mono">python -m mote.train.train</span> takes them; the run lands in the list below.</p>
</section>

{#if runsError}
  <p class="fail"><Icon name="alert" size={14} />{runsError}</p>
{:else if runs.length === 0}
  <p class="empty">No training runs found in the runs directory.</p>
{:else}
  <div class="picker" role="group" aria-label="Training runs">
    {#each runs as r (r.id)}
      {@const i = selected.indexOf(r.id)}
      <button class="run" class:on={i >= 0} aria-pressed={i >= 0} onclick={() => toggle(r.id)}>
        <span class="name">
          {#if i >= 0 && !single}<span class="dot" style:background={colorOf(i)}></span>{/if}
          {r.id}
        </span>
        <span class="meta">
          {r.steps.toLocaleString()} steps ·
          {r.last_val_bpb === null ? 'not evaluated yet' : `${num(r.last_val_bpb, 3)} bits/byte`}
          {#if r.running}<span class="live">running</span>{/if}
        </span>
      </button>
    {/each}
  </div>
  <p class="hint">Select more than one run to overlay their curves and compare.</p>

  {#if logError}
    <p class="fail small"><Icon name="alert" size={13} />{logError}</p>
  {/if}

  {#if single && primaryRun}
    <dl class="rows head">
      <dt>Started</dt>
      <dd>{when(primaryRun.started_at)}</dd>
      <dt>Records</dt>
      <dd>{primaryRecords.length.toLocaleString()} read{primaryRunning ? ' · polling' : ''}</dd>
      {#if lastTrain}
        <dt>Elapsed</dt>
        <dd>{minutes(lastTrain.elapsed_min ?? 0)}</dd>
        <dt>Throughput</dt>
        <dd>{count(lastTrain.bytes_per_sec)} bytes/s</dd>
        <dt>Learning rate</dt>
        <dd>{lastTrain.lr.toExponential(2)}</dd>
        <dt>Grad norm</dt>
        <dd>{num(lastTrain.grad_norm, 2)}</dd>
      {/if}
    </dl>
  {/if}

  {#if !single}
    <section class="compare">
      <h3>Side by side — last evaluation of each</h3>
      <div class="tablewrap">
        <table>
          <thead>
            <tr>
              <th>run</th>
              <th>steps</th>
              <th>min</th>
              <th>bytes/s</th>
              <th>val bpb</th>
              <th>B/chunk</th>
              <th>@sep</th>
              <th>mbp top-1</th>
            </tr>
          </thead>
          <tbody>
            {#each table as row (row.id)}
              <tr>
                <td class="name"><span class="dot" style:background={row.color}></span>{row.id}</td>
                <td>{row.steps.toLocaleString()}</td>
                <td>{row.minutes !== undefined ? num(row.minutes, 0) : '—'}</td>
                <td>{row.bps !== undefined ? count(row.bps) : '—'}</td>
                <td>{row.valBpb !== undefined ? num(row.valBpb, 3) : '—'}</td>
                <td>{row.bpic !== undefined ? num(row.bpic, 2) : '—'}</td>
                <td>{row.sep !== undefined ? pct(row.sep, 0) : '—'}</td>
                <td>{row.mbp !== undefined ? pct(row.mbp, 0) : '—'}</td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    </section>
  {/if}

  {#if !anyRecords}
    <p class="empty">The log for this run is empty.</p>
  {:else}
    <section>
      <Curve series={bpbSeries} yLabel="Bits per byte" xLabel="step" digits={2} />
    </section>

    <section>
      <Curve series={bpicSeries} yLabel="Bytes per chunk" xLabel="step" digits={1} />
    </section>

    {#if latest}
      <section>
        <h3>
          {single ? 'Last evaluation' : `Last evaluation of ${primary}`} — step {latest.step.toLocaleString()}
        </h3>
        <dl class="rows">
          <dt>Val bits/byte</dt>
          <dd>{num(latest.eval.val_bpb, 4)}</dd>
          <dt>Bytes per chunk</dt>
          <dd>{num(latest.eval.val_bpic, 2)}</dd>
          <dt>Boundaries on separators</dt>
          <dd>{pct(latest.eval.boundary_on_separator_frac, 1)}</dd>
          {#if latest.eval.mbp_top1_acc !== undefined}
            <dt>Multi-byte top-1</dt>
            <dd>{pct(latest.eval.mbp_top1_acc, 1)}</dd>
          {/if}
        </dl>

        {#if latest.eval.sample}
          <h4>Learned chunking of a fixed probe</h4>
          <p class="sample">
            {#each latest.eval.sample.split('|') as piece, i (i)}<span class="piece" class:alt={i % 2 === 1}>{piece}</span>{/each}
          </p>
        {/if}
      </section>
    {/if}
  {/if}
{/if}

<style>
  .jobs {
    margin: 0 0 1.1rem;
    padding: 0 0 0.9rem;
    border-bottom: 1px solid var(--line);
  }
  .jobrow {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  .jobrow.head {
    justify-content: space-between;
    margin-bottom: 0.3rem;
  }
  .jobrow input {
    flex: 1;
    font-size: 0.8rem;
    padding: 0.4rem 0.55rem;
    border: 1px solid var(--line);
    border-radius: calc(var(--radius) - 2px);
    background: var(--bg);
    color: var(--ink);
    min-width: 0;
  }
  .jobline {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin: 0.15rem 0;
    min-width: 0;
  }
  .jobline .mono {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    flex: 1;
  }
  .mono {
    font-family: var(--mono, ui-monospace, monospace);
    font-size: 0.78rem;
  }
  .state.failed { color: var(--danger, #b4413c); }

  .picker {
    display: grid;
    gap: 0.3rem;
    margin-bottom: 0.5rem;
  }

  .hint {
    margin: 0 0 1.2rem;
    font-size: 0.75rem;
    color: var(--ink-3);
  }

  .run {
    display: block;
    width: 100%;
    text-align: left;
    padding: 0.5rem 0.6rem;
    border: 1px solid var(--rule);
    border-radius: var(--radius-sm);
    background: transparent;
    cursor: pointer;
    transition: border-color 120ms ease, background-color 120ms ease;
  }
  .run:hover {
    background: var(--surface);
  }
  .run.on {
    border-color: var(--accent-line);
    background: var(--accent-soft);
  }

  .name {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-family: var(--font-mono);
    font-size: 0.8125rem;
    color: var(--ink);
  }

  .dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex: none;
  }

  .run :global(.meta) {
    display: block;
    margin-top: 0.1rem;
    font-size: 0.75rem;
  }

  .live {
    color: var(--accent-ink);
  }
  .live::before {
    content: '· ';
  }

  .head {
    margin-bottom: 1.5rem;
  }

  section + section {
    margin-top: 1.8rem;
  }

  .compare {
    margin-bottom: 1.6rem;
  }

  .tablewrap {
    overflow-x: auto;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.8125rem;
    font-variant-numeric: tabular-nums;
  }
  th,
  td {
    padding: 0.4rem 0.45rem;
    border-top: 1px solid var(--rule);
    text-align: right;
    white-space: nowrap;
  }
  th {
    border-top: 0;
    font-weight: 500;
    color: var(--ink-3);
  }
  th:first-child,
  td:first-child {
    text-align: left;
    padding-left: 0;
  }
  td.name {
    font-family: var(--font-mono);
  }
  td.name .dot {
    margin-right: 0.4rem;
    vertical-align: 0;
  }

  h3 {
    font-size: 0.9375rem;
    font-weight: 600;
    margin-bottom: 0.45rem;
  }

  h4 {
    margin: 1.2rem 0 0.4rem;
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: var(--ink-3);
  }

  .sample {
    margin: 0;
    font-family: var(--font-read);
    font-size: 0.9375rem;
    line-height: 1.85;
    color: var(--ink);
  }

  .piece {
    background: var(--chunk-a);
    padding: 0.05em 0;
    border-radius: 3px;
    box-decoration-break: clone;
    -webkit-box-decoration-break: clone;
  }
  .piece.alt {
    background: var(--chunk-b);
  }

  .empty {
    margin: 0;
    font-size: 0.875rem;
    color: var(--ink-2);
  }

  .fail {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    margin: 0 0 0.9rem;
    font-size: 0.875rem;
    color: var(--accent-ink);
  }
  .fail.small {
    font-size: 0.8125rem;
  }
</style>
