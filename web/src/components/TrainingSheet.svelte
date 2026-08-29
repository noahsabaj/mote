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

  // Jobs are started from the terminal (`mote train start … --serve`); the sheet is for watching them,
  // stopping them, and choosing which one is on the air (R8 + R1, signed 2026-08-25).
  async function serveJob(id: string | null, on: boolean) {
    if (jobBusy) return;
    jobBusy = true;
    try {
      jobs = await api.trainingServe(id, on);
      jobError = null;
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
  // The run is what a line is about; a dozen queued arms that all began `--preset flagship --data …` were
  // indistinguishable once truncated (QA 2026-08-24). The full argv stays in the tooltip.
  const runOf = (j: TrainingJob) => {
    const i = j.argv.indexOf('--out');
    return i >= 0 && j.argv[i + 1] ? j.argv[i + 1].replace(/^runs\//, '') : argline(j);
  };
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
        sep: last?.eval.boundary_on_separator_frac
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
    <p class="jobline" title={argline(jobs.current)}>
      <span class="live">running</span>
      {#if jobs.phase && jobs.phase !== 'train'}<span class="phase">{jobs.phase}</span>{/if}
      <span class="mono">{runOf(jobs.current)}</span>
      {#if jobs.current.serve}<span class="air">on the air</span>{/if}
      <span class="meta args">{argline(jobs.current)}</span>
      <button
        class="quiet"
        disabled={jobBusy}
        onclick={() => serveJob(jobs?.current?.id ?? null, !jobs?.current?.serve)}
        title={jobs.current.serve
          ? 'Stop answering chats from this run; the pinned checkpoint takes over'
          : 'Answer chats from this run’s EMA while it trains; its final checkpoint becomes the pin'}
      >
        {jobs.current.serve ? 'Take off the air' : 'Put on the air'}
      </button>
    </p>
  {:else}
    <p class="meta">
      Nothing running. Start one from the terminal: <span class="mono">mote train start … --serve</span> puts it on
      the air. The queue is sequential; while a job runs, chats are answered from the CPU.
    </p>
  {/if}
  {#each jobs?.queued ?? [] as j, i (j.id)}
    <p class="jobline" title={argline(j)}>
      <span class="meta">{i + 1} · queued{j.resumed ? ' (resume)' : ''}{j.retry_of ? ` · retry ${j.retries ?? ''}`.trimEnd() : ''}{j.needs_bytes ? ` · waits for ${(j.needs_bytes / 2 ** 30).toFixed(1)} GB free` : ''}</span>
      <span class="mono">{runOf(j)}</span>
      {#if j.serve}<span class="air">on the air</span>{/if}
      <span class="meta args">{argline(j)}</span>
      <button class="quiet" disabled={jobBusy} onclick={() => serveJob(j.id, !j.serve)} aria-label="{j.serve ? 'Take' : 'Put'} {runOf(j)} {j.serve ? 'off' : 'on'} the air">
        {j.serve ? 'Off the air' : 'On the air'}
      </button>
      <button class="quiet" disabled={jobBusy} onclick={() => stopJob(j.id)} aria-label="Cancel {runOf(j)}">Cancel</button>
    </p>
  {/each}
  {#each (jobs?.recent ?? []).slice(0, 3) as j (j.id)}
    <p class="jobline meta" title={argline(j)}>
      <span class="state {j.state}">{j.state}</span> <span class="mono">{runOf(j)}</span>
      <span class="meta args">{argline(j)}</span>
    </p>
  {/each}
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
    border-bottom: 1px solid var(--rule);
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
  /* what the running job is doing (eval 3/16, checkpoint) and which job answers chats */
  .phase,
  .air {
    flex: none;
    font-size: 0.6875rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--ink-3);
  }
  .air {
    padding: 0 0.35em;
    border: 1px solid var(--accent-line);
    border-radius: 999px;
    color: var(--accent-ink);
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
    flex: none;
    max-width: 55%;
  }
  /* the argv trails the name in the muted colour and takes whatever width is left */
  .jobline .args {
    flex: 1;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-family: var(--font-mono);
    font-size: 0.7rem;
  }
  .mono {
    font-family: var(--font-mono);
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
