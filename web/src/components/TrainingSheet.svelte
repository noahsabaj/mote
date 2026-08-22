<script lang="ts">
  import { api, ApiError } from '../lib/api';
  import { isEvalRecord, isTrainRecord, type LogRecord, type TrainingRun } from '../lib/types';
  import Curve from './Curve.svelte';
  import type { Point } from '../lib/chart';
  import Icon from './Icon.svelte';
  import { count, minutes, num, pct, when } from '../lib/format';

  const POLL_MS = 3000;

  let runs = $state<TrainingRun[]>([]);
  let runsError = $state<string | null>(null);
  let selected = $state<string | null>(null);
  let records = $state<LogRecord[]>([]);
  let logError = $state<string | null>(null);
  let cursor = 0;
  let pulling = false;

  const run = $derived(runs.find((r) => r.id === selected) ?? null);
  const running = $derived(run?.running === true);

  async function loadRuns() {
    try {
      runs = await api.runs();
      runsError = null;
      if (!selected && runs.length) selected = runs[0].id;
    } catch (e) {
      runsError = e instanceof ApiError ? e.message : String(e);
    }
  }

  async function pull(id: string) {
    if (pulling) return;
    pulling = true;
    try {
      const page = await api.runLog(id, cursor);
      if (page.records.length) records = [...records, ...page.records];
      cursor = page.next;
      logError = null;
    } catch (e) {
      logError = e instanceof ApiError ? e.message : String(e);
    } finally {
      pulling = false;
    }
  }

  $effect(() => {
    void loadRuns();
  });

  // A run change starts the log over; `since` then only ever asks for what is new.
  $effect(() => {
    const id = selected;
    if (!id) return;
    cursor = 0;
    records = [];
    void pull(id);
  });

  $effect(() => {
    const id = selected;
    if (!id || !running) return;
    const timer = setInterval(() => {
      void pull(id);
      void loadRuns();
    }, POLL_MS);
    return () => clearInterval(timer);
  });

  const trainPoints = $derived<Point[]>(
    records.filter(isTrainRecord).map((r) => ({ x: r.step, y: r.train_bpb }))
  );
  const evalRecords = $derived(records.filter(isEvalRecord));
  const valPoints = $derived<Point[]>(evalRecords.map((r) => ({ x: r.step, y: r.eval.val_bpb })));
  const bpicPoints = $derived<Point[]>(
    records.filter(isTrainRecord).map((r) => ({ x: r.step, y: r.bpic }))
  );
  const targetPoints = $derived<Point[]>(
    records.filter(isTrainRecord).map((r) => ({ x: r.step, y: r.target_ratio }))
  );
  const latest = $derived(evalRecords.length ? evalRecords[evalRecords.length - 1] : null);
  const lastTrain = $derived.by(() => {
    const t = records.filter(isTrainRecord);
    return t.length ? t[t.length - 1] : null;
  });
</script>

{#if runsError}
  <p class="fail"><Icon name="alert" size={14} />{runsError}</p>
{:else if runs.length === 0}
  <p class="empty">No training runs found in the runs directory.</p>
{:else}
  <div class="picker" role="group" aria-label="Training run">
    {#each runs as r (r.id)}
      <button
        class="run"
        class:on={r.id === selected}
        aria-pressed={r.id === selected}
        onclick={() => (selected = r.id)}
      >
        <span class="name">{r.id}</span>
        <span class="meta">
          {r.steps.toLocaleString()} steps · {num(r.last_val_bpb, 3)} bits/byte
          {#if r.running}<span class="live">running</span>{/if}
        </span>
      </button>
    {/each}
  </div>

  {#if logError}
    <p class="fail small"><Icon name="alert" size={13} />{logError}</p>
  {/if}

  {#if run}
    <dl class="rows head">
      <dt>Started</dt>
      <dd>{when(run.started_at)}</dd>
      <dt>Records</dt>
      <dd>{records.length.toLocaleString()} read{running ? ' · polling' : ''}</dd>
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

  {#if records.length === 0}
    <p class="empty">The log for this run is empty.</p>
  {:else}
    <section>
      <Curve
        series={[
          { points: trainPoints, label: 'train', weight: 'faint' },
          { points: valPoints, label: 'val', weight: 'solid', dots: true }
        ]}
        yLabel="Bits per byte"
        xLabel="step"
        digits={2}
      />
    </section>

    <section>
      <Curve
        series={[
          { points: bpicPoints, label: 'measured', weight: 'faint' },
          { points: targetPoints, label: 'target', weight: 'solid' }
        ]}
        yLabel="Bytes per chunk"
        xLabel="step"
        digits={1}
      />
    </section>

    {#if latest}
      <section>
        <h3>Last evaluation — step {latest.step.toLocaleString()}</h3>
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
            {#each latest.eval.sample.split('|') as piece, i (i)}<span class="piece" class:first={i === 0}>{piece}</span>{/each}
          </p>
        {/if}
      </section>
    {/if}
  {/if}
{/if}

<style>
  .picker {
    display: grid;
    gap: 0.3rem;
    margin-bottom: 1.2rem;
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
    display: block;
    font-family: var(--font-mono);
    font-size: 0.8125rem;
    color: var(--ink);
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
    border-left: 1px solid var(--accent-line);
    padding-left: 0.11em;
    margin-left: 0.05em;
  }
  .piece.first {
    border-left: 0;
    padding-left: 0;
    margin-left: 0;
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
