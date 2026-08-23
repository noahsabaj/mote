<script lang="ts">
  // Every byte of one reply, windowed: only the rows on screen exist in the DOM, so a
  // 4096-byte reply costs ~30 nodes rather than 4096.
  import type { ByteTrace } from '../lib/trace.svelte';
  import type { Turn } from '../lib/stores/chat.svelte';
  import Sparkline from './Sparkline.svelte';
  import { byteGlyph, hex, num, pct } from '../lib/format';

  let { trace, turn }: { trace: ByteTrace; turn?: Turn } = $props();

  // The stats line under a reply only flags the knobs that were off default. This is the
  // place that states all of them, because this is where you come to compare two draws.
  const drawnAt = $derived.by(() => {
    const p = turn?.params;
    if (!p) return null;
    return `T ${p.temperature} · top-p ${p.top_p} · n ${p.n_candidates} · max ${p.max_bytes} B`;
  });

  const ROW = 26;
  const OVERSCAN = 8;

  let viewport = $state<HTMLElement | null>(null);
  let scrollTop = $state(0);
  let viewportHeight = $state(420);

  const total = $derived((trace.version, trace.count));
  const first = $derived(Math.max(0, Math.floor(scrollTop / ROW) - OVERSCAN));
  const visible = $derived(
    Math.min(total - first, Math.ceil(viewportHeight / ROW) + OVERSCAN * 2)
  );

  const rows = $derived.by(() => {
    const out = [];
    for (let i = first; i < first + visible && i < total; i++) out.push(trace.byteAt(i));
    return out;
  });

  const probs = $derived((trace.version, trace.boundaryProbs(256)));

  function onscroll(e: Event) {
    scrollTop = (e.currentTarget as HTMLElement).scrollTop;
  }

  $effect(() => {
    if (!viewport) return;
    const ro = new ResizeObserver(([entry]) => {
      viewportHeight = entry.contentRect.height;
    });
    ro.observe(viewport);
    return () => ro.disconnect();
  });

  let copied = $state(false);
  async function copyJson() {
    try {
      const rows = [];
      for (let i = 0; i < total; i++) rows.push(trace.byteAt(i));
      await navigator.clipboard.writeText(JSON.stringify({ bytes: total, chunks: trace.chunkCount, rows }, null, 0));
      copied = true;
      setTimeout(() => (copied = false), 1600);
    } catch {
      /* clipboard blocked */
    }
  }
</script>

<section class="summary">
  <dl class="rows">
    <dt>Bytes</dt>
    <dd>{total}</dd>
    <dt>Chunks</dt>
    <dd>{trace.chunkCount}{trace.chunkCount ? ` · ${num(total / trace.chunkCount, 1)} B each` : ''}</dd>
    <dt>Parallel</dt>
    <dd>{pct(trace.mbpFraction(), 1)} of bytes came from the multi-byte head</dd>
    {#if drawnAt}
      <dt>Drawn at</dt>
      <dd>{drawnAt}</dd>
    {/if}
    {#if turn?.checkpointStep !== undefined}
      <dt>Checkpoint</dt>
      <dd>step {turn.checkpointStep}</dd>
    {/if}
  </dl>
  <button class="quiet" onclick={copyJson} title="Copy every row of this trace as JSON, for sharing or a notebook">
    {copied ? 'Copied' : 'Copy trace as JSON'}
  </button>
</section>

<section class="plot">
  <h3>Boundary probability</h3>
  <p class="meta">Last {probs.length} bytes. The router opens a chunk where this spikes.</p>
  <Sparkline values={probs} label="Boundary probability of the most recent bytes" height={40} />
</section>

<section class="table">
  <h3>Bytes</h3>
  <div class="frame">
  <div class="head" aria-hidden="true">
    <span class="c-i">#</span>
    <span class="c-hex">hex</span>
    <span class="c-ch">byte</span>
    <span class="c-src">from</span>
    <span class="c-n">p</span>
    <span class="c-n">H</span>
    <span class="c-n">p(b)</span>
  </div>
  <!-- svelte-ignore a11y_no_noninteractive_tabindex -- a scrollable region has to be
       reachable by keyboard, so the tabindex is deliberate. -->
  <div
    class="viewport"
    bind:this={viewport}
    onscroll={onscroll}
    tabindex="0"
    role="group"
    aria-label="Byte-by-byte trace, {total} rows"
  >
    <div class="spacer" style="height: {total * ROW}px">
      <div class="window" style="transform: translateY({first * ROW}px)">
        {#each rows as r (r.i)}
          <div class="row" class:boundary={r.boundary} style="height: {ROW}px">
            <span class="c-i">{r.i}</span>
            <span class="c-hex">{hex(r.byte)}</span>
            <span class="c-ch" class:sep={!r.chars}>{byteGlyph(r.byte)}</span>
            <span class="c-src" class:mbp={r.mbp} class:fix={r.fix}
              >{r.mbp ? 'parallel' : r.fix ? 'corrected' : 'sampled'}</span
            >
            <span class="c-n">{num(r.p, 2)}</span>
            <span class="c-n">{num(r.entropy, 2)}</span>
            <span class="c-n">{num(r.boundaryP, 2)}</span>
          </div>
        {/each}
      </div>
    </div>
  </div>
  </div>
  <p class="meta foot">
    A left rule marks a byte the router chose as a chunk start. <span class="k">parallel</span>
    came from the multi-byte head, <span class="k">corrected</span> replaced a draft byte that
    verification rejected. <span class="k">p</span> is the sampled byte's probability,
    <span class="k">H</span> the entropy of that step, <span class="k">p(b)</span> the boundary
    probability.
  </p>
</section>

<style>
  section + section {
    margin-top: 1.6rem;
  }

  h3 {
    font-size: 0.9375rem;
    font-weight: 600;
  }

  .plot :global(p) {
    margin: 0.15rem 0 0.5rem;
  }

  /* One horizontally scrollable frame around both, and fixed columns in both: the rows sit
     inside a vertical scroller and the head does not, so any `fr` column resolves to a
     different width in each and the last column loses its digits. */
  .frame {
    overflow-x: auto;
    overscroll-behavior-x: contain;
  }

  .head,
  .row {
    display: grid;
    grid-template-columns: 2.3rem 2.1rem 2.2rem 3.9rem 2rem 2rem 2.3rem;
    gap: 0.3rem;
    align-items: center;
    font-family: var(--font-mono);
    font-size: 0.75rem;
    font-variant-numeric: tabular-nums;
    padding: 0 0.4rem;
  }

  /* Both boxes hold the same floor, so the frame scrolls them together rather than letting
     the rows squeeze under the vertical scrollbar. The floor allows for that scrollbar. */
  .head,
  .viewport {
    min-width: 20.5rem;
  }

  .head {
    color: var(--ink-3);
    padding-bottom: 0.35rem;
    border-bottom: 1px solid var(--rule);
    margin-top: 0.5rem;
  }

  .viewport {
    height: min(26rem, 48vh);
    overflow-y: auto;
    scrollbar-gutter: stable;
    overscroll-behavior: contain;
    contain: strict;
  }

  .spacer {
    position: relative;
  }
  .window {
    position: absolute;
    inset: 0 0 auto 0;
    will-change: transform;
  }

  .row {
    border-left: 1px solid transparent;
    color: var(--ink-2);
  }
  .row.boundary {
    border-left-color: var(--accent-line);
    color: var(--ink);
  }

  .c-i {
    color: var(--ink-3);
  }
  .c-ch.sep {
    color: var(--ink-3);
  }
  .c-src {
    font-size: 0.6875rem;
    letter-spacing: 0.02em;
    color: var(--ink-3);
  }
  .c-src.mbp {
    color: var(--accent-ink);
  }
  .c-src.fix {
    color: var(--ink);
  }
  .c-n {
    text-align: right;
  }

  .foot {
    margin-top: 0.6rem;
    line-height: 1.5;
  }
  .k {
    font-family: var(--font-mono);
    color: var(--ink-2);
  }
</style>
