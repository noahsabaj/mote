<script lang="ts">
  import type { ByteTrace } from '../lib/trace.svelte';

  let {
    trace,
    structure,
    streaming
  }: { trace: ByteTrace; structure: boolean; streaming: boolean } = $props();

  // Structure view builds one span per chunk (a reply of 512 bytes is ~90 chunks), never one
  // per byte. Very long replies are capped so the paragraph cannot become a span forest.
  const CAP_CHUNKS = 1500;

  const rows = $derived.by(() => {
    if (!structure) return [];
    void trace.version;
    const all = trace.chunkRows();
    const kept = all.length > CAP_CHUNKS ? all.slice(all.length - CAP_CHUNKS) : all;
    return kept.map((c) => ({
      key: c.index,
      label: `chunk ${c.index} · bytes ${c.start}–${c.end} · ${c.bytes} B`,
      segments: trace.segmentsFor(c.start, c.end)
    }));
  });

  const hidden = $derived(structure ? Math.max(0, (trace.version, trace.runCount) - CAP_CHUNKS) : 0);
</script>

<!-- No whitespace between the spans: the container is `pre-wrap`, so stray template
     newlines would show up in the text. -->
{#if structure}<span class="prose structured"
    >{#if hidden}<span class="elided">… {hidden} earlier chunks not shown …&#10;</span
      >{/if}{#each rows as row, r (row.key)}<span class="chunk" class:alt={r % 2 === 1} title={row.label}
        >{#each row.segments as seg, i (i)}<span class:par={seg.mbp}>{seg.text}</span
          >{/each}</span
      >{/each}{#if streaming}<span
        class="caret"
        class:pending={trace.pending > 0}
        aria-hidden="true"
        title={trace.pending > 0
          ? `${trace.pending} byte${trace.pending > 1 ? 's' : ''} buffered — a multi-byte character is still arriving`
          : undefined}
      ></span>{/if}</span
  >{:else}<span class="prose"
    >{trace.text}{#if streaming}<span
        class="caret"
        class:pending={trace.pending > 0}
        aria-hidden="true"
        title={trace.pending > 0
          ? `${trace.pending} byte${trace.pending > 1 ? 's' : ''} buffered — a multi-byte character is still arriving`
          : undefined}
      ></span>{/if}</span
  >{/if}

{#if structure}
  <p class="legend" aria-label="How to read this view">
    <span class="key"><span class="swatch-a"></span><span class="swatch-b"></span>one learned chunk each</span>
    <span class="key"><span class="swatch-par">bytes</span> taken in parallel from the multi-byte head</span>
  </p>
{/if}

<style>
  .prose {
    /* break-spaces: a whitespace-only reply must wrap instead of hanging off the page (QA 2026-08-24) */
    white-space: break-spaces;
    overflow-wrap: anywhere;
  }

  /* Alternating tints: a chunk is a span of shading, not a rule through the text, so the
     paragraph still reads as prose with Structure on. */
  .structured .chunk {
    background: var(--chunk-a);
    padding: 0.05em 0;
    border-radius: 3px;
    box-decoration-break: clone;
    -webkit-box-decoration-break: clone;
  }
  .structured .chunk.alt {
    background: var(--chunk-b);
  }

  .structured .par {
    text-decoration: underline;
    text-decoration-color: var(--accent-line);
    text-decoration-thickness: 1.5px;
    text-underline-offset: 0.22em;
  }

  .elided {
    color: var(--ink-3);
    font-size: 0.8125rem;
    font-style: italic;
  }

  .caret {
    display: inline-block;
    width: 0.4em;
    height: 1.02em;
    margin-left: 0.06em;
    vertical-align: text-bottom;
    background: var(--ink-3);
    border-radius: 1px;
    animation: blink 1.15s steps(2, jump-none) infinite;
  }
  .caret.pending {
    background: var(--accent);
  }

  @keyframes blink {
    50% {
      opacity: 0.25;
    }
  }

  .legend {
    display: flex;
    flex-wrap: wrap;
    gap: 0.25rem 1.1rem;
    margin: 0.9rem 0 0;
    font-family: var(--font-ui);
    font-size: 0.75rem;
    color: var(--ink-3);
  }
  .key {
    display: inline-flex;
    align-items: center;
    gap: 0.35em;
  }
  .swatch-a,
  .swatch-b {
    display: inline-block;
    width: 0.9em;
    height: 0.9em;
    border-radius: 2px;
    background: var(--chunk-a);
  }
  .swatch-b {
    background: var(--chunk-b);
    margin-left: -0.2em;
  }
  .swatch-par {
    text-decoration: underline;
    text-decoration-color: var(--accent-line);
    text-decoration-thickness: 1.5px;
    text-underline-offset: 0.22em;
    color: var(--ink-2);
  }
</style>
