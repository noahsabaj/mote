<script lang="ts">
  import type { Turn } from '../lib/stores/chat.svelte';
  import { chat } from '../lib/stores/chat.svelte';
  import { ui } from '../lib/stores/ui.svelte';
  import ChunkedText from './ChunkedText.svelte';
  import Icon from './Icon.svelte';
  import { num, pct } from '../lib/format';

  let {
    turn,
    isLast,
    oninspect
  }: { turn: Turn; isLast: boolean; oninspect: (id: string) => void } = $props();

  const trace = $derived(chat.traces[turn.id]);
  const streaming = $derived(chat.streamingId === turn.id);
  const text = $derived(streaming && trace ? trace.text : turn.content);
  const empty = $derived(streaming && text.length === 0);

  let copied = $state(false);
  let copyTimer: ReturnType<typeof setTimeout> | null = null;

  async function copy() {
    try {
      await navigator.clipboard.writeText(turn.content || text);
      copied = true;
      if (copyTimer) clearTimeout(copyTimer);
      copyTimer = setTimeout(() => (copied = false), 1600);
    } catch {
      /* clipboard blocked — the text is selectable either way */
    }
  }

  const reasonNote = $derived(
    turn.reason === 'max_bytes'
      ? 'reached the byte limit'
      : turn.reason === 'stopped'
        ? 'stopped'
        : ''
  );
</script>

{#if turn.role === 'user'}
  <article class="turn user" aria-label="You said">
    <div class="said">{turn.content}</div>
  </article>
{:else}
  <article class="turn model" aria-label="Morpheme replied" aria-busy={streaming}>
    {#if turn.truncated}
      <p class="notice">
        <Icon name="alert" size={13} />
        The conversation was truncated to fit the {turn.contextLimit} byte context window.
      </p>
    {/if}

    <div class="body" class:awaiting={empty}>
      {#if trace && (streaming || ui.structure)}
        <ChunkedText {trace} structure={ui.structure} {streaming} />
      {:else}
        <span class="prose">{text}</span>
      {/if}
      {#if empty}<span class="waiting" aria-hidden="true"></span>{/if}
    </div>

    {#if turn.error}
      <p class="notice error">
        <Icon name="alert" size={13} />
        {turn.error}
      </p>
    {/if}

    {#if !streaming}
      <footer class:pinned={isLast}>
        <p class="stats meta">
          {#if turn.stats}
            {turn.stats.bytes} bytes · {turn.stats.chunks} chunks · {num(
              turn.stats.bytes_per_chunk,
              1
            )} B per chunk · {num(turn.stats.bytes_per_sec, 0)} B/s
            {#if turn.stats.mbp_proposed > 0}
              · {pct(turn.stats.mbp_accept_rate)} of multi-byte proposals accepted
            {/if}
            {#if reasonNote}· {reasonNote}{/if}
          {/if}
        </p>
        <div class="actions">
          {#if trace}
            <button
              class="quiet"
              aria-pressed={ui.structure}
              onclick={() => ui.toggleStructure()}
              title="Mark learned chunk boundaries and bytes taken in parallel"
            >
              <Icon name="structure" size={14} />
              Structure
            </button>
            <button class="quiet" onclick={() => oninspect(turn.id)}>Bytes</button>
          {/if}
          <button class="quiet" onclick={copy}>
            <Icon name={copied ? 'check' : 'copy'} size={14} />
            {copied ? 'Copied' : 'Copy'}
          </button>
          {#if isLast}
            <button class="quiet" onclick={() => chat.regenerate()} disabled={chat.busy}>
              <Icon name="redo" size={14} />
              Again
            </button>
          {/if}
        </div>
      </footer>
    {/if}
  </article>
{/if}

<style>
  .turn {
    margin: 0 0 1.9rem;
  }

  .user {
    display: flex;
    justify-content: flex-end;
  }

  .said {
    max-width: 85%;
    padding: 0.6rem 0.9rem;
    border-radius: var(--radius);
    background: var(--surface);
    color: var(--ink);
    font-size: 0.9375rem;
    line-height: 1.55;
    white-space: pre-wrap;
    overflow-wrap: break-word;
  }

  .model .body {
    font-family: var(--font-read);
    font-size: 1.0625rem;
    line-height: 1.68;
    color: var(--ink);
  }

  .prose {
    white-space: pre-wrap;
    overflow-wrap: break-word;
  }

  /* Before the first byte lands: one mark, not a skeleton screen. */
  .waiting {
    display: inline-block;
    width: 0.4em;
    height: 1.02em;
    vertical-align: text-bottom;
    background: var(--ink-3);
    border-radius: 1px;
    animation: breathe 1.5s ease-in-out infinite;
  }
  @keyframes breathe {
    0%,
    100% {
      opacity: 0.9;
    }
    50% {
      opacity: 0.25;
    }
  }

  .notice {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    margin: 0 0 0.5rem;
    font-size: 0.8125rem;
    color: var(--ink-3);
  }
  .notice.error {
    margin: 0.6rem 0 0;
    color: var(--accent-ink);
  }

  footer {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    flex-wrap: wrap;
    margin-top: 0.55rem;
    min-height: 28px;
  }

  .stats {
    margin: 0;
    font-size: 0.75rem;
    flex: 1 1 14rem;
  }

  .actions {
    display: flex;
    gap: 0.1rem;
    opacity: 0;
    transition: opacity 120ms ease;
  }
  .turn:hover .actions,
  footer:focus-within .actions,
  footer.pinned .actions {
    opacity: 1;
  }

  @media (hover: none) {
    .actions {
      opacity: 1;
    }
  }
</style>
