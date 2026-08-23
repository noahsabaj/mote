<script lang="ts">
  import { chat } from '../lib/stores/chat.svelte';
  import { notices } from '../lib/stores/notice.svelte';
  import Message from './Message.svelte';
  import FoldLine from './FoldLine.svelte';
  import Icon from './Icon.svelte';

  let { oninspect }: { oninspect: (id: string) => void } = $props();

  let scroller = $state<HTMLElement | null>(null);
  let pinned = $state(true);

  // A checkpoint can be hot-swapped mid-conversation, and two replies from two different
  // models look identical in a transcript. The break is a fact about the thread, so it is
  // drawn between the turns rather than tucked into one of them.
  const rows = $derived.by(() => {
    let prev: number | undefined;
    return chat.turns.map((turn, i) => {
      let rule: number | null = null;
      if (turn.role === 'assistant' && turn.checkpointStep !== undefined) {
        if (prev !== undefined && prev !== turn.checkpointStep) rule = turn.checkpointStep;
        prev = turn.checkpointStep;
      }
      return {
        turn,
        rule,
        isLast: i === chat.turns.length - 1 && turn.role === 'assistant'
      };
    });
  });

  // Where Mote's view will start on the next reply (docs/context.md); turns above it are dimmed.
  const fold = $derived(chat.busy ? null : (chat.preview?.fold ?? null));

  function onscroll() {
    if (!scroller) return;
    const gap = scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight;
    pinned = gap < 90;
  }

  function toBottom(smooth = true) {
    if (!scroller) return;
    scroller.scrollTo({
      top: scroller.scrollHeight,
      behavior: smooth && !matchMedia('(prefers-reduced-motion: reduce)').matches
        ? 'smooth'
        : 'auto'
    });
  }

  // Follow the stream only while the reader has not scrolled away.
  $effect(() => {
    const live = chat.streamingId ? chat.traces[chat.streamingId] : undefined;
    void chat.turns.length;
    void live?.version;
    if (pinned && scroller) scroller.scrollTop = scroller.scrollHeight;
  });
</script>

<div class="scroller" bind:this={scroller} onscroll={onscroll}>
  <div class="column" class:empty={chat.isEmpty}>
    {#if chat.isEmpty}
      <div class="opening">
        <p class="lede">
          Mote reads and writes raw UTF-8 bytes, and works out its own chunks as it goes.
        </p>
        <p class="sub">
          Ask it something. Turn on <em>Structure</em> under a reply to see where it drew the
          boundaries, or open <em>Bytes</em> for the byte-by-byte trace.
        </p>
      </div>
    {:else}
      {#each rows as row, i (row.turn.id)}
        {#if row.rule !== null}
          <div class="swap" role="separator">
            <span>checkpoint step {row.rule} loaded</span>
          </div>
        {/if}
        {#if fold && i === fold.from}
          <FoldLine {fold} card={chat.card} onedit={(c) => chat.setCard(c)} onunfold={() => chat.unfold()} />
        {/if}
        <div class="slot" class:folded={fold !== null && i < fold.from}>
          <Message turn={row.turn} isLast={row.isLast} {oninspect} />
        </div>
      {/each}
    {/if}
  </div>
</div>

{#if !pinned && !chat.isEmpty}
  <button class="jump" class:lifted={!!notices.current} onclick={() => toBottom()}>
    <Icon name="chevron" size={13} />
    {chat.busy ? 'Follow the reply' : 'Jump to latest'}
  </button>
{/if}

<style>
  .slot.folded {
    opacity: 0.55;
  }

  .scroller {
    flex: 1;
    overflow-y: auto;
    overscroll-behavior-y: contain;
    /* both-edges keeps the centred column aligned with the composer, which has no scrollbar */
    scrollbar-gutter: stable both-edges;
  }

  .column {
    max-width: var(--shell);
    margin: 0 auto;
    padding: 2rem var(--gutter) 1.5rem;
  }

  /* Empty state sits in the middle of the free space above the composer, not at the top. */
  .column.empty {
    min-height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: center;
  }

  .opening {
    max-width: 30rem;
    padding: 0 0 6vh;
  }

  .lede {
    margin: 0;
    font-family: var(--font-read);
    font-size: 1.3rem;
    line-height: 1.45;
    color: var(--ink);
    text-wrap: balance;
  }

  .sub {
    margin: 0.85rem 0 0;
    font-size: 0.875rem;
    line-height: 1.6;
    color: var(--ink-3);
    max-width: 32rem;
  }
  .sub em {
    font-style: normal;
    color: var(--ink-2);
  }

  .jump {
    position: absolute;
    left: 50%;
    bottom: 0.5rem;
    transform: translateX(-50%);
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.3rem 0.7rem;
    border: 1px solid var(--rule-strong);
    border-radius: 999px;
    background: var(--bg);
    box-shadow: var(--shadow);
    font-size: 0.75rem;
    color: var(--ink-2);
    cursor: pointer;
  }
  .jump :global(svg) {
    transform: rotate(90deg);
  }
  /* The undo bar owns this slot while it is up. */
  .jump.lifted {
    bottom: 3.3rem;
  }

  .swap {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    margin: 0 0 1.9rem;
    font-size: 0.75rem;
    color: var(--ink-3);
    font-variant-numeric: tabular-nums;
  }
  .swap::before,
  .swap::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--rule);
  }
</style>
