<script lang="ts">
  import { chat } from '../lib/stores/chat.svelte';
  import Message from './Message.svelte';
  import Icon from './Icon.svelte';

  let { oninspect }: { oninspect: (id: string) => void } = $props();

  let scroller = $state<HTMLElement | null>(null);
  let pinned = $state(true);

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
  <div class="column">
    {#if chat.isEmpty}
      <div class="opening">
        <p class="lede">
          Morpheme reads and writes raw UTF-8 bytes, and works out its own chunks as it goes.
        </p>
        <p class="sub">
          Ask it something. Turn on <em>Structure</em> under a reply to see where it drew the
          boundaries, or open <em>Bytes</em> for the byte-by-byte trace.
        </p>
      </div>
    {:else}
      {#each chat.turns as turn, i (turn.id)}
        <Message
          {turn}
          isLast={i === chat.turns.length - 1 && turn.role === 'assistant'}
          {oninspect}
        />
      {/each}
    {/if}
  </div>
</div>

{#if !pinned && chat.busy}
  <button class="jump" onclick={() => toBottom()}>
    <Icon name="chevron" size={13} />
    Follow the reply
  </button>
{/if}

<style>
  .scroller {
    flex: 1;
    overflow-y: auto;
    overscroll-behavior-y: contain;
    scrollbar-gutter: stable;
  }

  .column {
    max-width: calc(var(--measure) + 6rem);
    margin: 0 auto;
    padding: 2rem var(--gutter) 1.5rem;
  }

  .opening {
    max-width: 30rem;
    padding: min(14vh, 6rem) 0 0;
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
</style>
