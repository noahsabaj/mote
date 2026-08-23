<script lang="ts">
  // The list that opens when the composer starts with a slash. Selection lives in the
  // composer, because the arrow keys that move it never leave the textarea.

  import type { Command } from '../lib/commands';
  import { dismissable } from '../lib/actions';

  let {
    items,
    selected,
    trigger,
    onpick,
    ondismiss
  }: {
    items: Command[];
    selected: number;
    /** the composer's textarea — keyboard focus stays there, so it must not count as "outside" */
    trigger: HTMLElement | null;
    onpick: (c: Command) => void;
    ondismiss: () => void;
  } = $props();
</script>

<div class="menu" role="listbox" aria-label="Commands" use:dismissable={{ onDismiss: ondismiss, trigger }}>
  {#each items as c, i (c.name)}
    <button
      type="button"
      class="item"
      class:on={i === selected}
      role="option"
      aria-selected={i === selected}
      onclick={() => onpick(c)}
    >
      <span class="name">/{c.name}</span>
      <span class="hint">{c.hint}</span>
    </button>
  {/each}
</div>

<style>
  .menu {
    position: absolute;
    bottom: calc(100% + 0.5rem);
    left: 0;
    z-index: 30;
    width: min(26rem, calc(100vw - 2rem));
    padding: 0.25rem;
    background: var(--bg);
    border: 1px solid var(--rule-strong);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    animation: rise 130ms ease;
  }

  .item {
    display: flex;
    align-items: baseline;
    gap: 0.6rem;
    width: 100%;
    min-height: var(--tap);
    padding: 0.35rem 0.55rem;
    border: 0;
    border-radius: var(--radius-sm);
    background: transparent;
    color: var(--ink);
    font: inherit;
    text-align: left;
    cursor: pointer;
  }
  .item.on,
  .item:hover {
    background: var(--surface);
  }

  .name {
    flex: none;
    color: var(--accent-ink);
    font-size: 0.875rem;
  }
  .hint {
    font-size: 0.75rem;
    color: var(--ink-3);
  }

  @keyframes rise {
    from {
      opacity: 0;
      transform: translateY(4px);
    }
  }

  @media (prefers-reduced-motion: reduce) {
    .menu {
      animation: none;
    }
  }
</style>
