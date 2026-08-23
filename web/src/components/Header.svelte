<script lang="ts">
  import Wordmark from './Wordmark.svelte';
  import Icon from './Icon.svelte';
  import StatusBadge from './StatusBadge.svelte';
  import { chat } from '../lib/stores/chat.svelte';
  import { ui } from '../lib/stores/ui.svelte';
  import { dismissable } from '../lib/actions';
  import { when } from '../lib/format';
  import type { SheetView } from '../lib/views';

  let { open, onopen }: { open: SheetView | null; onopen: (v: SheetView) => void } = $props();

  let menuOpen = $state(false);
  let trigger = $state<HTMLElement | null>(null);

  const current = $derived(chat.index.find((c) => c.id === chat.id));
  const title = $derived(current?.title ?? 'New conversation');

  function pick(id: string) {
    menuOpen = false;
    chat.open(id);
  }

  function fresh() {
    menuOpen = false;
    chat.newConversation();
  }
</script>

<header>
  <div class="left">
    <Wordmark />
    <div class="menu-anchor">
      <button
        bind:this={trigger}
        class="quiet convo"
        aria-expanded={menuOpen}
        aria-haspopup="true"
        onclick={() => (menuOpen = !menuOpen)}
      >
        <span class="convo-title">{title}</span>
        <Icon name="chevron" size={13} />
      </button>

      {#if menuOpen}
        <div
          class="menu"
          aria-label="Conversations"
          use:dismissable={{ onDismiss: () => (menuOpen = false), trigger }}
        >
          <button class="item new" onclick={fresh}>
            <Icon name="plus" size={14} />
            New conversation
          </button>
          {#if chat.index.length}
            <div class="rule"></div>
            <ul>
              {#each chat.index as c (c.id)}
                <li class:current={c.id === chat.id}>
                  <button class="item" onclick={() => pick(c.id)} aria-current={c.id === chat.id}>
                    <span class="label">{c.title}</span>
                    <span class="meta stamp">{when(new Date(c.updatedAt).toISOString())}</span>
                  </button>
                  <button
                    class="quiet del"
                    aria-label="Delete conversation “{c.title}”"
                    onclick={() => chat.deleteConversation(c.id)}
                  >
                    <Icon name="trash" size={14} />
                  </button>
                </li>
              {/each}
            </ul>
          {/if}
        </div>
      {/if}
    </div>
  </div>

  <nav aria-label="Model surfaces">
    <StatusBadge onopen={() => onopen('diagnostics')} />
    <button class="quiet" aria-pressed={open === 'model'} onclick={() => onopen('model')}>
      Model
    </button>
    <button
      class="quiet"
      aria-pressed={open === 'diagnostics'}
      onclick={() => onopen('diagnostics')}
    >
      Diagnostics
    </button>
    <button class="quiet" aria-pressed={open === 'training'} onclick={() => onopen('training')}>
      Training
    </button>
    <button
      class="quiet theme"
      onclick={() => ui.cycleTheme()}
      title={`Theme: ${ui.theme} — click to change`}
      aria-label={`Theme: ${ui.theme}. Click to change.`}
    >
      <Icon name={ui.theme === 'light' ? 'sun' : ui.theme === 'dark' ? 'moon' : 'auto'} size={15} />
    </button>
  </nav>
</header>

<style>
  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.75rem;
    padding: 0.6rem var(--gutter) 0.55rem;
  }

  .left {
    display: flex;
    align-items: center;
    gap: 0.35rem;
    min-width: 0;
  }

  .menu-anchor {
    position: relative;
    min-width: 0;
  }

  .convo {
    max-width: 15rem;
  }
  .convo :global(svg) {
    transform: rotate(90deg);
    transition: transform 140ms ease;
  }
  .convo[aria-expanded='true'] :global(svg) {
    transform: rotate(-90deg);
  }
  .convo-title {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  nav {
    display: flex;
    align-items: center;
    gap: 0.1rem;
    flex: none;
  }
  .theme {
    margin-left: 0.35rem;
    padding: 0 0.45em;
  }

  .menu {
    position: absolute;
    top: calc(100% + 0.4rem);
    left: 0;
    z-index: 30;
    width: min(21rem, calc(100vw - 2rem));
    max-height: min(24rem, 70vh);
    overflow-y: auto;
    padding: 0.3rem;
    background: var(--bg);
    border: 1px solid var(--rule-strong);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    animation: pop 120ms ease;
  }

  .rule {
    height: 1px;
    background: var(--rule);
    margin: 0.3rem 0.15rem;
  }

  ul {
    list-style: none;
    margin: 0;
    padding: 0;
  }

  li {
    display: flex;
    align-items: center;
    gap: 0.2rem;
    border-radius: var(--radius-sm);
  }
  li:hover {
    background: var(--surface);
  }
  li.current .label {
    color: var(--accent-ink);
  }

  .item {
    flex: 1;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    min-width: 0;
    min-height: 32px;
    padding: 0.25rem 0.5rem;
    border: 0;
    border-radius: var(--radius-sm);
    background: transparent;
    text-align: left;
    font-size: 0.875rem;
    cursor: pointer;
  }
  .item.new {
    width: 100%;
    color: var(--ink);
  }
  .item.new:hover {
    background: var(--surface);
  }

  .label {
    flex: 1;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .stamp {
    flex: none;
    font-size: 0.75rem;
  }

  .del {
    flex: none;
    opacity: 0;
  }
  li:hover .del,
  .del:focus-visible {
    opacity: 1;
  }

  @keyframes pop {
    from {
      opacity: 0;
      transform: translateY(-3px);
    }
  }

  @media (max-width: 34rem) {
    .convo-title,
    .stamp {
      display: none;
    }
    nav :global(button) {
      padding: 0 0.3em;
      font-size: 0.78125rem;
    }
    .del {
      opacity: 1;
    }
  }
</style>
