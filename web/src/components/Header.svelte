<script lang="ts">
  import Wordmark from './Wordmark.svelte';
  import Icon from './Icon.svelte';
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

  // Below 34rem the labels are hidden and only the icon shows, so the header still fits a
  // phone without the wordmark and the conversation title fighting over the same pixels.
  const SURFACES = [
    { view: 'model', label: 'Model', icon: 'model' },
    { view: 'diagnostics', label: 'Diagnostics', icon: 'diagnostics' },
    { view: 'training', label: 'Training', icon: 'training' }
  ] as const;
</script>

<header>
  <div class="left">
    <span class="brand"><Wordmark /></span>
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
    {#each SURFACES as s (s.view)}
      <button
        class="quiet surface"
        aria-pressed={open === s.view}
        onclick={() => onopen(s.view)}
        title={s.label}
      >
        <span class="surface-icon"><Icon name={s.icon} size={15} /></span>
        <span class="surface-label">{s.label}</span>
      </button>
    {/each}
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
    width: 100%;
    max-width: var(--shell);
    margin: 0 auto;
    padding: 0.6rem var(--gutter) 0.55rem;
  }

  .left {
    display: flex;
    align-items: center;
    gap: 0.35rem;
    flex: 1 1 auto;
    min-width: 0;
  }

  .brand {
    flex: none;
  }

  .menu-anchor {
    position: relative;
    flex: 1 1 auto;
    min-width: 0;
    max-width: 15rem;
  }

  /* Both caps are relative to the shrinking anchor, so a long title ellipsises instead of
     pushing the button out from under it and into the nav. */
  .convo {
    max-width: 100%;
  }
  .convo :global(svg) {
    transform: rotate(90deg);
    transition: transform 140ms ease;
  }
  .convo[aria-expanded='true'] :global(svg) {
    transform: rotate(-90deg);
  }
  .convo-title {
    min-width: 0;
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
  /* Label on a laptop, icon on a phone — never both. */
  .surface-icon {
    display: none;
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

  /* Phone: the wordmark gives way to the conversation title, and the three surfaces become
     icons. Everything still fits at 320px with the longest status word. */
  @media (max-width: 34rem) {
    .brand,
    .stamp {
      display: none;
    }
    .menu-anchor {
      max-width: none;
    }
    .surface-icon {
      display: block;
    }
    .surface-label {
      position: absolute;
      width: 1px;
      height: 1px;
      padding: 0;
      margin: -1px;
      overflow: hidden;
      clip-path: inset(50%);
      white-space: nowrap;
    }
    /* Height is the cheap dimension on a phone: the icons stay narrow so the row still fits
       320px, but every target is 40px tall rather than 28. */
    .surface,
    .theme,
    .convo {
      min-height: 40px;
    }
    nav :global(button) {
      padding: 0 0.32em;
    }
    .del {
      opacity: 1;
    }

    /* Anchored to the trigger the menu would run off the right edge, so on a phone it spans
       the viewport instead. */
    .menu {
      position: fixed;
      top: calc(env(safe-area-inset-top) + 3rem);
      left: max(0.75rem, env(safe-area-inset-left));
      right: max(0.75rem, env(safe-area-inset-right));
      width: auto;
      max-height: min(24rem, 62vh);
    }
  }
</style>
