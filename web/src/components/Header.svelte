<script lang="ts">
  import Wordmark from './Wordmark.svelte';
  import Icon from './Icon.svelte';
  import { chat } from '../lib/stores/chat.svelte';
  import { ui } from '../lib/stores/ui.svelte';
  import { clock } from '../lib/clock.svelte';
  import { dismissable, tip } from '../lib/actions';
  import { ago } from '../lib/format';
  import type { SheetView } from '../lib/views';

  let { open, onopen }: { open: SheetView | null; onopen: (v: SheetView) => void } = $props();

  let trigger = $state<HTMLElement | null>(null);
  let field = $state<HTMLInputElement | null>(null);
  let query = $state('');
  let renaming = $state<string | null>(null);
  let draft = $state('');
  let cancelled = false;

  // The filter only earns its place once the list is long enough to scroll past.
  const showFilter = $derived(chat.index.length > 6);
  const listed = $derived.by(() => {
    const q = query.trim().toLowerCase();
    return q ? chat.index.filter((c) => c.title.toLowerCase().includes(q)) : chat.index;
  });

  const mac = typeof navigator !== 'undefined' && /mac/i.test(navigator.userAgent);
  const MOD = mac ? '⌘' : 'Ctrl+';
  const ALT = mac ? '⌥' : 'Alt+';

  $effect(() => {
    if (ui.switcher && field) field.focus();
  });
  $effect(() => {
    if (!ui.switcher) {
      query = '';
      renaming = null;
    }
  });

  function pick(id: string) {
    ui.switcher = false;
    chat.open(id);
  }

  function fresh() {
    ui.switcher = false;
    chat.newConversation();
  }

  function startRename(id: string, title: string) {
    draft = title;
    renaming = id;
  }

  const takeField = (node: HTMLInputElement) => {
    node.focus();
    node.select();
  };

  // Enter and clicking away both commit; only Escape throws the edit away. Blur fires again
  // as the field unmounts, so the guard keeps it to one commit.
  function endRename(id: string) {
    if (renaming !== id) return;
    renaming = null;
    if (!cancelled) chat.rename(id, draft);
    cancelled = false;
  }

  function onRenameKey(e: KeyboardEvent, id: string) {
    if (e.key === 'Enter') {
      e.preventDefault();
      endRename(id);
    } else if (e.key === 'Escape') {
      e.stopPropagation(); // do not close the whole menu
      cancelled = true;
      endRename(id);
    }
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
        aria-expanded={ui.switcher}
        aria-haspopup="true"
        onclick={() => (ui.switcher = !ui.switcher)}
      >
        <span class="convo-title">{chat.title}</span>
        <Icon name="chevron" size={13} />
      </button>

      {#if ui.switcher}
        <div
          class="menu"
          aria-label="Conversations"
          use:dismissable={{ onDismiss: () => (ui.switcher = false), trigger }}
        >
          <button class="item new" onclick={fresh}>
            <Icon name="plus" size={14} />
            New conversation
          </button>

          {#if showFilter}
            <div class="find">
              <Icon name="search" size={14} />
              <input
                bind:this={field}
                bind:value={query}
                type="search"
                placeholder="Filter by title"
                aria-label="Filter conversations by title"
              />
            </div>
          {/if}

          {#if chat.index.length}
            <div class="rule"></div>
            {#if listed.length === 0}
              <p class="meta none">Nothing matches “{query.trim()}”.</p>
            {/if}
            <ul>
              {#each listed as c (c.id)}
                <li class:current={c.id === chat.id}>
                  {#if renaming === c.id}
                    <input
                      class="rename"
                      bind:value={draft}
                      use:takeField
                      aria-label="Rename conversation"
                      onkeydown={(e) => onRenameKey(e, c.id)}
                      onblur={() => endRename(c.id)}
                    />
                  {:else}
                    <button class="item" onclick={() => pick(c.id)} aria-current={c.id === chat.id}>
                      <span class="label">{c.title}</span>
                      <span class="meta stamp">{ago(new Date(c.updatedAt).toISOString(), clock.now)}</span>
                    </button>
                    <button
                      class="quiet act"
                      aria-label="Rename conversation “{c.title}”"
                      onclick={() => startRename(c.id, c.title)}
                      use:tip={'Rename'}
                    >
                      <Icon name="pencil" size={14} />
                    </button>
                    <button
                      class="quiet act"
                      aria-label="Delete conversation “{c.title}”"
                      onclick={() => chat.deleteConversation(c.id)}
                      use:tip={'Delete'}
                    >
                      <Icon name="trash" size={14} />
                    </button>
                  {/if}
                </li>
              {/each}
            </ul>
          {/if}

          <div class="rule"></div>
          <div class="foot">
            <span class="meta">Export this one</span>
            <span class="formats">
              <button class="quiet" disabled={chat.isEmpty} onclick={() => chat.exportAs(chat.id, 'md')}>
                <Icon name="download" size={13} />
                Markdown
              </button>
              <button class="quiet" disabled={chat.isEmpty} onclick={() => chat.exportAs(chat.id, 'json')}>
                <Icon name="download" size={13} />
                JSON
              </button>
            </span>
          </div>
          <p class="meta keys">
            {MOD}K conversations · {ALT}1/2/3 panels · Esc to the composer
          </p>
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
    width: min(23rem, calc(100vw - 2rem));
    max-height: min(28rem, 70vh);
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

  .find {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    margin: 0.25rem 0.15rem 0;
    padding: 0 0.5rem;
    border: 1px solid var(--rule);
    border-radius: var(--radius-sm);
    color: var(--ink-3);
  }
  .find:focus-within {
    border-color: var(--accent-line);
  }
  .find input {
    flex: 1;
    min-width: 0;
    min-height: 32px;
    border: 0;
    outline: none;
    background: transparent;
    color: var(--ink);
    font: inherit;
    font-size: 0.875rem;
  }
  .find input::-webkit-search-cancel-button {
    filter: grayscale(1);
  }

  .none {
    margin: 0.4rem 0.65rem 0.5rem;
    font-size: 0.8125rem;
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

  .rename {
    flex: 1;
    min-width: 0;
    min-height: 32px;
    margin: 0 0.15rem;
    padding: 0 0.45rem;
    border: 1px solid var(--accent-line);
    border-radius: var(--radius-sm);
    outline: none;
    background: var(--bg);
    color: var(--ink);
    font: inherit;
    font-size: 0.875rem;
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

  .act {
    flex: none;
    opacity: 0;
  }
  li:hover .act,
  .act:focus-visible {
    opacity: 1;
  }

  .foot {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.5rem;
    padding: 0.1rem 0.15rem 0.1rem 0.5rem;
  }
  .foot .meta {
    font-size: 0.75rem;
  }
  .formats {
    display: inline-flex;
    gap: 0.1rem;
  }

  .keys {
    margin: 0.35rem 0.5rem 0.25rem;
    font-size: 0.6875rem;
    color: var(--ink-3);
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
    /* Nothing may hide behind hover on touch, and both row controls need a real target. */
    .act {
      opacity: 1;
      width: 40px;
      min-height: 40px;
    }
    .item,
    .rename,
    .find input {
      min-height: 40px;
    }
    .keys {
      display: none;
    }

    /* Anchored to the trigger the menu would run off the right edge, so on a phone it spans
       the viewport instead. */
    .menu {
      position: fixed;
      top: calc(env(safe-area-inset-top) + 3rem);
      left: max(0.75rem, env(safe-area-inset-left));
      right: max(0.75rem, env(safe-area-inset-right));
      width: auto;
      max-height: min(28rem, 62vh);
    }
  }
</style>
