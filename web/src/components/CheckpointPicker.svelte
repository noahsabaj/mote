<script lang="ts">
  // The composer's checkpoint pill and the short list behind it (docs/checkpoints.md).
  //
  // This is the fast path: the handful of checkpoints you actually swap between, in the corner
  // you type from. Rows do one thing — load — so the whole list stays tappable; the challenger
  // gets a line of its own underneath, and the full sortable list is one link away.
  import { model } from '../lib/stores/model.svelte';
  import { chat } from '../lib/stores/chat.svelte';
  import { ui } from '../lib/stores/ui.svelte';
  import { layout } from '../lib/layout.svelte';
  import { dismissable } from '../lib/actions';
  import { displayName, shortlist } from '../lib/checkpoints';
  import { num } from '../lib/format';
  import Icon from './Icon.svelte';
  import Sheet from './Sheet.svelte';

  let open = $state(false);
  let trigger = $state<HTMLElement | null>(null);

  $effect(() => {
    if (open) void model.refreshCheckpoints();
  });

  const loaded = $derived(model.checkpoints.find((c) => c.loaded));
  const challenger = $derived(model.checkpoints.find((c) => c.challenger));
  const rows = $derived(shortlist(model.checkpoints, model.recents));
  // Before the first list arrives the pill still has something true to say: /api/model names
  // the checkpoint it is serving.
  const label = $derived(displayName(loaded?.id ?? model.info?.checkpoint.path ?? ''));

  async function pick(id: string) {
    open = false;
    if (id === loaded?.id) return;
    if (chat.busy) chat.stop();
    await model.load(id);
  }

  function browse() {
    open = false;
    ui.sheet = 'checkpoints';
  }
</script>

{#snippet list()}
  {#if rows.length === 0}
    <p class="empty">No checkpoints reported.</p>
  {:else}
    <ul>
      {#each rows as c (c.id)}
        <li>
          <button class="row" onclick={() => pick(c.id)} aria-current={c.loaded} disabled={model.busy}>
            <span class="text">
              <span class="name">{displayName(c.id)}</span>
              <span class="sub">
                step {c.step.toLocaleString()} ·
                {c.val_bpb === null ? 'not evaluated yet' : `${num(c.val_bpb, 3)} bits/byte`}
              </span>
            </span>
            {#if c.loaded}
              <span class="tick" aria-hidden="true"><Icon name="check" size={15} /></span>
            {/if}
          </button>
        </li>
      {/each}
    </ul>
  {/if}

  <div class="rule"></div>

  <!-- The two footer rows are siblings of each other and of the list, so every line in the
       popover starts its text at the same x. -->
  {#if challenger}
    <div class="vs">
      <span class="vs-text">
        Challenger <span class="vs-name">{displayName(challenger.id)}</span>
      </span>
      <button class="quiet inline" onclick={() => model.clearChallenger()}>Clear</button>
    </div>
  {:else}
    <button class="quiet wide" onclick={browse}>
      Set a challenger
      <Icon name="chevron" size={13} />
    </button>
  {/if}

  <button class="quiet wide" onclick={browse}>
    All {model.checkpoints.length} checkpoints
    <Icon name="chevron" size={13} />
  </button>
{/snippet}

<div class="anchor">
  <button
    bind:this={trigger}
    class="quiet pill"
    aria-expanded={open}
    aria-haspopup="true"
    onclick={() => (open = !open)}
  >
    <Icon name="model" size={14} />
    <span class="label">{label || 'Checkpoint'}</span>
    <Icon name="chevron" size={12} />
  </button>

  {#if open && !layout.phone}
    <div class="panel" use:dismissable={{ onDismiss: () => (open = false), trigger }}>
      {@render list()}
    </div>
  {/if}
</div>

{#if open && layout.phone}
  <Sheet title="Checkpoint" subtitle="What Mote is answering as" onclose={() => (open = false)}>
    {@render list()}
  </Sheet>
{/if}

<style>
  .anchor {
    position: relative;
    min-width: 0;
  }

  .pill {
    max-width: min(14rem, 42vw);
  }
  .label {
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-family: var(--font-mono);
    font-size: 0.8125rem;
  }
  .pill :global(svg):last-child {
    transform: rotate(90deg);
    transition: transform 140ms ease;
  }
  .pill[aria-expanded='true'] :global(svg):last-child {
    transform: rotate(-90deg);
  }

  .panel {
    position: absolute;
    bottom: calc(100% + 0.5rem);
    left: 0;
    z-index: 30;
    width: min(20rem, calc(100vw - 2rem));
    max-height: min(60vh, 30rem);
    overflow-y: auto;
    overscroll-behavior: contain;
    padding: 0.3rem;
    background: var(--bg);
    border: 1px solid var(--rule-strong);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    animation: rise 130ms ease;
  }

  ul {
    list-style: none;
    margin: 0;
    padding: 0;
  }

  .row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    width: 100%;
    min-height: 40px;
    padding: 0.35rem 0.5rem;
    border: 0;
    border-radius: var(--radius-sm);
    background: transparent;
    color: var(--ink);
    font: inherit;
    text-align: left;
    cursor: pointer;
  }
  .row:hover:not(:disabled) {
    background: var(--surface);
  }
  .row:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .text {
    flex: 1;
    min-width: 0;
  }
  .name {
    display: block;
    font-family: var(--font-mono);
    font-size: 0.875rem;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .row[aria-current='true'] .name {
    color: var(--accent-ink);
  }
  .sub {
    display: block;
    margin-top: 0.05rem;
    font-size: 0.75rem;
    color: var(--ink-3);
    font-variant-numeric: tabular-nums;
  }

  .tick {
    flex: none;
    color: var(--accent-ink);
  }

  .empty {
    margin: 0.5rem 0.55rem;
    font-size: 0.8125rem;
    color: var(--ink-3);
  }

  .rule {
    height: 1px;
    background: var(--rule);
    margin: 0.3rem 0.15rem;
  }

  /* The challenger is a second engine, not a second row action, so it gets a line rather than
     a button on every checkpoint — the same place Claude's picker keeps Effort. */
  .vs {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.4rem;
    min-height: 36px;
    /* Matches the buttons' own padding, so the text sits on the same rail as every row. */
    padding: 0 0.5rem;
  }
  .vs-text {
    font-size: 0.8125rem;
    color: var(--ink-2);
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .vs-name {
    font-family: var(--font-mono);
    color: var(--accent-ink);
  }
  .inline {
    min-height: 0;
    padding: 0 0.2em;
    font-size: 0.75rem;
    text-decoration: underline;
    text-underline-offset: 2px;
  }

  .wide {
    width: 100%;
    justify-content: space-between;
    min-height: 36px;
    padding: 0 0.5rem;
    font-size: 0.8125rem;
  }
  .wide :global(svg) {
    color: var(--ink-3);
  }

  @keyframes rise {
    from {
      opacity: 0;
      transform: translateY(4px);
    }
  }

  @media (max-width: 34rem) {
    .row,
    .wide,
    .vs {
      min-height: 44px;
    }
    /* the pill was 28 px tall beside a 40 px Sampling button (QA 2026-08-24) */
    .pill {
      min-height: 40px;
    }
  }
</style>
