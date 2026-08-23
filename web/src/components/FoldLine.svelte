<script lang="ts">
  // The line in the conversation where Mote's view starts. Everything above it was folded into a
  // compaction card (docs/context.md) — the exact bytes are shown here and can be edited; the
  // edited version rides into the next reply in place of the generated one.
  import Icon from './Icon.svelte';
  import type { FoldInfo } from '../lib/types';

  let {
    fold,
    card,
    onedit,
    onunfold
  }: {
    fold: FoldInfo;
    /** the user's edited card, or null when the generated one is in use */
    card: string | null;
    onedit: (card: string | null) => void;
    onunfold: () => void;
  } = $props();

  let open = $state(false);
  let draft = $state('');

  const shown = $derived(card ?? fold.card);
  const bytes = $derived(new TextEncoder().encode(shown).length);

  function toggle() {
    open = !open;
    if (open) draft = shown;
  }
</script>

<div class="fold" role="separator" aria-label="Mote's view starts here">
  <div class="rule">
    <span class="label">
      <Icon name="structure" size={13} />
      Mote's view starts here · {fold.turns} earlier {fold.turns === 1 ? 'turn' : 'turns'} folded into a
      {bytes}-byte card{card !== null ? ' (edited)' : ''}
    </span>
    <span class="actions">
      <button class="quiet" onclick={toggle} aria-expanded={open}>{open ? 'Hide card' : 'Show card'}</button>
      <button class="quiet" onclick={onunfold} title="Send the whole conversation next time; older turns are dropped, not folded">
        Unfold
      </button>
    </span>
  </div>
  {#if open}
    <div class="card">
      <label class="sr-only" for="fold-card">Compaction card</label>
      <textarea id="fold-card" bind:value={draft} rows="4" spellcheck="false"></textarea>
      <div class="row">
        <span class="meta">These bytes are merged into the first kept user turn. What is not in them, Mote cannot see.</span>
        <span class="buttons">
          {#if card !== null}
            <button class="quiet" onclick={() => { onedit(null); draft = fold.card; }}>Reset</button>
          {/if}
          <button class="quiet" disabled={draft === shown} onclick={() => onedit(draft)}>Use my version</button>
        </span>
      </div>
    </div>
  {/if}
</div>

<style>
  .fold {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    margin: 0.9rem 0;
  }
  .rule {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.75rem;
    flex-wrap: wrap;
    padding-top: 0.55rem;
    border-top: 1px dashed var(--accent-line);
  }
  .label {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.8rem;
    color: var(--ink-2);
  }
  .actions {
    display: inline-flex;
    gap: 0.25rem;
  }
  .card {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    padding: 0.75rem;
    border: 1px solid var(--accent-line);
    border-radius: 10px;
    background: var(--accent-soft);
  }
  textarea {
    width: 100%;
    resize: vertical;
    font: inherit;
    font-size: 0.85rem;
    line-height: 1.45;
    color: var(--ink);
    background: var(--bg);
    border: 1px solid var(--accent-line);
    border-radius: 8px;
    padding: 0.5rem 0.6rem;
  }
  .row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.75rem;
    flex-wrap: wrap;
  }
  .meta {
    font-size: 0.75rem;
    color: var(--ink-3);
  }
  .buttons {
    display: inline-flex;
    gap: 0.25rem;
  }
</style>
