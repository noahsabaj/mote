<script lang="ts">
  // Every checkpoint on disk, sortable and filterable (docs/checkpoints.md). The composer's
  // picker offers the handful you actually use; this is where you go to choose from all of them.
  import { model } from '../lib/stores/model.svelte';
  import { chat } from '../lib/stores/chat.svelte';
  import { ckptView } from '../lib/stores/ckptview.svelte';
  import { tip } from '../lib/actions';
  import {
    applyView,
    chipFamilies,
    loadedBpb,
    rowParts,
    displayName,
    SORT_SPECS,
    type SortKey
  } from '../lib/checkpoints';
  import { num } from '../lib/format';
  import Icon from './Icon.svelte';

  $effect(() => {
    void model.refreshCheckpoints();
  });

  const view = $derived(ckptView.view);
  const baseline = $derived(loadedBpb(model.checkpoints));
  const families = $derived(chipFamilies(model.checkpoints));
  const result = $derived(applyView(model.checkpoints, view, baseline));
  const spec = $derived(SORT_SPECS.find((s) => s.key === view.sort) ?? SORT_SPECS[0]);

  async function load(id: string) {
    if (chat.busy) chat.stop();
    await model.load(id);
  }
</script>

{#if model.checkpointError}
  <p class="fail">
    <Icon name="alert" size={14} />
    {model.checkpointError}
  </p>
{/if}

{#if model.checkpoints.length === 0}
  <p class="meta">No checkpoints reported.</p>
{:else}
  <div class="controls">
    <div class="find">
      <Icon name="search" size={14} />
      <input
        type="search"
        value={view.query}
        oninput={(e) => ckptView.setQuery(e.currentTarget.value)}
        placeholder="Filter by run name"
        aria-label="Filter checkpoints by run name"
      />
    </div>

    <div class="sorter">
      <label class="sr-only" for="ckpt-sort">Sort checkpoints by</label>
      <select
        id="ckpt-sort"
        value={view.sort}
        onchange={(e) => ckptView.setSort(e.currentTarget.value as SortKey)}
      >
        {#each SORT_SPECS as s (s.key)}
          <option value={s.key}>{s.label}</option>
        {/each}
      </select>
      <button
        class="quiet dir"
        onclick={() => ckptView.flip()}
        aria-label="Sort {spec.label}: {view.dir === 'asc' ? spec.asc : spec.desc}. Click to reverse."
        use:tip={view.dir === 'asc' ? spec.asc : spec.desc}
      >
        <span class="arrow" aria-hidden="true">{view.dir === 'asc' ? '↑' : '↓'}</span>
      </button>
    </div>
  </div>

  <div class="chips">
    <button class="chip" class:on={view.evaluated} onclick={() => ckptView.toggleEvaluated()}>
      Evaluated
    </button>
    <button
      class="chip"
      class:on={view.better && baseline !== null}
      disabled={baseline === null}
      onclick={() => ckptView.toggleBetter()}
      title={baseline === null
        ? 'The loaded checkpoint has no bits/byte of its own to compare against'
        : `Better than the loaded ${num(baseline, 3)} bits/byte`}
    >
      Beats loaded
    </button>
    {#each families as f (f)}
      <button class="chip" class:on={view.families.includes(f)} onclick={() => ckptView.toggleFamily(f)}>
        {f}
      </button>
    {/each}
  </div>

  {#if result.filtering}
    <p class="count">
      <span>{result.shown.length} of {model.checkpoints.length} shown</span>
      <button class="quiet inline" onclick={() => ckptView.clearFilters()}>Clear</button>
    </p>
  {/if}

  {#if result.shown.length === 0}
    <p class="meta none">Nothing matches these filters.</p>
  {:else}
    <ul class="ckpts">
      {#each result.shown as c (c.id)}
        <li class:loaded={c.loaded}>
          <div class="who">
            <span class="id">{displayName(c.id)}</span>
            <!-- One span per part so a narrow row breaks between them and never inside a date. -->
            <span class="meta">
              {#each rowParts(c, view.sort) as part, i (part)}
                <span class="part">{i > 0 ? '· ' : ''}{part}</span>{' '}
              {/each}
            </span>
          </div>
          <div class="acts">
            {#if c.loaded}
              <span class="badge">loaded</span>
            {:else}
              <button
                class="btn act-load"
                onclick={() => load(c.id)}
                disabled={model.busy}
                aria-label="Load checkpoint {displayName(c.id)}"
                use:tip={'Load'}
              >
                <span class="act-word">{model.swapping === c.id ? 'Loading…' : 'Load'}</span>
                <span class="act-icon" aria-hidden="true"><Icon name="download" size={14} /></span>
              </button>
            {/if}
            {#if c.challenger}
              <span class="badge vs">challenger</span>
              <button
                class="quiet"
                onclick={() => model.clearChallenger()}
                aria-label="Clear the challenger"
                use:tip={'Clear challenger'}
              >
                <span class="act-word">Clear</span>
                <span class="act-icon" aria-hidden="true"><Icon name="close" size={14} /></span>
              </button>
            {:else if !c.loaded}
              <button
                class="quiet"
                onclick={() => model.setChallenger(c.id)}
                disabled={model.busy || model.challengerLoading !== null}
                aria-label="Load {displayName(c.id)} as the challenger"
                use:tip={"Set as challenger"}
              >
                <span class="act-word">
                  {model.challengerLoading === c.id ? 'Loading…' : 'Challenger'}
                </span>
                <span class="act-icon" aria-hidden="true"><Icon name="copy" size={14} /></span>
              </button>
            {/if}
          </div>
        </li>
      {/each}
    </ul>
  {/if}

  <p class="meta foot">
    Loading a checkpoint swaps the served model. Generation is refused while the swap runs. A
    challenger stays loaded beside it: Compare and arena mode draw their second reply from it, blind.
  </p>
{/if}

<style>
  .fail {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    margin: 0 0 0.8rem;
    font-size: 0.875rem;
    color: var(--accent-ink);
  }

  .controls {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.5rem;
  }

  .find {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    flex: 1 1 auto;
    min-width: 0;
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
    min-height: var(--tap);
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

  .sorter {
    display: flex;
    align-items: center;
    gap: 0.15rem;
    flex: none;
  }

  /* Native, so a phone gets its own picker — but wearing the app's own border and type
     rather than the platform's, which at default styling looks like a foreign object. */
  select {
    min-height: var(--tap);
    max-width: 8.5rem;
    padding: 0 1.5rem 0 0.5rem;
    border: 1px solid var(--rule);
    border-radius: var(--radius-sm);
    background: var(--bg);
    color: var(--ink);
    font: inherit;
    font-size: 0.875rem;
    cursor: pointer;
    appearance: none;
    background-image: linear-gradient(45deg, transparent 50%, currentcolor 50%),
      linear-gradient(135deg, currentcolor 50%, transparent 50%);
    background-position: calc(100% - 0.75rem) 55%, calc(100% - 0.5rem) 55%;
    background-size: 4px 4px, 4px 4px;
    background-repeat: no-repeat;
  }
  select:hover {
    background-color: var(--surface);
  }
  select:focus-visible {
    border-color: var(--accent-line);
    outline: none;
    box-shadow: 0 0 0 3px var(--accent-soft);
  }

  .dir {
    min-width: var(--tap);
    padding: 0 0.35em;
  }
  .arrow {
    font-size: 0.9375rem;
    line-height: 1;
  }

  .chips {
    display: flex;
    flex-wrap: wrap;
    gap: 0.3rem;
    margin-bottom: 0.55rem;
  }

  .chip {
    min-height: 28px;
    padding: 0 0.6em;
    border: 1px solid var(--rule);
    border-radius: 999px;
    background: transparent;
    color: var(--ink-2);
    font: inherit;
    font-size: 0.75rem;
    cursor: pointer;
  }
  .chip:hover:not(:disabled) {
    background: var(--surface);
  }
  .chip.on {
    border-color: var(--accent-line);
    background: var(--accent-soft);
    color: var(--accent-ink);
  }
  .chip:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }

  /* A filter that persists across reloads can hide rows you have forgotten about, so the
     sheet says how many and offers one tap back. */
  .count {
    display: flex;
    align-items: center;
    gap: 0.3rem;
    margin: 0 0 0.4rem;
    font-size: 0.75rem;
    color: var(--ink-3);
    font-variant-numeric: tabular-nums;
  }
  .inline {
    min-height: 0;
    padding: 0 0.2em;
    font-size: 0.75rem;
    text-decoration: underline;
    text-underline-offset: 2px;
  }

  .none {
    margin: 0.4rem 0 0;
    font-size: 0.8125rem;
    color: var(--ink-3);
  }

  .ckpts {
    list-style: none;
    margin: 0;
    padding: 0;
  }

  .ckpts li {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.6rem;
    padding: 0.7rem 0;
    border-top: 1px solid var(--rule);
  }
  .ckpts li:first-child {
    border-top: 0;
  }

  .who {
    min-width: 0;
  }

  .id {
    display: block;
    font-size: 0.875rem;
    font-family: var(--font-mono);
    overflow-wrap: anywhere;
  }
  .loaded .id {
    color: var(--accent-ink);
  }

  .who :global(.meta) {
    display: block;
    margin-top: 0.1rem;
    font-size: 0.75rem;
  }
  .part {
    display: inline-block;
    white-space: nowrap;
  }

  .acts {
    display: flex;
    align-items: center;
    gap: 0.2rem;
    flex: none;
  }

  .badge {
    flex: none;
    font-size: 0.6875rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--ink-3);
  }
  .badge.vs {
    color: var(--accent-ink);
  }

  /* Word on a laptop, icon on a phone — the same trade the header makes with its surfaces. */
  .act-icon {
    display: none;
  }

  .foot {
    margin: 0.75rem 0 0;
    font-size: 0.75rem;
    line-height: 1.5;
  }

  @media (max-width: 34rem) {
    .act-word {
      position: absolute;
      width: 1px;
      height: 1px;
      padding: 0;
      margin: -1px;
      overflow: hidden;
      clip-path: inset(50%);
      white-space: nowrap;
    }
    .act-icon {
      display: block;
    }
    .acts :global(button) {
      min-width: 40px;
      min-height: 40px;
      padding: 0;
      justify-content: center;
    }
    .chip {
      min-height: 34px;
    }
  }
</style>
