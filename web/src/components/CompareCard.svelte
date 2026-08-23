<script lang="ts">
  // Two replies to the same prompt, side by side and blind: which one served you better?
  // (docs/prefs.md, docs/rubric.md). Sources are revealed by Message.svelte after the vote.
  import type { Turn } from '../lib/stores/chat.svelte';
  import { chat } from '../lib/stores/chat.svelte';
  import { prefs } from '../lib/stores/prefs.svelte';
  import type { PairVote } from '../lib/types';
  import Icon from './Icon.svelte';

  let { turn }: { turn: Turn } = $props();

  const pool = $derived([...(turn.samples ?? []), turn]);
  const a = $derived(pool.find((t) => t.id === turn.compare?.aId) ?? null);
  const b = $derived(pool.find((t) => t.id === turn.compare?.bId) ?? null);
  const origin = $derived(turn.compare?.origin ?? 'compare');

  let reason = $state('');
  let rules = $state(false);
  let sending = $state(false);

  async function cast(vote: PairVote | null) {
    if (sending) return;
    sending = true;
    try {
      await chat.vote(turn.id, vote, reason.trim());
    } finally {
      sending = false;
    }
  }

  function showRules() {
    rules = !rules;
    if (rules) void prefs.loadRubric();
  }
</script>

<section class="compare" aria-label="Choose between two replies">
  <p class="lede meta">
    {#if origin === 'retry'}
      Two replies to the same prompt — the one you had, and the retry. Which serves you better?
    {:else}
      Two replies to the same prompt, unlabelled. Which serves you better?
    {/if}
    <button class="quiet inline" aria-pressed={rules} onclick={showRules}>Rules</button>
  </p>
  {#if rules}
    <pre class="rubric">{prefs.rubric?.text ?? 'Loading the rubric…'}</pre>
  {/if}

  <div class="pair">
    <article class="side" aria-label="Reply A">
      <header><span class="tag">A</span></header>
      <p class="prose">{a?.content ?? ''}</p>
    </article>
    <article class="side" aria-label="Reply B">
      <header><span class="tag">B</span></header>
      <p class="prose">{b?.content ?? ''}</p>
    </article>
  </div>

  <div class="verdict">
    <button class="btn accent" disabled={sending} onclick={() => cast('a')}>A is better</button>
    <button class="btn accent" disabled={sending} onclick={() => cast('b')}>B is better</button>
    <button class="btn" disabled={sending} onclick={() => cast('tie')}>Tie</button>
    <button class="btn" disabled={sending} onclick={() => cast('both_bad')}>Both bad</button>
    <button class="quiet" disabled={sending} onclick={() => cast(null)} title="Keep the pair unrated">
      <Icon name="close" size={13} />
      Skip
    </button>
  </div>
  <label class="why">
    <span class="meta">Why? (optional, one line — it is what gets discussed when the rater disagrees)</span>
    <input type="text" bind:value={reason} maxlength="200" placeholder="2 — B caves to a wrong correction" />
  </label>
  {#if prefs.error}
    <p class="meta fail"><Icon name="alert" size={13} /> {prefs.error}</p>
  {/if}
</section>

<style>
  .compare {
    border: 1px solid var(--line);
    border-radius: var(--radius);
    padding: 0.75rem 0.9rem 0.8rem;
    background: var(--surface);
  }
  .lede {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem 0.6rem;
    align-items: center;
    margin: 0 0 0.6rem;
  }
  .rubric {
    max-height: 16rem;
    overflow: auto;
    white-space: pre-wrap;
    font-size: 0.78rem;
    line-height: 1.45;
    margin: 0 0 0.7rem;
    padding: 0.6rem 0.7rem;
    border-radius: calc(var(--radius) - 2px);
    background: var(--bg);
    color: var(--ink-2);
  }
  .pair {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.7rem;
  }
  @media (max-width: 640px) {
    .pair {
      grid-template-columns: 1fr;
    }
  }
  .side {
    border: 1px solid var(--line);
    border-radius: calc(var(--radius) - 2px);
    padding: 0.55rem 0.7rem 0.65rem;
    background: var(--bg);
    min-width: 0;
  }
  .side header {
    margin-bottom: 0.35rem;
  }
  .tag {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    padding: 0.05rem 0.45rem;
    border-radius: 999px;
    border: 1px solid var(--line);
    color: var(--ink-2);
  }
  .prose {
    margin: 0;
    white-space: pre-wrap;
    overflow-wrap: break-word;
    font-size: 0.9375rem;
    line-height: 1.55;
  }
  .verdict {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-top: 0.7rem;
    align-items: center;
  }
  .why {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
    margin-top: 0.55rem;
  }
  .why input {
    font: inherit;
    font-size: 0.875rem;
    padding: 0.4rem 0.55rem;
    border: 1px solid var(--line);
    border-radius: calc(var(--radius) - 2px);
    background: var(--bg);
    color: var(--ink);
  }
  .why input:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 1px;
  }
  .fail {
    margin: 0.4rem 0 0;
    color: var(--danger, #b4413c);
  }
</style>
