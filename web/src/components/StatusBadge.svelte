<script lang="ts">
  // The honesty label, compressed to the one word the backend sends. The sentence behind it
  // is never editorialised — it moves to Diagnostics, one click away, instead of sitting
  // under the header on every screen.
  import { model } from '../lib/stores/model.svelte';
  import { auth } from '../lib/stores/auth.svelte';
  import { statusNote } from '../lib/status';

  let { onopen }: { onopen: () => void } = $props();

  const label = $derived(
    auth.required
      ? 'locked'
      : model.error
        ? 'offline'
        : model.info
          ? model.info.status
          : model.loading
            ? 'loading'
            : null
  );
  const off = $derived(auth.required || !!model.error || !model.info);
</script>

{#if label}
  <button
    class="badge"
    class:off
    onclick={onopen}
    title={statusNote()}
    aria-label="Model status: {label}. {statusNote()} Open diagnostics."
  >
    {label}
  </button>
{/if}

<style>
  .badge {
    flex: none;
    margin-right: 0.55rem;
    padding: 0.1rem 0.45em;
    border: 1px solid var(--rule-strong);
    border-radius: 999px;
    background: transparent;
    font-size: 0.625rem;
    font-weight: 600;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: var(--accent-ink);
    cursor: pointer;
  }
  .badge:hover {
    background: var(--surface);
  }
  .badge.off {
    color: var(--ink-3);
  }
</style>
