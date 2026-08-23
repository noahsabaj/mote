<script lang="ts">
  // The honesty label, always on screen, never editorialised: `status` and `status_note`
  // are printed exactly as the backend sends them.
  import { model } from '../lib/stores/model.svelte';
  import { auth } from '../lib/stores/auth.svelte';
</script>

<div class="strip" role="note">
  {#if auth.required}
    <span class="tag off">locked</span>
    <p>This studio needs its access token before anything here is measured.</p>
  {:else if model.error}
    <span class="tag off">offline</span>
    <p>Backend unreachable — {model.error} Nothing on this screen is measured.</p>
  {:else if model.info}
    <span class="tag">{model.info.status}</span>
    <p>{model.info.status_note}</p>
  {:else if model.loading}
    <span class="tag off">loading</span>
    <p>Reading the checkpoint the backend has open.</p>
  {/if}
</div>

<style>
  .strip {
    display: flex;
    align-items: baseline;
    gap: 0.7rem;
    padding: 0.45rem var(--gutter) 0.6rem;
    border-top: 1px solid var(--rule);
    border-bottom: 1px solid var(--rule);
    background: var(--bg);
  }

  .tag {
    flex: none;
    font-size: 0.6875rem;
    font-weight: 600;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: var(--accent-ink);
    padding-top: 0.12em;
  }
  .tag.off {
    color: var(--ink-3);
  }

  p {
    margin: 0;
    font-size: 0.8125rem;
    line-height: 1.45;
    color: var(--ink-2);
    max-width: 62ch;
  }

  @media (max-width: 34rem) {
    .strip {
      flex-direction: column;
      gap: 0.15rem;
    }
  }
</style>
