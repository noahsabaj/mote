<script lang="ts">
  // Sits above the composer, in the same slot the "jump to latest" pill uses; Conversation
  // lifts that pill while this is showing so they never overlap.
  import { notices } from '../lib/stores/notice.svelte';
  import Icon from './Icon.svelte';
</script>

{#if notices.current}
  {#key notices.current.id}
    <div class="bar" role="status" aria-live="polite">
      <span class="msg">{notices.current.message}</span>
      {#if notices.current.undo}
        <button class="act" onclick={() => notices.undo()}>Undo</button>
      {/if}
      <button class="close" onclick={() => notices.dismiss()} aria-label="Dismiss">
        <Icon name="close" size={13} />
      </button>
    </div>
  {/key}
{/if}

<style>
  .bar {
    position: absolute;
    left: 50%;
    bottom: 0.5rem;
    z-index: 40;
    transform: translateX(-50%);
    display: flex;
    align-items: center;
    gap: 0.2rem;
    max-width: calc(100vw - 2 * var(--gutter));
    padding: 0.25rem 0.3rem 0.25rem 0.8rem;
    border: 1px solid var(--rule-strong);
    border-radius: 999px;
    background: var(--bg);
    box-shadow: var(--shadow);
    animation: bar-in 140ms ease;
  }

  .msg {
    font-size: 0.8125rem;
    color: var(--ink-2);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .act {
    flex: none;
    min-height: 30px;
    padding: 0 0.6em;
    border: 0;
    border-radius: 999px;
    background: transparent;
    color: var(--accent-ink);
    font-size: 0.8125rem;
    font-weight: 550;
    cursor: pointer;
  }
  .act:hover {
    background: var(--accent-soft);
  }

  .close {
    flex: none;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 30px;
    height: 30px;
    padding: 0;
    border: 0;
    border-radius: 999px;
    background: transparent;
    color: var(--ink-3);
    cursor: pointer;
  }
  .close:hover {
    background: var(--surface);
    color: var(--ink);
  }

  @keyframes bar-in {
    from {
      opacity: 0;
      transform: translate(-50%, 6px);
    }
  }

  @media (max-width: 34rem) {
    .act,
    .close {
      min-height: 40px;
      height: 40px;
    }
    .close {
      width: 40px;
    }
  }
</style>
