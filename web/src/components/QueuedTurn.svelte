<script lang="ts">
  // One item waiting behind the reply in flight. It looks like the turn it is about to
  // become, so the queue reads as part of the conversation rather than as a control panel —
  // but it is not a turn yet, and nothing here is persisted, exported or traced.

  import { queue, type QueuedItem } from '../lib/stores/queue.svelte';
  import { chat } from '../lib/stores/chat.svelte';
  import { autosize, tip } from '../lib/actions';
  import Icon from './Icon.svelte';

  let {
    item,
    isLast,
    dragging,
    ongrab
  }: {
    item: QueuedItem;
    /** the bottom-most item carries Interrupt, which belongs to the stream, not to any one item */
    isLast: boolean;
    dragging: boolean;
    ongrab: (e: PointerEvent, id: string) => void;
  } = $props();

  let editing = $state(false);
  let draft = $state('');
  let area = $state<HTMLTextAreaElement | null>(null);

  const changed = $derived(draft.trim() !== item.text);

  function open() {
    draft = item.text;
    editing = true;
    queueMicrotask(() => {
      if (!area) return;
      area.focus();
      area.setSelectionRange(area.value.length, area.value.length);
    });
  }

  function save() {
    if (!changed) {
      editing = false;
      return;
    }
    editing = false;
    queue.edit(item.id, draft);
  }

  function onEditKey(e: KeyboardEvent) {
    if (e.key === 'Escape') {
      // The window handler behind this one stops generation; cancelling an edit must not.
      e.stopPropagation();
      editing = false;
    } else if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      save();
    }
  }

  // Dragging is a pointer gesture, so it is useless to a keyboard and to a screen reader.
  // The same reorder lives on the handle's own arrow keys.
  function onGripKey(e: KeyboardEvent) {
    if (!e.altKey) return;
    if (e.key === 'ArrowUp') {
      e.preventDefault();
      queue.nudge(item.id, -1);
    } else if (e.key === 'ArrowDown') {
      e.preventDefault();
      queue.nudge(item.id, 1);
    }
  }
</script>

<article class="queued" class:dragging data-qid={item.id} aria-label="Queued">
  {#if editing}
    <div class="edit">
      <textarea
        bind:this={area}
        bind:value={draft}
        use:autosize={320}
        rows="1"
        aria-label="Edit this queued message"
        onkeydown={onEditKey}
      ></textarea>
      <div class="edit-foot">
        <button class="btn" onclick={() => (editing = false)}>Cancel</button>
        <button class="btn accent" onclick={save} disabled={!changed}>Save</button>
      </div>
    </div>
  {:else}
    <button class="said" class:cmd={item.command !== null} onclick={open} title="Edit">
      {item.text}
    </button>
    <footer>
      <button
        class="quiet ico grip"
        aria-label="Reorder — hold to drag, or Alt with the arrow keys"
        onpointerdown={(e) => ongrab(e, item.id)}
        onkeydown={onGripKey}
        use:tip={'Drag, or Alt+↑ ↓'}
      >
        <Icon name="grip" size={14} />
      </button>
      {#if isLast}
        <button class="quiet interrupt" onclick={() => chat.stop()}>Interrupt</button>
      {/if}
      <button
        class="quiet ico"
        aria-label="Remove from the queue"
        onclick={() => queue.remove(item.id)}
        use:tip={'Remove'}
      >
        <Icon name="close" size={14} />
      </button>
    </footer>
  {/if}
</article>

<style>
  .queued {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    margin: 0 0 1.9rem;
  }
  .queued.dragging {
    opacity: 0.55;
  }

  /* The same bubble as a sent prompt, because that is what it is about to be. It is a
     button only so that clicking it opens the edit box. */
  .said {
    display: block;
    max-width: 85%;
    padding: 0.6rem 0.9rem;
    border: 0;
    border-radius: var(--radius);
    background: var(--surface);
    color: var(--ink);
    font: inherit;
    font-size: 0.9375rem;
    line-height: 1.55;
    text-align: left;
    white-space: pre-wrap;
    overflow-wrap: break-word;
    cursor: text;
  }
  /* A command is an instruction to the studio, not a prompt for Mote, and the difference
     matters most right here — while it can still be called off. */
  .said.cmd {
    color: var(--accent-ink);
  }

  footer {
    display: flex;
    align-items: center;
    gap: 0.1rem;
    margin-top: 0.25rem;
  }

  .ico {
    justify-content: center;
    width: var(--tap);
    min-height: var(--tap);
    padding: 0;
  }

  .grip {
    /* otherwise a touch drag scrolls the transcript instead of moving the item */
    touch-action: none;
    cursor: grab;
    color: var(--ink-3);
  }

  .interrupt {
    min-height: var(--tap);
    font-size: 0.75rem;
  }

  .edit {
    width: 100%;
  }
  .edit textarea {
    display: block;
    width: 100%;
    resize: none;
    padding: 0.6rem 0.9rem;
    border: 1px solid var(--accent-line);
    border-radius: var(--radius);
    outline: none;
    background: var(--bg);
    color: var(--ink);
    font: inherit;
    font-size: 0.9375rem;
    line-height: 1.55;
    max-height: 320px;
    box-shadow: 0 0 0 3px var(--accent-soft);
  }
  .edit-foot {
    display: flex;
    align-items: center;
    justify-content: flex-end;
    gap: 0.4rem;
    margin-top: 0.5rem;
  }

  @media (hover: none) {
    .ico {
      width: 44px;
      min-height: 44px;
    }
  }
</style>
