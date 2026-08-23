<script lang="ts">
  // The waiting items, below the reply in flight.
  //
  // Reordering is measured rather than animated: as the pointer crosses a neighbour's
  // midpoint the item swaps places for real, so what you see during the drag is the order
  // that will run. One pointer path covers mouse, pen and touch; the arrow keys on each
  // handle cover everything else.

  import { queue } from '../lib/stores/queue.svelte';
  import QueuedTurn from './QueuedTurn.svelte';

  let list = $state<HTMLElement | null>(null);
  let dragId = $state<string | null>(null);

  function grab(e: PointerEvent, id: string) {
    if (e.pointerType === 'mouse' && e.button !== 0) return;
    // No preventDefault: the handle must still take focus, or clicking it would cost you
    // the keyboard reorder it exists to advertise.
    (e.currentTarget as HTMLElement).focus();
    dragId = id;
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', drop);
    window.addEventListener('pointercancel', drop);
  }

  function onMove(e: PointerEvent) {
    if (!dragId || !list) return;
    const rows = [...list.querySelectorAll<HTMLElement>('[data-qid]')];
    const from = rows.findIndex((r) => r.dataset.qid === dragId);
    if (from < 0) return;
    for (let i = 0; i < rows.length; i += 1) {
      if (i === from) continue;
      const r = rows[i].getBoundingClientRect();
      const mid = r.top + r.height / 2;
      if ((i < from && e.clientY < mid) || (i > from && e.clientY > mid)) {
        queue.moveTo(dragId, i);
        return;
      }
    }
  }

  function drop() {
    dragId = null;
    window.removeEventListener('pointermove', onMove);
    window.removeEventListener('pointerup', drop);
    window.removeEventListener('pointercancel', drop);
  }

  // A drag interrupted by the component going away would otherwise leave listeners behind.
  $effect(() => () => drop());
</script>

{#if queue.items.length > 0}
  <div
    class="queue"
    class:dragging={dragId !== null}
    bind:this={list}
    role="group"
    aria-label="Waiting to run"
  >
    {#each queue.items as item, i (item.id)}
      <QueuedTurn
        {item}
        index={i}
        total={queue.items.length}
        isLast={i === queue.items.length - 1}
        dragging={dragId === item.id}
        ongrab={grab}
      />
    {/each}
  </div>
{/if}

<style>
  .queue.dragging {
    user-select: none;
    cursor: grabbing;
  }
</style>
