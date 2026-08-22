<script lang="ts">
  import Header from './components/Header.svelte';
  import type { SheetView } from './lib/views';
  import HonestyStrip from './components/HonestyStrip.svelte';
  import Conversation from './components/Conversation.svelte';
  import Composer from './components/Composer.svelte';
  import Sheet from './components/Sheet.svelte';
  import ModelSheet from './components/ModelSheet.svelte';
  import DiagnosticsSheet from './components/DiagnosticsSheet.svelte';
  import TrainingSheet from './components/TrainingSheet.svelte';
  import ByteInspector from './components/ByteInspector.svelte';
  import { untrack } from 'svelte';
  import { chat } from './lib/stores/chat.svelte';
  import { model } from './lib/stores/model.svelte';

  let sheet = $state<SheetView | null>(null);
  let inspecting = $state<string | null>(null);

  const inspectTrace = $derived(inspecting ? chat.traces[inspecting] : undefined);

  // untrack: this runs once on mount. Without it the effect would subscribe to the chat
  // state that `start()` reads and re-run on the first reply.
  $effect(() => {
    untrack(() => {
      chat.start();
      void model.refresh();
    });
    return () => chat.dispose();
  });

  function onkeydown(e: KeyboardEvent) {
    if (e.key === 'Escape' && chat.busy && !sheet && !inspecting) {
      e.preventDefault();
      chat.stop();
    }
  }

  function toggle(view: SheetView) {
    sheet = sheet === view ? null : view;
    inspecting = null;
  }

  function inspect(id: string) {
    sheet = null;
    inspecting = id;
  }

  const SHEET_TITLES: Record<SheetView, { title: string; subtitle: string }> = {
    model: { title: 'Model', subtitle: 'What is loaded, and what else could be' },
    diagnostics: { title: 'Diagnostics', subtitle: 'Measured on the running model' },
    training: { title: 'Training', subtitle: 'Read from the run logs on disk' }
  };
</script>

<svelte:window onkeydown={onkeydown} />

<div class="app">
  <Header open={sheet} onopen={toggle} />
  <HonestyStrip />
  <main>
    <Conversation oninspect={inspect} />
  </main>
  <Composer />
</div>

{#if sheet}
  <Sheet
    title={SHEET_TITLES[sheet].title}
    subtitle={SHEET_TITLES[sheet].subtitle}
    onclose={() => (sheet = null)}
  >
    {#if sheet === 'model'}
      <ModelSheet />
    {:else if sheet === 'diagnostics'}
      <DiagnosticsSheet />
    {:else}
      <TrainingSheet />
    {/if}
  </Sheet>
{/if}

{#if inspecting && inspectTrace}
  <Sheet
    title="Bytes"
    subtitle="Every byte of this reply, as the backend reported it"
    onclose={() => (inspecting = null)}
  >
    <ByteInspector trace={inspectTrace} />
  </Sheet>
{/if}

<style>
  .app {
    display: flex;
    flex-direction: column;
    height: 100dvh;
    max-height: 100dvh;
  }

  main {
    position: relative;
    flex: 1;
    min-height: 0;
    display: flex;
    flex-direction: column;
  }
</style>
