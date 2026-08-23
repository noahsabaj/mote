<script lang="ts">
  import Header from './components/Header.svelte';
  import type { SheetView } from './lib/views';
  import Conversation from './components/Conversation.svelte';
  import Composer from './components/Composer.svelte';
  import Sheet from './components/Sheet.svelte';
  import ModelSheet from './components/ModelSheet.svelte';
  import DiagnosticsSheet from './components/DiagnosticsSheet.svelte';
  import TrainingSheet from './components/TrainingSheet.svelte';
  import ByteInspector from './components/ByteInspector.svelte';
  import TokenPrompt from './components/TokenPrompt.svelte';
  import UndoBar from './components/UndoBar.svelte';
  import { auth } from './lib/stores/auth.svelte';
  import { untrack } from 'svelte';
  import { chat } from './lib/stores/chat.svelte';
  import { model } from './lib/stores/model.svelte';
  import { ui } from './lib/stores/ui.svelte';

  let sheet = $state<SheetView | null>(null);
  let inspecting = $state<string | null>(null);

  const inspectTrace = $derived(inspecting ? chat.traces[inspecting] : undefined);
  const inspectTurn = $derived(inspecting ? chat.turns.find((t) => t.id === inspecting) : undefined);

  // untrack: this runs once on mount. Without it the effect would subscribe to the chat
  // state that `start()` reads and re-run on the first reply.
  $effect(() => {
    untrack(() => {
      ui.applyTheme();
      chat.start();
      void model.refresh();
    });
    return () => chat.dispose();
  });

  // Modifier keys only: the composer is a prose field, so a bare letter can never be a
  // shortcut. Alt+digit rather than Cmd/Ctrl+digit because the browser reserves that one
  // for switching tabs, and `code` rather than `key` because Option+1 is not "1" on a Mac.
  const SHEET_BY_CODE: Record<string, SheetView> = {
    Digit1: 'model',
    Digit2: 'diagnostics',
    Digit3: 'training'
  };

  function onkeydown(e: KeyboardEvent) {
    const mod = e.metaKey || e.ctrlKey;

    if (mod && !e.altKey && e.key.toLowerCase() === 'k') {
      e.preventDefault();
      ui.switcher = !ui.switcher;
      return;
    }

    if (e.altKey && !mod && SHEET_BY_CODE[e.code]) {
      e.preventDefault();
      toggle(SHEET_BY_CODE[e.code]);
      return;
    }

    if (e.key !== 'Escape') return;
    // Each open layer cancels itself first; only when nothing is open does Escape mean
    // "stop generating", and only when nothing is generating does it mean "let me type".
    if (ui.editing) {
      ui.editing = null;
      return;
    }
    if (sheet || inspecting || ui.switcher) return;
    if (chat.busy) {
      e.preventDefault();
      chat.stop();
      return;
    }
    ui.focusComposer += 1;
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
  <main>
    <Conversation oninspect={inspect} />
    <UndoBar />
  </main>
  <Composer />
</div>

{#if auth.required}
  <TokenPrompt />
{:else if sheet}
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
    <ByteInspector trace={inspectTrace} turn={inspectTurn} />
  </Sheet>
{/if}

<style>
  .app {
    display: flex;
    flex-direction: column;
    height: 100dvh;
    max-height: 100dvh;
    padding-top: env(safe-area-inset-top);
    padding-left: env(safe-area-inset-left);
    padding-right: env(safe-area-inset-right);
  }

  main {
    position: relative;
    flex: 1;
    min-height: 0;
    display: flex;
    flex-direction: column;
  }
</style>
