<script lang="ts">
  import { chat } from '../lib/stores/chat.svelte';
  import { model } from '../lib/stores/model.svelte';
  import { diagnostics } from '../lib/stores/diagnostics.svelte';
  import { settings } from '../lib/stores/settings.svelte';
  import { ui } from '../lib/stores/ui.svelte';
  import { autosize, dismissable } from '../lib/actions';
  import SamplingControls from './SamplingControls.svelte';
  import Icon from './Icon.svelte';

  let value = $state('');
  let area = $state<HTMLTextAreaElement | null>(null);
  let panelOpen = $state(false);
  let panelTrigger = $state<HTMLElement | null>(null);
  let focused = $state(false);

  const swapping = $derived(model.swapping !== null);
  const disabled = $derived(swapping);
  const canSend = $derived(value.trim().length > 0 && !chat.busy && !disabled);

  function submit() {
    if (!canSend) return;
    chat.send(value);
    value = '';
    queueMicrotask(() => {
      if (area) {
        area.style.height = 'auto';
        area.focus();
      }
    });
  }

  function onkeydown(e: KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      submit();
      return;
    }
    // Up on an empty composer reaches back for the last prompt, the way a shell does. It
    // opens the edit box on that turn rather than pulling the text down here, so the edit
    // happens where the text lives.
    if (e.key === 'ArrowUp' && value === '' && !chat.busy && !e.isComposing) {
      const last = [...chat.turns].reverse().find((t) => t.role === 'user');
      if (!last) return;
      e.preventDefault();
      ui.editing = last.id;
    }
  }

  // Context is only worth a line when it is actually filling up.
  const context = $derived.by(() => {
    const live = diagnostics.stats;
    const last = [...chat.turns].reverse().find((t) => t.contextLimit);
    const used = live?.context_bytes ?? last?.contextBytes;
    const limit = live?.context_limit ?? last?.contextLimit ?? model.info?.context_limit_bytes;
    if (!used || !limit) return null;
    if (used / limit < 0.5) return null;
    return { used, limit, frac: Math.min(1, used / limit) };
  });
</script>

<div class="composer">
  {#if chat.link === 'offline'}
    <p class="line warn">
      <Icon name="alert" size={13} />
      Not connected to the backend.
      <button class="quiet inline" onclick={() => chat.reconnect()}>Retry now</button>
    </p>
  {:else if swapping}
    <p class="line">Loading {model.swapping} — the composer is disabled until the swap finishes.</p>
  {:else if context}
    <p class="line">
      Context {context.used} of {context.limit} bytes
      <span class="gauge" aria-hidden="true"
        ><span class="gauge-fill" style="width: {context.frac * 100}%"></span></span
      >
    </p>
  {/if}

  <div class="field" class:focused>
    <label class="sr-only" for="composer">Message Mote</label>
    <textarea
      id="composer"
      bind:this={area}
      bind:value
      use:autosize={220}
      rows="1"
      placeholder={disabled ? 'Swapping checkpoint…' : 'Ask Mote something'}
      {disabled}
      onkeydown={onkeydown}
      onfocus={() => (focused = true)}
      onblur={() => (focused = false)}
    ></textarea>

    <div class="tools">
      <div class="anchor">
        <button
          bind:this={panelTrigger}
          class="quiet"
          aria-expanded={panelOpen}
          onclick={() => (panelOpen = !panelOpen)}
        >
          <Icon name="sliders" size={14} />
          Sampling
          {#if settings.anyOverridden}
            <span class="dot" aria-hidden="true"></span>
            <span class="sr-only">(changed from the checkpoint defaults)</span>
          {/if}
        </button>
        {#if panelOpen}
          <div
            class="panel"
            use:dismissable={{ onDismiss: () => (panelOpen = false), trigger: panelTrigger }}
          >
            <SamplingControls />
          </div>
        {/if}
      </div>

      {#if chat.busy}
        <button class="btn accent send" onclick={() => chat.stop()} aria-label="Stop generating">
          <Icon name="stop" size={14} />
          Stop
        </button>
      {:else}
        <button class="btn accent send" onclick={submit} disabled={!canSend} aria-label="Send">
          <Icon name="send" size={15} />
        </button>
      {/if}
    </div>
  </div>

  <p class="hint" class:visible={focused || chat.busy}>
    Enter sends · Shift+Enter starts a line · ↑ edits the last prompt · Esc stops
  </p>
</div>

<style>
  .composer {
    /* same measure and centring as the transcript column */
    max-width: var(--shell);
    width: 100%;
    margin: 0 auto;
    padding: 0 var(--gutter) max(0.7rem, env(safe-area-inset-bottom));
  }

  .line {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    margin: 0 0 0.4rem;
    font-size: 0.75rem;
    color: var(--ink-3);
    font-variant-numeric: tabular-nums;
  }
  .line.warn {
    color: var(--accent-ink);
  }
  .inline {
    min-height: 0;
    padding: 0 0.2em;
    font-size: 0.75rem;
    text-decoration: underline;
    text-underline-offset: 2px;
  }

  .gauge {
    flex: none;
    width: 4.5rem;
    height: 3px;
    border-radius: 2px;
    background: var(--surface-2);
    overflow: hidden;
  }
  .gauge-fill {
    display: block;
    height: 100%;
    background: var(--accent);
  }

  .field {
    display: grid;
    border: 1px solid var(--rule-strong);
    border-radius: var(--radius);
    background: var(--bg);
    transition: border-color 120ms ease, box-shadow 120ms ease;
  }
  .field.focused {
    border-color: var(--accent-line);
    box-shadow: 0 0 0 3px var(--accent-soft);
  }

  textarea {
    resize: none;
    border: 0;
    outline: none;
    background: transparent;
    color: var(--ink);
    font: inherit;
    font-size: 1rem;
    line-height: 1.55;
    padding: 0.7rem 0.85rem 0.15rem;
    min-height: 2.4rem;
    max-height: 220px;
  }
  textarea::placeholder {
    color: var(--ink-3);
  }
  textarea:disabled {
    color: var(--ink-3);
  }

  .tools {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.5rem;
    padding: 0.3rem 0.4rem 0.4rem;
  }

  .anchor {
    position: relative;
  }

  /* The sliders are behind a click, so a reply drawn off-default would otherwise look
     exactly like one drawn at the checkpoint's recommendation. */
  .dot {
    width: 5px;
    height: 5px;
    margin-left: 0.05rem;
    border-radius: 50%;
    background: var(--accent);
  }

  .panel {
    position: absolute;
    bottom: calc(100% + 0.5rem);
    left: 0;
    z-index: 30;
    width: min(23rem, calc(100vw - 2rem));
    padding: 0.95rem 1rem;
    background: var(--bg);
    border: 1px solid var(--rule-strong);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    animation: rise 130ms ease;
  }

  .send {
    min-width: var(--tap);
    padding: 0 0.7em;
  }

  .hint {
    margin: 0.45rem 0 0;
    font-size: 0.75rem;
    color: var(--ink-3);
    opacity: 0;
    transition: opacity 140ms ease;
  }
  .hint.visible {
    opacity: 1;
  }

  @keyframes rise {
    from {
      opacity: 0;
      transform: translateY(4px);
    }
  }

  @media (max-width: 34rem) {
    .hint {
      display: none;
    }
  }
</style>
