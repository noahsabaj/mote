<script lang="ts">
  import { chat } from '../lib/stores/chat.svelte';
  import { model } from '../lib/stores/model.svelte';
  import { diagnostics } from '../lib/stores/diagnostics.svelte';
  import { settings, showParam } from '../lib/stores/settings.svelte';
  import { ui } from '../lib/stores/ui.svelte';
  import { autosize, dismissable, tip } from '../lib/actions';
  import { looksLikeCommand, matching, parseCommand, type Command } from '../lib/commands';
  import SamplingControls from './SamplingControls.svelte';
  import CommandMenu from './CommandMenu.svelte';
  import HelpPanel from './HelpPanel.svelte';
  import Icon from './Icon.svelte';

  let value = $state('');
  let area = $state<HTMLTextAreaElement | null>(null);
  let panelOpen = $state(false);
  let panelTrigger = $state<HTMLElement | null>(null);
  let focused = $state(false);

  const swapping = $derived(model.swapping !== null);
  const disabled = $derived(swapping);
  const hasText = $derived(value.trim().length > 0);
  const canSubmit = $derived(hasText && !disabled);

  // Whether this message *is* a command, which is the only thing the highlight may claim.
  // A slash further in — "/clear the table", or a command on a line of its own after some
  // prose — is an ordinary message, and colouring it would promise something Enter will not do.
  const command = $derived(parseCommand(value));

  // The menu is explicitly closed rather than derived away, so completing an entry does not
  // immediately reopen the list on the very text it just inserted.
  let menuClosed = $state(false);
  let selected = $state(0);
  const menuItems = $derived(looksLikeCommand(value) ? matching(value) : []);
  const menuOpen = $derived(!menuClosed && !disabled && menuItems.length > 0);

  function complete(c: Command) {
    value = `/${c.name}`;
    menuClosed = true;
    area?.focus();
  }

  function submit() {
    if (!canSubmit) return;
    chat.enter(value);
    value = '';
    menuClosed = false;
    selected = 0;
    queueMicrotask(() => {
      if (area) {
        area.style.height = 'auto';
        area.focus();
      }
    });
  }

  function onkeydown(e: KeyboardEvent) {
    // The menu takes the keys it needs first; picking an entry only fills the field, so a
    // command that deletes a conversation always costs a second, deliberate Enter.
    if (menuOpen && !e.isComposing) {
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        selected = (selected + 1) % menuItems.length;
        return;
      }
      if (e.key === 'ArrowUp') {
        e.preventDefault();
        selected = (selected - 1 + menuItems.length) % menuItems.length;
        return;
      }
      if (e.key === 'Enter' || e.key === 'Tab') {
        e.preventDefault();
        complete(menuItems[Math.min(selected, menuItems.length - 1)]);
        return;
      }
      if (e.key === 'Escape') {
        e.stopPropagation(); // closing the menu is not "stop generating"
        menuClosed = true;
        return;
      }
    }

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

  // Escape from anywhere hands the keyboard back to the composer.
  $effect(() => {
    if (ui.focusComposer === 0) return;
    area?.focus();
  });

  // What the *next* reply will be drawn at. The transcript already discloses what each past
  // reply was drawn at, so this reads forward: the two never say the same thing twice.
  // Only the randomness knobs are named — they are what changes the character of a reply, and
  // naming all four would reflow the toolbar mid-drag. A cap or draft-length change adds a dot.
  const summary = $derived(
    settings.offDefaultKeys
      .filter((k) => k === 'temperature' || k === 'top_p')
      .map((k) => showParam(k, settings.params[k]))
      .join(' · ')
  );
  const otherChanged = $derived(
    settings.offDefaultKeys.some((k) => k === 'max_bytes' || k === 'n_candidates')
  );
  const spoken = $derived(
    settings.offDefaultKeys.map((k) => showParam(k, settings.params[k])).join(', ')
  );

  // Context is only worth a line when it is actually filling up.
  const context = $derived.by(() => {
    const live = diagnostics.stats;
    const last = [...chat.turns].reverse().find((t) => t.contextLimit);
    const used = live?.context_bytes ?? chat.preview?.used ?? last?.contextBytes;
    const limit = live?.context_limit ?? chat.preview?.limit ?? last?.contextLimit ?? model.info?.context_limit_bytes;
    if (!used || !limit) return null;
    if (used / limit < 0.5) return null;
    return { used, limit, frac: Math.min(1, used / limit) };
  });
  // The meter shows the *next* prompt: re-ask the backend whenever what would be sent changes.
  $effect(() => {
    void chat.turns.length;
    void chat.foldMode;
    void chat.foldNow;
    void chat.card;
    void settings.params.max_bytes;
    chat.refreshPreview();
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
      {chat.busy ? 'Context' : 'Next prompt'} {context.used} of {context.limit} bytes
      <span class="gauge" aria-hidden="true"
        ><span class="gauge-fill" style="width: {context.frac * 100}%"></span></span
      >
      {#if chat.preview?.fold && !chat.busy}
        · {chat.preview.fold.turns} folded
      {/if}
      {#if chat.foldMode === 'off'}
        · folding off
        <button class="quiet inline" onclick={() => chat.refold()}>Fold again</button>
      {:else if !chat.busy && chat.turns.length >= 3}
        <button class="quiet inline" onclick={() => chat.foldNext()} disabled={chat.foldNow}>
          {chat.foldNow ? 'Folding on the next reply' : 'Fold now'}
        </button>
      {/if}
    </p>
  {/if}

  <div class="field-wrap">
    {#if menuOpen}
      <CommandMenu
        items={menuItems}
        {selected}
        trigger={area}
        onpick={complete}
        ondismiss={() => (menuClosed = true)}
      />
    {/if}
    {#if ui.help}
      <div class="help-pop" use:dismissable={{ onDismiss: () => (ui.help = false) }}>
        <HelpPanel />
      </div>
    {/if}

    <div class="field" class:focused>
      <label class="sr-only" for="composer">Message Mote</label>
      <!-- A textarea cannot colour its own text, so a command is drawn by a mirror behind a
           transparent one. This is safe only because a command is short and single-line: it
           never wraps or scrolls, so there is no scroll position to keep in sync. Ordinary
           prose is never mirrored at all. -->
      {#if command}
        <div class="mirror" aria-hidden="true">{value}</div>
      {/if}
      <textarea
        id="composer"
        class:masked={!!command}
        bind:this={area}
        bind:value
        use:autosize={220}
        rows="1"
        placeholder={disabled ? 'Swapping checkpoint…' : 'Ask Mote something, or / for commands'}
        {disabled}
        onkeydown={onkeydown}
        oninput={() => {
          menuClosed = false;
          selected = 0;
        }}
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
            {#if summary}<span class="summary tabular" aria-hidden="true">· {summary}</span>{/if}
            {#if otherChanged}<span class="dot" aria-hidden="true"></span>{/if}
            {#if settings.anyOverridden}
              <span class="sr-only">— next reply off the checkpoint defaults: {spoken}</span>
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

        <div class="acts">
          {#if chat.busy}
            <!-- Stop is the emergency control, so it keeps its place at the end of the row
                 and the queue button appears to its left rather than pushing it along. -->
            {#if hasText}
              <button class="btn" onclick={submit} aria-label="Add to the queue" use:tip={'Queue'}>
                <Icon name="plus" size={15} />
              </button>
            {/if}
            <button class="btn accent send" onclick={() => chat.stop()} aria-label="Stop generating">
              <Icon name="stop" size={14} />
              Stop
            </button>
          {:else}
            <button
              class="btn accent send"
              onclick={submit}
              disabled={!canSubmit}
              aria-label={command ? `Run /${command}` : 'Send'}
            >
              <Icon name="send" size={15} />
            </button>
          {/if}
        </div>
      </div>
    </div>
  </div>

  <p class="hint" class:visible={focused || chat.busy}>
    {#if chat.busy}
      Enter queues · Esc interrupts · / for commands
    {:else}
      Enter sends · Shift+Enter starts a line · ↑ edits the last prompt · / for commands
    {/if}
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

  /* Anchors both popovers — the command menu and /help — to the field rather than to the
     page, so they rise from the thing they belong to. */
  .field-wrap {
    position: relative;
  }

  .help-pop {
    position: absolute;
    bottom: calc(100% + 0.5rem);
    left: 0;
    z-index: 30;
    width: min(30rem, calc(100vw - 2rem));
    padding: 0.95rem 1rem;
    background: var(--bg);
    border: 1px solid var(--rule-strong);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    animation: rise 130ms ease;
  }

  .field {
    position: relative;
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
  /* Every metric here has to match the textarea above, since one sits exactly on the other. */
  .mirror {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    padding: 0.7rem 0.85rem 0.15rem;
    font: inherit;
    font-size: 1rem;
    line-height: 1.55;
    color: var(--accent-ink);
    white-space: pre-wrap;
    overflow-wrap: break-word;
    pointer-events: none;
  }
  textarea.masked {
    color: transparent;
    caret-color: var(--ink);
  }

  .acts {
    display: flex;
    align-items: center;
    gap: 0.35rem;
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

  .summary {
    color: var(--accent-ink);
    white-space: nowrap;
  }

  .panel {
    position: absolute;
    bottom: calc(100% + 0.5rem);
    left: 0;
    z-index: 30;
    width: min(23rem, calc(100vw - 2rem));
    /* Pinning every hint open, on a phone, used to run the panel off the top of the screen
       with no way back down to Reset. */
    max-height: min(60vh, 32rem);
    overflow-y: auto;
    overscroll-behavior: contain;
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
