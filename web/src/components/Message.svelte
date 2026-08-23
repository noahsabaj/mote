<script lang="ts">
  import type { Turn } from '../lib/stores/chat.svelte';
  import { chat } from '../lib/stores/chat.svelte';
  import { ui } from '../lib/stores/ui.svelte';
  import { clock } from '../lib/clock.svelte';
  import { autosize, tip } from '../lib/actions';
  import ChunkedText from './ChunkedText.svelte';
  import Icon from './Icon.svelte';
  import { ago, num, pct } from '../lib/format';
  import type { SamplingParams } from '../lib/types';

  let {
    turn,
    isLast,
    oninspect
  }: { turn: Turn; isLast: boolean; oninspect: (id: string) => void } = $props();

  const trace = $derived(chat.traces[turn.id]);
  const streaming = $derived(chat.streamingId === turn.id);
  const text = $derived(streaming && trace ? trace.text : turn.content);
  const empty = $derived(streaming && text.length === 0);
  // A turn that failed before its first byte has a trace object but nothing in it, so the
  // byte-level tools and Copy would open on an empty reply. Only Retry is any use there.
  const hasBytes = $derived(!!trace && trace.count > 0);
  const hasText = $derived((turn.content || text).length > 0);
  const stamp = $derived(ago(new Date(turn.at).toISOString(), clock.now));

  let copied = $state(false);
  let copyTimer: ReturnType<typeof setTimeout> | null = null;

  async function copy() {
    try {
      await navigator.clipboard.writeText(turn.content || text);
      copied = true;
      if (copyTimer) clearTimeout(copyTimer);
      copyTimer = setTimeout(() => (copied = false), 1600);
    } catch {
      /* clipboard blocked — the text is selectable either way */
    }
  }

  // ------------------------------------------------------------------ editing

  const editing = $derived(ui.editing === turn.id);
  const EDIT_NOTE = 'Saving replaces every reply after this one.';
  let noteOpen = $state(false);
  let draft = $state('');
  let editArea = $state<HTMLTextAreaElement | null>(null);
  const changed = $derived(draft.trim().length > 0 && draft.trim() !== turn.content);

  $effect(() => {
    if (!editing) return;
    draft = turn.content;
    queueMicrotask(() => {
      if (!editArea) return;
      editArea.focus();
      editArea.setSelectionRange(editArea.value.length, editArea.value.length);
    });
  });

  function saveEdit() {
    if (!changed) return;
    const text = draft;
    ui.editing = null;
    chat.editAndResend(turn.id, text);
  }

  function onEditKey(e: KeyboardEvent) {
    if (e.key === 'Escape') {
      // The window handler behind this one stops generation; cancelling an edit should not.
      e.stopPropagation();
      ui.editing = null;
    } else if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      saveEdit();
    }
  }

  // --------------------------------------------------------------- provenance

  const SHORT: Record<keyof SamplingParams, string> = {
    temperature: 'T',
    top_p: 'top-p',
    max_bytes: 'max',
    n_candidates: 'n'
  };

  // Only the knobs that were not the checkpoint's own recommendation. A reply drawn at the
  // defaults says nothing, because there is nothing to disclose.
  const offDefault = $derived.by(() => {
    const p = turn.params;
    if (!p || !turn.offDefault?.length) return '';
    return turn.offDefault.map((k) => `${SHORT[k]} ${p[k]}`).join(', ');
  });

  const reasonNote = $derived(
    turn.reason === 'max_bytes'
      ? 'reached the byte limit'
      : turn.reason === 'stopped'
        ? 'stopped'
        : ''
  );

  // All samples of this reply slot in the order they were generated; `turn` is the shown one.
  const pool = $derived([...(turn.samples ?? []), turn].sort((a, b) => a.at - b.at));
  const cur = $derived(pool.findIndex((t) => t.id === turn.id));
</script>

{#if turn.role === 'user'}
  <article class="turn user" aria-label="You said">
    {#if editing}
      <div class="edit">
        <textarea
          bind:this={editArea}
          bind:value={draft}
          use:autosize={320}
          rows="1"
          aria-label="Edit this prompt"
          onkeydown={onEditKey}
        ></textarea>
        <div class="edit-foot">
          <!-- A tooltip on its own would say nothing on a phone, where there is no hover;
               clicking pins the same sentence under the row instead. -->
          <button
            type="button"
            class="note"
            aria-expanded={noteOpen}
            aria-label="What saving does"
            onclick={() => (noteOpen = !noteOpen)}
            use:tip={EDIT_NOTE}
          >
            <Icon name="info" size={14} />
          </button>
          <button class="btn" onclick={() => (ui.editing = null)}>Cancel</button>
          <button class="btn accent" onclick={saveEdit} disabled={!changed}>Save</button>
        </div>
        {#if noteOpen}
          <p class="note-text meta">{EDIT_NOTE}</p>
        {/if}
      </div>
    {:else}
      <div class="said">{turn.content}</div>
      <footer class="asked">
        <span class="meta stamp">{stamp}</span>
        <button
          class="quiet ico"
          aria-label="Retry"
          disabled={chat.busy}
          onclick={() => chat.retryFrom(turn.id)}
          use:tip={'Retry'}
        >
          <Icon name="redo" size={14} />
        </button>
        <button
          class="quiet ico"
          aria-label="Edit"
          disabled={chat.busy}
          onclick={() => (ui.editing = turn.id)}
          use:tip={'Edit'}
        >
          <Icon name="pencil" size={14} />
        </button>
        <button
          class="quiet ico"
          aria-label="Copy"
          onclick={copy}
          use:tip={copied ? 'Copied' : 'Copy'}
        >
          <Icon name={copied ? 'check' : 'copy'} size={14} />
        </button>
      </footer>
    {/if}
  </article>
{:else}
  <article class="turn model" aria-label="Mote replied" aria-busy={streaming}>
    {#if turn.truncated}
      <p class="notice">
        <Icon name="alert" size={13} />
        The conversation was truncated to fit the {turn.contextLimit} byte context window.
      </p>
    {/if}

    <div class="body" class:awaiting={empty}>
      {#if trace && (streaming || ui.structure)}
        <ChunkedText {trace} structure={ui.structure} {streaming} />
      {:else}
        <span class="prose">{text}</span>
      {/if}
      {#if empty}<span class="waiting" aria-hidden="true"></span>{/if}
    </div>

    {#if turn.error}
      <p class="notice error">
        <Icon name="alert" size={13} />
        {turn.error}
      </p>
    {/if}

    {#if !streaming}
      <footer class:pinned={isLast}>
        <p class="stats meta">
          {#if turn.stats}
            {turn.stats.bytes} bytes · {turn.stats.chunks} chunks · {num(
              turn.stats.bytes_per_sec,
              0
            )} B/s
            {#if turn.stats.mbp_proposed > 0}
              · {pct(turn.stats.mbp_accept_rate)} multi-byte accepted
            {/if}
            {#if turn.ttfbMs !== undefined}· first byte in {num(turn.ttfbMs, 0)} ms{/if}
            {#if reasonNote}· {reasonNote}{/if}
            {#if offDefault}· <span class="off">off default: {offDefault}</span>{/if}
          {/if}
        </p>
        <div class="actions">
          {#if pool.length > 1}
            <span class="samples" role="group" aria-label="Samples of this reply">
              <button
                class="quiet prev"
                disabled={cur <= 0 || chat.busy}
                onclick={() => chat.chooseSample(turn.id, pool[cur - 1].id)}
                aria-label="Previous sample"
              >
                <Icon name="chevron" size={13} />
              </button>
              <span class="meta tabular">{cur + 1}/{pool.length}</span>
              <button
                class="quiet"
                disabled={cur >= pool.length - 1 || chat.busy}
                onclick={() => chat.chooseSample(turn.id, pool[cur + 1].id)}
                aria-label="Next sample"
              >
                <Icon name="chevron" size={13} />
              </button>
            </span>
          {/if}
          {#if hasBytes}
            <button
              class="quiet"
              aria-pressed={ui.structure}
              onclick={() => ui.toggleStructure()}
              title="Mark learned chunk boundaries and bytes taken in parallel"
            >
              <Icon name="structure" size={14} />
              Structure
            </button>
            <button class="quiet" onclick={() => oninspect(turn.id)}>Bytes</button>
          {/if}
          {#if hasText}
            <button class="quiet" onclick={copy}>
              <Icon name={copied ? 'check' : 'copy'} size={14} />
              {copied ? 'Copied' : 'Copy'}
            </button>
          {/if}
        </div>
      </footer>
    {/if}
  </article>
{/if}

<style>
  .turn {
    margin: 0 0 1.9rem;
  }

  /* A column, not a row: the bubble sits right and its controls sit under it, right. */
  .user {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
  }

  .said {
    max-width: 85%;
    padding: 0.6rem 0.9rem;
    border-radius: var(--radius);
    background: var(--surface);
    color: var(--ink);
    font-size: 0.9375rem;
    line-height: 1.55;
    white-space: pre-wrap;
    overflow-wrap: break-word;
  }

  .asked {
    display: flex;
    align-items: center;
    gap: 0.1rem;
    margin-top: 0.25rem;
    opacity: 0;
    transition: opacity 120ms ease;
  }
  .asked .stamp {
    margin-right: 0.35rem;
    font-size: 0.75rem;
  }
  .turn.user:hover .asked,
  .asked:focus-within {
    opacity: 1;
  }

  .ico {
    justify-content: center;
    width: var(--tap);
    min-height: var(--tap);
    padding: 0;
  }

  /* ------------------------------------------------------------------ editing */

  /* Editing takes the whole column: a multi-paragraph prompt in an 85%-wide right-aligned
     bubble is unusable, and the edit is the only thing happening on screen anyway. */
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

  .note {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: var(--tap);
    height: var(--tap);
    margin-right: auto;
    padding: 0;
    border: 0;
    border-radius: var(--radius-sm);
    background: transparent;
    color: var(--ink-3);
    cursor: help;
  }
  .note:hover,
  .note[aria-expanded='true'] {
    color: var(--ink);
    background: var(--surface);
  }

  .note-text {
    margin: 0.35rem 0 0;
    font-size: 0.75rem;
    text-align: right;
  }

  /* -------------------------------------------------------------------- model */

  .model .body {
    font-family: var(--font-read);
    font-size: 1.0625rem;
    line-height: 1.68;
    color: var(--ink);
  }

  .prose {
    white-space: pre-wrap;
    overflow-wrap: break-word;
  }

  /* Before the first byte lands: one mark, not a skeleton screen. */
  .waiting {
    display: inline-block;
    width: 0.4em;
    height: 1.02em;
    vertical-align: text-bottom;
    background: var(--ink-3);
    border-radius: 1px;
    animation: breathe 1.5s ease-in-out infinite;
  }
  @keyframes breathe {
    0%,
    100% {
      opacity: 0.9;
    }
    50% {
      opacity: 0.25;
    }
  }

  .notice {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    margin: 0 0 0.5rem;
    font-size: 0.8125rem;
    color: var(--ink-3);
  }
  .notice.error {
    margin: 0.6rem 0 0;
    color: var(--accent-ink);
  }

  footer {
    display: flex;
    /* The stats often run to two lines; the actions belong beside the first of them, not
       floating against the middle of the block. */
    align-items: flex-start;
    justify-content: space-between;
    gap: 0.5rem 1rem;
    flex-wrap: wrap;
    margin-top: 0.55rem;
    min-height: 28px;
  }

  .stats {
    margin: 0;
    padding-top: 0.3rem;
    font-size: 0.75rem;
    flex: 1 1 14rem;
  }
  .off {
    color: var(--accent-ink);
  }

  .actions {
    display: flex;
    align-items: center;
    flex: none;
    gap: 0.1rem;
    opacity: 0;
    transition: opacity 120ms ease;
  }
  .samples {
    display: inline-flex;
    align-items: center;
    gap: 0.05rem;
    margin-right: 0.4rem;
  }
  .samples .meta {
    font-size: 0.75rem;
    padding: 0 0.15rem;
  }
  .prev :global(svg) {
    transform: rotate(180deg);
  }
  .turn:hover .actions,
  footer:focus-within .actions,
  footer.pinned .actions {
    opacity: 1;
  }

  /* Nothing may be discoverable by hover alone, so on touch both footers stay put. */
  @media (hover: none) {
    .actions,
    .asked {
      opacity: 1;
    }
    .ico,
    .note {
      width: 44px;
      min-height: 44px;
      height: 44px;
    }
  }
</style>
