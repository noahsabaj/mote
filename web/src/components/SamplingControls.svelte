<script lang="ts">
  import {
    GROUP_LABELS,
    PARAM_SPECS,
    PRESETS,
    editValue,
    parseValue,
    settings,
    showParam,
    type ParamGroup,
    type ParamSpec
  } from '../lib/stores/settings.svelte';
  import { chat } from '../lib/stores/chat.svelte';
  import { num, pct } from '../lib/format';
  import { tip } from '../lib/actions';
  import Icon from './Icon.svelte';
  import type { SamplingParams } from '../lib/types';

  const id = Math.random().toString(36).slice(2, 8);

  const BANDS: { group: ParamGroup; specs: ParamSpec[] }[] = (
    ['randomness', 'limits'] as ParamGroup[]
  ).map((group) => ({ group, specs: PARAM_SPECS.filter((s) => s.group === group) }));

  // A hint belongs to the row you are working, not to the panel. Focus rather than hover:
  // dragging a slider focuses it, so this covers pointer and keyboard alike, and the panel does
  // not twitch as the cursor crosses it. The panel grows upward, so the composer never moves.
  let activeKey = $state<keyof SamplingParams | null>(null);

  // Each field holds text of its own while it is being edited, so a half-typed "10" on the way
  // to "1024" is never clamped up to the minimum mid-keystroke. Committing waits for blur or
  // Enter; Escape throws the draft away. Every field is bound rather than merely given a
  // `value`, because Svelte writes that as an attribute and a typed-in input ignores it —
  // the revert would then be real in the store and invisible on screen.
  let editing = $state<keyof SamplingParams | null>(null);
  let drafts = $state<Partial<Record<keyof SamplingParams, string>>>({});
  // $state: Svelte 5 warns (at runtime) when bind:this targets a plain object's property
  let fields = $state<Partial<Record<keyof SamplingParams, HTMLInputElement>>>({});

  // Any field you are not editing follows the store, so dragging a slider moves its number.
  $effect(() => {
    for (const spec of PARAM_SPECS) {
      if (editing !== spec.key) drafts[spec.key] = editValue(spec.key, settings.params[spec.key]);
    }
  });

  function commit(key: keyof SamplingParams): void {
    const v = parseValue(key, drafts[key] ?? '');
    editing = null;
    // Unparseable text is not a value: the field snaps back rather than inventing one. Written
    // after `editing` clears so the effect above restores the text either way.
    if (v !== null) settings.set(key, v);
  }

  function onFieldKey(e: KeyboardEvent, key: keyof SamplingParams): void {
    if (e.key === 'Enter') {
      // The composer sends on Enter; a number being typed into a panel above it must not.
      e.preventDefault();
      e.stopPropagation();
      commit(key);
      fields[key]?.blur();
      return;
    }
    if (e.key === 'Escape') {
      const dirty = drafts[key] !== editValue(key, settings.params[key]);
      editing = null;
      // Escape cancels the edit you are making. With no edit to cancel it belongs to the
      // panel, so it is allowed through to close it.
      if (dirty) e.stopPropagation();
    }
  }

  /** Presets are one choice, so the group holds one tab stop and the arrows move within it. */
  function onPresetKey(e: KeyboardEvent, i: number): void {
    const keys = ['ArrowRight', 'ArrowDown', 'ArrowLeft', 'ArrowUp', 'Home', 'End'];
    if (!keys.includes(e.key)) return;
    e.preventDefault();
    const last = PRESETS.length - 1;
    const next =
      e.key === 'Home'
        ? 0
        : e.key === 'End'
          ? last
          : e.key === 'ArrowRight' || e.key === 'ArrowDown'
            ? (i + 1) % PRESETS.length
            : (i - 1 + PRESETS.length) % PRESETS.length;
    const target = PRESETS[next];
    settings.apply(target.id);
    document.getElementById(`${id}-preset-${target.id}`)?.focus();
  }

  /** Which segment carries the group's tab stop; the first one when no preset is active. */
  function presetTabIndex(i: number): 0 | -1 {
    const active = settings.activePreset;
    const chosen = active ? PRESETS.findIndex((p) => p.id === active) : 0;
    return i === chosen ? 0 : -1;
  }

  /** Top-p is read only on the sampling path; at temperature 0 the engine takes the argmax. */
  function inert(key: keyof SamplingParams): boolean {
    return key === 'top_p' && settings.greedy;
  }

  function frac(spec: ParamSpec): number {
    return (settings.defaults[spec.key] - spec.min) / (spec.max - spec.min);
  }

  const lastReply = $derived.by(() => {
    for (let i = chat.turns.length - 1; i >= 0; i--) {
      const t = chat.turns[i];
      if (t.role === 'assistant' && t.stats) return t;
    }
    return null;
  });

  // What the knobs did, measured — not predicted. `mean p` is the probability of the byte the
  // model chose under the distribution it actually sampled from, so temperature and top-p move
  // it. (The stream's `entropy` would not: the engine computes that on the raw softmax, before
  // either knob applies, so showing it here would be a lie dressed as feedback.)
  const readout = $derived.by(() => {
    const t = lastReply;
    if (!t?.stats) return null;
    const trace = chat.traces[t.id];
    void trace?.version; // settle with the final flush
    const parts: string[] = [];
    if (trace && trace.size > 0) parts.push(`mean p ${num(trace.meanP, 2)}`);
    parts.push(`${t.stats.bytes} bytes`);
    if (t.stats.mbp_proposed > 0) parts.push(`${pct(t.stats.mbp_accept_rate)} of drafts accepted`);
    if (t.reason === 'max_bytes') parts.push('reached the byte limit');
    // The sliders may have moved since. Naming what it was drawn at keeps the number true
    // instead of quietly attributing it to the settings now on screen.
    const p = t.params;
    const drift = p
      ? PARAM_SPECS.filter((s) => p[s.key] !== settings.params[s.key]).map((s) =>
          showParam(s.key, p[s.key])
        )
      : [];
    // Built whole rather than assembled in the template: Svelte trims leading whitespace inside
    // an element, which silently ate the space before the parenthesis.
    const lede = drift.length ? `Last reply (at ${drift.join(', ')}):` : 'Last reply:';
    return { lede, text: parts.join(' · ') };
  });
</script>

<div class="controls">
  <!-- One track, three segments: the shape says pick one, which three loose buttons never did.
       Sliders dragged off every preset leave it empty, which is the truth — no preset is what
       the model is being run at, and inventing a "Custom" segment would only hide that. -->
  <div class="seg" role="radiogroup" aria-label="Presets">
    {#each PRESETS as preset, i (preset.id)}
      <button
        id="{id}-preset-{preset.id}"
        role="radio"
        aria-checked={settings.activePreset === preset.id}
        tabindex={presetTabIndex(i)}
        use:tip={preset.note}
        onclick={() => settings.apply(preset.id)}
        onkeydown={(e) => onPresetKey(e, i)}
      >
        {preset.label}
      </button>
    {/each}
  </div>

  {#each BANDS as band (band.group)}
    <section>
      <h3>{GROUP_LABELS[band.group]}</h3>

      {#each band.specs as spec (spec.key)}
        <div
          class="row"
          class:inert={inert(spec.key)}
          onfocusin={() => (activeKey = spec.key)}
          onfocusout={() => {
            if (activeKey === spec.key) activeKey = null;
          }}
        >
          <label for="{id}-{spec.key}">{spec.label}</label>
          <!-- The revert sits to the left of the number so the number keeps its place on the
               panel edge when one appears; a row that shifts as you drag is unreadable. -->
          <div class="val-cell">
            {#if settings.overridden(spec.key)}
              <button
                class="revert"
                use:tip={`Back to ${editValue(spec.key, settings.defaults[spec.key])}`}
                aria-label="Reset {spec.label} to the checkpoint default"
                onclick={() => settings.resetKey(spec.key)}
              >
                <Icon name="undo" size={13} />
              </button>
            {/if}
            <input
              bind:this={fields[spec.key]}
              class="val tabular"
              class:off={settings.overridden(spec.key)}
              type="text"
              inputmode="decimal"
              autocomplete="off"
              spellcheck="false"
              disabled={inert(spec.key)}
              aria-label="{spec.label}, {spec.min} to {spec.max}"
              bind:value={drafts[spec.key]}
              onfocus={(e) => {
                editing = spec.key;
                e.currentTarget.select();
              }}
              onblur={() => commit(spec.key)}
              onkeydown={(e) => onFieldKey(e, spec.key)}
            />
          </div>

          <div class="track" style="--frac: {frac(spec)}">
            <input
              id="{id}-{spec.key}"
              type="range"
              min={spec.min}
              max={spec.max}
              step={spec.step}
              value={settings.params[spec.key]}
              disabled={inert(spec.key)}
              aria-describedby="{id}-{spec.key}-hint"
              oninput={(e) => settings.set(spec.key, Number(e.currentTarget.value))}
            />
            <!-- where the checkpoint's own value sits, so home is findable without reading -->
            <span class="tick" aria-hidden="true"></span>
          </div>

          <p
            class="hint"
            id="{id}-{spec.key}-hint"
            hidden={!(settings.explain || activeKey === spec.key)}
          >
            {spec.hint}
          </p>
          {#if inert(spec.key)}
            <p class="hint reason">Not in use: temperature 0 takes the likeliest byte outright.</p>
          {/if}
        </div>
      {/each}
    </section>
  {/each}

  {#if readout}
    <p class="readout"><span class="at">{readout.lede}</span> {readout.text}</p>
  {/if}

  <div class="foot">
    <button class="quiet" aria-pressed={settings.explain} onclick={() => settings.toggleExplain()}>
      Explain
    </button>
    <button
      class="quiet"
      aria-pressed={settings.verifyPrefix}
      onclick={() => settings.toggleVerifyPrefix()}
      use:tip={'Re-read each prompt cold after the cached continuation and report any divergence (slower replies)'}
    >
      Verify cache
    </button>
    <button
      class="quiet"
      aria-pressed={settings.arena}
      onclick={() => settings.toggleArena()}
      use:tip={'Every prompt gets two replies, A and B, and you choose — from the challenger when one is loaded'}
    >
      Arena
    </button>
    <button class="quiet" onclick={() => settings.reset()} disabled={!settings.anyOverridden}>
      Reset all
    </button>
  </div>
</div>

<style>
  .controls {
    display: grid;
    gap: 0.85rem;
  }

  /* ---------------------------------------------------------------- bands */

  section {
    display: grid;
    gap: 0.55rem;
  }

  h3 {
    margin: 0;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid var(--rule);
    font-size: 0.6875rem;
    font-weight: 550;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--ink-3);
  }

  .row {
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 0.1rem 0.6rem;
    align-items: baseline;
  }
  .row.inert {
    opacity: 0.55;
  }

  label {
    font-size: 0.8125rem;
    color: var(--ink);
  }

  /* The number is typed as well as dragged: the slider steps in round units, the field reaches
     everything between them. It stays a plain number until you touch it, so a panel of four
     boxes does not read as a form waiting to be filled in. */
  .val-cell {
    display: flex;
    align-items: center;
    gap: 0.15rem;
    justify-self: end;
  }

  .val {
    /* Four tabular digits plus the padding and border the box adds around them: "4096" is the
       widest thing any of the four knobs can show, and it must not clip. */
    width: 5.6ch;
    padding: 0.1em 0.25em;
    border: 1px solid transparent;
    border-radius: 4px;
    background: transparent;
    color: var(--ink-3);
    font: inherit;
    font-size: 0.8125rem;
    text-align: right;
    cursor: text;
  }
  .val.off {
    color: var(--accent-ink);
  }
  .val:hover:not(:disabled) {
    border-color: var(--rule);
    background: var(--surface-2);
  }
  .val:focus {
    border-color: var(--accent-line);
    background: var(--surface-2);
    color: var(--ink);
    outline: none;
  }
  .val:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 1px;
  }
  .val:disabled {
    cursor: default;
  }

  /* One knob home again without disturbing the other three. Present only when there is
     something to undo, so it never sits there meaning nothing. */
  .revert {
    display: inline-flex;
    align-items: center;
    padding: 0.15em;
    border: 0;
    border-radius: 4px;
    background: transparent;
    color: var(--ink-3);
    cursor: pointer;
  }
  .revert:hover {
    background: var(--surface-2);
    color: var(--accent-ink);
  }

  .track {
    position: relative;
    grid-column: 1 / -1;
    display: flex;
    align-items: center;
    height: 20px;
    margin-top: 0.1rem;
  }

  input[type='range'] {
    width: 100%;
    height: 20px;
    margin: 0;
    accent-color: var(--accent);
    cursor: pointer;
  }
  input[type='range']:disabled {
    cursor: default;
  }

  .tick {
    position: absolute;
    top: 50%;
    /* centre of the thumb's travel: a ~16px thumb never brings its centre to either end */
    left: calc(8px + (100% - 16px) * var(--frac));
    width: 1px;
    height: 11px;
    margin-top: -5.5px;
    background: var(--ink-2);
    opacity: 0.8;
    pointer-events: none;
  }

  .hint {
    grid-column: 1 / -1;
    margin: 0.15rem 0 0;
    font-size: 0.75rem;
    line-height: 1.4;
    color: var(--ink-3);
  }
  .hint[hidden] {
    display: none;
  }
  .reason {
    color: var(--ink-2);
  }

  /* -------------------------------------------------------------- readout */

  .readout {
    margin: 0;
    padding-top: 0.7rem;
    border-top: 1px solid var(--rule);
    font-size: 0.75rem;
    line-height: 1.45;
    color: var(--ink-2);
    font-variant-numeric: tabular-nums;
  }
  .at {
    color: var(--ink-3);
  }

  .foot {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.75rem;
  }

  /* On a phone the number fields were 24 px tall and the slider tracks 20 (QA 2026-08-24). */
  @media (max-width: 34rem) {
    .val {
      min-height: 34px;
    }
    .track,
    input[type='range'] {
      height: 32px;
    }
    .foot {
      flex-wrap: wrap;
      gap: 0.25rem 0.5rem;
    }
  }
</style>
