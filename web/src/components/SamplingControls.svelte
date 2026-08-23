<script lang="ts">
  import {
    GROUP_LABELS,
    PARAM_SPECS,
    PRESETS,
    settings,
    showParam,
    showValue,
    type ParamGroup,
    type ParamSpec
  } from '../lib/stores/settings.svelte';
  import { chat } from '../lib/stores/chat.svelte';
  import { num, pct } from '../lib/format';
  import { tip } from '../lib/actions';
  import type { SamplingParams } from '../lib/types';

  const id = Math.random().toString(36).slice(2, 8);

  const BANDS: { group: ParamGroup; specs: ParamSpec[] }[] = (
    ['randomness', 'limits'] as ParamGroup[]
  ).map((group) => ({ group, specs: PARAM_SPECS.filter((s) => s.group === group) }));

  // A hint belongs to the row you are working, not to the panel. Focus rather than hover:
  // dragging a slider focuses it, so this covers pointer and keyboard alike, and the panel does
  // not twitch as the cursor crosses it. The panel grows upward, so the composer never moves.
  let activeKey = $state<keyof SamplingParams | null>(null);

  function value(key: keyof SamplingParams): string {
    return showValue(key, settings.params[key]);
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
  <div class="presets" role="group" aria-label="Presets">
    {#each PRESETS as preset (preset.id)}
      <button
        class="chip"
        aria-pressed={settings.activePreset === preset.id}
        use:tip={preset.note}
        onclick={() => settings.apply(preset.id)}
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
          <button
            class="val tabular"
            class:off={settings.overridden(spec.key)}
            disabled={!settings.overridden(spec.key)}
            use:tip={`Reset to ${settings.defaults[spec.key].toFixed(spec.digits)}`}
            aria-label="{spec.label} {value(spec.key)}, reset to the checkpoint default"
            onclick={() => settings.resetKey(spec.key)}
          >
            {value(spec.key)}
          </button>

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

  /* -------------------------------------------------------------- presets */

  .presets {
    display: flex;
    gap: 0.3rem;
  }

  .chip {
    flex: 1;
    min-height: 27px;
    padding: 0 0.5em;
    border: 1px solid var(--rule);
    border-radius: var(--radius-sm);
    background: transparent;
    color: var(--ink-2);
    font-size: 0.8125rem;
    cursor: pointer;
    transition: background-color 120ms ease, border-color 120ms ease, color 120ms ease;
  }
  .chip:hover {
    background: var(--surface);
    color: var(--ink);
  }
  .chip[aria-pressed='true'] {
    border-color: var(--accent-line);
    background: var(--accent-soft);
    color: var(--accent-ink);
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

  /* The value is the per-knob reset: it is what you are looking at when you want the default
     back, so it should be what you press. Dead while it already is the default. */
  .val {
    padding: 0 0.2em;
    margin: 0 -0.2em;
    border: 0;
    border-radius: 4px;
    background: transparent;
    color: var(--ink-3);
    font: inherit;
    font-size: 0.8125rem;
    cursor: pointer;
  }
  .val.off {
    color: var(--accent-ink);
  }
  .val:hover:not(:disabled) {
    background: var(--surface-2);
  }
  .val:disabled {
    cursor: default;
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
</style>
