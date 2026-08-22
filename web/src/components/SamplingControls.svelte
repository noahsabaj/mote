<script lang="ts">
  import { PARAM_SPECS, settings } from '../lib/stores/settings.svelte';
  import type { SamplingParams } from '../lib/types';

  const id = Math.random().toString(36).slice(2, 8);

  function show(key: keyof SamplingParams, digits: number): string {
    return settings.params[key].toFixed(digits);
  }
</script>

<div class="controls">
  {#each PARAM_SPECS as spec (spec.key)}
    <div class="row">
      <label for="{id}-{spec.key}">{spec.label}</label>
      <output class="tabular" for="{id}-{spec.key}">{show(spec.key, spec.digits)}</output>
      <input
        id="{id}-{spec.key}"
        type="range"
        min={spec.min}
        max={spec.max}
        step={spec.step}
        value={settings.params[spec.key]}
        aria-describedby="{id}-{spec.key}-hint"
        oninput={(e) => settings.set(spec.key, Number(e.currentTarget.value))}
      />
      <p class="hint" id="{id}-{spec.key}-hint">
        {spec.hint}
        {#if settings.overridden(spec.key)}
          <span class="default">Model default {settings.defaults[spec.key].toFixed(spec.digits)}.</span>
        {/if}
      </p>
    </div>
  {/each}

  <div class="foot">
    <p class="meta">Defaults come from the loaded checkpoint.</p>
    <button class="quiet" onclick={() => settings.reset()} disabled={!settings.anyOverridden}>
      Reset
    </button>
  </div>
</div>

<style>
  .controls {
    display: grid;
    gap: 0.9rem;
  }

  .row {
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 0.15rem 0.6rem;
  }

  label {
    font-size: 0.8125rem;
    color: var(--ink);
  }

  output {
    font-size: 0.8125rem;
    color: var(--accent-ink);
  }

  input[type='range'] {
    grid-column: 1 / -1;
    width: 100%;
    height: 18px;
    margin: 0.1rem 0 0;
    accent-color: var(--accent);
    cursor: pointer;
  }

  .hint {
    grid-column: 1 / -1;
    margin: 0.05rem 0 0;
    font-size: 0.75rem;
    line-height: 1.4;
    color: var(--ink-3);
  }

  .default {
    color: var(--ink-2);
  }

  .foot {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.75rem;
    padding-top: 0.7rem;
    border-top: 1px solid var(--rule);
  }
  .foot :global(p) {
    margin: 0;
    font-size: 0.75rem;
  }
</style>
