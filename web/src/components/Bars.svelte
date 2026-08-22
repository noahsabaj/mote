<script lang="ts">
  // A labelled row of horizontal bars. Values are shown as numbers as well as length, so the
  // reading never depends on judging a bar by eye.
  let {
    values,
    prefix,
    max = 1,
    digits = 2
  }: { values: number[]; prefix: string; max?: number; digits?: number } = $props();
</script>

{#if values.length === 0}
  <p class="none">Not reported yet.</p>
{:else}
  <ul>
    {#each values as v, i (i)}
      <li>
        <span class="name">{prefix}{i}</span>
        <span class="track" aria-hidden="true">
          <span class="fill" style="width: {Math.max(0, Math.min(1, v / max)) * 100}%"></span>
        </span>
        <span class="value tabular">{v.toFixed(digits)}</span>
      </li>
    {/each}
  </ul>
{/if}

<style>
  ul {
    list-style: none;
    margin: 0.35rem 0 0;
    padding: 0;
    display: grid;
    gap: 0.28rem;
  }
  li {
    display: grid;
    grid-template-columns: 3.4rem 1fr 2.7rem;
    align-items: center;
    gap: 0.6rem;
    font-size: 0.8125rem;
  }
  .name {
    color: var(--ink-3);
    font-family: var(--font-mono);
    font-size: 0.75rem;
  }
  .track {
    height: 6px;
    border-radius: 3px;
    background: var(--surface-2);
    overflow: hidden;
  }
  .fill {
    display: block;
    height: 100%;
    background: var(--accent);
    border-radius: 3px;
    transition: width 200ms ease;
  }
  .value {
    text-align: right;
    color: var(--ink-2);
  }
  .none {
    margin: 0.35rem 0 0;
    font-size: 0.8125rem;
    color: var(--ink-3);
  }
</style>
