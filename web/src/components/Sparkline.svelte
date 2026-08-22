<script lang="ts">
  // Inline SVG only. Values are plotted as given; the caller states the domain.
  let {
    values,
    min = 0,
    max = 1,
    height = 34,
    label,
    threshold
  }: {
    values: number[];
    min?: number;
    max?: number;
    height?: number;
    label: string;
    /** optional horizontal reference line, in data units */
    threshold?: number;
  } = $props();

  const W = 240;

  const path = $derived.by(() => {
    const n = values.length;
    if (n === 0) return '';
    const span = Math.max(1e-6, max - min);
    const dx = n > 1 ? W / (n - 1) : 0;
    let d = '';
    for (let i = 0; i < n; i++) {
      const y = height - ((values[i] - min) / span) * height;
      d += `${i === 0 ? 'M' : 'L'}${(i * dx).toFixed(2)} ${Math.max(0.5, Math.min(height - 0.5, y)).toFixed(2)}`;
      if (i < n - 1) d += ' ';
    }
    return d;
  });

  const thresholdY = $derived(
    threshold === undefined
      ? null
      : height - ((threshold - min) / Math.max(1e-6, max - min)) * height
  );
</script>

<svg viewBox="0 0 {W} {height}" preserveAspectRatio="none" role="img" aria-label={label} {height}>
  {#if thresholdY !== null}
    <line x1="0" x2={W} y1={thresholdY} y2={thresholdY} class="threshold" />
  {/if}
  <path d={path} class="trace" vector-effect="non-scaling-stroke" />
</svg>

<style>
  svg {
    width: 100%;
    display: block;
    overflow: visible;
  }
  .trace {
    fill: none;
    stroke: var(--accent);
    stroke-width: 1.25;
    stroke-linejoin: round;
    stroke-linecap: round;
  }
  .threshold {
    stroke: var(--rule-strong);
    stroke-width: 1;
    stroke-dasharray: 2 3;
    vector-effect: non-scaling-stroke;
  }
</style>
