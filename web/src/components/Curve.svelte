<script lang="ts">
  // Inline SVG line chart — no chart library. Axes are two hairlines and four labels.
  import type { Point, Series } from '../lib/chart';

  let {
    series,
    yLabel,
    xLabel,
    height = 170,
    digits = 2
  }: {
    series: Series[];
    yLabel: string;
    xLabel: string;
    height?: number;
    digits?: number;
  } = $props();

  const W = 320;
  const PAD_L = 34;
  const PAD_R = 4;
  const PAD_T = 8;
  const PAD_B = 20;

  const bounds = $derived.by(() => {
    let x0 = Infinity;
    let x1 = -Infinity;
    let y0 = Infinity;
    let y1 = -Infinity;
    for (const s of series) {
      for (const p of s.points) {
        if (!Number.isFinite(p.x) || !Number.isFinite(p.y)) continue;
        if (p.x < x0) x0 = p.x;
        if (p.x > x1) x1 = p.x;
        if (p.y < y0) y0 = p.y;
        if (p.y > y1) y1 = p.y;
      }
    }
    if (!Number.isFinite(x0)) return null;
    if (x1 === x0) x1 = x0 + 1;
    const pad = (y1 - y0) * 0.08 || 0.05;
    // Snap the y domain outwards to a round step so the axis labels are readable numbers.
    const step = niceStep((y1 + pad - (y0 - pad)) / 2);
    const nonNegative = y0 >= 0; // bits/byte, bytes/chunk, rates: never draw a negative axis for these
    return {
      x0,
      x1,
      y0: nonNegative ? Math.max(0, Math.floor((y0 - pad) / step) * step) : Math.floor((y0 - pad) / step) * step,
      y1: Math.ceil((y1 + pad) / step) * step,
      step
    };
  });

  /** Largest of 1, 2 or 5 × 10^k that is not greater than `span`. */
  function niceStep(span: number): number {
    if (!(span > 0)) return 1;
    const mag = Math.pow(10, Math.floor(Math.log10(span)));
    const norm = span / mag;
    return (norm >= 5 ? 5 : norm >= 2 ? 2 : 1) * mag;
  }

  function sx(x: number): number {
    if (!bounds) return PAD_L;
    return PAD_L + ((x - bounds.x0) / (bounds.x1 - bounds.x0)) * (W - PAD_L - PAD_R);
  }
  function sy(y: number): number {
    if (!bounds) return PAD_T;
    return PAD_T + (1 - (y - bounds.y0) / (bounds.y1 - bounds.y0)) * (height - PAD_T - PAD_B);
  }

  /** Cap the vertex count so a long run stays a cheap path. */
  function thin(points: Point[]): Point[] {
    const LIMIT = 700;
    if (points.length <= LIMIT) return points;
    const stride = Math.ceil(points.length / LIMIT);
    const out: Point[] = [];
    for (let i = 0; i < points.length; i += stride) out.push(points[i]);
    const last = points[points.length - 1];
    if (out[out.length - 1] !== last) out.push(last);
    return out;
  }

  function d(points: Point[]): string {
    const pts = thin(points);
    return pts
      .map((p, i) => `${i === 0 ? 'M' : 'L'}${sx(p.x).toFixed(2)} ${sy(p.y).toFixed(2)}`)
      .join(' ');
  }

  const ticks = $derived.by(() => {
    if (!bounds) return [];
    const out: { y: number; label: string }[] = [];
    // Enough decimals to tell adjacent ticks apart: a flat curve inside a 0.1 band gets a 0.05 step,
    // and rounding those to `digits` produced duplicate labels — which crashed the keyed block (QA 2026-08-24).
    const decimals = Math.max(digits, Math.max(0, -Math.floor(Math.log10(bounds.step))));
    for (let v = bounds.y0; v <= bounds.y1 + bounds.step / 2; v += bounds.step) {
      out.push({ y: sy(v), label: v.toFixed(decimals) });
    }
    return out;
  });
</script>

<figure>
  <figcaption>
    <span class="y">{yLabel}</span>
    <span class="keys">
      {#each series as s (s.label)}
        <span class="key"
          ><span class="swatch {s.weight}" style:background={s.color ?? null}></span>{s.label}</span
        >
      {/each}
    </span>
  </figcaption>

  {#if bounds}
    <svg viewBox="0 0 {W} {height}" role="img" aria-label="{yLabel} against {xLabel}">
      {#each ticks as t, i (i)}
        <line x1={PAD_L} x2={W - PAD_R} y1={t.y} y2={t.y} class="grid" />
        <text x={PAD_L - 5} y={t.y + 3} class="tick" text-anchor="end">{t.label}</text>
      {/each}
      {#each series as s (s.label)}
        <path d={d(s.points)} class="line {s.weight}" style:stroke={s.color ?? null} />
        {#if s.dots}
          {#each thin(s.points) as p, i (i)}
            <circle cx={sx(p.x)} cy={sy(p.y)} r="1.9" class="dot" style:fill={s.color ?? null} />
          {/each}
        {/if}
      {/each}
      <text x={PAD_L} y={height - 5} class="tick">{Math.round(bounds.x0)}</text>
      <text x={W - PAD_R} y={height - 5} class="tick" text-anchor="end">
        {Math.round(bounds.x1)}
      </text>
      <text x={(W + PAD_L) / 2} y={height - 5} class="tick" text-anchor="middle">{xLabel}</text>
    </svg>
  {:else}
    <p class="none">No records yet.</p>
  {/if}
</figure>

<style>
  figure {
    margin: 0;
  }

  figcaption {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 0.75rem;
    flex-wrap: wrap;
    margin-bottom: 0.35rem;
  }

  .y {
    font-size: 0.8125rem;
    color: var(--ink);
  }

  .keys {
    display: flex;
    gap: 0.75rem;
  }
  .key {
    display: inline-flex;
    align-items: center;
    gap: 0.3em;
    font-size: 0.75rem;
    color: var(--ink-3);
  }
  .swatch {
    width: 12px;
    height: 2px;
    border-radius: 1px;
    background: var(--accent);
  }
  .swatch.faint {
    opacity: 0.38;
  }

  svg {
    width: 100%;
    display: block;
  }

  .grid {
    stroke: var(--rule);
    stroke-width: 1;
    vector-effect: non-scaling-stroke;
  }

  .line {
    fill: none;
    stroke: var(--accent);
    stroke-width: 1.4;
    stroke-linejoin: round;
    stroke-linecap: round;
    vector-effect: non-scaling-stroke;
  }
  .line.faint {
    stroke-width: 1;
    opacity: 0.38;
  }

  .dot {
    fill: var(--accent);
  }

  .tick {
    font-family: var(--font-mono);
    font-size: 8px;
    fill: var(--ink-3);
  }

  .none {
    margin: 0;
    font-size: 0.8125rem;
    color: var(--ink-3);
  }
</style>
