// Shapes for the inline-SVG charts. Kept out of the components so both the chart and its
// callers can import them.

export interface Point {
  x: number;
  y: number;
}

export interface Series {
  points: Point[];
  label: string;
  /** 'faint' is the noisy per-step trace, 'solid' the evaluated one */
  weight: 'faint' | 'solid';
  dots?: boolean;
  /** CSS color; defaults to the accent. Used when several runs share one chart. */
  color?: string;
}
