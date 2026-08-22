const NBSP = '\u202f'; // narrow no-break space, keeps "12.7 M" together

export function bytes(n: number): string {
  if (!Number.isFinite(n)) return '—';
  if (n < 1024) return `${n}${NBSP}B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(n < 10240 ? 1 : 0)}${NBSP}KB`;
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(n < 10 * 1024 ** 2 ? 1 : 0)}${NBSP}MB`;
  return `${(n / 1024 ** 3).toFixed(2)}${NBSP}GB`;
}

export function count(n: number): string {
  if (!Number.isFinite(n)) return '—';
  if (n < 1000) return String(n);
  if (n < 1e6) return `${(n / 1e3).toFixed(n < 1e4 ? 1 : 0)}${NBSP}K`;
  if (n < 1e9) return `${(n / 1e6).toFixed(n < 1e7 ? 2 : 1)}${NBSP}M`;
  return `${(n / 1e9).toFixed(2)}${NBSP}B`;
}

export function num(n: number | undefined | null, digits = 2): string {
  if (n === undefined || n === null || !Number.isFinite(n)) return '—';
  return n.toFixed(digits);
}

export function pct(x: number | undefined | null, digits = 0): string {
  if (x === undefined || x === null || !Number.isFinite(x)) return '—';
  return `${(x * 100).toFixed(digits)}%`;
}

export function minutes(m: number): string {
  if (!Number.isFinite(m)) return '—';
  if (m < 1) return `${Math.round(m * 60)}${NBSP}s`;
  if (m < 90) return `${m < 10 ? m.toFixed(1) : Math.round(m)}${NBSP}min`;
  const h = Math.floor(m / 60);
  const rest = Math.round(m % 60);
  return rest ? `${h}${NBSP}h ${rest}${NBSP}min` : `${h}${NBSP}h`;
}

/** Absolute date, no "3 minutes ago" churn — this is a lab tool. */
export function when(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  });
}

export function hex(b: number): string {
  return b.toString(16).toUpperCase().padStart(2, '0');
}

const CONTROL_NAMES: Record<number, string> = {
  0: 'NUL',
  9: 'TAB',
  10: 'LF',
  13: 'CR',
  27: 'ESC',
  32: 'SP',
  127: 'DEL'
};

/** A printable stand-in for a raw byte in the inspector. */
export function byteGlyph(b: number): string {
  if (b in CONTROL_NAMES) return CONTROL_NAMES[b];
  if (b < 32) return `·${hex(b)}`;
  if (b < 127) return String.fromCharCode(b);
  return `·${hex(b)}`; // continuation / lead byte of a multi-byte character
}
