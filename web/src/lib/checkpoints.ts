// Sorting and filtering for the checkpoint list (docs/checkpoints.md). Pure functions over
// the API's rows: the sheet holds the state, this decides what the state means.
//
// Two of the six sort keys are sizes that differ by an order of magnitude and used to be
// confusable — `bytes_seen` is training data consumed, `file_size_bytes` is the .pt on disk —
// so both are named in full wherever they appear.

import { bytes, num, when } from './format';
import type { CheckpointListItem } from './types';

export type SortKey = 'name' | 'step' | 'val_bpb' | 'bytes_seen' | 'file_size' | 'created_at';
export type SortDir = 'asc' | 'desc';

export interface SortSpec {
  key: SortKey;
  label: string;
  /** what ascending means for this key, said plainly enough to put in an option */
  asc: string;
  desc: string;
}

export const SORT_SPECS: SortSpec[] = [
  { key: 'created_at', label: 'Saved', asc: 'oldest first', desc: 'newest first' },
  { key: 'val_bpb', label: 'Bits/byte', asc: 'best first', desc: 'worst first' },
  { key: 'step', label: 'Step', asc: 'fewest first', desc: 'most first' },
  { key: 'name', label: 'Name', asc: 'A to Z', desc: 'Z to A' },
  { key: 'file_size', label: 'File size', asc: 'smallest first', desc: 'largest first' },
  { key: 'bytes_seen', label: 'Bytes seen', asc: 'fewest first', desc: 'most first' }
];

export const SORT_LABELS: Record<SortKey, string> = Object.fromEntries(
  SORT_SPECS.map((s) => [s.key, s.label])
) as Record<SortKey, string>;

export interface CheckpointView {
  sort: SortKey;
  dir: SortDir;
  query: string;
  /** hide checkpoints with no eval record yet */
  evaluated: boolean;
  /** only checkpoints that beat the loaded model's bits/byte */
  better: boolean;
  /** run-name families to keep; empty means all of them */
  families: string[];
}

export const DEFAULT_VIEW: CheckpointView = {
  sort: 'created_at',
  dir: 'desc',
  query: '',
  evaluated: false,
  better: false,
  families: []
};

/** A checkpoint is a run: `runs/ab_muon_4096/last.pt` is the run `ab_muon_4096`. */
export function runName(id: string): string {
  const parts = id.split('/').filter(Boolean);
  return parts.length >= 2 ? parts[parts.length - 2] : (parts[0] ?? id);
}

/**
 * What a row calls itself. A run's `last.pt` is just the run — that is the usual case and the
 * name you think in. But `discover_checkpoints` globs every `.pt` in a run directory, so a run
 * that saved intermediates contributes several, and showing all of them as the bare run name
 * puts identical-looking rows in the list that load different models. Those keep their stem.
 */
export function displayName(id: string): string {
  const parts = id.split('/').filter(Boolean);
  const stem = (parts[parts.length - 1] ?? '').replace(/\.pt$/, '');
  const run = runName(id);
  return stem === 'last' || stem === run ? run : `${run}/${stem}`;
}

/** Runs are named `<family>_<variant>`, so the first segment groups an experiment together. */
export function family(id: string): string {
  const name = runName(id);
  const cut = name.indexOf('_');
  return cut === -1 ? name : name.slice(0, cut);
}

/**
 * Families worth a chip: those with more than one run. A one-off experiment would otherwise
 * add a chip that filters to exactly itself, and the chip row grows with every stray run —
 * search still reaches them.
 */
export function chipFamilies(items: CheckpointListItem[]): string[] {
  const counts = new Map<string, number>();
  for (const c of items) counts.set(family(c.id), (counts.get(family(c.id)) ?? 0) + 1);
  return [...counts.entries()]
    .filter(([, n]) => n > 1)
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .map(([f]) => f);
}

/** The loaded checkpoint's bits/byte, or null — which is what disables the "better than" chip. */
export function loadedBpb(items: CheckpointListItem[]): number | null {
  return items.find((c) => c.loaded)?.val_bpb ?? null;
}

function value(c: CheckpointListItem, key: SortKey): number | string | null {
  switch (key) {
    case 'name':
      return displayName(c.id).toLowerCase();
    case 'step':
      return c.step;
    case 'val_bpb':
      return c.val_bpb;
    case 'bytes_seen':
      return c.bytes_seen;
    case 'file_size':
      return c.file_size_bytes;
    case 'created_at':
      return Date.parse(c.created_at) || 0;
  }
}

function stamp(c: CheckpointListItem): number {
  return Date.parse(c.created_at) || 0;
}

/**
 * Nulls sink in *both* directions rather than flipping to the top, so sorting by bits/byte
 * ascending shows the best model first and the un-evaluated ones last — which is the question
 * being asked. Equal values fall back to newest first, since a dozen runs share a step count.
 */
export function compare(a: CheckpointListItem, b: CheckpointListItem, key: SortKey, dir: SortDir): number {
  const va = value(a, key);
  const vb = value(b, key);
  if (va === null || vb === null) {
    if (va === vb) return stamp(b) - stamp(a);
    return va === null ? 1 : -1;
  }
  let d = 0;
  if (typeof va === 'string' && typeof vb === 'string') d = va.localeCompare(vb);
  else d = (va as number) - (vb as number);
  if (dir === 'desc') d = -d;
  return d || stamp(b) - stamp(a);
}

export interface ViewResult {
  shown: CheckpointListItem[];
  /** how many the filters removed — the sheet says so rather than quietly showing fewer */
  hidden: number;
  filtering: boolean;
}

export function applyView(
  items: CheckpointListItem[],
  view: CheckpointView,
  baseline: number | null
): ViewResult {
  const q = view.query.trim().toLowerCase();
  // The chip is disabled without a baseline, but a persisted `better: true` can outlive the
  // checkpoint it was set against, so the filter itself has to be inert rather than empty.
  const better = view.better && baseline !== null;
  const shown = items
    .filter((c) => {
      if (q && !displayName(c.id).toLowerCase().includes(q)) return false;
      if (view.evaluated && c.val_bpb === null) return false;
      if (better && (c.val_bpb === null || c.val_bpb >= (baseline as number))) return false;
      if (view.families.length && !view.families.includes(family(c.id))) return false;
      return true;
    })
    .sort((a, b) => compare(a, b, view.sort, view.dir));
  return {
    shown,
    hidden: items.length - shown.length,
    filtering: q !== '' || view.evaluated || better || view.families.length > 0
  };
}

/**
 * A row's second line: step, bits/byte and when it was saved, always — plus whichever size
 * you are currently sorting by, so the number the order is based on is on screen. Both sizes
 * carry a word, because one row reading a bare "63 MB" was read as the file on disk when it
 * is the training data consumed.
 */
export function rowParts(c: CheckpointListItem, sort: SortKey): string[] {
  const parts = [
    `step ${c.step.toLocaleString()}`,
    c.val_bpb === null ? 'not evaluated yet' : `${num(c.val_bpb, 3)} bits/byte`
  ];
  if (sort === 'file_size') parts.push(`file ${bytes(c.file_size_bytes)}`);
  if (sort === 'bytes_seen') parts.push(`seen ${bytes(c.bytes_seen)}`);
  parts.push(when(c.created_at));
  return parts;
}

/** The five most recently loaded checkpoints, best-scoring ones until that history exists. */
export function shortlist(
  items: CheckpointListItem[],
  recents: string[],
  limit = 5
): CheckpointListItem[] {
  const byRecency = recents
    .map((id) => items.find((c) => c.id === id))
    .filter((c): c is CheckpointListItem => c !== undefined);
  if (byRecency.length >= Math.min(limit, items.length)) return byRecency.slice(0, limit);
  const best = items
    .filter((c) => c.val_bpb !== null && !byRecency.includes(c))
    .sort((a, b) => (a.val_bpb as number) - (b.val_bpb as number));
  return [...byRecency, ...best].slice(0, limit);
}
