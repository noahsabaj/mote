// How the checkpoint sheet is currently sorted and filtered. Kept apart from the model store,
// which owns the checkpoints themselves: this is a view of that list, and it survives reloads.
//
// Everything here persists, including the filters — which is why the sheet always says how many
// rows a filter is hiding and offers one tap to clear it. A saved filter you have forgotten is
// otherwise indistinguishable from a short list.

import * as persist from '../persist';
import { DEFAULT_VIEW, SORT_SPECS, type CheckpointView, type SortDir, type SortKey } from '../checkpoints';

const KEY = 'ckpt.view';
const KEYS = new Set<SortKey>(SORT_SPECS.map((s) => s.key));

/** Storage can hold a sort key from an older build, or families whose runs are long gone. */
function sane(raw: Partial<CheckpointView> | null): CheckpointView {
  if (!raw || typeof raw !== 'object') return { ...DEFAULT_VIEW };
  return {
    sort: KEYS.has(raw.sort as SortKey) ? (raw.sort as SortKey) : DEFAULT_VIEW.sort,
    dir: raw.dir === 'asc' || raw.dir === 'desc' ? raw.dir : DEFAULT_VIEW.dir,
    query: typeof raw.query === 'string' ? raw.query : '',
    evaluated: raw.evaluated === true,
    better: raw.better === true,
    families: Array.isArray(raw.families) ? raw.families.filter((f) => typeof f === 'string') : []
  };
}

class CheckpointViewStore {
  view = $state<CheckpointView>(sane(persist.read<Partial<CheckpointView> | null>(KEY, null)));

  private save(): void {
    persist.write(KEY, this.view);
  }

  setSort(key: SortKey): void {
    this.view.sort = key;
    this.save();
  }

  setDir(dir: SortDir): void {
    this.view.dir = dir;
    this.save();
  }

  flip(): void {
    this.view.dir = this.view.dir === 'asc' ? 'desc' : 'asc';
    this.save();
  }

  setQuery(q: string): void {
    this.view.query = q;
    this.save();
  }

  toggleEvaluated(): void {
    this.view.evaluated = !this.view.evaluated;
    this.save();
  }

  toggleBetter(): void {
    this.view.better = !this.view.better;
    this.save();
  }

  toggleFamily(f: string): void {
    this.view.families = this.view.families.includes(f)
      ? this.view.families.filter((x) => x !== f)
      : [...this.view.families, f];
    this.save();
  }

  /** Clears what hides rows. The sort order is a preference, not a filter, so it stays. */
  clearFilters(): void {
    this.view.query = '';
    this.view.evaluated = false;
    this.view.better = false;
    this.view.families = [];
    this.save();
  }
}

export const ckptView = new CheckpointViewStore();
