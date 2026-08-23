// Sampling parameters and small UI preferences.
//
// The model's own defaults (from /api/model) are the baseline; the user's changes are kept
// as overrides so "Reset" genuinely restores what the backend recommends, and a checkpoint
// swap that ships different defaults is respected for anything untouched.

import * as persist from '../persist';
import type { SamplingParams } from '../types';

export interface ParamSpec {
  key: keyof SamplingParams;
  label: string;
  hint: string;
  min: number;
  max: number;
  step: number;
  digits: number;
}

export const PARAM_SPECS: ParamSpec[] = [
  {
    key: 'temperature',
    label: 'Temperature',
    hint: 'Flattens or sharpens the byte distribution before sampling.',
    min: 0,
    max: 2,
    step: 0.05,
    digits: 2
  },
  {
    key: 'top_p',
    label: 'Top-p',
    hint: 'Sample only from the smallest set of bytes whose probability sums to this.',
    min: 0.1,
    max: 1,
    step: 0.01,
    digits: 2
  },
  {
    key: 'max_bytes',
    label: 'Max bytes',
    hint: 'Hard stop for one reply, counted in raw UTF-8 bytes.',
    min: 32,
    max: 4096,
    step: 32,
    digits: 0
  },
  {
    key: 'n_candidates',
    label: 'Draft length n',
    hint: 'Bytes the multi-byte head drafts at each chunk boundary; the model verifies them exactly (0 = off).',
    min: 0,
    max: 8,
    step: 1,
    digits: 0
  }
];

const FALLBACK: SamplingParams = {
  temperature: 0.8,
  top_p: 0.9,
  max_bytes: 512,
  n_candidates: 3
};

class Settings {
  /** Defaults reported by the loaded checkpoint. */
  defaults = $state<SamplingParams>({ ...FALLBACK });
  #overrides = $state<Partial<SamplingParams>>(persist.read('overrides', {}));

  params = $derived<SamplingParams>({ ...this.defaults, ...this.#overrides });

  overridden(key: keyof SamplingParams): boolean {
    return this.#overrides[key] !== undefined;
  }

  get anyOverridden(): boolean {
    return Object.keys(this.#overrides).length > 0;
  }

  setDefaults(d: SamplingParams): void {
    this.defaults = { ...d };
  }

  set<K extends keyof SamplingParams>(key: K, value: number): void {
    this.#overrides = { ...this.#overrides, [key]: value };
    persist.write('overrides', this.#overrides);
  }

  reset(): void {
    this.#overrides = {};
    persist.write('overrides', this.#overrides);
  }
}

export const settings = new Settings();
