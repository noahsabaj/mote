// Sampling parameters and small UI preferences.
//
// The model's own defaults (from /api/model) are the baseline; the user's changes are kept
// as overrides so "Reset" genuinely restores what the backend recommends, and a checkpoint
// swap that ships different defaults is respected for anything untouched.
//
// "Overridden" is decided by comparing values, not by which keys happen to sit in storage:
// a slider dragged back onto the checkpoint's own number is not a change, and must not light
// the composer's changed-dot.

import * as persist from '../persist';
import type { SamplingParams } from '../types';

/** Two knobs shape the distribution; two bound the reply. Saying so is the panel's job. */
export type ParamGroup = 'randomness' | 'limits';

export interface ParamSpec {
  key: keyof SamplingParams;
  label: string;
  hint: string;
  group: ParamGroup;
  min: number;
  max: number;
  step: number;
  digits: number;
}

/** Short names for provenance lines. One vocabulary for the transcript, the trigger and the panel. */
export const SHORT: Record<keyof SamplingParams, string> = {
  temperature: 'T',
  top_p: 'top-p',
  max_bytes: 'max'
};

export const GROUP_LABELS: Record<ParamGroup, string> = {
  randomness: 'Randomness',
  limits: 'Limits & speed'
};

export const PARAM_SPECS: ParamSpec[] = [
  {
    key: 'temperature',
    label: 'Temperature',
    hint: 'Flattens or sharpens the byte distribution before sampling. At 0 the model stops sampling and always takes its most likely byte.',
    group: 'randomness',
    min: 0,
    max: 2,
    step: 0.05,
    digits: 2
  },
  {
    key: 'top_p',
    label: 'Top-p',
    hint: 'Sample only from the smallest set of bytes whose probability sums to this.',
    group: 'randomness',
    min: 0.1,
    max: 1,
    step: 0.01,
    digits: 2
  },
  {
    key: 'max_bytes',
    label: 'Max bytes',
    hint: 'Hard stop for one reply, counted in raw UTF-8 bytes.',
    group: 'limits',
    min: 32,
    max: 4096,
    step: 32,
    digits: 0
  }
];

export interface Preset {
  id: string;
  label: string;
  /** Randomness only. A key left out means "whatever the checkpoint recommends". */
  values: Partial<Pick<SamplingParams, 'temperature' | 'top_p'>>;
  note: string;
}

/**
 * Three presets, no invented numbers. Greedy is argmax — the engine's own `temperature <= 0`
 * branch, and what `mote.eval.probe` runs at. Checkpoint is whatever shipped with the
 * weights. Raw is temperature 1 with no nucleus truncation, which is the distribution the
 * model actually learned, not a taste setting. Greedy leaves top-p at the checkpoint's value
 * rather than forcing one, because at temperature 0 the engine never reads it.
 */
export const PRESETS: Preset[] = [
  {
    id: 'greedy',
    label: 'Greedy',
    values: { temperature: 0 },
    note: 'Always the most likely byte. No sampling, so the same prompt gives the same reply.'
  },
  {
    id: 'checkpoint',
    label: 'Checkpoint',
    values: {},
    note: 'The values this checkpoint shipped with.'
  },
  {
    id: 'raw',
    label: 'Raw',
    values: { temperature: 1, top_p: 1 },
    note: "The model's own distribution, untouched — nothing flattened, nothing truncated."
  }
];

function specFor(key: keyof SamplingParams): ParamSpec | undefined {
  return PARAM_SPECS.find((s) => s.key === key);
}

/**
 * Two renderings of one value, so nothing can drift between them.
 *
 * `showValue` is for reading: temperature 0 says "Greedy", which is the whole truth in one
 * word, and the composer trigger and the transcript say it too. `editValue` is for the panel's
 * fields, which are always numeric — a box you can type into should never hold a word you are
 * not allowed to type.
 */
export function editValue(key: keyof SamplingParams, v: number): string {
  const spec = specFor(key);
  return v.toFixed(spec ? spec.digits : 2);
}

export function showValue(key: keyof SamplingParams, v: number): string {
  if (key === 'temperature' && v <= 0) return 'Greedy';
  return editValue(key, v);
}

/**
 * A typed value, clamped to the slider's range but *not* snapped to its step. The step is a
 * dragging convenience — it lands you on round numbers — while the engine takes anything in
 * range, so a typed 0.83 has to mean 0.83. Returns null for anything unparseable, which the
 * caller reads as "leave the value alone".
 */
export function parseValue(key: keyof SamplingParams, text: string): number | null {
  const spec = specFor(key);
  const v = Number(text.trim());
  if (!text.trim() || !Number.isFinite(v)) return null;
  if (!spec) return v;
  const clamped = Math.min(spec.max, Math.max(spec.min, v));
  // Fractional bytes do not exist, so a typed 1024.7 on that knob is a typo rather than a preference.
  return spec.digits === 0 ? Math.round(clamped) : clamped;
}

/** The same value with its short name — "T 1.35", or just "Greedy", which names itself. */
export function showParam(key: keyof SamplingParams, v: number): string {
  const shown = showValue(key, v);
  return shown === 'Greedy' ? shown : `${SHORT[key]} ${shown}`;
}

const FALLBACK: SamplingParams = {
  temperature: 0.8,
  top_p: 0.9,
  max_bytes: 512
};

/** Slider steps are decimal, so a dragged 0.8 and a default 0.8 can differ in the last bit. */
function same(a: number, b: number): boolean {
  return Math.abs(a - b) < 1e-9;
}

class Settings {
  /** Defaults reported by the loaded checkpoint. */
  defaults = $state<SamplingParams>({ ...FALLBACK });
  #overrides = $state<Partial<SamplingParams>>(persist.read('overrides', {}));
  /** Pin every hint open. Off by default: the explanations are for learning, not for every visit. */
  explain = $state<boolean>(persist.read('sampling.explain', false));
  /** debug: re-read every prompt cold after the cached continuation and report any divergence */
  verifyPrefix = $state<boolean>(persist.read('prefix.verify', false));
  /** arena mode: every prompt gets two replies, A and B, and you choose (docs/prefs.md) */
  arena = $state<boolean>(persist.read('prefs.arena', false));

  params = $derived<SamplingParams>({ ...this.defaults, ...this.#overrides });

  overridden(key: keyof SamplingParams): boolean {
    return !same(this.params[key], this.defaults[key]);
  }

  get anyOverridden(): boolean {
    return PARAM_SPECS.some((s) => this.overridden(s.key));
  }

  /** Keys whose value differs from the checkpoint's, in panel order. */
  get offDefaultKeys(): (keyof SamplingParams)[] {
    return PARAM_SPECS.filter((s) => this.overridden(s.key)).map((s) => s.key);
  }

  /** Temperature 0 is the engine's greedy branch; top-p is never read there. */
  get greedy(): boolean {
    return this.params.temperature <= 0;
  }

  /** The preset the sliders currently sit on, or null. Checkpoint wins a tie by listing first. */
  get activePreset(): string | null {
    const p = this.params;
    for (const preset of PRESETS) {
      const t = preset.values.temperature ?? this.defaults.temperature;
      const tp = preset.values.top_p ?? this.defaults.top_p;
      if (same(t, p.temperature) && same(tp, p.top_p)) return preset.id;
    }
    return null;
  }

  setDefaults(d: SamplingParams): void {
    this.defaults = { ...d };
  }

  set<K extends keyof SamplingParams>(key: K, value: number): void {
    const next = { ...this.#overrides };
    if (same(value, this.defaults[key])) delete next[key];
    else next[key] = value;
    this.#write(next);
  }

  resetKey(key: keyof SamplingParams): void {
    const next = { ...this.#overrides };
    delete next[key];
    this.#write(next);
  }

  apply(id: string): void {
    const preset = PRESETS.find((p) => p.id === id);
    if (!preset) return;
    for (const key of ['temperature', 'top_p'] as const) {
      const v = preset.values[key];
      if (v === undefined) this.resetKey(key);
      else this.set(key, v);
    }
  }

  reset(): void {
    this.#write({});
  }

  toggleExplain(): void {
    this.explain = !this.explain;
    persist.write('sampling.explain', this.explain);
  }

  toggleVerifyPrefix(): void {
    this.verifyPrefix = !this.verifyPrefix;
    persist.write('prefix.verify', this.verifyPrefix);
  }

  toggleArena(): void {
    this.arena = !this.arena;
    persist.write('prefs.arena', this.arena);
  }

  #write(next: Partial<SamplingParams>): void {
    this.#overrides = next;
    persist.write('overrides', next);
  }
}

export const settings = new Settings();
