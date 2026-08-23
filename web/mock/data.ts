// Dev-only fixtures. Everything here is invented and is labelled as such in the payloads
// themselves (see `status_note`, the device name, the run and checkpoint ids), so a screenshot
// taken against `npm run dev` can never be mistaken for a real measurement.

import type {
  CheckpointListItem,
  LogRecord,
  ModelInfo,
  TrainingRun
} from '../src/lib/types';

// Mirrors mote/serve/identity.py for the mock's 12.66M-parameter checkpoint.
export const MOCK_IDENTITY_CARD =
  'You are Mote, a small byte-level language model with about 13 million parameters, trained ' +
  'by Noah on a single GPU. You read and write raw UTF-8 bytes rather than tokens. You were ' +
  'trained on public web, educational and conversational text. You are small: you make ' +
  'mistakes, especially with arithmetic, dates and specific facts, and you should say so ' +
  'rather than guess. When someone corrects you, check the claim: agree if it is right, and ' +
  'politely keep your answer if it is wrong.';

export const MOCK_NOTE =
  'DEV MOCK — no model is loaded. Every number on this screen is fabricated by the ' +
  'development server so the interface can be reviewed without a GPU.';

interface CkptSpec {
  id: string;
  step: number;
  /** null mirrors the backend, which reports no bits/byte until a run has an eval record */
  val_bpb: number | null;
  bytes_seen: number;
  file_size_bytes: number;
  trained_minutes: number;
  created_at: string;
}

// The mock's runs live under `mock/` rather than `runs/`, so a path on screen still says what
// it is while the run names keep the real shape — `<family>_<variant>`, which is what the
// checkpoint sheet's family chips are derived from. The set below deliberately covers what the
// sorting has to survive: families of several and a family of one, two runs with no eval record
// yet, several runs tied at the same step, and file sizes that differ from bytes seen by an
// order of magnitude in the other direction.
export const CHECKPOINTS: CkptSpec[] = [
  {
    id: 'mock/pilot_1h/step_1200.pt',
    step: 1200,
    val_bpb: null,
    bytes_seen: 39321600,
    file_size_bytes: 152043520,
    trained_minutes: 23.4,
    created_at: '2026-08-21T09:41:02Z'
  },
  {
    id: 'mock/pilot_1h/step_2400.pt',
    step: 2400,
    val_bpb: 1.78,
    bytes_seen: 78643200,
    file_size_bytes: 152043520,
    trained_minutes: 46.9,
    created_at: '2026-08-21T10:05:18Z'
  },
  {
    id: 'mock/pilot_1h/last.pt',
    step: 3100,
    val_bpb: 1.63,
    bytes_seen: 101580800,
    file_size_bytes: 152043520,
    trained_minutes: 60.2,
    created_at: '2026-08-21T10:19:44Z'
  },
  {
    id: 'mock/pilot_4h/last.pt',
    step: 11800,
    val_bpb: 1.41,
    bytes_seen: 386662400,
    file_size_bytes: 152043520,
    trained_minutes: 241.6,
    created_at: '2026-08-22T02:52:07Z'
  },
  {
    id: 'mock/ab_muon_2048/last.pt',
    step: 2000,
    val_bpb: 1.761,
    bytes_seen: 65536000,
    file_size_bytes: 293601280,
    trained_minutes: 58.1,
    created_at: '2026-08-23T09:01:12Z'
  },
  {
    id: 'mock/ab_muon_4096/last.pt',
    step: 2000,
    val_bpb: 1.913,
    bytes_seen: 65536000,
    file_size_bytes: 293601280,
    trained_minutes: 61.4,
    created_at: '2026-08-23T08:01:55Z'
  },
  {
    id: 'mock/ab_muon_nombp/last.pt',
    step: 2000,
    val_bpb: 1.797,
    bytes_seen: 65536000,
    file_size_bytes: 264241152,
    trained_minutes: 54.7,
    created_at: '2026-08-23T09:16:03Z'
  },
  {
    id: 'mock/ab_adamw_4096/last.pt',
    step: 992,
    val_bpb: 2.79,
    bytes_seen: 32505856,
    file_size_bytes: 424673280,
    trained_minutes: 29.8,
    created_at: '2026-08-23T07:27:41Z'
  },
  {
    id: 'mock/ab_b299_2048/last.pt',
    step: 2000,
    val_bpb: 2.061,
    bytes_seen: 65536000,
    file_size_bytes: 424673280,
    trained_minutes: 57.2,
    created_at: '2026-08-23T06:27:19Z'
  },
  {
    id: 'mock/sweep_a0.1_n4/last.pt',
    step: 1500,
    val_bpb: 2.14,
    bytes_seen: 49152000,
    file_size_bytes: 152043520,
    trained_minutes: 41.0,
    created_at: '2026-08-22T18:12:36Z'
  },
  {
    id: 'mock/sweep_a0.3_n6/last.pt',
    step: 1500,
    val_bpb: 2.264,
    bytes_seen: 49152000,
    file_size_bytes: 152043520,
    trained_minutes: 42.3,
    created_at: '2026-08-22T19:04:50Z'
  },
  {
    id: 'mock/overnight_sft2/last.pt',
    step: 3299,
    val_bpb: 0.954,
    bytes_seen: 108068864,
    file_size_bytes: 424673280,
    trained_minutes: 188.5,
    created_at: '2026-08-23T08:31:07Z'
  },
  {
    id: 'mock/overnight_dpo2/last.pt',
    step: 3349,
    val_bpb: null,
    bytes_seen: 109707264,
    file_size_bytes: 141557760,
    trained_minutes: 194.2,
    created_at: '2026-08-23T10:39:28Z'
  },
  {
    id: 'mock/smoke_win/last.pt',
    step: 400,
    val_bpb: 3.02,
    bytes_seen: 13107200,
    file_size_bytes: 152043520,
    trained_minutes: 8.6,
    created_at: '2026-08-20T14:22:09Z'
  }
];

export const state = {
  loadedId: 'mock/pilot_1h/last.pt',
  swapping: false,
  challengerId: null as string | null
};

export function modelPayload(): ModelInfo {
  const ck = CHECKPOINTS.find((c) => c.id === state.loadedId) ?? CHECKPOINTS[2];
  return {
    name: ck.id,
    params: 12660000,
    status: ck.step > 8000 ? 'undertrained' : 'pilot',
    status_note: MOCK_NOTE,
    checkpoint: {
      path: ck.id,
      step: ck.step,
      bytes_seen: ck.bytes_seen,
      val_bpb: ck.val_bpb,
      trained_minutes: ck.trained_minutes,
      created_at: ck.created_at
    },
    architecture: {
      outer_width: 256,
      encoder_layers: 2,
      decoder_layers: 2,
      main: 'Relation 6L/384/8 heads',
      mbp_layers: 2
    },
    context_limit_bytes: 2048,
    device: { name: 'Mock device (development server, no GPU)', vram_total_mb: 8188, vram_used_mb: 912 },
    kernels: { mamba3: true, ssd: true },
    defaults: {
      temperature: 0.8,
      top_p: 0.9,
      max_bytes: 512,
      n_candidates: 3
    },
    // The backend sends these whenever a probe.json sits beside the checkpoint; the mock
    // reports them for the loaded run so the measured rendering is visible in dev, and leaves
    // the 1200-step checkpoint without one so the "not measured" branch is reachable too.
    probe:
      ck.step >= 2400
        ? {
            identity_acc: 0.9167,
            hold_rate: 0.7143,
            concede_rate: 0.625,
            n_identity: 24,
            n_facts: 40,
            identity_acc_seen: 1,
            hold_rate_seen: 0.875,
            concede_rate_seen: 0.75,
            n_identity_seen: 6,
            n_facts_seen: 8
          }
        : null,
    identity_card: MOCK_IDENTITY_CARD,
    challenger: (() => {
      const ch = CHECKPOINTS.find((c) => c.id === state.challengerId);
      return ch ? { id: ch.id, name: ch.id, step: ch.step, val_bpb: ch.val_bpb, loading: false } : null;
    })()
  };
}

export function checkpointList(): CheckpointListItem[] {
  return CHECKPOINTS.map((c) => ({
    id: c.id,
    step: c.step,
    val_bpb: c.val_bpb,
    bytes_seen: c.bytes_seen,
    file_size_bytes: c.file_size_bytes,
    created_at: c.created_at,
    loaded: c.id === state.loadedId,
    challenger: c.id === state.challengerId
  }));
}

// ------------------------------------------------------------------- training

const BOOT = Date.now();
const RUNNING_START = BOOT - 41 * 60_000;

/** The "running" run genuinely grows while the dev server is up, so `?since=` polling works. */
function runningSteps(): number {
  return Math.min(6000, 400 + Math.floor((Date.now() - BOOT) / 1000) * 20);
}

export function runs(): TrainingRun[] {
  const live = runningSteps();
  return [
    {
      id: 'pilot_4h',
      steps: live,
      last_val_bpb: valAt(live),
      running: true,
      started_at: new Date(RUNNING_START).toISOString()
    },
    {
      id: 'pilot_1h',
      steps: 3100,
      last_val_bpb: 1.63,
      running: false,
      started_at: '2026-08-21T09:19:31Z'
    }
  ];
}

// Deterministic pseudo-noise so a reload does not reshuffle the curve.
function noise(seed: number): number {
  const x = Math.sin(seed * 12.9898) * 43758.5453;
  return x - Math.floor(x) - 0.5;
}

function trainBpbAt(step: number): number {
  const base = 1.28 + 3.4 * Math.exp(-step / 900);
  return Math.max(0.9, base + noise(step) * 0.09);
}

function valAt(step: number): number {
  return Math.max(1.02, 1.3 + 3.2 * Math.exp(-step / 1000) + noise(step * 0.5) * 0.02);
}

const SAMPLES = [
  'The| rout|er| comp|ares| each| by|te| with| the| one| bef|ore| it|.',
  'The| router| compares| each| byte| with| the| one| before| it|.',
  'The| router| compares| each| byte| with| the| one| before| it|. Where| they| stop| looking| alike|, it| draws| a| boundary|.'
];

const EVERY = 10;
const EVAL_EVERY = 200;

/** Rebuild the whole JSONL log for a run, then slice from `since` — the real server tails a file. */
export function runLog(id: string, since: number): { records: LogRecord[]; next: number } {
  const total = id === 'pilot_4h' ? runningSteps() : 3100;
  const startedAt = id === 'pilot_4h' ? RUNNING_START : Date.parse('2026-08-21T09:19:31Z');
  const records: LogRecord[] = [];
  for (let step = EVERY; step <= total; step += EVERY) {
    const elapsed = (step / total) * (id === 'pilot_4h' ? (Date.now() - startedAt) / 60000 : 60.2);
    const ce = trainBpbAt(step) * Math.LN2;
    records.push({
      step,
      elapsed_min: Number(elapsed.toFixed(3)),
      lr: 3e-3 * (step < 300 ? step / 300 : Math.max(0.1, 1 - (step - 300) / (total * 1.2))),
      target_ratio: 5 + 1.5 * Math.min(1, step / (total * 0.6)),
      bytes_per_sec: 690000 + noise(step * 3) * 40000,
      train_bpb: Number(trainBpbAt(step).toFixed(4)),
      ce: Number(ce.toFixed(4)),
      ce_mbp: Number((ce * 1.18).toFixed(4)),
      ratio: Number((0.04 + Math.abs(noise(step * 7)) * 0.05).toFixed(4)),
      bpic: Number((4.4 + Math.min(1.8, step / total * 2) + noise(step * 11) * 0.2).toFixed(3)),
      grad_norm: Number((0.9 + Math.abs(noise(step * 5)) * 1.6).toFixed(3))
    });
    if (step % EVAL_EVERY === 0) {
      records.push({
        step,
        elapsed_min: Number(elapsed.toFixed(3)),
        eval: {
          val_bpb: Number(valAt(step).toFixed(4)),
          val_bpic: Number((4.6 + Math.min(1.7, (step / total) * 2)).toFixed(3)),
          target_ratio: 5 + 1.5 * Math.min(1, step / (total * 0.6)),
          boundary_on_separator_frac: Number(
            Math.min(0.94, 0.31 + 0.6 * (1 - Math.exp(-step / 1400))).toFixed(4)
          ),
          mbp_top1_acc: Number(Math.min(0.72, 0.12 + 0.7 * (1 - Math.exp(-step / 2200))).toFixed(4)),
          sample: SAMPLES[Math.min(SAMPLES.length - 1, Math.floor(step / (total / 3)))]
        }
      });
    }
  }
  const from = Math.max(0, Math.min(since, records.length));
  return { records: records.slice(from), next: records.length };
}
