// Types for the Mote Studio backend contract (docs/api.md v1).
// Every field here is produced by the real backend; nothing is invented client-side.

export type HonestyStatus = 'pilot' | 'undertrained' | 'flagship';

export interface Health {
  ok: boolean;
  model_loaded: boolean;
}

export interface CheckpointInfo {
  path: string;
  step: number;
  bytes_seen: number;
  /** null until the run log contains an eval record */
  val_bpb: number | null;
  trained_minutes: number | null;
  created_at: string;
}

export interface Architecture {
  outer_width: number;
  encoder_layers: number;
  decoder_layers: number;
  main: string;
  mbp_layers: number;
}

export interface DeviceInfo {
  name: string;
  vram_total_mb: number;
  vram_used_mb: number;
}

export interface Kernels {
  mamba3: boolean;
  ssd: boolean;
}

export interface SamplingParams {
  temperature: number;
  top_p: number;
  max_bytes: number;
  n_candidates: number;
}

export interface ModelInfo {
  name: string;
  params: number;
  status: HonestyStatus;
  status_note: string;
  checkpoint: CheckpointInfo;
  architecture: Architecture;
  context_limit_bytes: number;
  device: DeviceInfo;
  kernels: Kernels;
  defaults: SamplingParams;
  /** identity / pushback probe (mote.eval.probe), measured on this checkpoint; absent until run */
  probe?: {
    /** primary scores: held-out prompts, facts and pushback wordings absent from the identity training data */
    identity_acc: number; hold_rate: number; concede_rate: number; n_identity: number; n_facts: number;
    /** the same scores on prompts that share the training templates (present once the probe has both sets) */
    identity_acc_seen?: number; hold_rate_seen?: number; concede_rate_seen?: number; n_identity_seen?: number; n_facts_seen?: number;
  } | null;
  /** the system message the engine prepends so the model knows what it is */
  identity_card?: string;
  /** a second engine loaded for blind side-by-side comparisons (docs/prefs.md), or null */
  challenger?: ChallengerInfo | null;
  /** "<run>/ema@<step>" while a training job's EMA answers chats (docs/shape.md), else null */
  live?: string | null;
}

// ------------------------------------------------------------------- training jobs (docs/shape.md)

export interface TrainingJob {
  id: string;
  argv: string[];
  state: 'queued' | 'running' | 'done' | 'failed' | 'cancelled' | 'interrupted';
  created_at: number;
  started_at: number | null;
  ended_at: number | null;
  error: string | null;
  resumed: boolean;
}

export interface JobsStatus {
  current: TrainingJob | null;
  queued: TrainingJob[];
  recent: TrainingJob[];
}

export interface ChallengerInfo {
  id: string;
  name: string;
  step: number;
  val_bpb: number | null;
  loading: boolean;
}

export interface CheckpointListItem {
  id: string;
  step: number;
  val_bpb: number | null;
  /** training data consumed — step × batch × seq × grad_accum, *not* the file's size */
  bytes_seen: number;
  /** the .pt on disk; an order of magnitude larger than bytes_seen, so both are labelled */
  file_size_bytes: number;
  created_at: string;
  loaded: boolean;
  /** loaded as the challenger (docs/prefs.md) */
  challenger?: boolean;
}

// ------------------------------------------------------------------- preferences (docs/prefs.md)

export type EngineRole = 'current' | 'challenger';

/** where a reply came from, captured when it was requested */
export interface ReplySource {
  checkpoint: string;
  step: number;
  engine: EngineRole;
  params: SamplingParams;
}

export type PairVote = 'a' | 'b' | 'tie' | 'both_bad';
export type PairOrigin = 'retry' | 'compare' | 'arena';

/** a reply slot with two candidates up for a vote; ids point into the slot's sample pool */
export interface ComparePair {
  aId: string;
  bId: string;
  origin: PairOrigin;
  vote?: PairVote | null;
  reason?: string;
  skipped?: boolean;
}

export interface VoteBody {
  pair: {
    messages: { role: ChatRole; content: string }[];
    a: string;
    b: string;
    a_source: ReplySource;
    b_source: ReplySource;
    origin: PairOrigin;
  };
  vote: PairVote | null;
  reason: string;
}

export interface PrefsTableRow {
  a: string;
  b: string;
  a_wins: number;
  b_wins: number;
  ties: number;
  both_bad: number;
  n: number;
}

export interface PrefsSummary {
  pairs: number;
  votes: { user: number; claude: number };
  unrated_by_claude: number;
  table: PrefsTableRow[];
  agreement: { n: number; agree: number; rate: number | null };
  rubric: string | null;
  /** the id of the pair just stored (POST /api/prefs/vote only) */
  pair?: string;
}

export interface Rubric {
  text: string;
  hash: string | null;
}

export interface TrainingRun {
  id: string;
  steps: number;
  last_val_bpb: number | null;
  running: boolean;
  started_at: string;
}

/** Every log record carries `step` and `elapsed_min`; the rest depends on the record kind. */
export interface LogRecordBase {
  step: number;
  elapsed_min?: number;
}

export interface TrainLogRecord extends LogRecordBase {
  lr: number;
  target_ratio: number;
  bytes_per_sec: number;
  train_bpb: number;
  ce: number;
  ce_mbp?: number;
  ratio: number;
  bpic: number;
  grad_norm: number;
}

export interface EvalPayload {
  val_bpb: number;
  val_bpic: number;
  target_ratio?: number;
  boundary_on_separator_frac: number;
  mbp_top1_acc?: number;
  sample?: string;
}

export interface EvalLogRecord extends LogRecordBase {
  eval: EvalPayload;
}

/** The run log is JSONL and heterogeneous — checkpoint / done / stopped markers appear too. */
export interface LogRecord extends LogRecordBase {
  [key: string]: unknown;
}

export type EvalRecord = LogRecord & EvalLogRecord;
export type TrainRecord = LogRecord & TrainLogRecord;

export interface LogPage {
  records: LogRecord[];
  next: number;
}

export function isEvalRecord(r: LogRecord): r is EvalRecord {
  return typeof r.eval === 'object' && r.eval !== null;
}

export function isTrainRecord(r: LogRecord): r is TrainRecord {
  return typeof r.train_bpb === 'number';
}

// ---------------------------------------------------------------- websocket

/** nbp: sampled from the next-byte head · mbp: a draft byte accepted by exact verification · fix: the
 *  correction drawn when a draft byte was rejected (still distributed exactly as the model would sample). */
export type ByteSource = 'nbp' | 'mbp' | 'fix';

/** folding (docs/context.md): 'auto' folds when the prompt would overflow, 'now' folds everything before
 *  the last user turn, 'off' is plain truncation (drop oldest) */
export type FoldMode = 'auto' | 'now' | 'off';

/** the first `from` turns were folded into `card`, which rides inside the first kept user turn */
export interface FoldInfo {
  from: number;
  turns: number;
  card: string;
}

/** POST /api/context — what the next prompt would look like, without generating */
export interface ContextPreview {
  used: number;
  limit: number;
  reserve: number;
  fold: FoldInfo | null;
  truncated: boolean;
  /** bytes of the would-be prompt the engine already holds a state for (its prefix cache) */
  reusable?: number;
}

/** the client's last fold, sent back so the server keeps the same fold point and card while the prompt fits */
export interface PrevFold {
  from: number;
  card: string;
}

/** from `start`: how much of the prompt came from the engine's prefix cache (docs/context.md) */
export interface PrefixInfo {
  reused: number;
  prefilled: number;
  prefill_ms: number;
  snapshots: number;
  cache_bytes: number;
  cache_budget: number;
  hits: number;
  misses: number;
}

/** the "verify prefix cache" toggle: a cold re-read of the prompt compared with the warm continuation */
export interface PrefixCheck {
  reused: number;
  prefilled: number;
  boundary_flips: number;
  chunks_cold: number;
  chunks_warm: number;
  max_logit_diff: number;
  cold_ms: number;
}

export interface ClientGenerate {
  type: 'generate';
  messages: { role: ChatRole; content: string }[];
  params: SamplingParams;
  context?: { fold: FoldMode; card?: string | null; prev?: PrevFold | null; verify_prefix?: boolean };
  /** which loaded engine answers: the served one (default) or the challenger */
  engine?: EngineRole;
}

export interface ClientStop {
  type: 'stop';
}

export type ClientMessage = ClientGenerate | ClientStop;

export interface StartEvent {
  type: 'start';
  prompt_bytes: number;
  context_bytes: number;
  context_limit: number;
  /** only when even folding could not fit (a giant message) */
  truncated: boolean;
  fold?: FoldInfo | null;
  prefix?: PrefixInfo;
  /** the checkpoint that is answering */
  checkpoint?: { name: string; step: number };
}

export interface ByteEvent {
  type: 'byte';
  i: number;
  byte: number;
  text: string | null;
  pending: number;
  p: number;
  entropy: number;
  boundary: boolean;
  boundary_p: number;
  chunk: number;
  source: ByteSource;
  t_ms: number;
}

export interface ChunkEvent {
  type: 'chunk';
  index: number;
  start: number;
  end: number;
  bytes: number;
  text: string;
}

export interface StatsPayload {
  bytes: number;
  elapsed_ms: number;
  bytes_per_sec: number;
  chunks: number;
  bytes_per_chunk: number;
  mbp_proposed: number;
  mbp_accepted: number;
  mbp_accept_rate: number;
  spec_rounds?: number;
  spec_fixes?: number;
  spec_replays?: number;
  context_bytes: number;
  context_limit: number;
}

export interface StatsEvent extends StatsPayload {
  type: 'stats';
}

export interface DiagnosticsEvent {
  type: 'diagnostics';
  mamba3: { encoder_retention: number[]; decoder_retention: number[] };
  relation: { exchange_mass: number[] };
  boundary_probs: number[];
  /** only on the event sent right after `start` when the client asked to verify the prefix cache */
  prefix_check?: PrefixCheck;
}

export interface DoneEvent {
  type: 'done';
  reason: 'eos' | 'max_bytes' | 'stopped';
  text: string;
  stats: StatsPayload;
}

export interface ErrorEvent {
  type: 'error';
  message: string;
}

export type ServerEvent =
  | StartEvent
  | ByteEvent
  | ChunkEvent
  | StatsEvent
  | DiagnosticsEvent
  | DoneEvent
  | ErrorEvent;

// ------------------------------------------------------------------- client

export type ChatRole = 'user' | 'assistant' | 'system';
