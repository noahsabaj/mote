// Types for the Morpheme Studio backend contract (docs/api.md v1).
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
  /** identity / pushback probe (morpheme.eval.probe), measured on this checkpoint; absent until run */
  probe?: { identity_acc: number; hold_rate: number; concede_rate: number; n_identity: number; n_facts: number } | null;
  /** the system message the engine prepends so the model knows what it is */
  identity_card?: string;
}

export interface CheckpointListItem {
  id: string;
  step: number;
  val_bpb: number | null;
  bytes_seen: number;
  created_at: string;
  loaded: boolean;
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

export interface ClientGenerate {
  type: 'generate';
  messages: { role: ChatRole; content: string }[];
  params: SamplingParams;
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
  truncated: boolean;
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
