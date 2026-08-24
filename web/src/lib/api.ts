// HTTP client. Same origin in production (the FastAPI app serves web/dist at `/`);
// in `npm run dev` the same paths are answered by the dev-only mock plugin.

import type {
  ChatRole,
  CheckpointListItem,
  ContextPreview,
  FoldMode,
  Health,
  JobsStatus,
  LogPage,
  ModelInfo,
  PrefsSummary,
  PrevFold,
  Rubric,
  TrainingRun,
  VoteBody
} from './types';
import { auth } from './stores/auth.svelte';

export class ApiError extends Error {
  readonly status: number;
  constructor(status: number, message: string) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, {
      ...init,
      headers: {
        Accept: 'application/json',
        ...(auth.token ? { Authorization: `Bearer ${auth.token}` } : {}),
        ...(init?.headers ?? {})
      }
    });
  } catch {
    throw new ApiError(0, 'Cannot reach the backend.');
  }
  if (!res.ok) {
    if (res.status === 401) auth.require();
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = (await res.json()) as { detail?: string; message?: string };
      detail = body.detail ?? body.message ?? detail;
    } catch {
      /* response had no JSON body */
    }
    throw new ApiError(res.status, detail);
  }
  return (await res.json()) as T;
}

export const api = {
  health: () => request<Health>('/api/health'),
  model: () => request<ModelInfo>('/api/model'),
  checkpoints: () => request<CheckpointListItem[]>('/api/checkpoints'),
  loadCheckpoint: (id: string) =>
    request<ModelInfo>('/api/checkpoints/load', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id })
    }),
  runs: () => request<TrainingRun[]>('/api/training/runs'),
  /** redeem a 6-digit pairing code shown on the PC's /pair page for the access token */
  pair: (code: string) =>
    request<{ token: string }>('/api/pair', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ code })
    }),
  /** what the next prompt would look like: bytes used, fold point, card — no generation */
  trainingQueue: () => request<JobsStatus>('/api/training/queue'),
  trainingStart: (args: string[]) =>
    request<JobsStatus & { submitted: string }>('/api/training/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ args })
    }),
  trainingStop: (id: string | null = null) =>
    request<JobsStatus>('/api/training/stop', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id })
    }),
  loadChallenger: (id: string) =>
    request<ModelInfo>('/api/challenger/load', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id })
    }),
  dropChallenger: () => request<ModelInfo>('/api/challenger', { method: 'DELETE' }),
  prefsVote: (body: VoteBody) =>
    request<PrefsSummary>('/api/prefs/vote', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    }),
  prefsSummary: () => request<PrefsSummary>('/api/prefs/summary'),
  prefsRubric: () => request<Rubric>('/api/prefs/rubric'),
  context: (
    messages: { role: ChatRole; content: string }[],
    max_bytes: number,
    fold: FoldMode,
    card: string | null,
    prev: PrevFold | null = null
  ) =>
    request<ContextPreview>('/api/context', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages, max_bytes, fold, card, prev })
    }),
  runLog: (id: string, since = 0) =>
    request<LogPage>(`/api/training/runs/${encodeURIComponent(id)}/log?since=${since}`)
};
