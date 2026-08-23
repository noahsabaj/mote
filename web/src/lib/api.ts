// HTTP client. Same origin in production (the FastAPI app serves web/dist at `/`);
// in `npm run dev` the same paths are answered by the dev-only mock plugin.

import type {
  CheckpointListItem,
  Health,
  LogPage,
  ModelInfo,
  TrainingRun
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
  runLog: (id: string, since = 0) =>
    request<LogPage>(`/api/training/runs/${encodeURIComponent(id)}/log?since=${since}`)
};
