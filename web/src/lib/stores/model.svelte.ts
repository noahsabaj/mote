import { api, ApiError } from '../api';
import { settings } from './settings.svelte';
import type { CheckpointListItem, ModelInfo } from '../types';

class ModelStore {
  info = $state<ModelInfo | null>(null);
  error = $state<string | null>(null);
  loading = $state(true);
  checkpoints = $state<CheckpointListItem[]>([]);
  checkpointError = $state<string | null>(null);
  /** id of the checkpoint currently being hot-swapped in, if any */
  swapping = $state<string | null>(null);

  get busy(): boolean {
    return this.swapping !== null;
  }

  async refresh(): Promise<void> {
    this.loading = true;
    try {
      const info = await api.model();
      this.info = info;
      settings.setDefaults(info.defaults);
      this.error = null;
    } catch (e) {
      this.error = e instanceof ApiError ? e.message : String(e);
    } finally {
      this.loading = false;
    }
  }

  async refreshCheckpoints(): Promise<void> {
    try {
      this.checkpoints = await api.checkpoints();
      this.checkpointError = null;
    } catch (e) {
      this.checkpointError = e instanceof ApiError ? e.message : String(e);
    }
  }

  async load(id: string): Promise<void> {
    if (this.swapping) return;
    this.swapping = id;
    this.checkpointError = null;
    try {
      const info = await api.loadCheckpoint(id);
      this.info = info;
      settings.setDefaults(info.defaults);
      this.error = null;
      await this.refreshCheckpoints();
    } catch (e) {
      this.checkpointError = e instanceof ApiError ? e.message : String(e);
    } finally {
      this.swapping = null;
    }
  }
}

export const model = new ModelStore();
