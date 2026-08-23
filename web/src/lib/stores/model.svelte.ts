import { api, ApiError } from '../api';
import * as persist from '../persist';
import { settings } from './settings.svelte';
import type { CheckpointListItem, ModelInfo } from '../types';

const RECENTS_KEY = 'ckpt.recents';
const RECENTS_MAX = 8;

class ModelStore {
  info = $state<ModelInfo | null>(null);
  error = $state<string | null>(null);
  loading = $state(true);
  checkpoints = $state<CheckpointListItem[]>([]);
  checkpointError = $state<string | null>(null);
  /** id of the checkpoint currently being hot-swapped in, if any */
  swapping = $state<string | null>(null);
  /** id of the checkpoint being loaded as the challenger, if any */
  challengerLoading = $state<string | null>(null);
  /**
   * Checkpoints you have actually loaded, newest first — what the composer's picker offers.
   * A few more are remembered than it shows, so deleting a run does not shorten the list.
   */
  recents = $state<string[]>(persist.read<string[]>(RECENTS_KEY, []));

  get busy(): boolean {
    return this.swapping !== null;
  }

  private remember(id: string): void {
    this.recents = [id, ...this.recents.filter((x) => x !== id)].slice(0, RECENTS_MAX);
    persist.write(RECENTS_KEY, this.recents);
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
      this.remember(id); // only a load that succeeded counts as one you have used
      await this.refreshCheckpoints();
    } catch (e) {
      this.checkpointError = e instanceof ApiError ? e.message : String(e);
    } finally {
      this.swapping = null;
    }
  }

  /** Load a second engine next to the served one, for blind comparisons (docs/prefs.md). */
  async setChallenger(id: string): Promise<void> {
    if (this.challengerLoading || this.swapping) return;
    this.challengerLoading = id;
    this.checkpointError = null;
    try {
      this.info = await api.loadChallenger(id);
      await this.refreshCheckpoints();
    } catch (e) {
      this.checkpointError = e instanceof ApiError ? e.message : String(e);
    } finally {
      this.challengerLoading = null;
    }
  }

  async clearChallenger(): Promise<void> {
    try {
      this.info = await api.dropChallenger();
      await this.refreshCheckpoints();
    } catch (e) {
      this.checkpointError = e instanceof ApiError ? e.message : String(e);
    }
  }
}

export const model = new ModelStore();
