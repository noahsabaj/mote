import { api, ApiError } from '../api';
import type { JobsStatus, LogRecord, TrainingRun } from '../types';

const POLL_MS = 3000;

/**
 * The daemon's job queue and the runs on disk (docs/shape.md): what the Training sheet shows.
 *
 * The sheet used to own all of this — fetching, polling, the per-run log cursors — as component
 * state, the one domain of ten without a store. A store outlives the sheet: reopening it does not
 * refetch every log from line 0, and anything else that wants to know whether a job is running
 * (the header, a notice) reads the same object.
 */
class TrainingStore {
  runs = $state<TrainingRun[]>([]);
  runsError = $state<string | null>(null);
  jobs = $state<JobsStatus | null>(null);
  jobBusy = $state(false);
  jobError = $state<string | null>(null);
  /** runs whose curves are shown; the first is the primary (details, probe sample) */
  selected = $state<string[]>([]);
  logs = $state<Record<string, LogRecord[]>>({});
  logError = $state<string | null>(null);
  /** next line to ask for, per run: later pulls only ask for what is new */
  #cursors: Record<string, number> = {};
  #pulling = new Set<string>();
  #timers: ReturnType<typeof setInterval>[] = [];
  #watchers = 0;

  async loadJobs(): Promise<void> {
    try {
      this.jobs = await api.trainingQueue();
      this.jobError = null;
    } catch (e) {
      this.jobError = e instanceof ApiError ? e.message : String(e);
    }
  }

  // Jobs are started from the terminal (`mote train start … --serve`); the sheet is for watching them,
  // stopping them, and choosing which one is on the air (R8 + R1, signed 2026-08-25).
  async serveJob(id: string | null, on: boolean): Promise<void> {
    if (this.jobBusy) return;
    this.jobBusy = true;
    try {
      this.jobs = await api.trainingServe(id, on);
      this.jobError = null;
    } catch (e) {
      this.jobError = e instanceof ApiError ? e.message : String(e);
    } finally {
      this.jobBusy = false;
    }
  }

  async stopJob(id: string | null = null): Promise<void> {
    if (this.jobBusy) return;
    this.jobBusy = true;
    try {
      this.jobs = await api.trainingStop(id);
      this.jobError = null;
    } catch (e) {
      this.jobError = e instanceof ApiError ? e.message : String(e);
    } finally {
      this.jobBusy = false;
    }
  }

  async loadRuns(): Promise<void> {
    try {
      this.runs = await api.runs();
      this.runsError = null;
      if (!this.selected.length && this.runs.length) this.select(this.runs[0].id);
    } catch (e) {
      this.runsError = e instanceof ApiError ? e.message : String(e);
    }
  }

  async pull(id: string): Promise<void> {
    if (this.#pulling.has(id)) return;
    this.#pulling.add(id);
    try {
      const page = await api.runLog(id, this.#cursors[id] ?? 0);
      if (page.records.length) this.logs = { ...this.logs, [id]: [...(this.logs[id] ?? []), ...page.records] };
      this.#cursors[id] = page.next;
      this.logError = null;
    } catch (e) {
      this.logError = e instanceof ApiError ? e.message : String(e);
    } finally {
      this.#pulling.delete(id);
    }
  }

  /** Show a run; its log starts on the first selection. */
  select(id: string): void {
    if (!this.selected.includes(id)) this.selected = [...this.selected, id];
    if (this.#cursors[id] === undefined) {
      this.#cursors[id] = 0;
      void this.pull(id);
    }
  }

  toggle(id: string): void {
    if (this.selected.includes(id)) {
      if (this.selected.length > 1) this.selected = this.selected.filter((x) => x !== id);
    } else {
      this.select(id);
    }
  }

  /** A sheet is watching: refresh now and poll while it stays open. Returns the release. */
  watch(): () => void {
    this.#watchers += 1;
    void this.loadRuns();
    void this.loadJobs();
    if (this.#watchers === 1) {
      this.#timers = [
        setInterval(() => {
          void this.loadJobs();
          if (this.jobs?.current) void this.loadRuns();
        }, POLL_MS * 2),
        setInterval(() => {
          const live = this.selected.filter((id) => this.runs.find((r) => r.id === id)?.running);
          if (!live.length) return;
          for (const id of live) void this.pull(id);
          void this.loadRuns();
        }, POLL_MS)
      ];
    }
    return () => {
      this.#watchers -= 1;
      if (this.#watchers === 0) {
        for (const t of this.#timers) clearInterval(t);
        this.#timers = [];
      }
    };
  }
}

export const training = new TrainingStore();
