// Live diagnostics for the reply currently streaming. Nothing here is retained between
// replies and nothing is synthesised — if the backend has not sent a value, it shows as "—".

import type { DiagnosticsEvent, StatsPayload } from '../types';

class Diagnostics {
  stats = $state<StatsPayload | null>(null);
  latest = $state<DiagnosticsEvent | null>(null);
  /** bytes/s samples from consecutive `stats` events, for the throughput trace */
  rate = $state<number[]>([]);
  /** true while a reply is in flight — the difference between "0" and "not measured" */
  live = $state(false);
  updatedAt = $state(0);

  begin(): void {
    this.stats = null;
    this.latest = null;
    this.rate = [];
    this.live = true;
    this.updatedAt = Date.now();
  }

  applyStats(s: StatsPayload): void {
    this.stats = s;
    const next = this.rate.length >= 120 ? this.rate.slice(1) : this.rate.slice();
    next.push(s.bytes_per_sec);
    this.rate = next;
    this.updatedAt = Date.now();
  }

  applyDiagnostics(d: DiagnosticsEvent): void {
    this.latest = d;
    this.updatedAt = Date.now();
  }

  end(final: StatsPayload | null): void {
    if (final) this.stats = final;
    this.live = false;
    this.updatedAt = Date.now();
  }
}

export const diagnostics = new Diagnostics();
