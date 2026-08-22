// Compact per-byte model for one streamed reply.
//
// A reply can be thousands of `byte` events. Everything scalar lives in typed arrays that
// grow by doubling; nothing here creates a DOM node. `version` is the only fine-grained
// reactive signal — it is bumped once per animation frame by the flush loop, so components
// that show detail re-read the arrays at most once a frame.

import type { ByteEvent, ChunkEvent } from './types';

export interface ChunkRow {
  index: number;
  start: number;
  end: number;
  bytes: number;
  text: string;
}

/** A run of characters that shares one origin — used to draw legible structure, not per-byte boxes. */
export interface Segment {
  text: string;
  mbp: boolean;
}

const FLAG_BOUNDARY = 1;
const FLAG_MBP = 2;

export class ByteTrace {
  /** Bumped once per flush; read it to make a component depend on the whole trace. */
  version = $state(0);
  /** Decoded text so far. Updated on flush, not per byte. */
  text = $state('');
  /** Bytes currently buffered mid-character (backend's `pending`). */
  pending = $state(0);
  #pendingRaw = 0;
  /** Number of bytes received. Updated on flush. */
  count = $state(0);
  /** Chunks closed so far. Updated on flush. */
  chunkCount = $state(0);

  #cap = 1024;
  #n = 0;
  #byte = new Uint8Array(this.#cap);
  #p = new Float32Array(this.#cap);
  #entropy = new Float32Array(this.#cap);
  #boundaryP = new Float32Array(this.#cap);
  #chunk = new Int32Array(this.#cap);
  #tMs = new Float32Array(this.#cap);
  #flags = new Uint8Array(this.#cap);
  /** Cumulative length of `#text` after this byte, so a byte's characters are a slice. */
  #textEnd = new Int32Array(this.#cap);
  #text = '';
  #chunks: ChunkRow[] = [];
  #dirty = false;

  get size(): number {
    return this.#n;
  }

  get chunks(): readonly ChunkRow[] {
    return this.#chunks;
  }

  get liveText(): string {
    return this.#text;
  }

  push(ev: ByteEvent): void {
    if (this.#n === this.#cap) this.#grow();
    const s = this.#n;
    this.#byte[s] = ev.byte & 0xff;
    this.#p[s] = ev.p;
    this.#entropy[s] = ev.entropy;
    this.#boundaryP[s] = ev.boundary_p;
    this.#chunk[s] = ev.chunk;
    this.#tMs[s] = ev.t_ms;
    this.#flags[s] = (ev.boundary ? FLAG_BOUNDARY : 0) | (ev.source === 'mbp' ? FLAG_MBP : 0);
    if (ev.text) this.#text += ev.text;
    this.#textEnd[s] = this.#text.length;
    this.#n = s + 1;
    this.#pendingRaw = ev.pending;
    this.#dirty = true;
  }

  closeChunk(ev: ChunkEvent): void {
    this.#chunks.push({
      index: ev.index,
      start: ev.start,
      end: ev.end,
      bytes: ev.bytes,
      text: ev.text
    });
    this.#dirty = true;
  }

  /** Publish accumulated state to the reactive fields. Called once per frame. */
  flush(): boolean {
    if (!this.#dirty) return false;
    this.#dirty = false;
    this.text = this.#text;
    this.pending = this.#pendingRaw;
    this.count = this.#n;
    this.chunkCount = this.#chunks.length;
    this.version++;
    return true;
  }

  /** Replace the decoded text with the authoritative one from `done`. */
  settle(text: string): void {
    this.#text = text;
    this.text = text;
    this.#pendingRaw = 0;
    this.pending = 0;
    this.count = this.#n;
    this.chunkCount = this.#chunks.length;
    this.version++;
  }

  byteAt(i: number): {
    i: number;
    byte: number;
    p: number;
    entropy: number;
    boundaryP: number;
    chunk: number;
    tMs: number;
    boundary: boolean;
    mbp: boolean;
    chars: string;
  } {
    const from = i > 0 ? this.#textEnd[i - 1] : 0;
    return {
      i,
      byte: this.#byte[i],
      p: this.#p[i],
      entropy: this.#entropy[i],
      boundaryP: this.#boundaryP[i],
      chunk: this.#chunk[i],
      tMs: this.#tMs[i],
      boundary: (this.#flags[i] & FLAG_BOUNDARY) !== 0,
      mbp: (this.#flags[i] & FLAG_MBP) !== 0,
      chars: this.#text.slice(from, this.#textEnd[i])
    };
  }

  /** Boundary probability of every byte, for the sparkline. */
  boundaryProbs(last = 64): number[] {
    const from = Math.max(0, this.#n - last);
    const out: number[] = [];
    for (let i = from; i < this.#n; i++) out.push(this.#boundaryP[i]);
    return out;
  }

  /** Fraction of bytes that came from the multi-byte head. */
  mbpFraction(): number {
    if (this.#n === 0) return 0;
    let hits = 0;
    for (let i = 0; i < this.#n; i++) if (this.#flags[i] & FLAG_MBP) hits++;
    return hits / this.#n;
  }

  /**
   * Split one chunk's characters into runs by origin. A character is attributed to the
   * multi-byte head when the byte that completed it was accepted in parallel.
   */
  segmentsFor(start: number, end: number): Segment[] {
    const lo = Math.max(0, start);
    const hi = Math.min(this.#n, end);
    const out: Segment[] = [];
    for (let i = lo; i < hi; i++) {
      const from = i > 0 ? this.#textEnd[i - 1] : 0;
      const to = this.#textEnd[i];
      if (to === from) continue;
      const mbp = (this.#flags[i] & FLAG_MBP) !== 0;
      const chars = this.#text.slice(from, to);
      const tail = out[out.length - 1];
      if (tail && tail.mbp === mbp) tail.text += chars;
      else out.push({ text: chars, mbp });
    }
    return out;
  }

  /** Chunks plus the still-open trailing chunk, so live text is never dropped. */
  chunkRows(): ChunkRow[] {
    const rows = this.#chunks.slice();
    const last = rows.length ? rows[rows.length - 1].end : 0;
    if (last < this.#n) {
      rows.push({
        index: rows.length,
        start: last,
        end: this.#n,
        bytes: this.#n - last,
        text: this.#text.slice(last > 0 ? this.#textEnd[last - 1] : 0)
      });
    }
    return rows;
  }

  #grow(): void {
    const cap = this.#cap * 2;
    const u8 = (src: Uint8Array) => {
      const d = new Uint8Array(cap);
      d.set(src);
      return d;
    };
    const f32 = (src: Float32Array) => {
      const d = new Float32Array(cap);
      d.set(src);
      return d;
    };
    const i32 = (src: Int32Array) => {
      const d = new Int32Array(cap);
      d.set(src);
      return d;
    };
    this.#byte = u8(this.#byte);
    this.#flags = u8(this.#flags);
    this.#p = f32(this.#p);
    this.#entropy = f32(this.#entropy);
    this.#boundaryP = f32(this.#boundaryP);
    this.#tMs = f32(this.#tMs);
    this.#chunk = i32(this.#chunk);
    this.#textEnd = i32(this.#textEnd);
    this.#cap = cap;
  }
}
