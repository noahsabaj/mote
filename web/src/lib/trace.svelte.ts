// Compact per-byte model for one streamed reply.
//
// A reply can be thousands of `byte` events. Everything scalar lives in typed arrays that
// grow by doubling; nothing here creates a DOM node. `version` is the only fine-grained
// reactive signal — it is bumped once per animation frame by the flush loop, so components
// that show detail re-read the arrays at most once a frame.

import type { ByteEvent } from './types';

export interface ChunkRow {
  index: number;
  /** byte-event index of the first byte of this chunk */
  start: number;
  /** byte-event index just past the last byte of this chunk */
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
/** The byte that replaced a rejected draft byte. Still an exact sample, but not a plain step. */
const FLAG_FIX = 4;

export interface SerializedTrace {
  v: 1;
  n: number;
  text: string;
  closed: number;
  runStart: number[];
  byte: string;
  flags: string;
  p: string;
  entropy: string;
  boundaryP: string;
  tMs: string;
  chunk: string;
  textEnd: string;
}

function b64(bytes: Uint8Array): string {
  let s = '';
  for (let i = 0; i < bytes.length; i += 0x2000) {
    s += String.fromCharCode.apply(null, Array.from(bytes.subarray(i, i + 0x2000)));
  }
  return btoa(s);
}

function unb64(s: string): Uint8Array {
  const bin = atob(s);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

function fill<T extends { set(a: ArrayLike<number>): void }>(dst: T, src: ArrayLike<number>): T {
  dst.set(src);
  return dst;
}

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
  /** Running sum of `#p`, so the mean is O(1) and never a scan over a long reply. */
  #pSum = 0;
  #entropy = new Float32Array(this.#cap);
  #boundaryP = new Float32Array(this.#cap);
  #chunk = new Int32Array(this.#cap);
  #tMs = new Float32Array(this.#cap);
  #flags = new Uint8Array(this.#cap);
  /** Cumulative length of `#text` after this byte, so a byte's characters are a slice. */
  #textEnd = new Int32Array(this.#cap);
  #text = '';
  /**
   * Byte-event index at which each chunk run starts. Chunk spans are derived from the
   * `chunk` field carried by every byte rather than from the `chunk` event's start/end:
   * the serving engine reports those in whole-context coordinates (prompt included) while
   * `byte.i` is reply-local, and its `end` disagrees with its own `bytes` count. The per-byte
   * field is unambiguous and always in the same frame as everything else here.
   */
  #runStart: number[] = [];
  #closed = 0;
  #dirty = false;

  get size(): number {
    return this.#n;
  }

  /** Number of chunk runs present, open one included. */
  get runCount(): number {
    return this.#runStart.length;
  }

  get liveText(): string {
    return this.#text;
  }

  /**
   * Mean probability of the byte the model actually chose, under the distribution it actually
   * sampled from. Unlike `entropy` (which the engine computes on the raw softmax, before
   * temperature and top-p), this moves when the sampling knobs move, so it is the one honest
   * per-reply answer to "what did that setting do".
   */
  get meanP(): number {
    return this.#n === 0 ? 0 : this.#pSum / this.#n;
  }

  push(ev: ByteEvent): void {
    if (this.#n === this.#cap) this.#grow();
    const s = this.#n;
    this.#byte[s] = ev.byte & 0xff;
    this.#p[s] = ev.p;
    this.#pSum += ev.p;
    this.#entropy[s] = ev.entropy;
    this.#boundaryP[s] = ev.boundary_p;
    this.#chunk[s] = ev.chunk;
    this.#tMs[s] = ev.t_ms;
    this.#flags[s] =
      (ev.boundary ? FLAG_BOUNDARY : 0) |
      (ev.source === 'mbp' ? FLAG_MBP : 0) |
      (ev.source === 'fix' ? FLAG_FIX : 0);
    if (s === 0 || this.#chunk[s - 1] !== ev.chunk) this.#runStart.push(s);
    if (ev.text) this.#text += ev.text;
    this.#textEnd[s] = this.#text.length;
    this.#n = s + 1;
    this.#pendingRaw = ev.pending;
    this.#dirty = true;
  }

  /** A `chunk` event only tells us one closed; its spans are not used (see `#runStart`). */
  closeChunk(): void {
    this.#closed += 1;
    this.#dirty = true;
  }

  /** Publish accumulated state to the reactive fields. Called once per frame. */
  flush(): boolean {
    if (!this.#dirty) return false;
    this.#dirty = false;
    this.text = this.#text;
    this.pending = this.#pendingRaw;
    this.count = this.#n;
    this.chunkCount = this.#closed;
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
    this.chunkCount = this.#closed;
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
    fix: boolean;
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
      fix: (this.#flags[i] & FLAG_FIX) !== 0,
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

  /** Every chunk, including the one still open, tiling the reply with no gaps. */
  chunkRows(): ChunkRow[] {
    const rows: ChunkRow[] = [];
    for (let r = 0; r < this.#runStart.length; r++) {
      const start = this.#runStart[r];
      const end = r + 1 < this.#runStart.length ? this.#runStart[r + 1] : this.#n;
      if (end <= start) continue;
      rows.push({
        index: this.#chunk[start],
        start,
        end,
        bytes: end - start,
        text: this.#text.slice(start > 0 ? this.#textEnd[start - 1] : 0, this.#textEnd[end - 1])
      });
    }
    return rows;
  }

  // ------------------------------------------------------------ persistence
  // Typed arrays are stored as base64 of their raw bytes; text and the chunk-run table as-is.

  toJSON(): SerializedTrace {
    const n = this.#n;
    return {
      v: 1,
      n,
      text: this.#text,
      closed: this.#closed,
      runStart: this.#runStart.slice(),
      byte: b64(this.#byte.subarray(0, n)),
      flags: b64(this.#flags.subarray(0, n)),
      p: b64(new Uint8Array(this.#p.buffer, 0, n * 4)),
      entropy: b64(new Uint8Array(this.#entropy.buffer, 0, n * 4)),
      boundaryP: b64(new Uint8Array(this.#boundaryP.buffer, 0, n * 4)),
      tMs: b64(new Uint8Array(this.#tMs.buffer, 0, n * 4)),
      chunk: b64(new Uint8Array(this.#chunk.buffer, 0, n * 4)),
      textEnd: b64(new Uint8Array(this.#textEnd.buffer, 0, n * 4))
    };
  }

  static fromJSON(d: SerializedTrace): ByteTrace {
    const t = new ByteTrace();
    const n = d.n;
    const cap = Math.max(1024, n);
    t.#cap = cap;
    t.#n = n;
    t.#byte = fill(new Uint8Array(cap), unb64(d.byte));
    t.#flags = fill(new Uint8Array(cap), unb64(d.flags));
    t.#p = fill(new Float32Array(cap), new Float32Array(unb64(d.p).buffer));
    t.#entropy = fill(new Float32Array(cap), new Float32Array(unb64(d.entropy).buffer));
    t.#boundaryP = fill(new Float32Array(cap), new Float32Array(unb64(d.boundaryP).buffer));
    t.#tMs = fill(new Float32Array(cap), new Float32Array(unb64(d.tMs).buffer));
    t.#chunk = fill(new Int32Array(cap), new Int32Array(unb64(d.chunk).buffer));
    t.#textEnd = fill(new Int32Array(cap), new Int32Array(unb64(d.textEnd).buffer));
    t.#runStart = d.runStart.slice();
    t.#closed = d.closed;
    t.#text = d.text;
    for (let i = 0; i < n; i++) t.#pSum += t.#p[i];
    t.settle(d.text);
    return t;
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
