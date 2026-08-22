// Dev-only byte-stream simulator for /ws/generate.
//
// It exists so the streaming interface can be judged in motion: realistic pacing
// (25-60 bytes/s), word-like chunk boundaries, occasional parallel multi-byte acceptances,
// and at least one multi-byte UTF-8 character per reply so the "pending" path is exercised.
// The text is drawn from a small fixed set of sentences — no model is involved.

import type { WebSocket } from 'ws';
import type { ClientGenerate, SamplingParams, StatsPayload } from '../src/lib/types';
import { state } from './data';

const REPLIES = [
  'The router looks at each byte next to the one before it. Where they stop resembling ' +
    'each other it opens a chunk — so the boundaries land near word edges without anyone ' +
    'writing a tokenizer. A chunk is not a word, though: it is whatever the encoder found ' +
    'worth separating.',
  'Dynamic chunking is a routing decision, not a lookup. Each byte gets a boundary ' +
    'probability; the ones that clear the bar start a new chunk, and the main network only ' +
    'ever sees the compacted chunk sequence. The decoder expands it back to bytes.',
  'Bytes are the whole vocabulary — 256 of them plus a few control symbols. Nothing is ' +
    'out of vocabulary, so a naïve spelling, an emoji or a stray control code all cost ' +
    'exactly what they weigh in UTF-8.',
  'The multi-byte head proposes several bytes at once from the chunk it just read. When ' +
    'its confidence clears the accept threshold τ, those bytes are taken in parallel and ' +
    'the sampler skips ahead; otherwise generation falls back to one byte at a time.',
  'Retention is the mean of exp(A·Δt) per head — how much of the state each head carries ' +
    'forward over a step. Low retention means the head is forgetting quickly, which is not ' +
    'automatically bad; short-range heads are supposed to.'
];

function hash(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

const SEP = new Set<number>([32, 10, 9]);
function isSeparator(b: number): boolean {
  if (SEP.has(b)) return true;
  if (b >= 33 && b <= 47) return true;
  if (b >= 58 && b <= 64) return true;
  return false;
}

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

export interface Session {
  cancel(): void;
  active: boolean;
}

export function runGeneration(ws: WebSocket, req: ClientGenerate): Session {
  const session: Session = { active: true, cancel: () => (session.active = false) };
  void stream(ws, req, session);
  return session;
}

function send(ws: WebSocket, obj: unknown): boolean {
  if (ws.readyState !== 1) return false;
  ws.send(JSON.stringify(obj));
  return true;
}

async function stream(ws: WebSocket, req: ClientGenerate, session: Session): Promise<void> {
  if (state.swapping) {
    send(ws, { type: 'error', message: 'A checkpoint swap is in progress. Try again in a moment.' });
    session.active = false;
    return;
  }

  const params: SamplingParams = req.params;
  const enc = new TextEncoder();
  const dec = new TextDecoder('utf-8');

  const contextLimit = 2048;
  const conversation = req.messages.map((m) => `${m.role}: ${m.content}`).join('\n');
  const promptBytesTotal = enc.encode(conversation).length;
  const truncated = promptBytesTotal > contextLimit;
  const promptBytes = Math.min(promptBytesTotal, contextLimit);

  const last = req.messages.filter((m) => m.role === 'user').pop();
  const seed = hash((last?.content ?? '') + String(req.messages.length));
  let reply = REPLIES[seed % REPLIES.length];
  // Long-form prompts get a longer answer, so the byte inspector has something to virtualize.
  if ((last?.content.length ?? 0) > 120) reply = `${reply} ${REPLIES[(seed + 2) % REPLIES.length]}`;

  const bytes = enc.encode(reply).slice(0, Math.max(32, params.max_bytes));

  send(ws, {
    type: 'start',
    prompt_bytes: promptBytes,
    context_bytes: promptBytes,
    context_limit: contextLimit,
    truncated
  });

  const t0 = Date.now();
  let chunkIndex = 0;
  let chunkStart = 0;
  let chunkTextStart = 0;
  let emitted = 0;
  let chunksClosed = 0;
  let mbpProposed = 0;
  let mbpAccepted = 0;
  let pendingBuf: number[] = [];
  let decoded = '';
  const boundaryHistory: number[] = [];

  const stats = (): StatsPayload => {
    const elapsed = Date.now() - t0;
    return {
      bytes: emitted,
      elapsed_ms: elapsed,
      bytes_per_sec: emitted / Math.max(0.001, elapsed / 1000),
      chunks: chunksClosed,
      bytes_per_chunk: chunksClosed ? emitted / chunksClosed : emitted,
      mbp_proposed: mbpProposed,
      mbp_accepted: mbpAccepted,
      mbp_accept_rate: mbpProposed ? mbpAccepted / mbpProposed : 0,
      context_bytes: promptBytes + emitted,
      context_limit: contextLimit
    };
  };

  const rate = 26 + (seed % 32); // 26-57 bytes/s
  let mbpRun = 0;

  for (let i = 0; i < bytes.length; i++) {
    if (!session.active || ws.readyState !== 1) break;

    const b = bytes[i];
    const prev = i > 0 ? bytes[i - 1] : 32;
    const boundary = i === 0 || isSeparator(prev);
    const boundaryP = boundary
      ? 0.72 + Math.random() * 0.27
      : Math.max(0.002, Math.random() * 0.18);
    boundaryHistory.push(Number(boundaryP.toFixed(3)));
    if (boundaryHistory.length > 64) boundaryHistory.shift();

    if (boundary && i > 0) {
      const text = decoded.slice(chunkTextStart);
      send(ws, {
        type: 'chunk',
        index: chunkIndex,
        start: chunkStart,
        end: i,
        bytes: i - chunkStart,
        text
      });
      chunksClosed += 1;
      chunkIndex += 1;
      chunkStart = i;
      chunkTextStart = decoded.length;
      send(ws, {
        type: 'diagnostics',
        mamba3: {
          encoder_retention: bars(4, seed + chunkIndex, 0.55, 0.98),
          decoder_retention: bars(4, seed + chunkIndex * 3, 0.4, 0.95)
        },
        relation: { exchange_mass: bars(6, seed + chunkIndex * 7, 0.2, 0.88) },
        boundary_probs: boundaryHistory.slice()
      });
    }

    // Multi-byte head: once in a while it proposes ahead and the proposal is accepted.
    let source: 'nbp' | 'mbp' = 'nbp';
    if (mbpRun > 0) {
      mbpRun -= 1;
      source = 'mbp';
      mbpAccepted += 1;
    } else if (boundary && Math.random() < 0.42) {
      mbpProposed += params.n_candidates;
      const take = 1 + Math.floor(Math.random() * Math.max(1, params.n_candidates - 1));
      mbpAccepted += 1;
      mbpRun = take - 1;
      source = 'mbp';
    }

    // UTF-8 assembly: continuation bytes complete nothing, so `text` stays null.
    pendingBuf.push(b);
    let text: string | null = null;
    let pending = pendingBuf.length;
    if ((b & 0xc0) !== 0x80) {
      const need = b < 0x80 ? 1 : b < 0xe0 ? 2 : b < 0xf0 ? 3 : 4;
      if (need === 1) {
        text = dec.decode(new Uint8Array(pendingBuf));
        pendingBuf = [];
        pending = 0;
      }
    } else {
      const lead = pendingBuf[0];
      const need = lead < 0xe0 ? 2 : lead < 0xf0 ? 3 : 4;
      if (pendingBuf.length >= need) {
        text = dec.decode(new Uint8Array(pendingBuf));
        pendingBuf = [];
        pending = 0;
      }
    }
    if (text) decoded += text;

    const ok = send(ws, {
      type: 'byte',
      i,
      byte: b,
      text,
      pending,
      p: Number((0.16 + Math.random() * 0.7).toFixed(4)),
      entropy: Number((0.4 + Math.random() * 3.1).toFixed(3)),
      boundary,
      boundary_p: Number(boundaryP.toFixed(4)),
      chunk: chunkIndex,
      source,
      t_ms: Number((Date.now() - t0).toFixed(1))
    });
    if (!ok) return;
    emitted += 1;

    if (emitted % 16 === 0) send(ws, { type: 'stats', ...stats() });

    // Parallel acceptances arrive together; single bytes arrive at the sampling rate.
    await sleep(source === 'mbp' && mbpRun > 0 ? 2 : (1000 / rate) * (0.65 + Math.random() * 0.8));
  }

  if (ws.readyState !== 1) return;

  if (chunkStart < emitted) {
    send(ws, {
      type: 'chunk',
      index: chunkIndex,
      start: chunkStart,
      end: emitted,
      bytes: emitted - chunkStart,
      text: decoded.slice(chunkTextStart)
    });
    chunksClosed += 1;
  }

  const reason = !session.active ? 'stopped' : emitted >= params.max_bytes ? 'max_bytes' : 'eos';
  send(ws, { type: 'done', reason, text: decoded, stats: stats() });
  session.active = false;
}

function bars(n: number, seed: number, lo: number, hi: number): number[] {
  const out: number[] = [];
  for (let i = 0; i < n; i++) {
    const x = Math.sin((seed + i * 37) * 0.7351) * 0.5 + 0.5;
    out.push(Number((lo + x * (hi - lo)).toFixed(3)));
  }
  return out;
}
