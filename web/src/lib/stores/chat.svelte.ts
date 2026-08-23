// Conversation state and the streaming pipeline.
//
// The backend keeps no conversation, so the full message list is sent every turn and the
// transcript lives here (and in localStorage). Socket events are queued and applied once
// per animation frame: a reply is thousands of `byte` events and each one must not touch
// the DOM on its own.

import { api } from '../api';
import { GenerateSocket, type LinkState } from '../ws';
import { auth } from './auth.svelte';
import { ByteTrace, type SerializedTrace } from '../trace.svelte';
import * as persist from '../persist';
import { download } from '../download';
import { MODEL_NAME } from '../brand';
import { settings } from './settings.svelte';
import { diagnostics } from './diagnostics.svelte';
import { model } from './model.svelte';
import { notices } from './notice.svelte';
import { queue } from './queue.svelte';
import { ui } from './ui.svelte';
import { parseCommand, unescapeCommand } from '../commands';
import type {
  ChatRole,
  ContextPreview,
  DoneEvent,
  FoldInfo,
  PrefixCheck,
  PrefixInfo,
  PrevFold,
  SamplingParams,
  ServerEvent,
  StatsPayload
} from '../types';

export interface Turn {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  at: number;
  /** assistant only — final counters from `done` */
  stats?: StatsPayload | null;
  reason?: DoneEvent['reason'];
  error?: string | null;
  /** from `start`: even folding could not fit the prompt (a giant message) */
  truncated?: boolean;
  /** from `start`: what this reply saw in place of the oldest turns (docs/context.md) */
  fold?: FoldInfo | null;
  contextBytes?: number;
  contextLimit?: number;
  /** milliseconds from send to the first byte of the reply */
  ttfbMs?: number;
  /** from `start`: bytes of the prompt the engine reused from its prefix cache vs read afresh */
  prefix?: PrefixInfo | null;
  /** the cold re-read comparison, when "verify prefix cache" was on for this reply */
  prefixCheck?: PrefixCheck | null;
  /** earlier samples of this reply slot (Retry keeps them); each is a complete Turn */
  samples?: Turn[];
  /** assistant only — what this reply was actually drawn at, captured when it was sent */
  params?: SamplingParams;
  /** the params above that differed from the checkpoint's own defaults at that moment */
  offDefault?: (keyof SamplingParams)[];
  /** assistant only — step of the checkpoint that produced it, for the swap rule */
  checkpointStep?: number;
}

interface StoredConversation {
  id: string;
  title: string;
  turns: Turn[];
  updatedAt: number;
  /** the title was typed rather than derived, so saving must not overwrite it */
  titleLocked?: boolean;
}

export interface ConversationSummary {
  id: string;
  title: string;
  updatedAt: number;
}

const INDEX_KEY = 'conversations';
const CURRENT_KEY = 'current';
const convKey = (id: string) => `conv.${id}`;
const tracesKey = (id: string) => `conv.${id}.traces`;
const MAX_CONVERSATIONS = 30;
/** Per conversation, the byte traces of the most recent replies survive a reload. */
const MAX_TRACES = 8;

function newId(): string {
  return Math.random().toString(36).slice(2, 10) + Date.now().toString(36).slice(-4);
}

/** A conversation as prose, with each reply's counters and provenance under it as a quote. */
function toMarkdown(title: string, turns: Turn[], modelName: string): string {
  const out: string[] = [`# ${title}`, ''];
  for (const t of turns) {
    if (t.role === 'user') {
      out.push('## You', '', t.content, '');
      continue;
    }
    out.push(`## ${modelName}`, '', t.content || '_(no reply)_', '');
    const meta: string[] = [];
    if (t.stats) {
      meta.push(
        `${t.stats.bytes} bytes`,
        `${t.stats.chunks} chunks`,
        `${t.stats.bytes_per_sec.toFixed(0)} B/s`
      );
    }
    if (t.params) {
      const p = t.params;
      meta.push(`T ${p.temperature} · top-p ${p.top_p} · n ${p.n_candidates}`);
    }
    if (t.checkpointStep !== undefined) meta.push(`step ${t.checkpointStep}`);
    if (t.error) meta.push(`error: ${t.error}`);
    if (meta.length) out.push(`> ${meta.join(' · ')}`, '');
  }
  return out.join('\n');
}

function titleFor(turns: Turn[]): string {
  const first = turns.find((t) => t.role === 'user');
  if (!first) return 'New conversation';
  const line = first.content.replace(/\s+/g, ' ').trim();
  if (!line) return 'New conversation';
  return line.length > 52 ? `${line.slice(0, 51)}…` : line;
}

class Chat {
  id = $state<string>(newId());
  title = $state<string>('New conversation');
  turns = $state<Turn[]>([]);
  index = $state<ConversationSummary[]>([]);
  traces = $state<Record<string, ByteTrace>>({});
  #titleLocked = $state(false);

  /** id of the assistant turn being streamed, or null */
  streamingId = $state<string | null>(null);
  /** true between "send" and the backend's `start` */
  awaitingStart = $state(false);
  link = $state<LinkState>('offline');

  /** folding for the next reply: 'auto' folds when the prompt would overflow, 'off' is plain truncation */
  foldMode = $state<'auto' | 'off'>('auto');
  /** one-shot: fold everything before the last user turn on the next send */
  foldNow = $state(false);
  /** the user's edited compaction card, sent in place of the generated one until reset */
  card = $state<string | null>(null);
  /** what the next prompt would look like (POST /api/context), refreshed as the turns change */
  preview = $state<ContextPreview | null>(null);
  #previewTimer: ReturnType<typeof setTimeout> | null = null;

  #socket: GenerateSocket;
  #queue: ServerEvent[] = [];
  #frame: number | null = null;
  #guard: ReturnType<typeof setTimeout> | null = null;
  #saveTimer: ReturnType<typeof setTimeout> | null = null;
  #sentAt = 0;

  constructor() {
    this.#socket = new GenerateSocket({
      onEvent: (ev) => this.#enqueue(ev),
      onUnauthorized: () => auth.require(),
      onLinkState: (s) => {
        this.link = s;
      },
      onInterrupted: () => this.#fail('The connection dropped mid-reply.'),
      onAborted: () => this.#abort()
    });
  }

  get busy(): boolean {
    return this.streamingId !== null || this.awaitingStart;
  }

  get isEmpty(): boolean {
    return this.turns.length === 0;
  }

  // ---------------------------------------------------------------- lifecycle

  start(): void {
    this.index = persist.read<ConversationSummary[]>(INDEX_KEY, []);
    const current = persist.read<string | null>(CURRENT_KEY, null);
    if (current) this.open(current);
    this.#socket.connect();
  }

  dispose(): void {
    this.#socket.dispose();
    if (this.#frame !== null) cancelAnimationFrame(this.#frame);
    if (this.#guard !== null) clearTimeout(this.#guard);
  }

  reconnect(): void {
    this.#socket.retryNow();
  }

  // ------------------------------------------------------------ conversations

  open(id: string): void {
    this.#resetFolding();
    if (this.busy) this.stop();
    const stored = persist.read<StoredConversation | null>(convKey(id), null);
    if (!stored) return;
    this.id = stored.id;
    this.title = stored.title;
    this.#titleLocked = !!stored.titleLocked;
    this.turns = stored.turns;
    const saved = persist.read<Record<string, SerializedTrace>>(tracesKey(id), {});
    const traces: Record<string, ByteTrace> = {};
    for (const [tid, s] of Object.entries(saved)) {
      try {
        traces[tid] = ByteTrace.fromJSON(s);
      } catch {
        /* a corrupt trace is simply not shown */
      }
    }
    this.traces = traces;
    persist.write(CURRENT_KEY, this.id);
  }

  newConversation(): void {
    this.#resetFolding();
    if (this.busy) this.stop();
    this.id = newId();
    this.title = 'New conversation';
    this.#titleLocked = false;
    this.turns = [];
    this.traces = {};
    persist.write(CURRENT_KEY, this.id);
  }

  /** A typed title survives saving; the derived one keeps tracking the first prompt. */
  rename(id: string, next: string): void {
    const clean = next.replace(/\s+/g, ' ').trim().slice(0, 80);
    if (!clean) return;
    if (id === this.id) {
      this.title = clean;
      this.#titleLocked = true;
      this.#saveNow();
      return;
    }
    const stored = persist.read<StoredConversation | null>(convKey(id), null);
    if (stored) persist.write(convKey(id), { ...stored, title: clean, titleLocked: true });
    this.index = this.index.map((c) => (c.id === id ? { ...c, title: clean } : c));
    persist.write(INDEX_KEY, this.index);
  }

  /** Deletes at once and hands the undo bar everything needed to put it back verbatim. */
  deleteConversation(id: string): void {
    // A debounced save may still be pending, and the snapshot the undo bar holds has to be
    // the conversation as it actually stands, not as it stood 300 ms ago.
    if (id === this.id && this.#saveTimer !== null) this.#saveNow();
    const stored = persist.read<StoredConversation | null>(convKey(id), null);
    const storedTraces = persist.read<Record<string, SerializedTrace>>(tracesKey(id), {});
    const before = this.index;
    const wasCurrent = id === this.id;
    const label = this.index.find((c) => c.id === id)?.title ?? 'conversation';

    persist.drop(convKey(id));
    persist.drop(tracesKey(id));
    this.index = this.index.filter((c) => c.id !== id);
    persist.write(INDEX_KEY, this.index);
    if (wasCurrent) this.newConversation();

    notices.show(`Deleted “${label}”.`, () => {
      if (stored) persist.write(convKey(id), stored);
      persist.write(tracesKey(id), storedTraces);
      this.index = before;
      persist.write(INDEX_KEY, before);
      if (wasCurrent) this.open(id);
    });
  }

  // ------------------------------------------------------------------ sending

  /**
   * Everything typed into the composer arrives here, message or command alike.
   *
   * While a reply is running the text waits in the queue instead of being refused, so a
   * thought had mid-reply is not lost. /help is the exception: it changes nothing about the
   * conversation, so making it wait would be pure ceremony.
   */
  enter(text: string): void {
    const t = text.trim();
    if (!t) return;
    if (parseCommand(t) === 'help') {
      ui.help = true;
      return;
    }
    if (this.busy) {
      queue.add(t);
      return;
    }
    this.#run(t);
  }

  #run(text: string): void {
    switch (parseCommand(text)) {
      case 'help':
        ui.help = true;
        return;
      case 'clear':
        // Nothing to delete and nothing to undo — announcing a deletion here would be a
        // small untruth about an empty conversation.
        if (this.turns.length > 0) this.deleteConversation(this.id);
        return;
      default:
        this.send(unescapeCommand(text));
    }
  }

  /**
   * Run whatever has been waiting. A command finishes synchronously, so the loop keeps
   * going; a message takes the socket and stops it until the reply lands. Deferred by a
   * microtask so the drain that called this can unwind first.
   */
  #pump(): void {
    queueMicrotask(() => {
      while (!this.busy) {
        const next = queue.shift();
        if (!next) return;
        this.#run(next.text);
      }
    });
  }

  send(text: string): void {
    const content = text.trim();
    if (!content || this.busy) return;
    this.turns = [...this.turns, { id: newId(), role: 'user', content, at: Date.now() }];
    this.#dispatch();
  }

  /**
   * Draw the reply to this prompt again. On the newest prompt nothing is lost: the old
   * reply is kept as a sample you can flip back to, which is the whole point of being able
   * to re-roll at a different temperature. On an earlier prompt the replies that followed
   * cannot survive a different continuation, so they go and the undo bar holds them.
   */
  retryFrom(turnId: string): void {
    const idx = this.turns.findIndex((t) => t.id === turnId);
    if (idx < 0 || this.turns[idx].role !== 'user' || this.busy) return;
    const tail = this.turns.slice(idx + 1);

    if (tail.every((t) => t.role === 'assistant')) {
      const samples: Turn[] = [];
      for (const t of tail) samples.push(...(t.samples ?? []), { ...t, samples: undefined });
      this.turns = this.turns.slice(0, idx + 1);
      this.#dispatch(samples);
      return;
    }

    const restore = this.#snapshot();
    this.turns = this.turns.slice(0, idx + 1);
    this.#dispatch();
    notices.show(`Replaced ${tail.length} turns after that prompt.`, restore);
  }

  /** Rewrite a prompt and run it again. Everything after it is replaced; undo restores it. */
  editAndResend(turnId: string, text: string): void {
    const content = text.trim();
    const idx = this.turns.findIndex((t) => t.id === turnId);
    if (!content || idx < 0 || this.turns[idx].role !== 'user' || this.busy) return;
    if (content === this.turns[idx].content) {
      this.retryFrom(turnId);
      return;
    }
    const dropped = this.turns.length - idx - 1;
    const restore = this.#snapshot();
    this.turns = [...this.turns.slice(0, idx), { ...this.turns[idx], content, at: Date.now() }];
    this.#dispatch();
    if (dropped > 0) {
      notices.show(
        `Replaced ${dropped} ${dropped === 1 ? 'reply' : 'turns'} after that prompt.`,
        restore
      );
    }
  }

  /** The transcript as it stands, as a closure that puts it back verbatim. */
  #snapshot(): () => void {
    const turns = this.turns;
    const traces = this.traces;
    const title = this.title;
    return () => {
      if (this.busy) this.stop();
      this.turns = turns;
      this.traces = traces;
      this.title = title;
      this.#save();
    };
  }

  /** Show another sample of a reply slot; it becomes the one the next turn is conditioned on. */
  chooseSample(turnId: string, sampleId: string): void {
    if (this.busy) return;
    const turn = this.turns.find((t) => t.id === turnId);
    if (!turn?.samples) return;
    const idx = turn.samples.findIndex((s) => s.id === sampleId);
    if (idx < 0) return;
    const pool = turn.samples.slice();
    const chosen = pool[idx];
    pool[idx] = { ...turn, samples: undefined };
    this.turns = this.turns.map((t) => (t.id === turnId ? { ...chosen, samples: pool } : t));
    this.#save();
  }

  stop(): void {
    if (!this.busy) return;
    this.#socket.stop();
  }

  // ------------------------------------------------------------------ folding (docs/context.md)

  /** Ask the backend what the next prompt would look like; debounced, for the meter and the fold line. */
  refreshPreview(): void {
    if (this.#previewTimer) clearTimeout(this.#previewTimer);
    this.#previewTimer = setTimeout(async () => {
      this.#previewTimer = null;
      const messages = this.turns.map((t) => ({ role: t.role, content: t.content }));
      if (!messages.length) {
        this.preview = null;
        return;
      }
      try {
        this.preview = await api.context(
          messages,
          settings.params.max_bytes,
          this.foldNow ? 'now' : this.foldMode,
          this.card,
          this.lastFold
        );
      } catch {
        /* offline: the meter falls back to the last reply's numbers */
      }
    }, 250);
  }

  /** The last reply's fold, sent back with the next prompt so the server keeps the same fold point and
   *  card while they still fit — the bytes before the newest turn then stay identical, which is what the
   *  engine's prefix cache reuses (docs/context.md). */
  get lastFold(): PrevFold | null {
    for (let i = this.turns.length - 1; i >= 0; i--) {
      const t = this.turns[i];
      if (t.role !== 'assistant') continue;
      return t.fold ? { from: t.fold.from, card: t.fold.card } : null;
    }
    return null;
  }

  /** The user's version of the compaction card (null = the generated one). */
  setCard(card: string | null): void {
    this.card = card;
    this.refreshPreview();
  }

  /** Send the whole conversation next time; older turns are dropped, not folded. */
  unfold(): void {
    this.foldMode = 'off';
    this.foldNow = false;
    this.refreshPreview();
  }

  refold(): void {
    this.foldMode = 'auto';
    this.refreshPreview();
  }

  /** Fold everything before the last user turn on the next send. */
  foldNext(): void {
    this.foldNow = true;
    this.refreshPreview();
  }

  #resetFolding(): void {
    this.foldMode = 'auto';
    this.foldNow = false;
    this.card = null;
    this.preview = null;
  }

  #dispatch(samples?: Turn[]): void {
    const history: { role: ChatRole; content: string }[] = this.turns.map((t) => ({
      role: t.role,
      content: t.content
    }));
    // Provenance is captured here, not read back later: by the time you compare two samples
    // the sliders have moved and the checkpoint may have been swapped underneath them.
    const params = { ...settings.params };
    // Same value-based comparison the panel and the composer trigger use, so a reply is never
    // labelled off-default by one surface and on-default by another.
    const offDefault = settings.offDefaultKeys;
    const reply: Turn = {
      id: newId(),
      role: 'assistant',
      content: '',
      at: Date.now(),
      stats: null,
      error: null,
      samples: samples?.length ? samples : undefined,
      params,
      offDefault: offDefault.length ? offDefault : undefined,
      checkpointStep: model.info?.checkpoint.step
    };
    this.traces = { ...this.traces, [reply.id]: new ByteTrace() };
    this.turns = [...this.turns, reply];
    this.streamingId = reply.id;
    this.awaitingStart = true;
    this.#sentAt = performance.now();
    diagnostics.begin();
    const context = {
      fold: this.foldNow ? ('now' as const) : this.foldMode,
      card: this.card,
      prev: this.lastFold,
      verify_prefix: settings.verifyPrefix
    };
    this.foldNow = false;
    this.#socket.generate(history, settings.params, context);
    this.#save();
  }

  // ------------------------------------------------------------ event pipeline

  #enqueue(ev: ServerEvent): void {
    this.#queue.push(ev);
    if (this.#frame === null) {
      this.#frame = requestAnimationFrame(() => this.#drain());
      // Backstop: requestAnimationFrame is throttled to a standstill in a hidden tab.
      if (this.#guard === null) this.#guard = setTimeout(() => this.#drain(), 400);
    }
  }

  #drain(): void {
    if (this.#frame !== null) cancelAnimationFrame(this.#frame);
    if (this.#guard !== null) clearTimeout(this.#guard);
    this.#frame = null;
    this.#guard = null;

    const queue = this.#queue;
    this.#queue = [];
    if (queue.length === 0) return;

    const id = this.streamingId;
    const trace = id ? this.traces[id] : undefined;
    let stats: StatsPayload | null = null;
    let diag: Extract<ServerEvent, { type: 'diagnostics' }> | null = null;
    let finished: DoneEvent | null = null;

    for (const ev of queue) {
      switch (ev.type) {
        case 'start':
          this.awaitingStart = false;
          this.#patch(id, {
            truncated: ev.truncated,
            fold: ev.fold ?? null,
            prefix: ev.prefix ?? null,
            contextBytes: ev.context_bytes,
            contextLimit: ev.context_limit
          });
          break;
        case 'byte':
          if (trace && trace.size === 0) {
            this.#patch(id, { ttfbMs: performance.now() - this.#sentAt });
          }
          trace?.push(ev);
          break;
        case 'chunk':
          trace?.closeChunk();
          break;
        case 'stats':
          stats = ev;
          break;
        case 'diagnostics':
          diag = ev;
          if (ev.prefix_check) this.#patch(id, { prefixCheck: ev.prefix_check });
          break;
        case 'done':
          finished = ev;
          break;
        case 'error':
          this.#fail(ev.message);
          break;
      }
    }

    trace?.flush();
    if (stats) diagnostics.applyStats(stats);
    if (diag) diagnostics.applyDiagnostics(diag);

    if (finished) {
      trace?.settle(finished.text);
      this.#patch(id, {
        content: finished.text,
        stats: finished.stats,
        reason: finished.reason,
        contextBytes: finished.stats.context_bytes,
        contextLimit: finished.stats.context_limit
      });
      this.streamingId = null;
      this.awaitingStart = false;
      diagnostics.end(finished.stats);
      this.#save();
      this.#pump();
    }

    if (this.#queue.length > 0 && this.#frame === null) {
      this.#frame = requestAnimationFrame(() => this.#drain());
    }
  }

  #patch(id: string | null, patch: Partial<Turn>): void {
    if (!id) return;
    this.turns = this.turns.map((t) => (t.id === id ? { ...t, ...patch } : t));
  }

  /** The request never reached the backend, so there is nothing to keep. */
  #abort(): void {
    const id = this.streamingId;
    if (id) this.turns = this.turns.filter((t) => t.id !== id || t.content.length > 0);
    if (id) delete this.traces[id];
    this.streamingId = null;
    this.awaitingStart = false;
    diagnostics.end(null);
    this.#save();
    this.#pump();
  }

  #fail(message: string): void {
    const id = this.streamingId;
    const trace = id ? this.traces[id] : undefined;
    if (id) this.#patch(id, { error: message, content: trace?.liveText ?? '' });
    this.streamingId = null;
    this.awaitingStart = false;
    diagnostics.end(null);
    this.#save();
    // A failed reply does not cancel what you asked for next.
    this.#pump();
  }

  // -------------------------------------------------------------- persistence

  #save(): void {
    if (this.#saveTimer !== null) clearTimeout(this.#saveTimer);
    this.#saveTimer = setTimeout(() => this.#saveNow(), 300);
  }

  #saveNow(): void {
    if (this.#saveTimer !== null) clearTimeout(this.#saveTimer);
    this.#saveTimer = null;
    if (this.turns.length === 0) return;
    const title = this.#titleLocked ? this.title : titleFor(this.turns);
    this.title = title;
    const record: StoredConversation = {
      id: this.id,
      title,
      turns: this.turns,
      updatedAt: Date.now(),
      titleLocked: this.#titleLocked || undefined
    };
    persist.write(convKey(this.id), record);
    persist.write(tracesKey(this.id), this.#serializeTraces());
    const rest = this.index.filter((c) => c.id !== this.id);
    const next = [{ id: this.id, title, updatedAt: record.updatedAt }, ...rest];
    const evicted = next.slice(MAX_CONVERSATIONS);
    for (const stale of evicted) {
      persist.drop(convKey(stale.id));
      persist.drop(tracesKey(stale.id));
    }
    this.index = next.slice(0, MAX_CONVERSATIONS);
    persist.write(INDEX_KEY, this.index);
    persist.write(CURRENT_KEY, this.id);
    // Their storage is already gone so there is nothing to undo, but they should not
    // vanish in silence either.
    if (evicted.length === 1) {
      notices.show(`“${evicted[0].title}” dropped — only ${MAX_CONVERSATIONS} are kept.`);
    } else if (evicted.length > 1) {
      notices.show(`${evicted.length} old conversations dropped — only ${MAX_CONVERSATIONS} are kept.`);
    }
  }

  // ------------------------------------------------------------------- export

  /** The open conversation comes from live state; any other is read back from storage. */
  #materialise(id: string): { title: string; turns: Turn[] } | null {
    if (id === this.id) return { title: this.title, turns: this.turns };
    const stored = persist.read<StoredConversation | null>(convKey(id), null);
    return stored ? { title: stored.title, turns: stored.turns } : null;
  }

  /** Markdown to read, JSON to analyse — the JSON carries provenance and counters. */
  exportAs(id: string, format: 'md' | 'json'): void {
    const conv = this.#materialise(id);
    if (!conv) return;
    const slug = conv.title.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
    const name = `${slug || 'conversation'}.${format}`;
    const text =
      format === 'json'
        ? JSON.stringify(
            {
              id,
              title: conv.title,
              model: MODEL_NAME,
              // What is loaded right now, which is not necessarily what drew every turn —
              // each reply carries its own checkpointStep for that.
              loadedCheckpoint: model.info?.name ?? null,
              turns: conv.turns
            },
            null,
            2
          )
        : toMarkdown(conv.title, conv.turns, MODEL_NAME);
    download(name, format === 'json' ? 'application/json' : 'text/markdown', text);
  }

  /** Traces of the newest replies (samples included), newest first, capped; a reply still streaming is skipped. */
  #serializeTraces(): Record<string, SerializedTrace> {
    const candidates: Turn[] = [];
    for (const t of this.turns) {
      if (t.role !== 'assistant') continue;
      candidates.push(t, ...(t.samples ?? []));
    }
    candidates.sort((a, b) => b.at - a.at);
    const out: Record<string, SerializedTrace> = {};
    let kept = 0;
    for (const t of candidates) {
      if (kept >= MAX_TRACES) break;
      if (t.id === this.streamingId) continue;
      const trace = this.traces[t.id];
      if (!trace || trace.size === 0) continue;
      out[t.id] = trace.toJSON();
      kept += 1;
    }
    return out;
  }
}

export const chat = new Chat();
