// Conversation state and the streaming pipeline.
//
// The backend keeps no conversation, so the full message list is sent every turn and the
// transcript lives here (and in localStorage). Socket events are queued and applied once
// per animation frame: a reply is thousands of `byte` events and each one must not touch
// the DOM on its own.

import { GenerateSocket, type LinkState } from '../ws';
import { ByteTrace } from '../trace.svelte';
import * as persist from '../persist';
import { settings } from './settings.svelte';
import { diagnostics } from './diagnostics.svelte';
import type { ChatRole, DoneEvent, ServerEvent, StatsPayload } from '../types';

export interface Turn {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  at: number;
  /** assistant only — final counters from `done` */
  stats?: StatsPayload | null;
  reason?: DoneEvent['reason'];
  error?: string | null;
  /** from `start`: the prompt did not fit the context window */
  truncated?: boolean;
  contextBytes?: number;
  contextLimit?: number;
}

interface StoredConversation {
  id: string;
  title: string;
  turns: Turn[];
  updatedAt: number;
}

export interface ConversationSummary {
  id: string;
  title: string;
  updatedAt: number;
}

const INDEX_KEY = 'conversations';
const CURRENT_KEY = 'current';
const convKey = (id: string) => `conv.${id}`;
const MAX_CONVERSATIONS = 30;

function newId(): string {
  return Math.random().toString(36).slice(2, 10) + Date.now().toString(36).slice(-4);
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
  turns = $state<Turn[]>([]);
  index = $state<ConversationSummary[]>([]);
  traces = $state<Record<string, ByteTrace>>({});

  /** id of the assistant turn being streamed, or null */
  streamingId = $state<string | null>(null);
  /** true between "send" and the backend's `start` */
  awaitingStart = $state(false);
  link = $state<LinkState>('offline');

  #socket: GenerateSocket;
  #queue: ServerEvent[] = [];
  #frame: number | null = null;
  #guard: ReturnType<typeof setTimeout> | null = null;
  #saveTimer: ReturnType<typeof setTimeout> | null = null;

  constructor() {
    this.#socket = new GenerateSocket({
      onEvent: (ev) => this.#enqueue(ev),
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
    if (this.busy) this.stop();
    const stored = persist.read<StoredConversation | null>(convKey(id), null);
    if (!stored) return;
    this.id = stored.id;
    this.turns = stored.turns;
    this.traces = {};
    persist.write(CURRENT_KEY, this.id);
  }

  newConversation(): void {
    if (this.busy) this.stop();
    this.id = newId();
    this.turns = [];
    this.traces = {};
    persist.write(CURRENT_KEY, this.id);
  }

  deleteConversation(id: string): void {
    persist.drop(convKey(id));
    this.index = this.index.filter((c) => c.id !== id);
    persist.write(INDEX_KEY, this.index);
    if (id === this.id) this.newConversation();
  }

  // ------------------------------------------------------------------ sending

  send(text: string): void {
    const content = text.trim();
    if (!content || this.busy) return;
    this.turns = [...this.turns, { id: newId(), role: 'user', content, at: Date.now() }];
    this.#dispatch();
  }

  regenerate(): void {
    if (this.busy) return;
    let cut = this.turns.length;
    while (cut > 0 && this.turns[cut - 1].role === 'assistant') cut -= 1;
    if (cut === this.turns.length) return;
    for (const dropped of this.turns.slice(cut)) delete this.traces[dropped.id];
    this.turns = this.turns.slice(0, cut);
    this.#dispatch();
  }

  stop(): void {
    if (!this.busy) return;
    this.#socket.stop();
  }

  #dispatch(): void {
    const history: { role: ChatRole; content: string }[] = this.turns.map((t) => ({
      role: t.role,
      content: t.content
    }));
    const reply: Turn = {
      id: newId(),
      role: 'assistant',
      content: '',
      at: Date.now(),
      stats: null,
      error: null
    };
    this.traces = { ...this.traces, [reply.id]: new ByteTrace() };
    this.turns = [...this.turns, reply];
    this.streamingId = reply.id;
    this.awaitingStart = true;
    diagnostics.begin();
    this.#socket.generate(history, settings.params);
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
            contextBytes: ev.context_bytes,
            contextLimit: ev.context_limit
          });
          break;
        case 'byte':
          trace?.push(ev);
          break;
        case 'chunk':
          trace?.closeChunk(ev);
          break;
        case 'stats':
          stats = ev;
          break;
        case 'diagnostics':
          diag = ev;
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
  }

  #fail(message: string): void {
    const id = this.streamingId;
    const trace = id ? this.traces[id] : undefined;
    if (id) this.#patch(id, { error: message, content: trace?.liveText ?? '' });
    this.streamingId = null;
    this.awaitingStart = false;
    diagnostics.end(null);
    this.#save();
  }

  // -------------------------------------------------------------- persistence

  #save(): void {
    if (this.#saveTimer !== null) clearTimeout(this.#saveTimer);
    this.#saveTimer = setTimeout(() => this.#saveNow(), 300);
  }

  #saveNow(): void {
    this.#saveTimer = null;
    if (this.turns.length === 0) return;
    const title = titleFor(this.turns);
    const record: StoredConversation = {
      id: this.id,
      title,
      turns: this.turns,
      updatedAt: Date.now()
    };
    persist.write(convKey(this.id), record);
    const rest = this.index.filter((c) => c.id !== this.id);
    const next = [{ id: this.id, title, updatedAt: record.updatedAt }, ...rest];
    for (const stale of next.slice(MAX_CONVERSATIONS)) persist.drop(convKey(stale.id));
    this.index = next.slice(0, MAX_CONVERSATIONS);
    persist.write(INDEX_KEY, this.index);
    persist.write(CURRENT_KEY, this.id);
  }
}

export const chat = new Chat();
