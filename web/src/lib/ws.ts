// WebSocket client for /ws/generate: one generation per socket message, with
// automatic reconnect (capped exponential backoff) and cancellation.

import type { ChatRole, ClientMessage, SamplingParams, ServerEvent } from './types';

export type LinkState = 'connecting' | 'open' | 'offline';

export interface GenerateSocketHandlers {
  onEvent(event: ServerEvent): void;
  onLinkState(state: LinkState): void;
  /** The socket dropped while a generation was in flight. */
  onInterrupted(): void;
}

const BACKOFF_MS = [400, 800, 1600, 3000, 5000, 8000];

function socketUrl(): string {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${proto}//${location.host}/ws/generate`;
}

export class GenerateSocket {
  #ws: WebSocket | null = null;
  #attempt = 0;
  #retryTimer: ReturnType<typeof setTimeout> | null = null;
  #disposed = false;
  #generating = false;
  #queued: ClientMessage | null = null;
  #handlers: GenerateSocketHandlers;
  #state: LinkState = 'offline';

  constructor(handlers: GenerateSocketHandlers) {
    this.#handlers = handlers;
  }

  get linkState(): LinkState {
    return this.#state;
  }

  connect(): void {
    if (this.#disposed) return;
    if (this.#ws && (this.#ws.readyState === WebSocket.OPEN || this.#ws.readyState === WebSocket.CONNECTING)) {
      return;
    }
    this.#setState('connecting');
    let ws: WebSocket;
    try {
      ws = new WebSocket(socketUrl());
    } catch {
      this.#scheduleRetry();
      return;
    }
    this.#ws = ws;

    ws.addEventListener('open', () => {
      if (this.#disposed) {
        ws.close();
        return;
      }
      this.#attempt = 0;
      this.#setState('open');
      if (this.#queued) {
        const pending = this.#queued;
        this.#queued = null;
        this.#rawSend(pending);
      }
    });

    ws.addEventListener('message', (ev: MessageEvent<string>) => {
      let parsed: ServerEvent;
      try {
        parsed = JSON.parse(ev.data) as ServerEvent;
      } catch {
        return;
      }
      if (parsed.type === 'done' || parsed.type === 'error') this.#generating = false;
      this.#handlers.onEvent(parsed);
    });

    ws.addEventListener('close', () => {
      if (this.#ws === ws) this.#ws = null;
      if (this.#disposed) return;
      if (this.#generating) {
        this.#generating = false;
        this.#handlers.onInterrupted();
      }
      this.#scheduleRetry();
    });

    ws.addEventListener('error', () => {
      // `close` always follows; retry logic lives there.
    });
  }

  generate(messages: { role: ChatRole; content: string }[], params: SamplingParams): void {
    this.#generating = true;
    this.#send({ type: 'generate', messages, params });
  }

  stop(): void {
    if (!this.#generating) return;
    this.#send({ type: 'stop' });
  }

  /** Drop any queued work and reconnect immediately (used by the "retry" affordance). */
  retryNow(): void {
    this.#queued = null;
    if (this.#retryTimer !== null) {
      clearTimeout(this.#retryTimer);
      this.#retryTimer = null;
    }
    this.#attempt = 0;
    this.connect();
  }

  dispose(): void {
    this.#disposed = true;
    if (this.#retryTimer !== null) clearTimeout(this.#retryTimer);
    this.#retryTimer = null;
    this.#ws?.close();
    this.#ws = null;
  }

  #send(msg: ClientMessage): void {
    if (this.#ws?.readyState === WebSocket.OPEN) {
      this.#rawSend(msg);
    } else {
      // Hold the request until the socket comes back; a stop cancels a held request outright.
      this.#queued = msg.type === 'stop' ? null : msg;
      if (msg.type === 'stop') this.#generating = false;
      this.connect();
    }
  }

  #rawSend(msg: ClientMessage): void {
    try {
      this.#ws?.send(JSON.stringify(msg));
    } catch {
      this.#queued = msg.type === 'stop' ? null : msg;
    }
  }

  #scheduleRetry(): void {
    this.#setState('offline');
    if (this.#retryTimer !== null) return;
    const delay = BACKOFF_MS[Math.min(this.#attempt, BACKOFF_MS.length - 1)];
    this.#attempt += 1;
    this.#retryTimer = setTimeout(() => {
      this.#retryTimer = null;
      this.connect();
    }, delay);
  }

  #setState(state: LinkState): void {
    if (this.#state === state) return;
    this.#state = state;
    this.#handlers.onLinkState(state);
  }
}
