// WebSocket client for /ws/generate: one generation per socket message, with
// automatic reconnect (capped exponential backoff) and cancellation.

import type { ChatRole, ClientGenerate, SamplingParams, ServerEvent } from './types';

/**
 * 'connecting' is only ever the very first attempt. Once a connection has succeeded or
 * failed once, every retry keeps the state at 'offline' — a status line that flickered
 * between "connecting" and "not connected" on each backoff tick would be noise.
 */
export type LinkState = 'connecting' | 'open' | 'offline';

export interface GenerateSocketHandlers {
  onEvent(event: ServerEvent): void;
  onLinkState(state: LinkState): void;
  /** The socket dropped while a generation was actually running. */
  onInterrupted(): void;
  /** A request was cancelled before it ever reached the backend. */
  onAborted(): void;
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
  #everConnected = false;
  /** true only between a request reaching the wire and its `done`/`error` */
  #generating = false;
  /** a request accepted while the socket was down, waiting for the next open */
  #pending: ClientGenerate | null = null;
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
    const live = this.#ws?.readyState;
    if (live === WebSocket.OPEN || live === WebSocket.CONNECTING) return;
    if (!this.#everConnected && this.#attempt === 0) this.#setState('connecting');

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
      this.#everConnected = true;
      this.#setState('open');
      this.#flush();
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
      // `close` always follows; the retry logic lives there.
    });
  }

  /** Queued rather than rejected when the link is down, then sent on the next open. */
  generate(messages: { role: ChatRole; content: string }[], params: SamplingParams): void {
    this.#pending = { type: 'generate', messages, params };
    if (this.#ws?.readyState === WebSocket.OPEN) this.#flush();
    else this.connect();
  }

  stop(): void {
    if (this.#generating) {
      this.#write({ type: 'stop' });
      return;
    }
    if (this.#pending) {
      this.#pending = null;
      this.#handlers.onAborted();
    }
  }

  /** Reconnect immediately instead of waiting out the backoff. */
  retryNow(): void {
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
    this.#pending = null;
    this.#ws?.close();
    this.#ws = null;
  }

  #flush(): void {
    if (!this.#pending) return;
    const request = this.#pending;
    if (!this.#write(request)) return;
    this.#pending = null;
    this.#generating = true;
  }

  #write(msg: ClientGenerate | { type: 'stop' }): boolean {
    if (this.#ws?.readyState !== WebSocket.OPEN) return false;
    try {
      this.#ws.send(JSON.stringify(msg));
      return true;
    } catch {
      return false;
    }
  }

  #scheduleRetry(): void {
    this.#setState('offline');
    if (this.#retryTimer !== null) return;
    // A failed attempt counts even if it never opened, so the first failure ends 'connecting'.
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
