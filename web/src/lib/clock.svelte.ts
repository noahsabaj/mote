// A single coarse clock for relative timestamps.
//
// `ago()` is a pure function of a time, so nothing would re-render it as it goes stale.
// One 30-second tick shared by every timestamp is cheaper than a timer per message, and it
// stops while the tab is hidden: a backgrounded studio has nobody to lie to.

const TICK_MS = 30_000;

class Clock {
  now = $state(Date.now());

  #timer: ReturnType<typeof setInterval> | null = null;

  constructor() {
    if (typeof document === 'undefined') return;
    document.addEventListener('visibilitychange', () => this.#sync());
    this.#sync();
  }

  #sync(): void {
    const visible = document.visibilityState === 'visible';
    if (visible && this.#timer === null) {
      this.now = Date.now(); // catch up on whatever elapsed while hidden
      this.#timer = setInterval(() => (this.now = Date.now()), TICK_MS);
    } else if (!visible && this.#timer !== null) {
      clearInterval(this.#timer);
      this.#timer = null;
    }
  }
}

export const clock = new Clock();
