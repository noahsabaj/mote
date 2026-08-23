// The one transient bar at the foot of the app: "X deleted — Undo".
//
// Undo is snapshot-and-restore, not deferred destruction: the action happens immediately
// and the caller hands over a closure that puts things back. That way a notice can be
// replaced, ignored or lost without ever leaving the app in a half-deleted state.

const LIFETIME_MS = 8000;

export interface Notice {
  id: number;
  message: string;
  /** absent for notices that only report something already irreversible */
  undo?: () => void;
}

class Notices {
  current = $state<Notice | null>(null);

  #seq = 0;
  #timer: ReturnType<typeof setTimeout> | null = null;

  /** Show a bar with an Undo action. Any bar already up is dropped. */
  show(message: string, undo?: () => void): void {
    this.#clearTimer();
    this.#seq += 1;
    this.current = { id: this.#seq, message, undo };
    this.#timer = setTimeout(() => this.dismiss(), LIFETIME_MS);
  }

  undo(): void {
    const n = this.current;
    this.dismiss();
    n?.undo?.();
  }

  dismiss(): void {
    this.#clearTimer();
    this.current = null;
  }

  #clearTimer(): void {
    if (this.#timer !== null) clearTimeout(this.#timer);
    this.#timer = null;
  }
}

export const notices = new Notices();
