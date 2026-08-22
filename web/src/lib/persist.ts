// Thin, failure-tolerant localStorage wrapper. Private-mode browsers and quota errors
// must never take the app down; the conversation just stops surviving reloads.

const PREFIX = 'morpheme.';

export function read<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(PREFIX + key);
    if (raw === null) return fallback;
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
}

export function write(key: string, value: unknown): void {
  try {
    localStorage.setItem(PREFIX + key, JSON.stringify(value));
  } catch {
    /* quota or disabled storage — non-fatal */
  }
}

export function drop(key: string): void {
  try {
    localStorage.removeItem(PREFIX + key);
  } catch {
    /* non-fatal */
  }
}
