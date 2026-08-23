// Anything typed while Mote is generating waits here until the reply lands.
//
// The queue is not the transcript. It lives in memory only: a reload drops it, it never
// reaches localStorage, it is not exported, and it carries no byte trace. An item becomes a
// real turn at the moment it fires and not before, which is also when its sampling
// parameters are captured — the sliders may well have moved while it was waiting.

import { parseCommand, type CommandName } from '../commands';

export interface QueuedItem {
  id: string;
  text: string;
  /** the command this item will run, or null if it is a message for Mote */
  command: CommandName | null;
}

let seq = 0;

class Queue {
  items = $state<QueuedItem[]>([]);

  get size(): number {
    return this.items.length;
  }

  add(text: string): void {
    const t = text.trim();
    if (!t) return;
    this.items = [...this.items, { id: `q${(seq += 1)}`, text: t, command: parseCommand(t) }];
  }

  remove(id: string): void {
    this.items = this.items.filter((i) => i.id !== id);
  }

  /** Editing to nothing means you changed your mind, which is the same as removing it. */
  edit(id: string, text: string): void {
    const t = text.trim();
    if (!t) {
      this.remove(id);
      return;
    }
    this.items = this.items.map((i) =>
      i.id === id ? { ...i, text: t, command: parseCommand(t) } : i
    );
  }

  /** Move one item by `delta` places, clamped to the ends. True if anything actually moved. */
  nudge(id: string, delta: number): boolean {
    const from = this.items.findIndex((i) => i.id === id);
    if (from < 0) return false;
    return this.moveTo(id, Math.min(Math.max(from + delta, 0), this.items.length - 1));
  }

  /** Put `id` at exactly this index — what a drag reports as it crosses a neighbour. */
  moveTo(id: string, to: number): boolean {
    const from = this.items.findIndex((i) => i.id === id);
    if (from < 0 || to === from || to < 0 || to >= this.items.length) return false;
    const next = this.items.slice();
    const [item] = next.splice(from, 1);
    next.splice(to, 0, item);
    this.items = next;
    return true;
  }

  /** Take the next item to run, if there is one. */
  shift(): QueuedItem | undefined {
    const [first, ...rest] = this.items;
    if (!first) return undefined;
    this.items = rest;
    return first;
  }
}

export const queue = new Queue();
