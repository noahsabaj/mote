// Preference votes (docs/prefs.md): the numbers the Model sheet shows and the rubric the Compare card
// can unfold. The votes themselves are posted from the chat store, which owns the turns.

import { api, ApiError } from '../api';
import type { MarkBody, PairVote, PrefsSummary, ReplyMark, Rubric, VoteBody } from '../types';

class Prefs {
  summary = $state<PrefsSummary | null>(null);
  rubric = $state<Rubric | null>(null);
  error = $state<string | null>(null);

  async refresh(): Promise<void> {
    try {
      this.summary = await api.prefsSummary();
      this.error = null;
    } catch (e) {
      this.error = e instanceof ApiError ? e.message : String(e);
    }
  }

  async loadRubric(): Promise<void> {
    if (this.rubric) return;
    try {
      this.rubric = await api.prefsRubric();
    } catch {
      /* the card still works without the rules unfolded */
    }
  }

  /** Marks already given, by turn id, so the buttons can show their state without a round trip. */
  marked = $state<Record<string, ReplyMark>>({});

  /** One thumb on one reply. Unlike a vote this needs no second sample and no comparison, which is the
   *  point: a pair costs two generations and a judgement, a mark costs a click on something you were
   *  already reading. mote.train.kto consumes them directly. */
  async mark(body: MarkBody, turnId: string): Promise<PrefsSummary | null> {
    const prev = this.marked[turnId];
    this.marked = { ...this.marked, [turnId]: body.mark }; // optimistic: the button should not lag the click
    try {
      const s = await api.prefsMark(body);
      this.summary = s;
      this.error = null;
      return s;
    } catch (e) {
      const next = { ...this.marked };
      if (prev) next[turnId] = prev;
      else delete next[turnId];
      this.marked = next;
      this.error = e instanceof ApiError ? e.message : String(e);
      return null;
    }
  }

  /** Store a pair and (unless vote is null) your verdict on it. Resolves to the new summary. */
  async vote(pair: VoteBody['pair'], vote: PairVote | null, reason = ''): Promise<PrefsSummary | null> {
    try {
      const s = await api.prefsVote({ pair, vote, reason });
      this.summary = s;
      this.error = null;
      return s;
    } catch (e) {
      this.error = e instanceof ApiError ? e.message : String(e);
      return null;
    }
  }
}

export const prefs = new Prefs();
