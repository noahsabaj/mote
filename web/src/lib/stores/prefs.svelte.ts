// Preference votes (docs/prefs.md): the numbers the Model sheet shows and the rubric the Compare card
// can unfold. The votes themselves are posted from the chat store, which owns the turns.

import { api, ApiError } from '../api';
import type { PairVote, PrefsSummary, Rubric, VoteBody } from '../types';

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
