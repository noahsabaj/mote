# Preferences — votes, the challenger, and the rater loop

Settled 2026-08-23 (grilling, five rounds) after the question "is the 'which response do you prefer?'
prompt on the big assistants a form of RLHF with the customer as the rater?" — it is, used first to judge
model changes by win rate and second as (prompt, chosen, rejected) training data, with the known catch
that users reward what pleases them. Mote does the same in miniature, with two raters: Noah, and Claude
under a written rubric, and with disagreements discussed rather than resolved by rule.

## What is built (phase 1, 2026-08-23)

* **Pairs in the studio.** *Retry* already draws a second sample of a reply; a vote is now offered
  between the previous and the new one (not blind — you saw the first). *Compare* draws a second
  sample on purpose: from the **challenger** when one is loaded, else from the current checkpoint, and
  shows the two side by side, blind (A/B, sources revealed after the vote). **Arena mode** (sampling
  panel) does that for every prompt. Votes: **A / B / Tie / Both bad**, with an optional one-line reason;
  *Skip* stores the pair unrated. After a vote the preferred reply is the one the conversation continues
  from.
* **Challenger slot.** Model sheet → a checkpoint row → *Challenger* loads a second engine next to the
  served one (`POST /api/challenger/load`, `DELETE /api/challenger`; `/api/model.challenger`). Each
  reply records its source: checkpoint name, step, engine role, sampling params.
* **Storage.** `data/prefs/pairs.jsonl` and `data/prefs/votes.jsonl`, append-only, local, gitignored
  (`mote/serve/prefs.py`). The newest vote of a rater for a pair counts, so a changed mind is a new
  line and the history stays.
* **Numbers.** Model sheet → *Preferences*: wins / ties / both-bad per checkpoint pair from your votes,
  and the you-vs-rater agreement rate once the rater has voted.
* **Rubric.** `docs/rubric.md`, drafted by Claude from the identity card, edited by Noah, shown in the
  Compare card; every AI verdict records its hash.
* **Rater loop.** `mote prefs export` writes unrated pairs to `data/prefs/to_rate.jsonl`, your own
  voted pairs first (they calibrate the rater), then ranked by how different the two replies are (edit
  distance, length gap, rubric markers on one side only); sources and your votes are stripped. Claude
  rates them in a Claude Code session and writes `{"id", "vote", "reason"}` lines; `mote prefs import
  <file>` stores them as rater `claude`; `mote prefs disagreements` prints the pairs where the two of
  you differ (hard: opposite sides; soft: one of you said tie or both bad) — those get discussed in the
  session, and your label stands unless you change it after the discussion.

## Pair sources (phase 2)

Volume comes from pairs the rater judges, not from your votes:

1. Your own studio conversations: each prompt gets two fresh samples from the checkpoint under training.
2. A prompt bank written to cover the rubric (identity, pushback both ways, unknowable facts,
   arithmetic, formatting, flattery bait), versioned in the repo.
3. Prompts from the public chat SFT data (filler).
4. **Synthetic negatives**: a good reply corrupted on purpose — a cave-in, flattery, a wrong fact, a
   truncation, a broken UTF-8 run. Labels by construction, no rater, unlimited.
5. **Verifiable pairs**: small arithmetic, yes/no facts from the Wikipedia index, "one word", length
   limits — a rule decides the winner.
6. Real user prompts from WildChat / LMSYS-Chat, after Noah accepts their terms on Hugging Face; filtered
   to short, single-turn, English, no personal data; labels ignored.
7. Mote's own generated prompts (sample the user side of the chat template) and Noah's conversations
   forked at a middle turn, so pairs carry realistic history including folded cards.

## Gate for training (phase 3, GPU)

The first DPO round on this data runs when there are **≥ ~1 000 rated pairs with ≥ ~150 of Noah's**
(`mote.train.dpo --pairs … --sft-weight … --epochs 1 --lr 5e-7`), and ships only if the identity
held-out probe, hold/concede, and val bpb do not regress. Both-bad pairs are excluded from training.

## Files

```
data/prefs/pairs.jsonl      {"id","ts","messages","a","b","a_source","b_source","origin"}
data/prefs/votes.jsonl      {"pair","ts","rater","vote","reason","rubric"}
data/prefs/to_rate.jsonl    the rater's batch (export)
data/prefs/verdicts.jsonl   what the rater writes back (import)
docs/rubric.md              the rules; its hash is stamped on every AI verdict
```
