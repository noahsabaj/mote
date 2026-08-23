# Search

Design settled 2026-08-23 (grilling rounds 1–3). **Status: designed, not built.** Nothing here is live
yet; the order at the bottom says what lands when.

## Why

Mote knows almost nothing — 35M parameters now, ~105M for the flagship — so retrieval is its only
route to facts, which makes search matter *more* for it than for a frontier model. The constraint is
the window: 2048 bytes, of which the identity card takes ~300 and the question and answer a few
hundred, leaving about 1 KB for what it reads. Everything below is sized to that.

## Protocol

* Two spare vocab ids (the embedding is padded 262 → 264, so nothing changes shape):
  `SEARCH_ID = 262` (`<|search|>`), `RESULT_ID = 263` (`<|result|>`).
* A searched reply is one assistant turn:
  `<|assistant|> <|search|> query bytes <|result|> [server: result bytes] <|assistant|> answer <|eos|>`.
  The model writes `<|search|>`, the query, then `<|result|>`; the server stops there, runs the
  search, appends the result bytes and an `<|assistant|>`, and resumes from the saved recurrent
  state (`forward_from_state`) — no prefix recomputation.
* Result bytes: up to 3 hits, `1. Title — snippet` per line, ≤ 1024 bytes in total; no hits →
  `(no results)`. Special ids are stripped from anything that comes back from the web.
* Caps: at most 2 searches per reply, 5 s per search. After the turn, the result bytes are dropped
  from the model's history (the answer stays; the sources stay in the UI), so old snippets never
  eat the window.
* Misses: trained on real retrieval misses, the model says *"I searched and didn't find that."* It
  does not guess — a 35M model's memory is mostly noise.
* Snippets only. No page fetching, so there is no SSRF surface, no robots question, and the only
  untrusted text the model sees is what the engine already shows publicly.

## Backend

* **SearXNG** in Docker (Docker Desktop here; podman/docker on Fedora after the move), JSON API on
  localhost, reachable only from the studio server. All engines enabled, Google included — it will
  captcha or block at times; the others cover. Queries leave the house via the home IP to whichever
  engines SearXNG asks; they are shown in Diagnostics and logged locally only.
* **Offline index**: English Wikipedia intros (the first ≤ 1024 bytes of every article, from
  FineWiki) in SQLite FTS5 with BM25 ranking — `data/wiki_intros.sqlite`, ~6 GB, no new
  dependencies. It is the training-data retriever, the deterministic eval, and the fallback engine
  when SearXNG is down.

## Data

* Questions from NQ-open, TriviaQA and SQuAD; the query is the question (plus paraphrases); hits
  come from *our* index, top-3, misses included; the answer is the gold span when it is in the
  hits, the miss reply when it is not. Chit-chat turns that must *not* search stay in the mix.
* The share of search dialogues in SFT is a lab A/B (10 / 20 / 35 %), judged on val bpb, the
  search eval and the identity probe together.

## Measurement

1. **Reading probe** — `python -m morpheme.eval.read_probe --checkpoint <pt>`: SQuAD passages
   (≤ 1024 bytes) in the user turn, greedy answers, exact match and F1 against the gold spans, plus
   the no-passage baseline on the same questions. Runs on the 35M as soon as `overnight_sft` lands.
2. **Search eval** — end to end on held-out NQ questions against the offline index: exact match,
   *miss honesty* (says "didn't find" exactly when the answer was not retrieved), and the
   unnecessary-search rate on chit-chat.

**Gate:** ≥ 50 % exact match after a small QA SFT with the gold snippet in context. Below that the
live loop waits for the flagship — a model that cannot copy the span cannot use search.

## Studio

* "Searching: …" inline while a search runs; source chips with links under the reply.
* Grounded bytes: bytes inside a ≥ 12-byte match with a snippet are tinted as a fourth
  `ByteSource` (`web`), next to nbp / mbp / fix — what was copied versus made up is visible.
* Diagnostics: queries, hits, latency, searches per reply.

## Order

1. Reading probe on the 35M (when tonight's SFT lands); the index and the dialogues are built
   CPU-side by agents meanwhile.
2. Rename → Fedora move.
3. Tool ids, server loop, SearXNG, studio surfaces; search eval; the SFT-share A/B in `mote lab`.
4. The flagship SFT inherits the data.
