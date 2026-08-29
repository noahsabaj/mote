# Context

Design settled 2026-08-23 (grilling rounds 1–4). Two halves: what the studio does when a conversation
outgrows the window (**built 2026-08-23**: `mote/infer/context.py`, `POST /api/context`, the fold line and
meter in the studio, `mote.eval.needle_probe`), and how long the window is and whether it is a hard edge
(**decided for the flagship, gated on measurements**).

## Facts that shape it

* The 35M serves a 2048-byte window; the identity card takes 479 of those bytes and stays (decided);
  a typical exchange is 300–600 bytes, so the start of a chat falls out after three or four.
* Chunking divides the Relation main network's length by ~5.5, so per-byte training compute grows
  mildly with the window: flagship 4096 = 1.13×, 8192 = 1.40×, 16384 = 1.94× the 2048 cost
  (35M: 1.22× / 1.66× / 2.54×). Memory and position generalisation are the real limits, not FLOPs.
* The outer Mamba-3 layers carry a fixed-size state; Relation is positional (RoPE + λ log i) and has
  only been trained to the window, so continuing past it is untested, not free.

## Studio: folding

* **Trigger**: automatic — when the next prompt would exceed `limit − reserve`, the server folds the
  oldest turns instead of dropping them. Manual "Fold now" and "Unfold" as well.
* **In batches** (decided 2026-08-23): an automatic fold frees a quarter of the window at once
  (`FOLD_SLACK`), and the studio sends its last fold (`context.prev = {from, card}`) back with every
  prompt, so the fold point and the card stay verbatim until the prompt no longer fits. The bytes before
  the newest turn therefore stay identical from turn to turn — which is what the prefix cache reuses.
* **Fold line**: the chat shows where Mote's view starts. It expands to the exact bytes Mote sees in
  place of the folded turns, and those bytes are editable by the user.
* **Compaction card** (heuristic, no model involved — a 35M model cannot read a summary it wrote):
  the first user message (what the chat is about, ≤ 200 bytes), user-stated facts picked by rules
  (names, "I am / I live / I like / I work …", numbers and dates the user gave), then the most recent
  turns verbatim, as many as fit. Deterministic and visible.
* **Meter**: bytes used of the window in the composer, with the fold state.
* **Probe**: needle-in-chat — a fact stated in turn 1, asked after 2 / 4 / 8 KB of filler turns;
  answer rate with folding vs plain truncation, on the 35M. Numbers, not claims.

## Studio: the prefix store

Settled 2026-08-23 (grilling, after CacheRoute 2608.19677 made the point at fleet scale) and rebuilt
at the root 2026-08-24 (after FreeToken 2608.16157; docs/shape.md § "Serving root"): the engine used
to re-read the whole conversation every turn — on the Windows studio that was 0.35 s at 50 B, 1.3 s
at 900 B and 3.5 s at 1800 B before the first byte. The router is causal, so an identical byte prefix
gives identical chunks and an identical state: `mote/infer/prefix_cache.py` keeps what it takes to
continue, and `Engine._read_prompt` reads only the bytes after the longest anchor.

* **Two kinds of state**: the Relation per-chunk cache (the only thing that grows with the context)
  lives in the device **arena** and is paged to the CPU per **branch** (one per linear conversation;
  256-chunk pages, full pages shared when a regenerate or an edit forks a branch). Everything else —
  Mamba-3 encoder/decoder, routing and dechunk states, the next-byte logits — is an **anchor**
  (~3 MB on the flagship, against ~108 MB for the full snapshots of the first version).
* **Anchors**: `card` — after the identity card, shared by every conversation, pinned (a cold start
  reads the card on its own first); `prompt` — the end of a turn's prompt (a regenerate reads
  nothing); `reply` — the end of the reply, where the next turn starts. Tool-result boundaries are
  reserved for search. A stopped reply's transcript is exactly what the engine consumed (the graph
  path drains whole batches), so its `reply` anchor is exact.
* **Hot arena**: the arena's contents stay valid between turns of the same conversation — a
  continue copies nothing, a switch copies the other branch's pages up once. Gated on the flagship
  memory measurement; the fallback flushes between turns.
* **Storage**: CPU memory (pinned when the model is on CUDA) under a byte budget over unique pages +
  anchors — `MOTE_PREFIX_CACHE_MB`, default 1024, 0 disables — eviction drops whole least-recently-used
  branches. Cleared on every weight swap, then `Engine.rewarm()` re-reads the conversations used in
  the last 10 minutes so the next message is warm.
* **Reporting**: the `start` event carries `prefix: {reused, prefilled, prefill_ms, snapshots, branches,
  …}`; the stats line under a reply says "N B reused"; the composer meter says "N already read";
  `/api/context` reports `reusable`; `/api/model.arena` shows the hot branch. **Verify cache**
  (sampling panel) re-reads each prompt cold in a private arena and reports moved cuts and the largest
  next-byte logit difference (`prefix_check` on a diagnostics event) — a debug toggle, off by default.
* **Measured**: `python -m mote.eval.prefix_probe` — 40 greedy turns on the 35M, warm vs cold per
  turn, mean and worst turn (`docs/results/2026-08-24-serving-root.md`; the first version's numbers are
  in `docs/results/2026-08-23-prefix-cache.md`).
* Input_States: wired 2026-08-24 (the Mamba-3 kernel now takes the cached states on a warm
  continuation; `tests/test_mamba3_states.py`).

## Flagship: the window

Decided and gated 2026-08-23, frozen 2026-08-24: **16384 bytes from the first step** (`mote/config.py`
`mote_96m`; ~3000 chunks in the main network; the memory profile that passed it is in docs/shape.md § "The
model", and the windowed-main alternative was measured and rejected there — long-range chunk lookups are
load-bearing). Long documents (≥ 8 KB) are ~10 % of the mix; SFT packs multi-turn conversations to the window.
