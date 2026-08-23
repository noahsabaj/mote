# Context

Design settled 2026-08-23 (grilling rounds 1–4). Two halves: what the studio does when a conversation
outgrows the window (**built 2026-08-23**: `mote/serve/context.py`, `POST /api/context`, the fold line and
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

## Studio: the prefix cache

Settled 2026-08-23 (grilling, after CacheRoute 2608.19677 made the point at fleet scale): the engine
used to re-read the whole conversation every turn — on the Windows studio (served on the CPU while
training holds the GPU; reference Mamba-3 and Relation paths) that was 0.35 s at 50 B, 1.3 s at 900 B
and 3.5 s at 1800 B before the first byte, superlinear because the Relation reference is O(S²). The router is causal, so an identical byte
prefix gives identical chunks and an identical state: `mote/serve/prefix_cache.py` keeps snapshots
of the inference state and `Engine._prefill_with_cache` reads only the bytes after the longest one.

* **Snapshots**: `card` — after the identity card, shared by every conversation (a cold start reads the
  card on its own first); `prompt` — the end of a turn's prompt (a regenerate reads nothing);
  `reply` — the end of the reply, where the next turn starts. A stopped reply whose transcript lacks
  the streamer's pending UTF-8 bytes simply matches the `prompt` snapshot instead.
* **Storage**: CPU memory (pinned when the model is on CUDA), most-recently-used first, under a byte
  budget — `MOTE_PREFIX_CACHE_MB`, default 1024, 0 disables. ~10 MB a snapshot on the 35M at 2048,
  ~100–190 MB on the flagship at 16384. Nothing stays on the GPU between turns, since training shares it.
  Cleared with the engine on a checkpoint swap.
* **Reporting**: the `start` event carries `prefix: {reused, prefilled, prefill_ms, snapshots, …}`; the
  stats line under a reply says "N B reused"; the composer meter says "N already read"; `/api/context`
  reports `reusable`. **Verify cache** (sampling panel) re-reads each prompt cold after the warm
  continuation and reports moved cuts and the largest next-byte logit difference (`prefix_check` on a
  diagnostics event) — it doubles the read time, so it is a debug toggle, off by default.
* **Measured**: `python -m mote.eval.prefix_probe` — a 40-turn greedy conversation on the 35M,
  warm vs cold per turn (`docs/results/2026-08-23-prefix-cache.md`).
* **Fedora follow-up**: the upstream Mamba-3 kernel accepts `Input_States`
  (`mamba3_siso_combined.py`), but the wrapper gates the kernel on `initial_states is None`, so a warm
  continuation runs the reference path there too; wire the states through with a GPU exactness test.

## Flagship: the window

* **16384 bytes from the first step** (decided; ~1.94× per-byte cost, so ≈2 weeks local instead of
  ≈1 at the same MFU). ~3000 chunks in the main network; positions trained to that length.
* **Gate — passed 2026-08-23**: `profile_step --preset flagship --chunk-bytes 6 --seq-len 16384 --batch-size 1
  --ckpt-main` on the 4060 Ti: **6.34 GB peak**, 42 KB/s, 19.3 TFLOPS (≈ 44 % MFU) — 8192 was 4.17 GB at the
  same 42 KB/s, 4096 3.06 GB at 29 KB/s. The preset now says 16384. (Fallback 8192 stays on paper.)
* **Long documents**: a ~10 % shard of documents ≥ 8 KB (full Wikipedia articles, Project Gutenberg
  books, long fineweb-edu pages; code when an ungated source is picked) mixed in with `--mix`; SFT
  packs multi-turn conversations to the window.
* **Windowed-main A/B on the 35M before the flagship config is frozen**: Relation restricted to the
  last W chunks (with the Mamba state carried) vs full attention, equal bytes, val bpb at 1× / 2× /
  4× the training length. A win makes the window a compute knob instead of a capability limit
  (conversations never hard-overflow; forgetting becomes gradual). Needs a key-window in the
  FlashRelation kernels first — GPU work, after the rename and the Fedora move.

## Order

1. Now: folding in the server and studio, the needle probe. Done 2026-08-23, plus the prefix cache and batch folding.
2. When the GPU frees: the 16384 profile (decides 16384 vs 8192); the long-document shard builds
   CPU-side meanwhile.
3. After rename + Fedora: key-window in FlashRelation, the windowed-vs-full A/B, then the flagship
   preset is frozen and the run starts.
