# Results — serving root: arena, anchors, one graph per byte (2026-08-24)

Grilled and signed 2026-08-24 after reading FreeToken (arXiv 2608.16157). What was built the same day
(design in `docs/shape.md` § "Serving root", `docs/context.md` § "Studio: the prefix store"):

* `mote/model/arena.py` — the Relation per-chunk cache lives in one static arena; layers write rows
  in place and read views (the flash kernel takes the row stride, no copy, no `torch.cat`).
* `mote/serve/prefix_cache.py` — branches of CPU pages (256 chunks, shared between forks) + anchors
  holding only the Mamba-3/routing/dechunk states and the logits. The card anchor is pinned.
* `mote/serve/graph.py` — decode as one CUDA graph per byte: IF node on the device router bit,
  device-side nucleus sampling, K=8 replays per host sync, `done` freezes the step at a stop.
* `Engine.rewarm()` — after every EMA swap the recent conversations are re-read into fresh anchors.

## Tests (105 passed, `pytest tests`)

`tests/test_prefix_cache.py`: warm continuation == cold prefill; branch extend / refresh / fork with
shared pages; hot arena copies nothing; budget + pinned card; rewarm restores the anchors.
`tests/test_graph_decode.py` (GPU): graph == eager greedy byte for byte over 40 bytes with boundaries,
same state and arena rows after; stop id and `max_bytes` freeze the state exactly despite 8 replays
per sync; the device sampler matches `_dist` (TV 0.02 over 6000 draws, exact nucleus support);
engine graph path == eager path incl. p/entropy/chunk events, next turn warm.

## Prefix probe, 35M (`overnight_sft2`), CPU, 40 greedy turns (`docs/results/2026-08-24-prefix-probe-cpu.json`)

| | 2026-08-23 (snapshots) | 2026-08-24 (store) |
|---|---|---|
| prompt bytes served from the cache | 92.5 % | **92.5 %** |
| chunk cuts moved / replies differ | 0 / 0 | **0 / 0** |
| largest next-byte logit difference | 1.5e-4 | **1.4e-4** |
| warm read, mean | 71 ms | **63 ms** |
| cold read, mean | 835 ms | 921 ms |
| warm read, worst turn | — | **531 ms** (turn 37, a fold: 977 B re-read from the card) |
| arena rows hydrated from CPU pages | — | **3** (one card row per fold; every other turn was hot) |

Typical warm turns read 16–34 bytes in 13–25 ms. The worst turn is now a reported number.

## Prefix probe, 35M, GPU (Triton kernels), 40 greedy turns, beside a running 2h arm (`2026-08-24-prefix-probe-gpu.json`)

| | GPU |
|---|---|
| prompt bytes served from the store / cuts moved / replies differ | 92.5 % / 0 / 0 |
| largest next-byte logit difference (kernel resume vs one pass) | **0.12** — kernel rounding, not the store (CPU reference: 1.4e-4); greedy replies identical |
| warm read, mean / worst (non-fold turns, ~30 B) | **47 ms / 57 ms** |
| fold turns (~1000 B re-read from the card) | 55–63 ms |
| cold read, mean / worst (~1500 B) | 44 ms / 56 ms |

Two fixes came out of this run, both in the model: `move_state` copied every tensor with a blocking
`.to()` (one stream sync each — 19–36 ms per restore beside the trainer; now non-blocking with one
sync: 4–10 ms), and the Mamba-3 kernel was handed `v` in fp32 after any decode (the eager `step`
returns fp32, the kernel returns bf16), so Triton compiled a second variant on the first warm turn of
every process (3.5 s; now the states are normalised at the kernel boundary, 41 ms). Honest reading:
**at 35M on the GPU the store is a wash for latency** — a 30-byte continuation costs ~37 ms of launches
and syncs, about what a cold 1.5 KB prefill costs through the kernels. It pays on the CPU (Windows
studio, 63 vs 921 ms) and it is built for the flagship's 16 KB contexts, where a cold read is a full
16 KB / 105M-parameter pass and the hot arena saves the row copies; that number waits for an idle GPU.

Note on this checkpoint: its router treats the 482-byte identity card as **one chunk** (boundary
probability mean 0.02 on prose; prefill, step and continuation agree 482/482 on CPU and GPU), so its
anchors and arena rows are tiny. The flagship's rows will be ~3000 per full context.

## Still to measure (needs an idle GPU — the pre-launch arms hold it)

* Per-byte decode time on the flagship preset, graph vs eager (spike numbers: toy replay 10.8/23.5 µs,
  92 µs with unfused sampling; the sort over 262 logits wants one Triton kernel).
* The memory gate for the hot arena on the flagship: trainer peak (6.34 GB at 16384, ckpt-main) +
  arena (147 MB at 4096 rows) + serving working set < 8 GB with margin; fallback is a flag that
  flushes the arena between turns.
* The 40-turn probe on the GPU with Triton (kernel rounding is the question there).

## JEPA-minimal arm (read the same day)

Equal wall-clock (120 min): **1.1943 vs the Muon control's 1.1773 (+0.017)**, router quality equal
(val bpic 3.23 both). The step-matched number (−0.032 at 26.3k) is confounded by cooldown position and
is not the gate number. EMA and SigReg arms pending.
