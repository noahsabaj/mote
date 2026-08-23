# Results — prefix cache and batch folding, 2026-08-23

The serving engine now keeps snapshots of its inference state and reads only the bytes it has not seen
(`morpheme/serve/prefix_cache.py`, `Engine._prefill_with_cache`; design in `docs/context.md`). Auto folds
happen in batches (a quarter of the window at a time) and the studio sends its last fold back, so the
prefix stays identical between folds. Everything below is the 35M on the CPU, which is where the Windows
studio runs while training holds the GPU.

## Live studio, before (cold read every turn)

`/v1/chat/completions`, `max_tokens` 1, prompt = identity card (479 B) + a user turn of the stated size:

| user turn | time to first byte |
|---|---|
| 50 B | 0.35 s |
| 900 B | 1.3 s |
| 1800 B | 3.5 s |

Superlinear in the prompt: the Relation reference path materialises the S×S evidence matrix.

## Live studio, after

Same route, `runs/overnight_sft/last.pt` (the checkpoint the service had loaded):

| turn | bytes read | wall |
|---|---|---|
| turn 1, cold (~1 400 B with the card), 24 reply bytes | all | 2.14 s |
| turn 1 again (regenerate — the prompt snapshot is an exact hit) | 0 | 55 ms |
| turn 2 (reply + a 20-byte question appended) | ~30 | 127 ms |
| turn 2 again | 0 | 49 ms |
| a new ~600 B conversation (only the card is warm) | ~600 | 0.53 s |

Nine snapshots, 145 MB on the CPU at that point (budget 1 GB, `MORPHEME_PREFIX_CACHE_MB`).

## Probe: 40 greedy turns, warm vs cold (`morpheme.eval.prefix_probe`, `docs/results/2026-08-23-prefix-probe.json`)

A scripted conversation of 40 short questions, `max_bytes` 48, greedy. Each turn is generated once
through the cache (with `verify_prefix`, which also reads the prompt cold and compares) and once by a
second engine with the cache disabled.

| | |
|---|---|
| prompt bytes served from the cache | **92.5 %** |
| turns where a chunk cut moved (warm vs cold) | **0 / 40** |
| largest next-byte logit difference | 1.5 × 10⁻⁴ |
| greedy replies that differed from the cache-less engine | **0 / 40** |
| mean read time per turn, warm | **71 ms** |
| mean read time per turn, cold | 835 ms |

Typical warm turns read 16–40 bytes in 31–54 ms against 170–1 440 ms cold. Folds happened at turns 21,
29 and 37 (batch folding: eight turns apart at this reply length); the turn right after a fold reuses only
the card (482 B) and reads ~1 000 B in ~0.4 s, the next turn is back to ~35 ms.

## What this does not cover

* GPU serving with Triton: prefill runs the fused Mamba-3 kernel but a warm continuation runs the
  reference path (`initial_states is None` gates the kernel) — the upstream kernel accepts `Input_States`;
  wire it through and re-run the probe on Fedora, where the kernel-vs-reference rounding is the new question.
* Sampling at temperature > 0: the probe is greedy so replies can be compared byte for byte; a 1e-4 logit
  difference cannot change a sampled distribution measurably.
* The `/v1/chat/completions` route has no `prev` hint, so a folded conversation there refolds every turn and
  reuses only the card after the window fills.
