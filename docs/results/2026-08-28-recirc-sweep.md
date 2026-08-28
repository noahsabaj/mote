# Training-free recirculation (2608.17981) on Mote — harness, partial grid, and where it lands

**Paper.** Mozer, Siddiqui, Sawyer, Sanyal, Liu (DeepMind, 18 Aug 2026). At every step, the residual
stream of a deep layer *s* for the current token is rescaled to the norm of shallow layer *d*'s stream and
mixed in, `z'_d = (1−α) z_d + α (‖z_d‖/‖z_s‖) z_s` (α ≈ 0.07–0.15; β = 1 for Gemma3 4B/12B); layers
*d+1…L* are re-run for that token and its KV is overwritten. The token's own readout stays the first pass;
later tokens attend to the recirculated cache. Recurrence in depth **and** step (their Fig. 3c/4b), so a
single layer can hold z(t) and z(t+1) and state tracking is no longer bounded by depth — unlike looping,
which is recurrence in depth only. Decode is ≈ free (two stacks in parallel); **prefill becomes serial**.

Evidence they give: Gemma3 1B/4B/12B ppl −1…−16 % (12B −25…−35 %, which their own footnote calls a weak
LM); best pairs {11→4}, {18→9}, {35→16}; not temperature (additive with T=1.2 tuning); not looping
(training-free looping shows no robust region on Gemma); early tokens (t<10) harmed in 1B → ramp; the
benefit vs lag is a power law with a tail to 256 tokens; adverbs/adjectives/verbs gain most. Downstream:
instruction-following error −25 %/−75 % (4B/12B), GSM8k up at pass@1 and pass@128, single-token
benchmarks within ±0.5 pt, Racing-Thoughts mixed. Adaptive recirculation (2-layer MLP → per-dimension α,β
from [z_s, z_d]; base frozen; 100 steps × 32) −23.0 % mean ppl vs −8.5 % fixed vs −21.6 % full fine-tune.
**Ministral3, Qwen3, Pythia, Phi2: a robust region but <0.5 %** vs ~5 % on Gemma (Peri-LN suspected).

## Why it was measured, not argued

The main network is a pre-norm fp32-residual stack of Relation blocks with a per-chunk {P2, I~} cache —
the paper's preconditions exactly (a residual "blackboard" aligned across layers; KV-like rows that a
second pass can overwrite). Chunk resolution makes the serial part 3.3× cheaper than a byte-level model
would pay. `scripts/recirc_sweep.py` replicates their (s, d) grid on a checkpoint, eval-only, on the CPU:
the encoder/router/chunking run once per window, the main network runs chunk by chunk with a per-layer
cache, the recirculated second pass rewrites chunk *t*'s rows at layers > *d*, and the dechunk/decoder/head
run once per configuration. **Validated:** at α = 0 the sequential path reproduces the parallel forward
exactly (val_bpb 1.21116 = 1.21116 on 4 windows, 1.15802 = 1.15802 on 32).

## Partial grid — `runs/t3l_dense_4e-4` (35M, 12 h, raw weights, step 189958)

32 spread val windows × 2048 bytes of `data/local_mix` (65 536 targets, bpic 3.817), α = 0.10, β = 0.9,
source rescaled to the destination norm. Baseline val_bpb **1.15802** on this subset.

| pair (s→d) | val_bpb | Δ |
|---|---|---|
| 1→0 | 1.15828 | +0.02 % |
| 2→0 | 1.15857 | +0.05 % |
| 3→0 | 1.15857 | +0.05 % |
| 4→1 (4 windows, baseline 1.21116) | 1.21066 | −0.04 % |

Three of 28 pairs at 32 windows, one at 4. Destination 0 is the layer the paper also finds uninterpretable
("not contextualized to the point that deeper layers can interpret the recirculated signal"), so the
interesting region — destination 1–2, source 3–6 — is exactly what is still unmeasured. Every |Δ| so far is
an order of magnitude under the 0.005-bpb (0.45 %) gate.

**Why it stopped.** The sweep ran beside `elr_gate/muon_ref`, a `--max-minutes 120` arm. Its throughput
fell from a steady 110.6 KB/s (elapsed 60–83 min, n = 466) to a mean of 93 KB/s (−16 %, troughs to 62) over
elapsed 83–94.5 min — the sweep's lifetime — recovered to 110.5 within a minute of the stop, and the arm
will end ~2 % short of its steps (prereg addendum: read the ELR pair at
matched steps). Every queued arm is wall-clock-budgeted and the two probes measure `bytes_per_sec`
directly, so the remaining 25 pairs, the α scan and the β = 1 / ramp variants run only on an idle queue:

    PYTHONPATH=. .venv/bin/python scripts/recirc_sweep.py --ckpt runs/t3l_dense_4e-4/last.pt \
        --windows 32 --batch-size 8 --alpha 0.10 --variants --threads 16     # ~45 min on the CPU

## Where it lands

1. **Training-time recurrence: no.** A serial or block-serial main network removes the parallel pass the
   47 KB/s trunk depends on; K = 64-chunk blocks would be ~78 serial passes per 16 KB sequence, and a K
   large enough to be cheap puts the lag past where the paper measures any benefit. The Feedback
   Transformer (Fan et al. 2021) is the trained precedent and pays exactly this.
2. **Serving-time recirculation: a post-trunk option, gated on the grid.** Decode at batch 1 is
   latency-bound, so a 2-row main step is ≈ free, but the decode graph needs a 2-row capture, the arena's
   append-only contract changes (row *t* at layers > *d* is rewritten), and prefill goes serial per chunk
   (~1200 steps for a 4 KB prompt: seconds on the GPU, tens of seconds on the CPU where serving sits while
   a job runs). The paper's real numbers are the adaptive MLP (frozen base, 100 steps — cheap post-trunk),
   never a trunk change.
3. **Prior:** the non-Gemma <0.5 % is below our own gate; the partial grid agrees so far.
4. **Parked looped-main arm (2608.18222): unchanged.** Looping and recirculation are different principles
   and looping has no training-free gain — nothing here promotes it.
5. **RLVR-1:** the paper reserves chain-of-thought for multi-step inference and recirculation for basic
   state tracking; the k-step planning tasks stay a CoT/RL matter.
