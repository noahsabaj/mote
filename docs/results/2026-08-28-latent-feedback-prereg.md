# Latent feedback (2608.08888) — pre-registration, signed 2026-08-28

Full-bandwidth transformer, Wang et al. (JHU/Princeton/Microsoft, 9 Aug 2026). Reading and the
mapping: docs/shape.md § Reading 2026-08-28 (2608.08888).

## Mechanism as built for Mote

A GLU fusion of the previous position's top state into the next input,

    u_t = RMSNorm( W_U · h_{t−1} ⊙ σ(W_G · x_t) )

state on the value path, the plain input only as the gate (the paper's asymmetry: an additive path lets
the model ignore the state). The first position stays plain. Trained by parallel multi-pass teacher
forcing: pass 1 is the ordinary forward; pass k shifts pass k−1's top states one position right, fuses,
re-runs; next-byte loss on every pass with λ = 1, gradients not detached; **prefix mixin** (a random plain
prefix per fused pass) and **jitter** (uniform ±0.02 on the carried state).

Two levels:

- **chunk-level** — the main network's top state (768-d, post final norm) fused into the next chunk's
  main input, gate on `pad(hc_t)`. Feedback passes re-run the main network, dechunk, decoder and head;
  the encoder and routing are reused from pass 1 (they see plain bytes either way).
- **byte-level** — the decoder's top state `h3` (512-d, post final norm) fused into the next byte's
  encoder input, gate on the byte embedding. Feedback passes re-run the whole model, routing included —
  the router sees the fully processed past.

## The arms

Three arms on the **trunk snapshot**, pre mix, the trunk's constant lr, **24 h each on the 4060 Ti**,
run after the trunk and before mid: `control` (single-pass continuation), `chunk`, `byte`. Feedback arms
use the paper's whole-run mixture from their first step: **75 % one-pass / 22 % two-pass / 3 %
three-pass** micro-batches (1.28× compute per batch → ~22 % fewer bytes than the control at equal
wall-clock). Multi-pass micro-batch memory is profiled before the arms are queued (`profile_step` with a
3-pass step at the flagship shape); fallback order for an arm that does not fit the 6.2 GB ceiling:
**feedback micro-batches on 8192-byte windows**, then **detach** (per-pass backward, one pass of
activations at a time). Nothing in the trunk, the branch trigger, or the mid 2×2 protocol changes; mid
starts three days later than it otherwise would.

## The gate (read at 24 h, equal wall-clock, the trunk's eval set)

An arm goes forward **iff** its k=1 fused-prefill val_bpb ≤ control − 0.005 **and** its k=0 val_bpb
≤ control + 0.005 (under Soft decoding the prompt is read at k=0). Both feedback arms clear → the better
k=1; within 0.005 of each other → chunk-level. The matched-bytes read (the control at the feedback arm's
byte count) is recorded as a secondary diagnostic, never the decider. The serving bench under Soft
decoding (prefs rubric, parked engine — never the live daemon) confirms the winner; **mid's 2×2 starts
from it**. Neither clears → mid starts from the trunk snapshot and this doc records the numbers.

## Serving

Soft decoding by default for a feedback checkpoint (two matmuls per step inside the decode graph;
`h_prev` in a static buffer and in prefix anchors); **Fused** prefill (one extra parallel pass over the
prompt) a per-request GenParams option, measured by the bench. MBP is off in the flagship, so there is no
draft-verification conflict (the fused input at t needs h_{t−1}; exact parallel verification would not).

## Post stages

SFT-1 / DPO / RLVR-1 train **multi-pass, k = 3 throughout** (the paper's long-context + instruction-tuning
recipe); rollouts under Soft decoding.

## Build

`mote/model/feedback.py` (fusion, shift, mixin, jitter), `FeedbackCfg`, the feedback forward in
`HNetForCausalLM`, multi-pass `compute_losses` with the schedule / window / detach flags,
`--eval-feedback-passes` (val at k = 0/1/2), Soft/Fused in `step` / `prefill` / graph / anchors,
`scripts/latent_feedback_arms.sh`. CPU tests during the trunk; targeted GPU tests and the memory profile
in the gap before the arms.
