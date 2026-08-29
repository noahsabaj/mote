# The throughput line — pre-registered (signed 2026-08-29, ~14:20 EDT)

Grilled in three rounds on the afternoon of 2026-08-29, out of "what if we wrote parts of the pipeline in
C / Rust / CUDA?". The answer to that question is in the candidate table: the compute already lives in
Triton and cuBLAS, Python is the launcher, and the measured cost is the *number* of launches and the memory
traffic between them — so the work is fusion, fewer syncs, graphs and two kernels, not a language. The
signed design comes first; the hypotheses follow.

## Frame

- **Objective: KB/s at the frozen recipe on the 4060 Ti.** Every other benefit is ranked in this unit.
- **Bar, two-tier.** A change that claims exactness (a refactor, a fused copy of an op, dropping a recompute)
  must be forward-bitwise on a fixed batch and its 100-step trajectory must sit inside a ≥3-run same-code
  envelope (the backward is not bitwise on any GPU — `tl.atomic_add` in the SSD / Mamba-3 backward). A
  change that moves numerics (Inductor reorders, tf32, bf16 activation storage, fp8) passes a matched-seed
  pair: val bpb within seed noise at matched steps and not worse at matched wall-clock.
- **Landing: the signed hot-swap rule stands, exact-only.** An exact change enters the running trunk at a
  daily snapshot once it clears the swap gate; **exact changes may be bundled** so that sub-10 % pieces clear
  the ≥ 10 % bar together (each piece forward-bitwise on its own, the bundle measured once). Tier-2 winners
  land for the arms and the next trunk, never into the running one.
- **Swap gate.** (1) forward-bitwise on a fixed batch from the snapshot, new code vs the trunk's code;
  (2) a 100-step resume from that snapshot on an L4 under both codes, the new inside the old's A/A envelope;
  (3) the speed ratio measured on the L4 (same Ada family); (4) the first local hour after the swap confirms
  the ratio, and the next snapshot reverts if it does not.
- **Dev GPU: a cloud L4 on the org account**, the remaining credits being this line's budget, each session
  reported with its cost. The local card is booked through the trunk and the latent-feedback arms.
- **My time:** this line through the trunk week (each day earlier is more D); the process line's probe suite
  starts when the trunk ends (≈ 2026-09-07).

## Hypotheses (predictions to be tested, not claims)

| # | Hypothesis | Predicted step gain | Test | Tier |
|---|---|---|---|---|
| 1 | Dropping `--ckpt-main` fits in 6.2 GB (serving is on the CPU during the trunk); forward kernels are deterministic, so the gradients are bitwise the same without the recompute | +25–30 % | 20-step probe at 16384 / accum 2 without the flag; read peak GB | exact |
| 2 | FlashRelation v2 at 16384 (signed 08-24; 1.7–2.2× on a kernel that is 30–40 % of the step; only measured at 4096, where it is 1.006×) | +12–22 % | 30-min v1 / v2 pair at 16384 | exact (fixed tile tables) |
| 3 | The per-step host syncs are the 20 % idle: `train.py:984` reads the step's CE for ATDC every step; `dc.py:82/85/167` do `bool(x.any()/all())` inside the routing forward. Each drains the queue and exposes the launch latency of ~5,300 kernels | +10–15 % | nsys busy % before / after | routing asserts exact; ATDC read once per log interval (10 steps) = tier 2, screened |
| 4 | Inductor on the block stacks fuses the 35 % elementwise / copy / cat bucket (FlashRelation is already `mote::relation_fwd/bwd`; the mamba ops trace as autograd Functions); `--compile` exists and was never measured | +15–20 % | eager vs `--compile` twins, 500 steps each, s/step over the last 300 | tier 2 |
| 5 | The layernorm backward (10 % of the step, 8× off its bandwidth floor) reaches the floor as one Triton kernel | +5–9 % | write + microbench, then a step read | exact-claimable |
| 6 | A whole-step CUDA graph once routing is static (`--bucket` rounds M up) removes what 3–4 leave | +5–10 % | after 3–4 | exact |
| 7 | Persistent autotune tables kill the 60-s compile tax and the autotune nondeterminism | 0 on the trunk, ~5 % of a 30-min arm | first bundle | exact |
| 8 | bf16 activation storage, tf32 on the fp32 path, fp8 GEMM (in that order of expected value; GEMM is 14 % of the step) | small; bf16 storage is the memory lever behind #1 | pairs | tier 2 |
| 9 | Ada-specific tiles for the upstream Mamba-3 / SSD kernels (tuned for Hopper) | 2–4 % | tile sweep on the L4 | exact |

The buckets overlap, so the gains do not add; 1.5–2× on the trunk (75 → 110–150 KB/s) is the defensible
upper hypothesis. The H100 is 4.8× for reference and is not this line.

**Memory fallback for #1:** checkpoint the outer (byte-level) layers only, keeping the main's recompute
off; bf16 activation storage is the tier-2 fallback behind it.

## Session 1 — Sunday 2026-08-30, one org L4, ≈ $1.2, when the lr arms free a slot

0. GPU forward-only bisect of the five housekeeping commits at the flagship shape (`--lr 0 --max-steps 2`):
   names the commit behind the 1.4e-5 forward change for the record. (On the CPU reference path all five
   are bitwise identical at 96M — `scratchpad/fw96`, 2026-08-29 14:04 — so the change is in the kernel
   dispatch, and it sits inside the envelope: nothing reverts.)
1. #1: the `--ckpt-main` memory probe.
2. #2: the v1 / v2 pair at 16384.
3. #4: the eager / `--compile` twins.

Their numbers order the engineering. Default order if they come out as predicted: #3 → #5 → #4 / #6 →
v2 tiles → #7 in the first bundle.

## Deferred, named

Serving speed (a Rust/C++ CPU decoder: 3–6× at 32M, 2–4× at 96M — mostly bytes moved, not Python left
behind); a deterministic backward (two-pass reductions in Triton); reversible outer layers and host-pinned
optimizer state unless #1's probe needs them; moving the trunk to an H100 (a different decision, Noah's).
