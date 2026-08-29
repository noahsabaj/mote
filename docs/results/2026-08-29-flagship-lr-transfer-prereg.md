# Flagship-shape lr transfer: the check the 35M fit never had (2026-08-29)

Signed 2026-08-29 on Noah's "Go" (the org Lightning account's 25 credits, spent this weekend in case they
expire). Nothing frozen moves on these numbers alone; they decide whether the signed launch rule
(`2026-08-28-lr-prereg.md` § Signed 3) may fire on its own or the lr goes to Noah first.

## Why

The horizon fit that sets the trunk lr — ln lr\* = 2.1238 − 0.4371 ln D (R² 0.984 over 66 budgets) — is
built entirely on 35M `local` arms (4×4 @2048, `data/local_mix`). The trunk is `mote-96m` at 1×16384 on
mix A. The only flagship-shape lr evidence is the 30-min `runs/lr_sweep_*` (a throughput confound). So the
launch rule extrapolates 6× in D *and* transfers across shape and data, untested. The local GPU cannot test
it before the gap (the queue drains ≈ Mon 2026-08-31 19:30 EDT — the resumes carry their clock — and a
3 × 12-h flagship sweep would push the launch by 36 h); the cloud can, in parallel.

## Arms

GCP L4, on-demand at $0.48/h (its spot price is higher, $0.73; the L4 is Ada sm_89 like the 4060 Ti, so
the same Triton kernels run). The trunk argv, step-driven:

    --preset mote-96m --data data/flagship_mix --batch-size 1 --seq-len 16384 --grad-accum 2
    --optimizer muon --weight-decay 0.1 --schedule trunk --bound-floor 2048 --ckpt-main
    --eval-ema 0.9999 --eval-every 500 --eval-batches 16 --eval-spread --seed 42 --ckpt-minutes 60
    --max-steps 61035

32,768 bytes per step; D = 2.0e9 at the end; warmup 6,103 steps (the recipe's 10 % of the horizon),
identical across the four arms because `--max-steps` is identical:

- `lr/3.6e-4`, `lr/7.2e-4`, `lr/14.4e-4` — to step 61,035 (D = 2.0e9).
- `lr/28.8e-4` — the same argv, stopped at step 15,259 (D = 5.0e8) by `scripts/cloud_arm.sh`.

Data: the local shards themselves, uploaded (md5-identical to `data/flagship_mix.*` and
`data/local_mix.*`). A rebuild from the HF streams (`scripts/cloud_build_mixes.sh`) was tried first and is
*not* byte-identical: `local_mix` predates the 08-24 fresh-document skipping (a few KB per source moved)
and `flagship_mix` predates the 08-25 swap of the code source to codeparrot-clean — so a rebuilt mix A is
not the trunk's mix A. The runs are pulled into `runs/cloud/lr/<lr>`.

## What the fit predicts

lr\*(2.0e9) = 7.2e-4. Two arms 2× apart cross where lr\*(D) equals their geometric mean:

| pair | lr\*(D_x) | D_x (bytes) | step |
|---|---|---|---|
| 28.8e-4 → 14.4e-4 | 2.04e-3 | 1.9e8 | 5,800 |
| 14.4e-4 → 7.2e-4 | 1.02e-3 | 9.1e8 | 27,700 |
| 7.2e-4 → 3.6e-4 | 5.1e-4 | 4.4e9 | 135,000 — beyond the arms |

## Read

On `val_bpb_ema` (the curve the fit reads; raw `val_bpb` is reported beside it). A crossover is the first
eval at which the lower-lr arm leads and keeps the lead to the end of the shorter arm.

**Transfer holds iff** both observed crossovers fall within [½×, 2×] of their predicted D **and** 3.6e-4
still trails 7.2e-4 at step 61,035. Then the signed launch rule fires as written. Otherwise the two observed
crossovers give a two-point (intercept, β) at the flagship shape; its lr\* at 2.53e10 is reported next to
the 35M fit's 2.38e-4, and the numbers go to Noah before anything moves. A 28.8e-4 arm that diverges or
trips the norm guard is an upper bound on the stable lr at this shape, not a crossover — the other two
prongs still decide. The windows are a factor of 2 in D, not a bpb margin (seed noise at 35M is 0.0003).

## Also on the credits

- **The QK-Norm set stays on the local queue.** It was going to move to one L4 (step-budgeted 24k/12k), but
  the L4 measures 0.73× the 4060 Ti at the flagship shape (54.8 KB/s at accum 2) and only ~0.5× at 35M
  (69 KB/s), and the org account runs **two GPU jobs at a time** — the six arms would have cost ≈ $6 and a
  slot for 13 h, pushing the lr arms past Monday's gap. Cancelled at step ~1,300 ($0.15); the gap still
  opens ≈ Mon 19:30 EDT.
- **The housekeeping bitwise check ran early** (`runs/cloud/bitwise/{old,new}`: 100 steps of the trunk
  argv with `--log-every 1` on `58e8672` and on HEAD `bbb3d36`, one L4, `scripts/bitwise_diff.py`):
  **DIFFERS** — from step 1 (`grad_norm` 1388.99 vs 1389.04, `train_bpb` 438.6795 vs 438.6858, `w_norm`
  in the 9th digit; 1003 differing log values over 100 steps; every tensor in `last.pt`). The size says a
  changed reduction order or kernel path, not a different init or a semantic change. Whether the L4 is
  run-to-run deterministic at this shape is being measured by `mote-bitwise-ctl` (HEAD twice, same
  machine, plus the GPU test suite): identical → the refactor is not bit-inert and the gap protocol's
  revert applies to whichever housekeeping step moved the bits (bisect: five commits, ~6 min each);
  differing → the verdict cannot be read on this GPU and the local check in the gap decides.

Spend: four lr arms ≈ $16 of the 25 at the measured rate; the two-slot cap means the three long arms
finish ≈ Sun 2026-08-30 13:00 EDT.
