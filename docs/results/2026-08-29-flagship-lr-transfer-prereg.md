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
- **Control read (~13:40 EDT):** `mote-bitwise-ctl` **DIFFERS** too — HEAD twice on one L4 differs from
  step 1 in the backward (`grad_norm` 1388.8065 vs 1388.7786; `tl.atomic_add` in the SSD / Mamba-3 backward
  kernels). But the *forward* is deterministic across processes and jobs (three HEAD runs: step-1
  `train_bpb` 438.67949915366603 identically) and `58e8672` differs there (438.68575107, 1.4e-5 relative):
  the housekeeping moved the forward by float noise, reproducibly, and after step 1 its deltas sit inside
  the same-code envelope on every logged field. The gap protocol is re-stated in the housekeeping prereg's
  amendment (forward-only bisect + a ≥3-run envelope). The cloud `pytest` also caught a stale GPU-only test
  (`c7cc1d9`).


**Amendment 2026-08-30 00:55 EDT.** `mote-lr-28p8e-4` was meant to stop at 15,259 steps (D = 5.0e8, shared warmup)
and ran to the full 61,035-step horizon instead: `cloud_arm.sh`'s poller did fire ("step 15290 >= 15259, sending
SIGTERM", "trainer exited with 143"), but `$!` on the L4 studio was a launcher shim — the trainer ran on as an orphan,
the job billed 12.8 h ($6.19) for a 3-h point, and its `log.jsonl` is one continuous run to 61,035. Read it as a fourth
full-horizon point, not the short one (its `--max-steps` and warmup were the horizon's all along). The trainer now
stops itself (`--stop-step` / `--stop-minutes`, schedule untouched); the poller is gone. Measured L4 throughput at this
recipe is 39–45 KB/s (evals every 500 steps included), not the 74.8 the budget assumed: each full arm ≈ $6.2, the four
≈ $25 against 25 credits — session 1 (submitted 00:54, pending) and the ladder pilot run only if credits remain;
the pilot's waiter was disarmed.

## Read (2026-08-30, 13:37–14:15 EDT)

**What ran.** Lightning stopped `mote-lr-14p4e-4` (step 52,810, $6.12) and `mote-lr-7p2e-4` (step 45,630,
$5.22) at 13:37:42 EDT — the org credits ran out ($24.64 across every job; the four lr arms $24.31) — and
`mote-s1b` failed to get a machine three seconds later ("job reconciliation failed", $0). A stopped job's
artifact folder is empty (`Teamspace.download_folder` finds no files), so the runs were recovered from the
jobs' stdout — the trainer's own log lines — into `runs/cloud/lr/<lr>/log.jsonl`: 28.8e-4 and 3.6e-4
complete at 61,035; 14.4e-4 to 52,810 (86 %); 7.2e-4 to 45,630 (75 %). Matched-step comparisons are fair
(the four arms share `--max-steps`, hence the schedule at every step); only the end-of-horizon read is
missing for two arms.

**The preregistered curve is unreadable at this horizon.** `val_bpb_ema` at decay 0.9999 starts from the
init weights with no bias correction (`train.py:716`, `:934`): the init still weighs 0.9999^n = 0.37 at 10k
steps, 0.05 at 30k, 0.011 at 45.5k. The ema's val_bpb is 5.7–12.9 at step 5,000 against raw 1.9 and only
drops below raw at ≈ 50k steps. Its "crossovers" (33,500 / 8,000 / 8,500 — in inverted order; a two-point
fit through them has β = +0.48) are the arms shedding the init at different rates, not quality. The read
below is on raw `val_bpb`, which the prereg reports beside the ema; the ema column is recorded, not used.

**Crossovers on raw `val_bpb`** (first eval at which the lower lr leads and keeps the lead to the end of
the shorter arm; evals every 500 steps):

| pair | predicted step | window (½×–2× in D) | observed | in window |
|---|---|---|---|---|
| 28.8e-4 → 14.4e-4 | 5,800 | 2,900–11,600 | **4,000** | yes |
| 14.4e-4 → 7.2e-4 | 27,700 | 13,850–55,400 | **13,000** | no (0.47×) |
| 7.2e-4 → 3.6e-4 | 135,000 | 67,500–270,000 | **18,000** | no (0.13×) |

Third prong: 3.6e-4 does not trail 7.2e-4 — it leads from step 18,000 to the end of the 7.2e-4 arm, by
0.009–0.021 over the last 15k steps (0.0173 at 45,500), and leads every arm at every step past 18k. Raw at
45,500: 28.8e-4 1.598 · 14.4e-4 1.246 · 7.2e-4 1.183 · 3.6e-4 1.166; 3.6e-4 ends 61,035 at 1.1396 raw /
1.1102 ema. 28.8e-4 finished normally at 1.599 (the norm guard never fired): an upper bound on the usable
lr — too hot, not unstable.

**Transfer does not hold.** The 35M fit puts lr\*(2.0e9) at 7.2e-4; at the flagship shape lr\*(2.0e9) ≤
3.6e-4 — the sweep's floor, unbracketed from below — so the fit is ≥ 2× high here, and every crossover
came *earlier* than predicted (lr\* lower at every D). Two-point fit through the first two raw crossovers:
β = −0.59, lr\*(2.0e9) = 4.1e-4, lr\*(2.53e10) = 0.92e-4; through all three: β = −0.83, lr\*(2.53e10) =
0.27e-4. The three crossovers span only 4.5× in D, so β is poorly determined and the trunk-horizon numbers
are 60× extrapolations; what is supported is the factor: the fit's 2.38e-4 for the trunk is high by ≥ 2×
if the correction measured at 2.0e9 carries — ≈ 1.2e-4 is the least-extrapolated estimate, and the true
value may be lower. **The signed launch rule does not fire on its own; the trunk lr goes to Noah.** The
local three-point rule (`2026-08-28-lr-prereg.md` § Signed 3) still reads Monday on `t3l24_dense_*`; it
reads `val_bpb_ema` too, which is sound only past ≈ 50k steps — its budgets must start there.

**Two defects this read found.** (1) The EMA has no bias correction and starts at the init: as a read it
is blind for the first ≈ 50k steps of any run, and the trunk serves this EMA from step one
(`load_weights(prefer_ema=True)`), so the trunk's first 6–11 h (42–75 KB/s) would serve weights carrying a
large init component (0.37 at 10k steps, 0.05 at 30k). Fix before the launch: bias-correct at read and
serve time — `(ema − β^n·θ₀)/(1 − β^n)` with θ₀ kept once beside the run and `ema_steps` in the checkpoint,
training bitwise untouched — or start the EMA at the end of warmup. (2) A stopped job leaves no artifacts;
its stdout was the record. Reads should come from the log stream (as here), or `cloud_arm.sh` should copy
`log.jsonl` into the artifacts path at each log interval.

Cost: $24.64 of the 25 org credits (lr arms $24.31, failed submissions $0.33). Session 1 of the throughput
line and the ladder pilot move to the local card after the Monday drain.
