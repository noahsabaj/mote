#!/usr/bin/env bash
# Does a hyper-connection spine buy Mote anything, and does the memory-free form buy as much?
# Signed 2026-08-26, re-signed 2026-08-27 after the first version was found unable to detect its
# own effect. The reading is docs/research/spine-2026-08-26.md.
#
# One n-stream residual at BYTE resolution, seven sites: the three encoder sublayers, the whole
# chunk stage, the three decoder sublayers. The main network keeps its own plain residual inside —
# different width, different resolution. No paper in the HC family crosses a resolution boundary,
# so the topology is ours and the arms are the only evidence there will be.
#
#   frac    n slices of the same 512-wide state   +0.56 GB @ 16384   ~half of HC's gain in the lit
#   expand  n copies of it                        +1.8 GB  @ 16384   the form every paper measures
# If frac lands within noise of expand, the expanded form never has to be paid for.
#
# ---------------------------------------------------------------------------------------------
# WHY THIS RUNS AT MOTE-96M AND NOT AT THE 138M IT IS FOR
#
# Mote-96M's byte level is IDENTICAL to Mote-138M's: d_model_outer 512, three encoder and three
# decoder sublayers, seven sites, 128-wide frac slices, the same Spine modules with the same
# parameter count. Only the main network differs (12 Relation layers against 18) and the spine
# never touches it. So nothing about the spine has to transfer between decider and target.
#
# Mote-35M would have been cheaper and was rejected: enc 2 + chunk + dec 2 is FIVE sites, and the
# doc's first stated risk is that seven may already be too shallow for streams to differentiate.
# A null at five sites cannot distinguish "hyper-connections do not help" from "five is not enough",
# and a null at the decider kills the line before the seven-site confirm ever runs.
#
# WHY --schedule trunk, WHICH THE FIRST VERSION DID NOT SET
#
# The gate compares at equal wall-clock, and the spine costs throughput, so a slower arm gets fewer
# steps. Measured off the three T3 12-h curves, a 15 % step shortfall costs +0.005 bpb and a 30 %
# shortfall +0.011 to +0.016 — against a literature effect of -0.010. frac at 0.85x and expand at
# 0.69x were therefore spending half to all of the effect before the spine did anything: the gate
# as first written could not detect what it was looking for, and a spine performing exactly as the
# papers claim would have read as break-even or a loss.
#
# Two things fix it. The kernels (mote/model/spine_kernel.py) shrink the tax at its source. And
# trunk holds the learning rate constant after warmup, so at step k every arm is at the same lr and
# a MATCHED-TOKEN read is valid — under the wsd default the lr follows wall-clock, so a slower arm
# sits later in its decay at step k and no post-hoc correction separates the two effects. With
# trunk both readings fall out of one set of runs at no extra GPU time:
#
#   MATCHED TOKENS decides   the architecture question, artifact-free
#   WALL-CLOCK reports       the cost of shipping it as it stands today
#
# No pre-registered threshold. The evidence is the four numbers scripts/spine_report.py prints and
# the call is joint — crossing a resolution boundary is unprecedented enough here that a fixed bar
# would throw away the interesting outcome.
#
# ELR-MATCHED on the shared parameters. The spine arm carries ~244 K parameters the control does
# not, so matched nominal lr is not matched ELR — the confound that reopened the Muon vs Muon-SW
# freeze. A0 records its per-matrix ELR; A1 and A2 replay it onto the matrices they share.
#
#   bash scripts/spine_gate.sh 2>&1 | tee docs/results/$(date +%F)-spine-gate.log
set -euo pipefail
OUT="${OUT:-runs/mote-96m}"
MIN="${MIN:-480}"
LR="${LR:-8e-4}"
DATA="${DATA:-data/flagship_mix}"
[ -f "$DATA.meta.json" ] || DATA="data/local_mix"
mkdir -p "$OUT" docs/results

# Nothing has ever completed a training step at Mote-96M or Mote-138M — runs/flagship_shape_v2 has
# a 0-byte log. A0 is the first arm ever at this shape, so if the SHAPE is wrong all three arms
# fail together and the gate says nothing about the spine. scripts/spine_shakedown.sh runs ~1 h of
# A0 first and is not optional before committing 24 h.
echo "== 0/3  profile all three modes at the real shape =="
for SPINE in off frac expand; do
  python -m mote.train.profile_step --preset mote-96m --data "$DATA" --ckpt-main \
      --batch-size 1 --seq-len 16384 --chunk-bytes 6 --spine "$SPINE" \
      --out "docs/results/$(date +%F)-spine96-profile-$SPINE.json" || {
        echo "profile failed for --spine $SPINE — fix that before spending 24 GPU-hours"; exit 1; }
done

# --ckpt-main on every arm, uniformly. Mote-138M at 16384 does not fit without it even with the
# spine OFF, and 96M expand is close enough to the edge to want it; more to the point, a
# checkpointing difference BETWEEN arms is a throughput difference, and throughput is exactly what
# the wall-clock reading charges for. It has to be the same on all three or the comparison is not
# one. --schedule trunk + --eval-ema match the T3 arms, which is what makes the matched-token read
# valid and the two gates comparable to each other.
common=(--preset mote-96m --data "$DATA" --optimizer muon --lr "$LR" --ckpt-main
        --batch-size 1 --grad-accum 4 --seq-len 16384 --bucket 64
        --schedule trunk --eval-ema 0.9999
        --max-minutes "$MIN" --eval-every 1000 --eval-batches 16 --eval-spread
        --log-every 100 --ckpt-minutes 15 --no-mbp --seed 42)

echo "== 1/3  A0 control: no spine, recording its per-matrix ELR =="
python -m mote.cli train start -- "${common[@]}" \
    --spine off --elr-trace-out elr_trace.json --out "$OUT/spine-ctl"

echo "== 2/3  A1 frac: n=4 slices, ELR matched to A0 =="
python -m mote.cli train start -- "${common[@]}" \
    --spine frac --spine-n 4 --elr-match "$OUT/spine-ctl/elr_trace.json" --out "$OUT/spine-frac-n4"

echo "== 3/3  A2 expand: n=4 copies, ELR matched to A0 =="
python -m mote.cli train start -- "${common[@]}" \
    --spine expand --spine-n 4 --elr-match "$OUT/spine-ctl/elr_trace.json" --out "$OUT/spine-expand-n4"

python -m mote.cli train queue
cat <<'NEXT'

When they finish:
  python scripts/spine_report.py runs/mote-96m/spine-{ctl,frac-n4,expand-n4}

Then two things are still open and deliberately not queued here:

  the Mote-138M confirm  deferred until this returns a winner. frac fits at 138M/16384 (5.74 GB
                         measured); n=4 expand does not fit on this card at all, so if expand wins
                         the confirm needs the memory work first. Nothing has trained at 138M, so
                         a confirm is two arms — control and winner — not one.

  A3 sinkhorn            only if A2 clears. It compares sHC's spectral sphere against the Birkhoff
                         manifold DeepSeek-V4 (1.6T, with Muon) and Motif 3 (314B) actually ship,
                         and it only earns four hours once the expanded form has shown it is worth
                         doing at all. Read h_res_drift first: if the static mixer never left the
                         identity, A3 is comparing two ways of doing nothing.

      python -m mote.cli train start -- ... --spine expand --spine-project sinkhorn \
          --elr-match runs/mote-96m/spine-ctl/elr_trace.json --out runs/mote-96m/spine-sinkhorn-n4
NEXT
