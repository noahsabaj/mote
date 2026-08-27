#!/usr/bin/env bash
# Does a hyper-connection spine buy Mote anything, and does the memory-free form buy as much?
# Signed 2026-08-26; the reading is docs/research/spine-2026-08-26.md.
#
# One n-stream residual at BYTE resolution, seven sites: the three encoder sublayers, the whole chunk
# stage, the three decoder sublayers. The main network keeps its own plain residual inside — different
# width, different resolution. No paper in the HC family crosses a resolution boundary, so the
# topology is ours and the arms are the only evidence there will be.
#
# Two arms, not one, because the memory question and the loss question come apart:
#   frac    n slices of the same 512-wide state   +0.56 GB   ~half of HC's gain in the literature
#   expand  n copies of it                        does not fit   the form every paper measures
# If frac lands within noise of expand, the expanded form never has to be paid for.
#
# Both memory figures above were ESTIMATES when this was signed and both were wrong; the numbers
# now shown are measured (docs/results/2026-08-26-spine-profile.md). frac is not free — the
# residual is the same width, but the spine's own byte-resolution intermediates at seven sites are
# not covered by --ckpt-main, which checkpoints the Relation blocks. And n=4 expanded does not fit
# on the 4060 Ti at 16384 at all, so arm A2 below is BLOCKED pending a call on what to do about
# it: drop to 8192 (everything fits, but that is not the frozen flagship context), run off+frac
# only at 16384, try n=2, or rent the card.
#
# ELR-MATCHED on the shared parameters. The spine arm carries ~244 K parameters the control does not,
# so matched nominal lr is not matched ELR — the exact confound that reopened the Muon vs Muon-SW
# freeze. A0 records its per-matrix ELR; A1 and A2 replay it onto the matrices they share with it and
# let the new spine parameters follow the schedule.
#
# Gate: endpoint val_bpb at matched seed. Seed noise is 0.00025 bpb and the standing gates are
# 0.003/0.005; the literature's effect converts to about -0.010 bpb, so a real effect is ~40x noise.
# Per-stream norm spread is a DIAGNOSTIC, not a gate: stream collapse is a norm runaway, and a run
# that wins while collapsing to one stream has not shown that multiple streams did anything.
#
#   bash scripts/spine_gate.sh 2>&1 | tee docs/results/$(date +%F)-spine-gate.log
set -euo pipefail
OUT="${OUT:-runs/mote-138m}"
MIN="${MIN:-240}"
LR="${LR:-8e-4}"
DATA="${DATA:-data/local_mix}"
mkdir -p "$OUT" docs/results

# Before anything else: nothing has ever run at this shape. 138M at 16384 was estimated at ~5.04 GB
# from Mote-96M's measured 4.31 GB, and an estimate is not a measurement — it needed >7.65 GiB and
# OOM'd with the spine OFF. With --ckpt-main it is 5.18 GB (off) and 5.74 GB (frac). That flag is
# not a tuning knob at this shape; without it none of the three arms below can start.
echo "== 0/3  profile Mote-138M — does the shape fit, and what does the spine cost in throughput? =="
for SPINE in off frac; do
  python -m mote.train.profile_step --preset mote-138m --data "$DATA" --ckpt-main \
      --batch-size 1 --seq-len 16384 --chunk-bytes 6 --spine "$SPINE" \
      --out "docs/results/$(date +%F)-spine-profile-$SPINE.json" || {
        echo "profile failed for --spine $SPINE; fix that before spending 8 GPU-hours"; exit 1; }
done
# `expand` is not in that loop on purpose: it OOMs here, and an expected failure that aborts the
# script is worse than no check at all. It is profiled only when A2 is unblocked.
[ "${FORCE_EXPAND:-0}" = "1" ] && python -m mote.train.profile_step --preset mote-138m --data "$DATA" \
      --ckpt-main --batch-size 1 --seq-len 16384 --chunk-bytes 6 --spine expand \
      --out "docs/results/$(date +%F)-spine-profile-expand.json"

common=(--preset mote-138m --data "$DATA" --optimizer muon --lr "$LR" --ckpt-main
        --batch-size 1 --grad-accum 4 --seq-len 16384 --bucket 64
        --max-minutes "$MIN" --eval-every 2000 --eval-batches 16 --eval-spread
        --log-every 100 --ckpt-minutes 15 --no-mbp --seed 42)

echo "== 1/3  A0 control: Mote-138M with no spine, recording its per-matrix ELR =="
python -m mote.cli train start -- "${common[@]}" \
    --spine off --elr-trace-out elr_trace.json --out "$OUT/spine-ctl"

echo "== 2/3  A1 frac: n=4 slices, no extra memory, ELR matched to A0 =="
python -m mote.cli train start -- "${common[@]}" \
    --spine frac --spine-n 4 --elr-match "$OUT/spine-ctl/elr_trace.json" --out "$OUT/spine-frac-n4"

echo "== 3/3  A2 expand: n=4 copies, ELR matched to A0 — BLOCKED, does not fit at 16384 =="
echo "   measured 2026-08-26: OOM at 16384 even with --ckpt-main (off 5.18 GB, frac 5.74 GB)."
echo "   Fits at 8192 (4.74 GB). Not queued: see the header for the four ways out."
if [ "${FORCE_EXPAND:-0}" = "1" ]; then
  python -m mote.cli train start -- "${common[@]}" \
      --spine expand --spine-n 4 --elr-match "$OUT/spine-ctl/elr_trace.json" --out "$OUT/spine-expand-n4"
fi

python -m mote.cli train queue
cat <<'NEXT'

When they finish:
  python scripts/spine_report.py runs/mote-138m/spine-{ctl,frac-n4,expand-n4}

A3 (sHC vs the manifold DeepSeek-V4 and Motif 3 actually ship) is deliberately NOT queued here.
It only earns its four hours if A2 clears the gate — until then it compares two ways of doing
something that has not been shown to be worth doing:

  python -m mote.cli train start -- ... --spine expand --spine-project sinkhorn \
      --elr-match runs/mote-138m/spine-ctl/elr_trace.json --out runs/mote-138m/spine-sinkhorn-n4
NEXT
