"""Collapse error between an ELR-matched pair, on 2608.24814's metric (scripts/elr_optimizer_gate.sh step 3).

    python scripts/elr_report.py runs/elr_gate/muon_ref runs/elr_gate/muonsw_matched

Δ_coll = mean |L_matched − L_ref| over the post-warmup steps, on losses smoothed with an EMA of 0.99 —
their protocol exactly, so the number is comparable to the ones they report (a few ×10⁻³ across 26
configurations, all under 5×10⁻³).

Two details that are easy to get wrong and both change the answer by an order of magnitude:

  * seed the EMA AT the post-warmup cut, not before it. The first logged step differs by ~26 bpb between
    two runs (raw initialisation), and an EMA with a 100-point time constant carries that spike ~500
    points into the window — it reported 2.7 bpb for a pair whose raw mean difference was 0.055;
  * keep only the last fresh-start segment of log.jsonl. A restart into an existing directory appends,
    and an unsegmented read silently compares two different runs.

The yardstick, measured on Mote 2026-08-26 from `ab2_muon_2h` vs `ab2_muon_seed7`: one seed change is
0.0126 bpb (0.0088 nats) on this metric and 0.00025 bpb on final val_bpb. A collapse error below the
first says the ELR matching worked; a residual gap above the second is a real optimizer difference.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mote import runinfo  # noqa: E402

LN2 = math.log(2)
EMA = 0.99


def curve(run: Path):
    """{step: train_bpb} for the last fresh start in the directory."""
    recs = runinfo.last_segment(runinfo.records(run))
    out, elr, lr = {}, {}, {}
    for r in recs:
        if "train_bpb" not in r or "lr" not in r:
            continue
        out[r["step"]] = r["train_bpb"]
        lr[r["step"]] = r["lr"]
        if "elr" in r:
            elr[r["step"]] = r["elr"]
    return out, elr, lr


def final_val(run: Path):
    last = runinfo.last_eval(run)
    if not last:
        return None
    return last.get("val_bpb_ema", last.get("val_bpb"))


def ema(pairs):
    s, out = None, {}
    for k, x in pairs:
        s = x if s is None else EMA * s + (1 - EMA) * x
        out[k] = s
    return out


def main(argv) -> int:
    if len(argv) != 2:
        print(__doc__.strip().splitlines()[2])
        return 2
    ref, mat = Path(argv[0]), Path(argv[1])
    a, ea, la = curve(ref)
    b, eb, lb = curve(mat)
    shared = sorted(set(a) & set(b))
    if len(shared) < 20:
        print(f"only {len(shared)} shared steps — did both runs finish?")
        return 1
    warm = shared[int(0.1 * len(shared))]
    post = [k for k in shared if k >= warm]
    A, B = ema([(k, a[k]) for k in post]), ema([(k, b[k]) for k in post])
    d = sum(abs(A[k] - B[k]) for k in post) / len(post)

    print(f"{ref.name} vs {mat.name}: {len(post)} post-warmup points, steps {post[0]}..{post[-1]}\n")
    print(f"  Δ_coll                = {d:.5f} bpb = {d*LN2:.5f} nats")
    print(f"  seed noise (Mote)     = 0.01263 bpb = 0.00875 nats   [ab2_muon_2h vs ab2_muon_seed7]")
    print(f"  paper's collapses     = 0.0018–0.0049 nats across 26 configurations")
    verdict = ("COLLAPSED — the norm-control difference acted through ELR and nothing else"
               if d * LN2 < 0.005 else
               "did NOT collapse — either the matching failed or the law does not hold here")
    print(f"  -> {verdict}\n")

    if ea and eb:
        gap = [abs(ea[k] - eb[k]) / max(ea[k], 1e-30) for k in post if k in ea and k in eb]
        if gap:
            print(f"  ELR tracking error    = {100*sum(gap)/len(gap):.3f} % mean, {100*max(gap):.3f} % worst")
    if la and lb:
        r = [lb[k] / la[k] for k in post if la.get(k)]
        if r:
            print(f"  applied lr ratio      = {min(r):.3f}..{max(r):.3f} × the reference's "
                  f"(the adaptation the matching had to make)")
    va, vb = final_val(ref), final_val(mat)
    if va and vb:
        print(f"\n  final val bpb         = {va:.5f} (ref) vs {vb:.5f} (matched), gap {vb-va:+.5f}")
        print(f"  seed noise on val_bpb = 0.00025 bpb — a gap under ~0.0008 is not an optimizer difference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
