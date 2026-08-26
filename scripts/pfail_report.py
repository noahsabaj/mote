"""Read the p_fail 3x3 and say what it decides (scripts/pfail_matrix.sh step 2).

The rate gates the sim regeneration, which gates the flagship's mid-training data, so this is the one
number the whole stage waits on. It is chosen by measurement because nothing in the 2026 literature gives
a failure rate for a synthetic world — the sim is Mote's own.
"""

import json
import pathlib
import statistics

RATES = (5, 15, 30)
OUT = pathlib.Path("runs/pfail")


def _read(train: int, probe: int):
    p = OUT / f"pf_{train}" / f"probe_at_{probe}.json"
    return json.loads(p.read_text()) if p.exists() else None


def main() -> int:
    grid = {(t, p): _read(t, p) for t in RATES for p in RATES}
    if any(v is None for v in grid.values()):
        print("incomplete matrix; run scripts/pfail_matrix.sh")
        return 1

    print("sim-QA EM — rows are the rate the arm TRAINED at, columns the rate it is PROBED at\n")
    print(f"{'train\\probe':>12s}" + "".join(f"{p:>9d}%" for p in RATES) + f"{'mean':>9s}")
    for t in RATES:
        row = [grid[(t, p)]["em"] for p in RATES]
        print(f"{t:11d}%" + "".join(f"{v:10.3f}" for v in row) + f"{statistics.mean(row):9.3f}")

    print(f"\n{'arm':>12s}{'recovery':>10s}{'repeat':>9s}{'unparseable':>13s}")
    rec = {}
    for t in RATES:
        p = OUT / f"pf_{t}" / "recovery.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        rec[t] = d
        print(f"{t:11d}%{d['recovery_rate']:10.3f}{d['repeat_rate']:9.3f}{d.get('unparseable_rate', 0):13.3f}")

    print("\nreading it:")
    # transfer: how much does an arm lose on the EASIEST worlds relative to the arm trained there
    for t in RATES:
        own = grid[(t, t)]["em"]
        at5 = grid[(t, 5)]["em"]
        base = grid[(5, 5)]["em"]
        print(f"  trained {t:2d}%: own-rate EM {own:.3f}, at 5 % {at5:.3f} "
              f"({at5 - base:+.3f} vs the arm trained at 5 %)")
    best_mean = max(RATES, key=lambda t: statistics.mean([grid[(t, p)]["em"] for p in RATES]))
    best_rec = max(rec, key=lambda t: rec[t]["recovery_rate"]) if rec else None
    print(f"\n  best mean EM across all three exams : p_fail={best_mean}%")
    if best_rec is not None:
        print(f"  best recovery rate                  : p_fail={best_rec}%")
    print("\nTake the highest rate that does not lose EM on the 5 % column relative to the 5 % arm — it "
          "teaches recovery for free. If every rate loses there, the rate is a real trade and 15 % is the "
          "safe pick. Record the choice in docs/results and set P_FAIL for scripts/mid_2x2.sh.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
