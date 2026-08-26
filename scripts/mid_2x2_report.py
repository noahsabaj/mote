"""Read the 2x2 off the five SFT checkpoints' proxy.json files (scripts/mid_2x2.sh step 5).

Both axes with their noise bands. A difference inside the combined standard error has decided nothing:
measured 2026-08-26, the gap between the best and worst of three checkpoints whose ordering is known from
12-hour runs is only 2.3 sem at n=120, so a branch comparison needs to clear that bar before it means
anything.
"""

import json
import pathlib

ARMS = ("b_decayed", "b_constant", "c_decayed", "c_constant", "floor")


def main() -> int:
    d, sem = {}, {}
    print(f"{'arm':14s} {'track':>9s} {'+/-':>8s} {'agree':>8s} {'ce':>8s}")
    for a in ARMS:
        p = pathlib.Path("runs/mid") / f"{a}_sft" / "proxy.json"
        if not p.exists():
            print(f"{a:14s} {'missing':>9s}")
            continue
        r = json.loads(p.read_text())
        d[a], sem[a] = r.get("recip_rank_uniform"), r.get("recip_rank_uniform_sem", 0.0)
        print(f"{a:14s} {d[a]:9.4f} {sem[a]:8.4f} {r.get('agree', 0):8.4f} {r.get('ce', 0):8.4f}")

    def cmp(lo: str, hi: str, label: str) -> None:
        if lo not in d or hi not in d:
            return
        delta = d[hi] - d[lo]
        noise = (sem[hi] ** 2 + sem[lo] ** 2) ** 0.5
        print(f"{label:34s} {delta:+.4f}  (noise +/-{noise:.4f})  "
              f"{'clear' if abs(delta) > noise else 'INSIDE THE NOISE'}")

    print()
    cmp("b_decayed", "c_decayed", "mixture (C - B), decayed")
    cmp("b_constant", "c_constant", "mixture (C - B), constant")
    cmp("b_constant", "b_decayed", "decay (decayed - constant), B")
    cmp("c_constant", "c_decayed", "decay (decayed - constant), C")
    cmp("floor", "c_decayed", "the whole stage (C decayed - floor)")
    print("\nIf both decay rows sit inside the noise, the 20 % decay is not earning its bytes and the "
          "branch should be --schedule constant. That is the question the old gate could not ask, "
          "because both of its arms were cooldowns.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
