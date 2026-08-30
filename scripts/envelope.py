"""Same-code envelope check (docs/results/2026-08-29-housekeeping-prereg.md, amendment 2026-08-29): the per-step
deltas between the old code and HEAD must sit inside the spread of ≥ 3 same-code HEAD runs, on every logged
field. A bitwise-identical trajectory is unattainable with these kernels (`tl.atomic_add` in the SSD / Mamba-3
backward), so this is the GPU half of the housekeeping bar.

    python scripts/envelope.py runs/gap/env a b c --old old [--max-ratio 1.5]

Each name is a run directory under the first argument holding a `log.jsonl` written with `--log-every 1`.
Time-derived fields are skipped. Per field: the max relative diff over same-code pairs and over old-vs-HEAD
pairs, and their ratio, for steps 1–10 and 1–100. VERDICT INSIDE (exit 0) iff every field's ratio is
≤ --max-ratio on both windows — 1.0 means the refactor's deltas are no larger than run-to-run noise; the L4
check of 2026-08-29 saw 0.7–1.4×. Step 1 is the forward on the init: it is deterministic across runs and
its old-vs-HEAD value is the forward check, printed on its own."""
import argparse
import itertools
import json
import sys

SKIP = {"elapsed_min", "elapsed_sec", "bytes_per_sec", "sec_per_step", "probe_sec_per_step", "eta_min", "time",
        "ts", "mem_gb", "peak_gb", "tok_per_sec", "tflops", "wall", "step", "lr", "target_ratio", "bpic"}


def load(path):
    d = {}
    for line in open(path):
        if not line.startswith("{"):
            continue
        r = json.loads(line)
        if "step" in r and "train_bpb" in r and r["step"] not in d:
            d[r["step"]] = r
    return d


def rel(x, y):
    return abs(x - y) / max(abs(x), abs(y), 1e-300)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("same", nargs="+", help="≥ 2 same-code (HEAD) run names")
    ap.add_argument("--old", required=True, help="the old-code run name")
    ap.add_argument("--max-ratio", type=float, default=1.5)
    args = ap.parse_args()
    R = {n: load(f"{args.root}/{n}/log.jsonl") for n in [*args.same, args.old]}
    steps = sorted(set.intersection(*[set(v) for v in R.values()]))
    if not steps:
        print("no common steps"); return 2
    first = R[args.same[0]][steps[0]]
    fields = sorted(k for k in first if k not in SKIP and isinstance(first[k], (int, float))
                    and all(k in R[n][s] for n in R for s in steps))
    pairs = {"same": list(itertools.combinations(args.same, 2)), "refactor": [(args.old, n) for n in args.same]}
    print(f"runs {list(R)}  common steps {len(steps)} (to {steps[-1]})  fields {fields}")
    print("step 1 train_bpb (forward on the init, deterministic):", {n: R[n][steps[0]]["train_bpb"] for n in R})
    ok = True
    for lo, hi, name in [(1, 10, "steps 1-10"), (1, 100, "steps 1-100")]:
        ss = [s for s in steps if lo <= s <= hi]
        if len(ss) < 2:
            continue
        print(f"\n== {name}: max rel diff over pairs × steps")
        print(f"{'field':14s} {'same':>9s} {'refactor':>9s} {'ratio':>6s}")
        for f in fields:
            m = {g: max(rel(R[x][s][f], R[y][s][f]) for x, y in ps for s in ss) for g, ps in pairs.items()}
            ratio = m["refactor"] / max(m["same"], 1e-300)
            flag = "" if ratio <= args.max_ratio else "  <-- outside"
            if ratio > args.max_ratio and m["refactor"] > 1e-12:  # both zero = identical, fine
                ok = False
            print(f"{f:14s} {m['same']:9.2e} {m['refactor']:9.2e} {ratio:6.1f}{flag}")
    print("\nVERDICT", "INSIDE" if ok else "OUTSIDE", f"(max-ratio {args.max_ratio})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
