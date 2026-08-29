"""Compare two runs of the same argv step for step: every logged number and every tensor in last.pt.

    python scripts/bitwise_diff.py runs/bitwise/old runs/bitwise/new

The housekeeping bar (docs/results/2026-08-29-housekeeping-prereg.md): the refactor is inert iff the per-step
losses and the saved state dict are bitwise identical. Time-derived fields are skipped; config keys that only
exist on one side are listed but do not fail the check (the refactor renamed some deliberately)."""
import json
import sys

import torch

SKIP = {"elapsed_min", "elapsed_sec", "bytes_per_sec", "sec_per_step", "probe_sec_per_step", "eta_min", "time", "ts",
        "mem_gb", "peak_gb", "tok_per_sec", "tflops", "wall"}  # time- and memory-derived, not numerics


def records(d):
    out = {}
    for line in open(f"{d}/log.jsonl"):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if "step" in r:
            out.setdefault(r["step"], {}).update(r)
    return out


def walk(x, y, path, bad):
    if isinstance(x, dict) and isinstance(y, dict):
        for k in sorted(set(x) | set(y)):
            if k not in x or k not in y:
                bad.append(f"{path}.{k}: only in {'new' if k not in x else 'old'}")
            else:
                walk(x[k], y[k], f"{path}.{k}", bad)
    elif torch.is_tensor(x) and torch.is_tensor(y):
        if x.shape != y.shape or x.dtype != y.dtype or not torch.equal(x, y):
            bad.append(f"{path}: tensor differs ({tuple(x.shape)} {x.dtype})")
    elif isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
        if len(x) != len(y):
            bad.append(f"{path}: len {len(x)} vs {len(y)}")
        else:
            for i, (u, v) in enumerate(zip(x, y)):
                walk(u, v, f"{path}[{i}]", bad)
    elif x != y:
        bad.append(f"{path}: {str(x)[:40]} vs {str(y)[:40]}")


def main(old, new):
    ra, rb = records(old), records(new)
    steps = sorted(set(ra) & set(rb))
    log_bad = []
    for s in steps:
        for k in sorted(set(ra[s]) & set(rb[s])):
            if k in SKIP:
                continue
            if ra[s][k] != rb[s][k]:
                log_bad.append((s, k, ra[s][k], rb[s][k]))
    print(f"log: {len(steps)} matched steps (old {len(ra)}, new {len(rb)}); differing values: {len(log_bad)}")
    for s, k, a, b in log_bad[:10]:
        print(f"  step {s} {k}: {a} vs {b}")
    ca = torch.load(f"{old}/last.pt", map_location="cpu", weights_only=False)
    cb = torch.load(f"{new}/last.pt", map_location="cpu", weights_only=False)
    bad = []
    for key in ("model", "optimizer", "step"):
        walk(ca.get(key), cb.get(key), key, bad)
    walk(ca.get("extra", {}).get("generator_state"), cb.get("extra", {}).get("generator_state"), "extra.generator_state", bad)
    cfg_notes = []
    walk(ca.get("config"), cb.get("config"), "config", cfg_notes)
    print(f"state: {len(bad)} differing entries; config notes: {len(cfg_notes)}")
    for p in bad[:20]:
        print("  ", p)
    for p in cfg_notes[:10]:
        print("   (config)", p)
    ok = not log_bad and not bad
    print("VERDICT:", "BITWISE IDENTICAL" if ok else "DIFFERS")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
