"""Fixed-seed serving diff and the served-continuation boundary check (the gap protocol,
docs/results/2026-08-29-housekeeping-prereg.md; docs/results/2026-08-28-lr-prereg.md § Signed 2).

    PYTHONPATH=<checkout> python scripts/serve_diff.py CKPT OUT.json [--bytes 100] [--raw]
    python scripts/serve_diff.py --compare OLD.json NEW.json

Runs a fixed prompt set through the engine on the CPU, greedy (temperature 0), no tools, and records every
generated id with its boundary flag. Run it once under the old code (`58e8672`: the engine was `mote.serve.engine`
and served the raw weights) and once under HEAD (`mote.infer.engine`, serves the EMA since f18f93b); `--raw`
strips the EMA into a temporary copy of the checkpoint so both runs serve the same weights and the diff is engine
against engine. The boundary check is printed for every run: a ~100-byte continuation must carry boundaries at the
natural rate (3.2–3.8 B/chunk → 25–35), not one per byte — the absolute floor did that before `655ac24`."""
import argparse
import hashlib
import json
import os
import sys
import tempfile
import threading

import torch

try:
    from mote.infer.engine import Engine, GenParams
except ImportError:  # before housekeeping 2/6 (58e8672) the engine lived in mote.serve
    from mote.serve.engine import Engine, GenParams

PROMPTS = [
    "The router compares each byte with the one before it. Where they stop looking alike, it draws a boundary.",
    "def fibonacci(n):\n    ",
    "Q: What is the capital of France?\nA:",
    "Once upon a time, in a village by the sea, there lived",
]


def strip_ema(path: str) -> str:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    ex = dict(ck.get("extra") or {})
    for k in ("ema", "ema_steps", "ema_zero_init", "ema_decay"):
        ex.pop(k, None)
    ck["extra"] = ex
    out = os.path.join(tempfile.mkdtemp(prefix="serve_diff_"), "last.pt")
    torch.save(ck, out)
    return out


def run(ckpt: str, n_bytes: int, raw: bool) -> dict:
    path = strip_ema(ckpt) if raw else ckpt
    eng = Engine(path, device="cpu")
    fields = GenParams.__dataclass_fields__
    params = GenParams(**{k: v for k, v in dict(temperature=0.0, top_p=1.0, max_bytes=n_bytes, max_calls=0).items() if k in fields})
    out = {"ckpt": ckpt, "raw": raw, "weights": getattr(eng, "weights", "raw"), "bytes": n_bytes, "prompts": []}
    for prompt in PROMPTS:
        events = []
        torch.manual_seed(0)
        eng.generate([{"role": "user", "content": prompt}], params, events.append, threading.Event())
        ids = [int(e["byte"]) for e in events if e.get("type") == "byte"]
        bounds = sum(1 for e in events if e.get("type") == "byte" and e.get("boundary"))
        text = bytes(i for i in ids if i < 256).decode("utf-8", errors="replace")
        out["prompts"].append({"prompt": prompt, "n": len(ids), "boundaries": bounds,
                               "sha": hashlib.sha1(json.dumps(ids).encode()).hexdigest()[:12], "ids": ids, "text": text})
        print(f"{len(ids):4d} bytes, {bounds:3d} boundaries | {text[:70]!r}")
    return out


def boundary_check(res: dict) -> bool:
    ok = True
    for p in res["prompts"]:
        n, b = p["n"], p["boundaries"]
        good = n < 20 or (0.1 * n <= b <= 0.6 * n)  # natural rate 3.2–3.8 B/chunk; 1/byte is the old floor bug
        ok &= good
        print(f"boundary check: {b} boundaries in {n} bytes -> {'ok' if good else 'NOT the natural rate'}")
    return ok


def compare(a: dict, b: dict) -> bool:
    same = True
    for pa, pb in zip(a["prompts"], b["prompts"]):
        if pa["ids"] == pb["ids"]:
            print(f"identical ({pa['n']} bytes, {pa['boundaries']} boundaries): {pa['prompt'][:40]!r}")
        else:
            same = False
            i = next((i for i, (x, y) in enumerate(zip(pa["ids"], pb["ids"])) if x != y), min(len(pa["ids"]), len(pb["ids"])))
            print(f"DIFFER at byte {i} of {pa['n']}/{pb['n']} (boundaries {pa['boundaries']}/{pb['boundaries']}): {pa['prompt'][:40]!r}\n"
                  f"   old: {pa['text'][:80]!r}\n   new: {pb['text'][:80]!r}")
    print("VERDICT", "REPLIES IDENTICAL" if same else "REPLIES DIFFER", f"(weights {a['weights']} vs {b['weights']})")
    return same


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="?")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--bytes", type=int, default=100)
    ap.add_argument("--raw", action="store_true", help="serve the raw weights (strip the EMA into a temp copy)")
    ap.add_argument("--compare", nargs=2, metavar=("OLD", "NEW"))
    args = ap.parse_args()
    if args.compare:
        a, b = (json.load(open(p)) for p in args.compare)
        return 0 if compare(a, b) else 1
    if not (args.ckpt and args.out):
        ap.error("CKPT OUT.json, or --compare OLD NEW")
    res = run(args.ckpt, args.bytes, args.raw)
    res["boundary_check"] = boundary_check(res)
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"wrote {args.out} (weights: {res['weights']})")
    return 0 if res["boundary_check"] else 1


if __name__ == "__main__":
    sys.exit(main())
