"""Memory and throughput of the frozen 16384 recipe across chunk rates (signed 2026-08-28).

    python scripts/gate_sweep.py [--rates 2.1 2.5 3.3 4.0] [--no-park]

Why a sweep and not one number: the router's rate is an observable, not a setting (mote/runinfo.py), and
it moves over a run — an untrained flagship router segments real text at 2.1-2.4 bytes/chunk (measured
2026-08-28, two seeds), trained 35M routers sit at 3.2-3.45, ATDC's target is 6.5. The main network's
memory scales with the chunk count, so the trunk's peak is at step 1, not at steady state. One profile per
rate gives the memory-vs-rate curve; any future ceiling (dc.bound_ceiling, the init guardrail) is a lookup.

Each rate is a fresh `mote.train.profile_step` process (its own CUDA context and peak counter). The studio's
engine is parked on the CPU for the sweep through POST /api/engine/device so the card is whole, and put
back afterwards even if a profile fails. Pass bar: peak <= 6.2 GB — what the daemon gets beside the locked
desktop (docs/results/2026-08-24-evening-gates.md), not shape.md's 7.2.

Writes docs/results/<date>-gate-sweep.json and .md.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CEILING_GB = 6.2  # decimal GB, the same unit profile_step reports (max_memory_allocated / 1e9)

RECIPE = ["--preset", "flagship", "--data", "data/flagship_mix", "--seq-len", "16384", "--batch-size", "1",
          "--grad-accum", "4", "--bucket", "64", "--ckpt-main"]


def _api(method: str, path: str, body=None):
    from mote.client import api
    return api(method, path, body)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", type=float, nargs="+", default=[2.1, 2.5, 3.3, 4.0])
    ap.add_argument("--no-park", action="store_true", help="do not move the studio's engine off the card first")
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--timed", type=int, default=6)
    ap.add_argument("--tag", default=time.strftime("%Y-%m-%d"))
    args = ap.parse_args(argv)

    out_dir = ROOT / "docs" / "results" / f"{args.tag}-gate-sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    parked = False
    if not args.no_park:
        try:
            info = _api("POST", "/api/engine/device", {"device": "cpu"})
            parked = bool(info.get("parked"))
            print(f"engine parked on the cpu (vram_used_mb now {info.get('device', {}).get('vram_used_mb')})", flush=True)
        except Exception as ex:
            print(f"could not park the engine ({ex!r}); profiling beside it", flush=True)
    rows = []
    try:
        for rate in args.rates:
            out = out_dir / f"bpic_{rate:.1f}.json"
            cmd = [sys.executable, "-m", "mote.train.profile_step", *RECIPE, "--chunk-bytes", str(rate),
                   "--warmup", str(args.warmup), "--timed", str(args.timed), "--out", str(out)]
            print(f"\n=== bpic {rate}: {' '.join(cmd[2:])}", flush=True)
            t0 = time.time()
            p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            took = time.time() - t0
            if p.returncode != 0 or not out.exists():
                tail = "\n".join((p.stdout + p.stderr).strip().splitlines()[-6:])
                print(f"bpic {rate}: FAILED after {took:.0f} s (exit {p.returncode})\n{tail}", flush=True)
                rows.append({"bpic": rate, "failed": True, "exit": p.returncode, "tail": tail, "seconds": round(took)})
                continue
            r = json.loads(out.read_text())
            r["seconds"] = round(took)
            rows.append(r)
            print(f"bpic {rate}: peak {r['peak_mem_GB']} GB, {r['bytes_per_sec']} B/s, {r['sec_per_step']} s/step "
                  f"({took:.0f} s)", flush=True)
    finally:
        if parked:
            try:
                _api("POST", "/api/engine/device", {"device": "cuda"})
                print("engine back on the gpu", flush=True)
            except Exception as ex:
                print(f"could not restore the engine ({ex!r}): `mote engine restore`", flush=True)

    res = {"recipe": RECIPE, "ceiling_GB": CEILING_GB, "rows": rows,
           "init_rate_measured": "2.07-2.42 bytes/chunk on real val text at flagship init, two seeds (2026-08-28)"}
    (out_dir.parent / f"{args.tag}-gate-sweep.json").write_text(json.dumps(res, indent=1))
    md = [f"# 16384 gate sweep — {args.tag}", "",
          f"Frozen recipe (`{' '.join(RECIPE)}`), one `profile_step` process per chunk rate, engine parked on the CPU.",
          f"Pass bar: peak ≤ {CEILING_GB} GB (the daemon's share beside the locked desktop, "
          "docs/results/2026-08-24-evening-gates.md). The untrained router segments real text at 2.1–2.4 bytes/chunk, "
          "so the 2.1 row is step 1 of the trunk; 3.3 is where trained routers sit; 6.5 is only ATDC's target.", "",
          "| bytes/chunk | chunks / 16384 | peak GB | fits ≤ 6.2 | B/s | s/step | MFU | FLOPs/byte (M) |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("failed"):
            md.append(f"| {r['bpic']} | {int(16384 / r['bpic'])} | FAILED (exit {r['exit']}) | no | — | — | — | — |")
        else:
            md.append(f"| {r['chunk_bytes_forced']} | {int(16384 / r['chunk_bytes_forced'])} | {r['peak_mem_GB']} | "
                      f"{'yes' if r['peak_mem_GB'] <= CEILING_GB else 'NO'} | {r['bytes_per_sec']} | {r['sec_per_step']} | "
                      f"{r['mfu']} | {r['flops_per_byte_M']} |")
    (out_dir.parent / f"{args.tag}-gate-sweep.md").write_text("\n".join(md) + "\n")
    print("\n" + "\n".join(md[6:]), flush=True)
    print("written:", out_dir.parent / f"{args.tag}-gate-sweep.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
