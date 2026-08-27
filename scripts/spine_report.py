"""Read the spine gate and print the four numbers the call is made on (signed 2026-08-27).

There is deliberately no pass/fail line. The gate has no pre-registered threshold: the literature's
effect converts to about -0.010 bpb, seed noise is 0.00025, and crossing a resolution boundary is
unprecedented in this family, so a partial transfer is likely enough that a fixed bar would throw
away the interesting outcome. What this prints is the evidence, ranked, and the decision is joint.

    MATCHED TOKENS   decides. Every arm sees the same data at the same step and, because the arms
                     run --schedule trunk, the same learning rate there too. Under the wsd default
                     a slower arm sits later in its decay at step k and this comparison is invalid,
                     which is why the gate sets trunk.
    WALL-CLOCK       the cost line. Endpoint val_bpb at equal minutes, which is what you would
                     actually get if you shipped it today.
    THROUGHPUT TAX   the difference between those two, in bpb, measured rather than modelled: read
                     off each arm's own curve at its own step shortfall.
    STREAM COS       whether the streams ever became distinct. HC starts every stream as a copy
                     (cos ~ 0.9998); if it has not fallen by the end, nothing differentiated and a
                     null result says the topology never engaged, not that the idea is wrong.

    python scripts/spine_report.py runs/mote-96m/spine-{ctl,frac-n4,expand-n4}
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Any, Dict, List, Optional, Tuple


def _load(d: pathlib.Path) -> Tuple[List[Tuple[int, float]], List[Dict[str, Any]], Optional[float]]:
    """(step, val_bpb_ema) points, the logging records, and the run's wall-clock minutes."""
    p = d / "log.jsonl"
    if not p.exists():
        raise SystemExit(f"no {p} — has this arm run?")
    curve, logs, minutes = [], [], None
    for line in p.read_text().splitlines():
        r = json.loads(line)
        minutes = r.get("elapsed_min", minutes)
        if "eval" in r:
            e = r["eval"]
            key = "val_bpb_ema" if "val_bpb_ema" in e else "val_bpb"
            if key in e:
                curve.append((r["step"], e[key]))
        elif "train_bpb" in r:
            logs.append(r)
    if not curve:
        raise SystemExit(f"{d.name} has no evals yet")
    return curve, logs, minutes


def _at(curve: List[Tuple[int, float]], step: int) -> Optional[float]:
    """The arm's val_bpb_ema at the last eval on or before `step`."""
    prior = [v for s, v in curve if s <= step]
    return prior[-1] if prior else None


def main(argv: List[str]) -> int:
    if len(argv) < 2:
        raise SystemExit(__doc__)
    dirs = [pathlib.Path(a) for a in argv]
    arms = {d.name: _load(d) for d in dirs}
    ctl = dirs[0].name  # the control is the first argument by convention

    common = min(c[-1][0] for c, _, _ in arms.values())
    print(f"control: {ctl}\nmatched-token step: {common} (the last eval every arm reached)\n")

    w = max(len(n) for n in arms) + 2
    print(f"{'arm':{w}s}{'steps':>9s}{'min':>7s}{'B/s':>10s}{'matched':>10s}{'vs ctl':>9s}"
          f"{'endpoint':>10s}{'vs ctl':>9s}")
    base_m = _at(arms[ctl][0], common)
    base_e = arms[ctl][0][-1][1]
    rows = {}
    for name, (curve, logs, minutes) in arms.items():
        m, e = _at(curve, common), curve[-1][1]
        bps = sum(r["bytes_per_sec"] for r in logs[1:]) / max(len(logs) - 1, 1) if len(logs) > 1 else 0.0
        rows[name] = (curve[-1][0], m, e, bps)
        dm = "" if name == ctl else f"{m - base_m:+9.4f}"
        de = "" if name == ctl else f"{e - base_e:+9.4f}"
        print(f"{name:{w}s}{curve[-1][0]:9d}{(minutes or 0):7.0f}{bps:10.0f}{m:10.4f}{dm:>9s}{e:10.4f}{de:>9s}")

    ctl_steps, _, _, ctl_bps = rows[ctl]
    print(f"\n{'arm':{w}s}{'throughput':>12s}{'step short':>12s}{'tax (bpb)':>11s}   the tax is what "
          f"wall-clock charges the arm before the spine does anything")
    for name, (steps, _, _, bps) in rows.items():
        if name == ctl:
            continue
        # the tax, measured on the CONTROL's own curve: what the control would have scored had it
        # been stopped at this arm's step count. No fitted constant, no scaling law.
        short = steps / max(ctl_steps, 1)
        at_short = _at(arms[ctl][0], steps)
        tax = (at_short - base_e) if at_short is not None else float("nan")
        print(f"{name:{w}s}{bps / max(ctl_bps, 1e-9):11.3f}x{short:11.1%}{tax:+11.4f}")

    print(f"\n{'arm':{w}s}{'stream_cos':>12s}{'spread':>9s}{'h_res_drift':>13s}{'alpha_res':>11s}"
          f"{'read_out_max':>14s}")
    for name, (_, logs, _) in arms.items():
        last = next((r for r in reversed(logs) if "stream_cos" in r or "h_res_drift" in r), None)
        if last is None:
            print(f"{name:{w}s}{'—':>12s}   (no spine diagnostics: this is the control, or the run "
                  f"predates the instrumentation)")
            continue
        print(f"{name:{w}s}{last.get('stream_cos', float('nan')):12.5f}"
              f"{last.get('stream_spread', float('nan')):9.3f}"
              f"{last.get('h_res_drift', float('nan')):13.5f}"
              f"{last.get('alpha_res', float('nan')):11.5f}"
              f"{last.get('read_out_max', float('nan')):14.4f}")

    print("""
reading it:
  matched vs ctl   the architecture question, artifact-free. Seed noise is 0.00025 bpb.
  endpoint vs ctl  the same arm judged on what it cost. matched minus endpoint is the tax below.
  stream_cos       started near 0.9998. Still there means the streams never differentiated, and a
                   null is evidence about the topology's depth, not about hyper-connections.
  h_res_drift      ||H_res - I||_F of the static mixer, starting near 0. Still near 0 means no
                   cross-stream mixing ever happened and the spine was a plain residual with
                   overhead — which is the identity degeneration A3 exists to test.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
