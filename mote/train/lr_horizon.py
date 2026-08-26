"""Optimal learning rate vs token horizon (Kakao 2608.20061 §2.2.1, adapted; signed 2026-08-24).

The freeze's lr came from 30-min arms at 0.2 tokens/param; the trunk runs at ~75. Kakao's proxies show the
optimum falling with horizon (≈2.3× over 20× tokens). This fits that slope from our own constant-LR runs:
for several learning rates, read the eval records of `runs/<name>/log.jsonl` (val_bpb_ema when the runs
carry `--eval-ema`, else val_bpb) at a set of shared token budgets, fit val bpb = a·x² + b·x + c in
x = ln lr per budget, read the vertex η*(D) = exp(−b / 2a), regress ln η* on ln D past the warm-up, and
extrapolate to a target horizon.

    python -m mote.train.lr_horizon runs/lrh_4e-4 runs/lrh_8e-4 runs/lrh_16e-4 --target-tokens 7e9 --n-params 35e6
    python -m mote.train.lr_horizon ... --json out.json

Reports per budget: the three points, the vertex, whether it lies inside the sweep; then β (slope), R², the
predicted η* at the target and the parabola's bpb gain at the last budget for moving from the lr in use to
η*. Every caveat Kakao would state applies: the slope is measured at our batch and scale only.

`--coord elr` fits in ln η/‖W‖_F instead of ln η (2608.24814; Yang et al. 2026 report the same for optimal-LR
extrapolation). At a fixed weight decay with the norms at equilibrium this is a relabeling — measured on the
three 12-h arms, ‖W‖ ∝ lr^0.478, so ln ELR = ½ ln lr + const and the vertex maps exactly. What it buys is
comparability the moment anything about norm control differs: a weight-decay change, Muon-SW's η²-scaled
decay, or a different width. It also states the compression the lr coordinate hides — 4e-4 → 16e-4 is 4× in
lr and 2.06× in ELR. Runs logged before ELR logging existed have no per-eval norm; `--coord elr` then falls
back to the endpoint ‖W‖ from last.pt for every budget and says so, which is only sound near equilibrium.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple


class EvalPoint(NamedTuple):
    """One eval: bytes seen, val bpb, and the ELR in force there (None for runs predating ELR logging).
    Indexes as (tokens, bpb) for everything written before the ELR coordinate existed."""

    tokens: float
    bpb: float
    elr: Optional[float]


def read_run(run: Path) -> Tuple[float, List[Tuple[float, float]]]:
    """(lr, [(tokens_seen, val_bpb), ...]) from a run directory. Each point also carries the ELR in force
    at that eval when the run logged one (`elr`, carried forward from the preceding train line)."""
    rj = json.loads((run / "run.json").read_text(encoding="utf-8"))
    lr = float(rj["lr"])
    tokens_per_step = int(rj["batch_size"]) * int(rj["seq_len"]) * int(rj["grad_accum"])
    pts = []
    recs = []
    for line in (run / "log.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    # a fresh start into an existing directory appends after the old run's lines (a resume continues the step
    # count; a fresh start logs its throughput probe at step 0): keep the last fresh start's segment only
    starts = [i for i, r in enumerate(recs) if r.get("step") == 0 and "probe_sec_per_step" in r]
    if starts:
        recs = recs[starts[-1]:]
    elr = None
    for rec in recs:
        if "elr" in rec:
            elr = float(rec["elr"])
        ev = rec.get("eval")
        if not ev:
            continue
        bpb = ev.get("val_bpb_ema", ev.get("val_bpb"))
        if bpb is None:
            continue
        pts.append(EvalPoint(rec["step"] * tokens_per_step, float(bpb), elr))
    pts.sort(key=lambda q: q.tokens)
    return lr, pts


def endpoint_elr(run: Path, lr: float) -> Optional[float]:
    """η/‖W‖_F from a run's final checkpoint — the fallback for arms that predate ELR logging."""
    import torch

    from .elr import muon_named_matrices  # noqa: F401  (kept for symmetry; names come from the state dict)

    ck = Path(run) / "last.pt"
    if not ck.exists():
        return None
    sd = torch.load(ck, map_location="cpu", weights_only=False)
    for k in ("model", "state_dict", "ema", "weights"):
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
            sd = sd[k]
            break
    skip = ("embeddings.weight", "lm_head.weight", "in_proj.weight")  # AdamW's, not Muon's
    tot = 0.0
    for k, v in sd.items():
        if not hasattr(v, "ndim") or v.ndim != 2 or v.numel() <= 4096 or any(k.endswith(e) for e in skip):
            continue
        tot += float(v.float().pow(2).sum())
    return lr / math.sqrt(tot) if tot else None


def _near(pts, tokens: float, tol: float = 0.15):
    """The eval nearest to `tokens` (within tol relative), else None."""
    best = min(pts, key=lambda p: abs(p[0] - tokens)) if pts else None
    if best is None or abs(best[0] - tokens) > tol * max(tokens, 1.0):
        return None
    return best


def _at(pts, tokens: float, tol: float = 0.15) -> Optional[float]:
    """val bpb of the eval nearest to `tokens` (within tol relative), else None."""
    best = _near(pts, tokens, tol)
    return None if best is None else best[1]


def parabola_vertex(xs: List[float], ys: List[float]) -> Tuple[float, float, float, float]:
    """Least-squares a·x² + b·x + c; returns (a, b, c, x*)."""
    n = len(xs)
    sx = [sum(x ** k for x in xs) for k in range(5)]
    sxy = [sum((x ** k) * y for x, y in zip(xs, ys)) for k in range(3)]
    # normal equations for [a, b, c]
    M = [[sx[4], sx[3], sx[2]], [sx[3], sx[2], sx[1]], [sx[2], sx[1], n]]
    v = [sxy[2], sxy[1], sxy[0]]
    a, b, c = _solve3(M, v)
    xstar = -b / (2 * a) if a > 0 else float("nan")
    return a, b, c, xstar


def _solve3(M, v):
    import copy

    A = copy.deepcopy(M)
    y = list(v)
    for i in range(3):
        piv = max(range(i, 3), key=lambda r: abs(A[r][i]))
        A[i], A[piv] = A[piv], A[i]
        y[i], y[piv] = y[piv], y[i]
        for r in range(i + 1, 3):
            f = A[r][i] / A[i][i]
            for c in range(i, 3):
                A[r][c] -= f * A[i][c]
            y[r] -= f * y[i]
    x = [0.0, 0.0, 0.0]
    for i in (2, 1, 0):
        x[i] = (y[i] - sum(A[i][c] * x[c] for c in range(i + 1, 3))) / A[i][i]
    return x


def linreg(xs: List[float], ys: List[float]) -> Tuple[float, float, float]:
    """(slope, intercept, R²) of ys on xs."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx if sxx > 0 else float("nan")
    inter = my - slope * mx
    ss_res = sum((y - (inter + slope * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return slope, inter, r2


def fit(runs: List[Tuple[float, List[Tuple[float, float]]]], budgets: Optional[List[float]] = None,
        min_frac: float = 0.15, coord: str = "lr", fallback_elr: Optional[List[Optional[float]]] = None) -> Dict:
    """`runs` = [(lr, points)]; budgets default to the shortest run's eval budgets past `min_frac` of it.

    `coord` is the axis the parabola is fitted in: "lr" (as signed 2026-08-24) or "elr" (η/‖W‖_F).
    `fallback_elr` supplies one endpoint ELR per run for arms that logged none.
    """
    lrs = [lr for lr, _ in runs]
    if budgets is None:
        shortest = min(runs, key=lambda r: r[1][-1].tokens if r[1] else 0)[1]
        end = shortest[-1].tokens
        budgets = sorted({q.tokens for q in shortest if q.tokens >= min_frac * end})
    per_budget = []
    approximated = False
    for D in budgets:
        near = [_near(pts, D) for _, pts in runs]
        if any(q is None for q in near):
            continue
        ys = [q[1] for q in near]
        if coord == "elr":
            rates = []
            for i, q in enumerate(near):
                e = q.elr if isinstance(q, EvalPoint) and q.elr else None
                if e is None:
                    e = (fallback_elr or [None] * len(runs))[i]
                    approximated = True
                rates.append(e)
            if any(not e for e in rates):
                continue
        else:
            rates = lrs
        xs = [math.log(r) for r in rates]
        a, b, c, xstar = parabola_vertex(xs, ys)
        inside = (min(xs) <= xstar <= max(xs)) if xstar == xstar else False
        per_budget.append({"tokens": D, "bpb": dict(zip([f"{lr:g}" for lr in lrs], ys)), "a": a,
                           "rates": rates, "lr_star": math.exp(xstar) if xstar == xstar else None,
                           "inside_sweep": inside})
    usable = [r for r in per_budget if r["lr_star"] is not None and r["a"] > 0]
    out = {"per_budget": per_budget, "n_usable": len(usable), "coord": coord, "elr_approximated": approximated}
    if len(usable) >= 2:
        slope, inter, r2 = linreg([math.log(r["tokens"]) for r in usable], [math.log(r["lr_star"]) for r in usable])
        out.update({"beta": slope, "intercept": inter, "r2": r2})
    return out


def predict(fit_out: Dict, tokens: float) -> Optional[float]:
    if "beta" not in fit_out:
        return None
    return math.exp(fit_out["intercept"] + fit_out["beta"] * math.log(tokens))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", help="run directories of constant-LR arms at different learning rates")
    ap.add_argument("--target-tokens", type=float, required=True, help="horizon to extrapolate to (tokens through the network, e.g. chunks)")
    ap.add_argument("--n-params", type=float, default=None, help="for the tokens/param readout only")
    ap.add_argument("--lr-in-use", type=float, default=None, help="report the parabola's bpb gain at the last budget vs this lr")
    ap.add_argument("--coord", default="lr", choices=["lr", "elr"], help="fit in ln lr (as signed) or in ln \u03b7/\u2016W\u2016_F. At a fixed weight decay with norms at equilibrium the two are a relabeling; they part company the moment norm control differs (2608.24814)")
    ap.add_argument("--json", default=None, help="write the fit here")
    args = ap.parse_args(argv)
    runs = [read_run(Path(r)) for r in args.runs]
    fb = [endpoint_elr(Path(r), lr) for r, (lr, _) in zip(args.runs, runs)] if args.coord == "elr" else None
    res = fit(runs, coord=args.coord, fallback_elr=fb)
    if res.get("elr_approximated"):
        print("note: some arms predate ELR logging — their endpoint \u2016W\u2016 from last.pt stands in at every "
              "budget, which is only sound near the norm equilibrium\n")
    for r in res["per_budget"]:
        tp = f" ({r['tokens']/args.n_params:.1f} tok/param)" if args.n_params else ""
        star = f"{r['lr_star']:.3g}" if r["lr_star"] else "n/a"
        if args.coord == "elr":
            star += "  [ELR; " + " ".join(f"{e:.3g}" for e in r["rates"]) + "]"
        print(f"D={r['tokens']:.3g}{tp}: " + "  ".join(f"lr {k}: {v:.4f}" for k, v in r["bpb"].items()) + f"  -> lr* {star}{'' if r['inside_sweep'] else ' (outside sweep!)'}")
    if "beta" in res:
        eta = predict(res, args.target_tokens)
        tp = f" ({args.target_tokens/args.n_params:.0f} tok/param)" if args.n_params else ""
        print(f"ln lr* = {res['intercept']:.3f} + {res['beta']:.3f} ln D   R²={res['r2']:.3f}   -> lr* at D={args.target_tokens:.3g}{tp}: {eta:.3g}")
        res["lr_star_target"] = eta
        if args.lr_in_use is not None:
            last = res["per_budget"][-1]
            xs = [math.log(float(k)) for k in last["bpb"]]
            ys = list(last["bpb"].values())
            a, b, c, _ = parabola_vertex(xs, ys)
            f = lambda x: a * x * x + b * x + c  # noqa: E731
            gain = f(math.log(args.lr_in_use)) - f(math.log(last["lr_star"]))
            res["gain_last_budget_bpb"] = gain
            print(f"at the last budget the parabola puts lr {args.lr_in_use:g} {gain:+.4f} bpb above lr* {last['lr_star']:.3g}")
    else:
        print(f"not enough usable budgets ({res['n_usable']}) for a slope")
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


if __name__ == "__main__":
    main()
