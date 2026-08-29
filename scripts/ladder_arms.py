"""The hybrid ladder's arms (signed 2026-08-29, docs/results/2026-08-29-hybrid-ladder-prereg.md).

Every arm is the control's preset with a main-network pattern and the FFN width trimmed so the TOTAL
parameter count matches the control's (matched parameters and bytes; wall-clock reported). Relation and
Mamba-3 blocks at the main width have different sizes (768: 2.36M vs 3.71M at expand 2), so `d_ff` is
solved per arm by building the model on the CPU and counting — the count, not an estimate, is what the
prereg matches.

    python scripts/ladder_arms.py --preset mote-11m            # the table: arm, pattern, d_ff, params
    python scripts/ladder_arms.py --preset mote-11m --argv 3to1_evidence   # the trainer flags of one arm

Positions scale by depth fraction: an arm is defined by the fractions of depth at which its Relation
layers sit, rounded to layers at 6 / 8 / 12 main layers (11M / 32M / 96M). Layer 0 is always Mamba-3 in a
hybrid and every hybrid keeps >= 2 Relation layers (mote/model/blocks.py main_pattern).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from mote.config import MoteConfig, resolve_preset
from mote.model.hnet import HNetForCausalLM


@dataclass
class Arm:
    name: str
    ratio_r: Optional[float]  # Relation share of the main layers; None = all Relation (the control)
    priority: List[float]  # depth fractions, in the order the Relation layers are placed ("uniform" = evenly spaced)
    expand: int = 2
    flags: str = "--relation-out-gate --mamba-out-norm"  # the hybrid defaults; "" for the controls
    note: str = ""


def _layer(frac: float, n: int) -> int:
    return min(n - 1, max(1, int(frac * (n - 1) + 0.5)))  # never layer 0 (Mamba-3 there in every hybrid)


def positions(arm: Arm, n: int) -> List[int]:
    """The Relation layers of `arm` in an n-layer main: k = max(2, round(ratio * n)) layers, placed by the
    arm's priority list (evenly spaced for 'uniform', else the fractions in order, skipping collisions)."""
    k = max(2, int(arm.ratio_r * n + 0.5))
    if arm.priority == UNIFORM:
        return [int((i + 1) * n / k + 0.5) - 1 for i in range(k)]
    out: List[int] = []
    for f in arm.priority:
        l = _layer(f, n)
        if l not in out:
            out.append(l)
        if len(out) == k:
            break
    assert len(out) == k, (arm.name, n, out)
    return sorted(out)


def pattern_for(arm: Arm, n_layers: int) -> Optional[str]:
    if arm.ratio_r is None:
        return None
    pos = set(positions(arm, n_layers))
    return "".join("R" if i in pos else "M" for i in range(n_layers))


UNIFORM = ["uniform"]
EVIDENCE = [0.5, 0.8, 0.1, 0.65, 0.35]  # mid, 75-80 %, early — the order three lines of evidence rank them
MID = [0.5, 0.55, 0.45, 0.6, 0.4, 0.65, 0.35]
LATE = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]

# At 12 main layers (96M): 1:1 = 6R, 3:1 = 3R, 5:1 = 2R. At 6 (11M) and 8 (32M) layers the 3:1 and 5:1
# arms both round to 2R, so the ratio axis there is {1:1, 2R} and the 5:1 arm is the 32M/96M rungs' only.
ARMS: Dict[str, Arm] = {
    "R":             Arm("R", None, [], flags="", note="control: today's all-Relation main"),
    "1to1":          Arm("1to1", 0.5, UNIFORM, note="half Relation, evenly spaced [1,3,5,7,9,11] at 12"),
    "3to1_uniform":  Arm("3to1_uniform", 0.25, UNIFORM, note="evenly spaced [3,7,11] at 12 — the Kimi/Relation-paper interleave"),
    "3to1_evidence": Arm("3to1_evidence", 0.25, EVIDENCE, note="early+mid+75 % [1,6,9] at 12 — the placement three lines agree on"),
    "3to1_mid":      Arm("3to1_mid", 0.25, MID, note="mid band [5,6,7] at 12"),
    "3to1_late":     Arm("3to1_late", 0.25, LATE, note="late [9,10,11] at 12 — the falsifier"),
    "5to1":          Arm("5to1", 1 / 6, UNIFORM, note="[5,11] at 12; identical to 3to1_uniform below 12 layers"),
    "3to1_expand1":  Arm("3to1_expand1", 0.25, EVIDENCE, expand=1, note="evidence placement at expand 1 (lighter Mamba-3, wider FFN)"),
}


def solve_d_ff(cfg: MoteConfig, target: int, lo: int = 64, hi: int = 8192) -> int:
    """The main FFN width at which the model has `target` parameters (to within one unit of 3*d_model)."""
    per_unit = 3 * cfg.main.d_model * cfg.main.n_layers  # SwiGLU adds 3*D params per unit of d_ff per layer

    def count(d_ff: int) -> int:
        c = MoteConfig.from_dict(cfg.to_dict())
        c.main.d_ff = d_ff
        with torch.device("meta"):
            return HNetForCausalLM(c).num_params()

    base = count(cfg.main.d_ff)
    guess = cfg.main.d_ff + round((target - base) / per_unit)
    guess = max(lo, min(hi, guess))
    # one correction step (the count is affine in d_ff)
    guess += round((target - count(guess)) / per_unit)
    return max(lo, min(hi, guess))


def arm_config(preset: str, arm: Arm) -> MoteConfig:
    ctl = resolve_preset(preset)
    cfg = resolve_preset(preset)
    cfg.main.pattern = pattern_for(arm, cfg.main.n_layers)
    cfg.main.mamba_expand = arm.expand
    if cfg.main.pattern:
        cfg.main.out_gate = "--relation-out-gate" in arm.flags
        cfg.main.mamba_out_norm = "--mamba-out-norm" in arm.flags
        with torch.device("meta"):
            target = HNetForCausalLM(ctl).num_params()
        cfg.main.d_ff = solve_d_ff(cfg, target)
    return cfg


def argv_for(preset: str, arm: Arm) -> str:
    cfg = arm_config(preset, arm)
    if not cfg.main.pattern:
        return f"--preset {preset}"
    return (f"--preset {preset} --main-pattern {cfg.main.pattern} --main-mamba-expand {arm.expand} "
            f"--main-d-ff {cfg.main.d_ff} {arm.flags}").strip()


def table(preset: str, names: Optional[List[str]] = None) -> List[dict]:
    rows = []
    for name in names or ARMS:
        arm = ARMS[name]
        cfg = arm_config(preset, arm)
        with torch.device("meta"):
            n = HNetForCausalLM(cfg).num_params()
        rows.append(dict(arm=name, pattern=cfg.main.pattern or "R" * cfg.main.n_layers, expand=arm.expand, d_ff=cfg.main.d_ff, params=n, note=arm.note))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="mote-11m")
    ap.add_argument("--argv", default=None, help="print the trainer flags of this arm")
    ap.add_argument("--arms", default=None, help="comma-separated subset")
    args = ap.parse_args()
    if args.argv:
        print(argv_for(args.preset, ARMS[args.argv]))
        return
    names = args.arms.split(",") if args.arms else None
    for r in table(args.preset, names):
        print(f"{r['arm']:14} {r['pattern']:14} e{r['expand']} d_ff {r['d_ff']:5d} params {r['params']:>12,}  {r['note']}")


if __name__ == "__main__":
    main()
