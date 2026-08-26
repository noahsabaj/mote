"""Effective learning rate: the coordinate every norm-control gate is reported on (signed 2026-08-26).

2608.24814 measures that LR and parameter norm govern loss dynamics through their ratio,
η^eff = η / ‖W‖_F, and not independently: runs whose LR and norm trajectories differ substantially
collapse onto the same loss curve when their ELR schedules match, to a few ×10⁻³ — four to nine times
tighter than re-running the identical configuration under a different seed.

Why Mote needs it rather than merely finds it interesting. Three things were measured here on 2026-08-26,
from checkpoints already on disk:

  * the trunk sits at the weight-decay norm equilibrium. Across three 12-h arms identical but for lr,
    ‖W‖_F ∝ lr^0.478 (√lr predicts 0.5), and all 61 Muon matrices cluster 1.23–1.46 around the predicted
    1.414. So **lr and weight decay are one axis, not two**: at equilibrium ELR ∝ √(lr·λ);
  * therefore the LR sweep did not sweep what it looked like. 4e-4 → 16e-4 spans **2.06× in ELR**, not 4×;
  * entrywise RMS is ≈ 0.66·√(lr/λ), flat to 1.4× across a 6× range of matrix size, which makes the
    relative step lr·‖U‖/‖W‖ ≈ 0.30·√(lr·λ) shape-independent. That is what transfers across width —
    ELR itself does not, and 2608.24814 §8 poses cross-scale transfer as open rather than answered.

The gate this exists to stop repeating: Muon vs Muon-SW (1.17734 vs 1.18006) decided the freeze's
optimizer, and Muon-SW ran at 0.914× Muon's ELR at the same nominal lr — because it *is* Muon with a
different decay schedule (muon.py, one line). The local slope d(val_bpb)/d(ln ELR) flips sign with
horizon (−0.073 bpb/nat at 30 min, +0.026 at 12 h), so that ELR gap explains between 0.000 and 0.0066 bpb
of a 0.00272 bpb effect. Precise (seed noise is 0.00025 bpb) and unattributable at the same time.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

# Norms are sampled on the trainer's logging cadence (`--log-every`), which is fine at any setting it takes:
# the norm's own time constant is 1/(lr·λ) ≈ 12,500 steps, far slower than anything that is logged.
SAVE_EVERY = 20  # samples between rewrites of the trace file, plus one when the run ends


def muon_named_matrices(model, opt) -> List[Tuple[str, torch.nn.Parameter]]:
    """[(name, param)] for the matrices Muon actually owns, in a stable order.

    Read off the optimizer rather than the model so it always matches what is being stepped: an AdamW run
    has no Muon groups and gets an empty list, which switches every ELR feature here off.
    """
    from .muon import Muon

    owned = set()
    for o in getattr(opt, "opts", [opt]):
        if isinstance(o, Muon):
            for g in o.param_groups:
                owned.update(id(p) for p in g["params"])
    return [(n, p) for n, p in model.named_parameters() if id(p) in owned]


@dataclass
class NormSample:
    step: int
    lr: float
    norms: Dict[str, float]

    @property
    def total(self) -> float:
        return math.sqrt(sum(v * v for v in self.norms.values()))


class NormTracker:
    """Per-matrix ‖W‖_F on a cadence, in one sync.

    The 61 norms are stacked into a single tensor and moved once, so a sample costs one device→host
    transfer rather than 61 — it rides the logging interval's existing sync.
    """

    def __init__(self, named: Sequence[Tuple[str, torch.nn.Parameter]]):
        self.names = [n for n, _ in named]
        self.params = [p for _, p in named]
        self.numel = {n: p.numel() for n, p in named}
        self.last: Optional[NormSample] = None

    def __bool__(self) -> bool:
        return bool(self.params)

    @torch.no_grad()
    def sample(self, step: int, lr: float) -> NormSample:
        vals = torch.stack([p.detach().float().norm() for p in self.params]).cpu().tolist()
        self.last = NormSample(step, lr, dict(zip(self.names, vals)))
        return self.last

    def record(self, s: Optional[NormSample] = None) -> Dict[str, float]:
        """The log line's ELR block: global ELR, the entrywise-RMS law's coefficient, and the spread.

        `rms_coeff` is RMS/entry ÷ √(lr/wd) — the 0.66 measured on the 12-h arms. It is the number that
        says whether a run is on the same equilibrium as the arms it is being compared with, and it is
        scale-free, so it reads the same at 35M and at the flagship.
        """
        s = s or self.last
        if s is None:
            return {}
        total = s.total
        # a zero-norm matrix is not on the equilibrium and would divide the aggregates by zero:
        # `residual_proj` is zero-initialised and stays zero until the first update moves it.
        rms = sorted(s.norms[n] / math.sqrt(self.numel[n]) for n in self.names if s.norms[n] > 0)
        out = {"w_norm": total, "elr": s.lr / total if total else 0.0}
        if rms:
            out["rms_per_entry"] = sum(rms) / len(rms)
            out["rms_spread"] = rms[-1] / rms[0]
        if len(rms) != len(self.names):
            out["n_zero_norm"] = len(self.names) - len(rms)
        return out


class NormGuard:
    """Trip when ‖W‖_F falls while the learning rate is flat.

    The rule is relative on purpose. The failure this is built from — `lr_sweep_12e-4`, whose norm ended
    at 221.5, *below* the 8e-4 arm's 294.6 at 1.5× the learning rate — is a collapse, not a drift, and a
    relative rule needs no fitted constant. The absolute alternative would compare against
    0.66·√(lr/λ)·√(mn), and that 0.66 was fitted on the 35M preset at 12 h; an uncalibrated constant on a
    7-day unattended run is a false positive that halts a healthy trunk.

    Arming only during a flat-lr phase is what keeps warmup and an intended decay tail from firing it:
    both move the equilibrium down by design, and a norm that follows them down is correct behaviour.
    """

    def __init__(self, drop: float = 0.05, consecutive: int = 3, arm_after: int = 20):
        self.drop, self.consecutive, self.arm_after = drop, consecutive, arm_after
        self.baseline: Optional[float] = None
        self.flat = 0          # consecutive samples at an unchanged lr
        self.below = 0         # consecutive samples under the threshold
        self._lr: Optional[float] = None

    def update(self, s: NormSample) -> Optional[str]:
        """The trip reason, or None. Call once per norm sample."""
        if self._lr is None or abs(s.lr - self._lr) > 1e-12 * max(abs(s.lr), 1.0):
            self._lr, self.flat, self.baseline, self.below = s.lr, 0, None, 0  # lr moved: disarm and re-arm
            return None
        self.flat += 1
        if self.flat < self.arm_after:
            return None
        total = s.total
        if self.baseline is None or total > self.baseline:
            self.baseline = total
            self.below = 0
            return None
        if total < self.baseline * (1.0 - self.drop):
            self.below += 1
            if self.below >= self.consecutive:
                return (f"norm collapse: ‖W‖_F {total:.2f} is {100*(1-total/self.baseline):.1f} % below its "
                        f"flat-lr baseline {self.baseline:.2f} for {self.below} samples at lr {s.lr:g}")
        else:
            self.below = 0
        return None


@dataclass
class ELRTrace:
    """A reference run's ELR schedule, per Muon matrix, for another run to track.

    Stored as sampled ‖W_ref‖ plus the reference lr at each sample rather than as ELR at every step:
    the norm's time constant is ~12,500 steps, so 100-step samples resolve it, and the file is 100×
    smaller. ELR is reconstructed by interpolating both, which is exact while lr is flat and good to
    ~0.1 % of lr across a 100-step window of a decay tail.
    """

    samples: List[NormSample] = field(default_factory=list)
    meta: Dict[str, object] = field(default_factory=dict)

    def add(self, s: NormSample) -> None:
        self.samples.append(s)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.write_text(json.dumps({
            "meta": self.meta,
            "samples": [{"step": s.step, "lr": s.lr, "norms": s.norms} for s in self.samples],
        }), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ELRTrace":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        t = cls(meta=d.get("meta", {}))
        for r in d["samples"]:
            t.add(NormSample(int(r["step"]), float(r["lr"]), {k: float(v) for k, v in r["norms"].items()}))
        return t

    def elr_at(self, step: int) -> Dict[str, float]:
        """{name: η^eff} at `step`, by linear interpolation of lr and of every ‖W‖; clamped at both ends."""
        ss = self.samples
        if not ss:
            return {}
        if step <= ss[0].step:
            a = b = ss[0]
            t = 0.0
        elif step >= ss[-1].step:
            a = b = ss[-1]
            t = 0.0
        else:
            lo, hi = 0, len(ss) - 1
            while hi - lo > 1:                     # bisect on step
                mid = (lo + hi) // 2
                if ss[mid].step <= step:
                    lo = mid
                else:
                    hi = mid
            a, b = ss[lo], ss[hi]
            t = (step - a.step) / max(b.step - a.step, 1)
        lr = a.lr + t * (b.lr - a.lr)
        out = {}
        for n, va in a.norms.items():
            w = va + t * (b.norms.get(n, va) - va)
            out[n] = lr / w if w else 0.0
        return out


class ELRMatcher:
    """Drive a run's per-matrix learning rates so it tracks a reference run's ELR schedule.

    This is 2608.24814's reference-run protocol (App. B.2): the matched run keeps its own optimizer and
    its own norm-control mechanism, and only its LR is adapted, η_i = η^eff,ref · ‖W_i‖. If the loss
    trajectories then collapse, the norm-control difference acted through ELR and nothing else.

    Per matrix, not globally, and without one param_group per matrix: Muon's batched Newton-Schulz runs
    before its per-parameter apply loop, so `param_lr` on the existing groups gives exact per-matrix rates
    at no throughput cost (a group per matrix would turn ~5 kernel launches into 61).
    """

    def __init__(self, trace: ELRTrace, named: Sequence[Tuple[str, torch.nn.Parameter]]):
        self.trace = trace
        self.named = list(named)
        self.by_name = {n: p for n, p in self.named}
        missing = [n for n in (trace.samples[0].norms if trace.samples else {}) if n not in self.by_name]
        if missing:
            raise SystemExit(f"ELR trace does not match this model: {len(missing)} unknown parameters, "
                             f"e.g. {missing[:3]}")
        self._norms: Dict[str, float] = {}
        self.last: Dict[str, float] = {}

    def refresh(self, s: NormSample) -> None:
        """Take this run's own norms from a sample the tracker already paid a sync for."""
        self._norms = dict(s.norms)

    @torch.no_grad()
    def apply(self, opt, step: int) -> float:
        """Set `param_lr` on every Muon group; returns the mean applied lr (for the log line)."""
        from .muon import Muon

        target = self.trace.elr_at(step)
        lrs = {}
        for n, p in self.named:
            e = target.get(n)
            w = self._norms.get(n)
            if e is None or not w:
                # no target, or a norm of zero. η_i = η^eff·‖W_i‖ would pin a zero-initialised matrix at
                # zero forever, so it keeps the schedule's own lr until its first update gives it a norm
                # (`residual_proj` is the case; the paper sidesteps it by arming matching after warmup).
                continue
            lrs[id(p)] = e * w
        if not lrs:
            return 0.0
        for o in getattr(opt, "opts", [opt]):
            if isinstance(o, Muon):
                for g in o.param_groups:
                    g["param_lr"] = lrs
        self.last = lrs
        return sum(lrs.values()) / len(lrs)
