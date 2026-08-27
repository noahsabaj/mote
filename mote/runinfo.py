"""What a finished run measured about itself, read back from its log.

One function matters here: `measured_bpic`. The router's compression rate — bytes per innermost
chunk — is an OBSERVABLE of a trained model, not a setting (see config.py::DCCfg for the provenance
and the numbers). Everything that needs to know how many chunks a prompt will produce has to read it
from a run that actually produced them:

    the serving arena          sizes its capacity from it instead of max_seq_len // 4
    profile_step               defaults --chunk-bytes to it instead of a literal

Before 2026-08-27 both used a constant of 6, inherited from ATDC's H-Net baseline target. Three
trained runs measured 3.20-3.45, which is 2.21x the main-network FLOPs per byte that constant
implies at Mote-138M/16384. A number nobody measures is a number that drifts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

# Fallback when a run recorded nothing. Deliberately the measured value and not ATDC's target: if
# this is ever used, being close to what routers actually do beats being close to what they aim at.
DEFAULT_BPIC = 3.3


def _log_path(target: str | Path) -> Optional[Path]:
    """`log.jsonl` for a checkpoint path, a run directory, or the log itself."""
    p = Path(target)
    if p.is_file() and p.name == "log.jsonl":
        return p
    d = p.parent if p.suffix == ".pt" else p
    log = d / "log.jsonl"
    return log if log.exists() else None


def last_eval(target: str | Path) -> dict:
    """The most recent `eval` record in a run's log (empty when there is none)."""
    log = _log_path(target)
    if log is None:
        return {}
    out: dict = {}
    try:
        for line in log.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if isinstance(rec, dict) and isinstance(rec.get("eval"), dict):
                out = rec["eval"]
    except Exception:  # a truncated or half-written log is not a reason to fail a load
        pass
    return out


def measured_bpic(target: str | Path, default: Optional[float] = None) -> Optional[float]:
    """Bytes per chunk this run last measured on its validation set, or `default`.

    `default=None` (the caller's own fallback) is deliberate: a caller that can do something better
    with "unknown" than with a guess should get to see the unknown.
    """
    v = last_eval(target).get("val_bpic")
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return v if v > 0 else default


def chunks_for(target: str | Path, seq_len: int, default_bpic: float = DEFAULT_BPIC) -> int:
    """How many chunks `seq_len` bytes will produce under this run's measured rate."""
    bpic = measured_bpic(target, default_bpic) or default_bpic
    return max(int(seq_len / max(bpic, 1e-6)) + 1, 1)
