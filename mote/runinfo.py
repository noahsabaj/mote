"""What a run wrote about itself, read back from its directory — the one reader of `log.jsonl` and `run.json`.

`measured_bpic` is the reason this module exists. The router's compression rate — bytes per innermost
chunk — is an OBSERVABLE of a trained model, not a setting (see config.py::DCCfg for the provenance and
the numbers). Everything that needs to know how many chunks a prompt will produce has to read it from a
run that actually produced them:

    the serving arena          sizes its capacity from it instead of max_seq_len // 4
    profile_step               defaults --chunk-bytes to it instead of a literal

Before 2026-08-27 both used a constant of 6, inherited from ATDC's H-Net baseline target. Three
trained runs measured 3.20-3.45, which is 2.21x the main-network FLOPs per byte that constant
implies at Mote-138M/16384. A number nobody measures is a number that drifts.

The rest of this module is the same idea applied to the log as a whole. Eight places used to parse
`log.jsonl` by hand (the engine's checkpoint description, the studio's run list and log endpoint, the
queue's progress check, the trainer's resume clock, the branch gate, the lr fit, the report scripts),
each with its own tolerance for a torn line, its own idea of where a fresh start begins and its own
`bytes_seen` arithmetic. They read through here now.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Iterator, List, Optional

# Fallback when a run recorded nothing. Deliberately the measured value and not ATDC's target: if
# this is ever used, being close to what routers actually do beats being close to what they aim at.
DEFAULT_BPIC = 3.3


# ---- files ---------------------------------------------------------------------------------------
def run_dir(target: str | Path) -> Path:
    """The run directory for a checkpoint path, a run directory, or one of its files."""
    p = Path(target)
    return p.parent if p.suffix in (".pt", ".json", ".jsonl") else p


def log_path(target: str | Path) -> Optional[Path]:
    """`log.jsonl` for a checkpoint path, a run directory, or the log itself (None when there is none)."""
    p = Path(target)
    if p.is_file() and p.name == "log.jsonl":
        return p
    log = run_dir(p) / "log.jsonl"
    return log if log.exists() else None


def run_json(target: str | Path) -> dict:
    """The run's `run.json` (its argv and provenance), or {} without one."""
    p = run_dir(target) / "run.json"
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except Exception:
        return {}


# ---- records -------------------------------------------------------------------------------------
def iter_records(target: str | Path) -> Iterator[dict]:
    """Every JSON record in the run's log, in order; a torn or half-written line is skipped (a run that
    is still writing is not a reason to fail a read)."""
    log = log_path(target)
    if log is None:
        return
    with open(log, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if isinstance(rec, dict):
                yield rec


def records(target: str | Path) -> List[dict]:
    return list(iter_records(target))


def records_since(target: str | Path, since: int = 0):
    """(the records from line `since` on, the next line cursor) — `GET /api/training/runs/{id}/log`. A torn
    line is skipped but still counts, so the cursor stays a line number."""
    log = log_path(target)
    if log is None:
        return [], since
    lines = log.read_text(encoding="utf-8").splitlines()
    out = []
    for line in lines[since:]:
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out, since + len(lines[since:])


def is_fresh_start(rec: dict) -> bool:
    """The first line a fresh run writes: its throughput probe at step 0. A resume continues the step
    count and never writes one."""
    return rec.get("step") == 0 and "probe_sec_per_step" in rec


def last_segment(recs: List[dict]) -> List[dict]:
    """The records of the last fresh start only. A fresh start into a used directory appends after the old
    run's lines (the trainer renames the old log aside since 2026-08-24, but older directories carry both)."""
    starts = [i for i, r in enumerate(recs) if is_fresh_start(r)]
    return recs[starts[-1]:] if starts else recs


def last_step(target: str | Path, tail_bytes: Optional[int] = None) -> int:
    """The largest `step` the run logged, 0 without one. `tail_bytes` reads only the end of the log — the
    queue asks this on every boot and a 7-day log is hundreds of MB."""
    log = log_path(target)
    if log is None:
        return 0
    step = 0
    if tail_bytes:
        try:
            with open(log, "rb") as f:
                f.seek(0, os.SEEK_END)
                f.seek(max(f.tell() - tail_bytes, 0))
                text = f.read().decode("utf-8", "replace")
        except OSError:
            return 0
        lines = text.splitlines()
    else:
        lines = log.read_text(encoding="utf-8").splitlines()
    for line in lines:
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict) and isinstance(rec.get("step"), int):
            step = max(step, rec["step"])
    return step


def last_elapsed_sec(target: str | Path) -> float:
    """The last `elapsed_min` the run logged, in seconds (0 without a log) — the clock of a checkpoint
    from before 2026-08-24, which carried none of its own."""
    last = 0.0
    for rec in iter_records(target):
        if "elapsed_min" in rec:
            last = float(rec["elapsed_min"]) * 60.0
    return last


def last_eval(target: str | Path) -> dict:
    """The most recent `eval` record in a run's log (empty when there is none)."""
    out: dict = {}
    for rec in iter_records(target):
        if isinstance(rec.get("eval"), dict):
            out = rec["eval"]
    return out


def final_val_bpb(target: str | Path) -> Optional[float]:
    """The last `val_bpb` the run evaluated, or None."""
    v = last_eval(target).get("val_bpb")
    return float(v) if v is not None else None


def is_done(target: str | Path) -> bool:
    """The run wrote its `done` record (a stopped-and-resumable run has not)."""
    return any(rec.get("done") is True for rec in iter_records(target))


def is_running(target: str | Path, quiet_s: float = 120.0) -> bool:
    """Written to in the last `quiet_s` seconds and not finished — the studio's run list."""
    log = log_path(target)
    if log is None:
        return False
    if time.time() - log.stat().st_mtime >= quiet_s:
        return False
    return not is_done(target)


# ---- derived numbers -----------------------------------------------------------------------------
def tokens_per_step(rj: dict) -> int:
    """Bytes an optimizer step consumes, from the run's argv: batch × window × accumulation."""
    try:
        return int(rj["batch_size"]) * int(rj["seq_len"]) * int(rj.get("grad_accum", 1))
    except (KeyError, TypeError, ValueError):
        return 0


def bytes_seen(target: str | Path, step: int, extra: Optional[dict] = None) -> int:
    """Bytes the run had consumed at `step`: from its argv when the run recorded one, else from a
    checkpoint's own `extra["bytes_seen"]`, else 0."""
    n = step * tokens_per_step(run_json(target))
    if n == 0 and extra and "bytes_seen" in extra:
        n = int(extra["bytes_seen"])
    return n


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


def describe(target: str | Path, step: int, extra: Optional[dict] = None) -> dict:
    """What a checkpoint's row and its serving card show: `val_bpb` and `elapsed_min` as of `step`
    (the log's last eval / train line at or before it) and `bytes_seen`."""
    val_bpb, minutes = None, None
    for rec in iter_records(target):
        if rec.get("step", 0) > step:
            continue
        if isinstance(rec.get("eval"), dict):
            val_bpb = rec["eval"].get("val_bpb", val_bpb)
        if "elapsed_min" in rec:
            minutes = rec["elapsed_min"]
    return {"val_bpb": val_bpb, "trained_minutes": minutes, "bytes_seen": bytes_seen(target, step, extra)}
