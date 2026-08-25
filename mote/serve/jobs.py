"""Training jobs inside the studio server — the resident daemon's half that owns long-running work
(docs/shape.md, decided 2026-08-23).

One worker thread drives `Trainer.run()` one yield at a time. Every yield is an accumulation slice, and
each slice runs under the GPU gate the serving engine also takes for a whole reply — so a chat request
makes training yield at the next slice (~0.1 s at the 35M, ~0.35 s at the flagship) and training resumes
when the reply is done. An EMA shadow of the weights follows the run and is synced into the serving
engine every `sync_steps` optimizer steps (the prefix cache clears on each sync); when a job finishes,
the engine loads the run's final checkpoint.

The queue is sequential and persists to `.mote/jobs.json`: on boot, a job that was `running` when the
process died is marked `interrupted` and re-enqueued in front with `--resume` — a cancelled job stays
cancelled. That is what "resident" means: the flagship run survives crashes and reboots unattended.

A CUDA out-of-memory failure is retried (2026-08-24: the desktop took 1.8 GB of the 8 GB card and eight
flagship arms died in a row): the failed record stays `failed`, a `--resume` copy goes to the front of the
queue with a growing delay, and it only starts once free + cached GPU memory covers the failed run's
tracked peak plus a margin — a structurally too-big job waits visibly instead of burning attempts, while
the rest of the queue keeps flowing around it. Three retries per lineage.
"""

from __future__ import annotations

import json
import secrets
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch

from ..train.train import Trainer, build_argparser


def _job_args(argv: List[str]) -> List[str]:
    return argv[1:] if argv and argv[0] == "rlvr" else argv


def _argparser_for(argv: List[str]):
    """`rlvr --init-from ...` is the RL job (mote/train/rlvr.py); anything else is the trainer."""
    if argv and argv[0] == "rlvr":
        from ..train.rlvr import build_argparser as rl_argparser
        return rl_argparser()
    return build_argparser()


def make_trainer(argv: List[str]):
    """The job's driver: Trainer or RlvrTrainer, both generators of ("slice"|"step", None) with
    model / cfg / out_dir / step / request_stop / close (the queue and the EMA only use those)."""
    if argv and argv[0] == "rlvr":
        from ..train.rlvr import RlvrTrainer
        return RlvrTrainer(argv[1:])
    return Trainer(argv)

STATES = ("queued", "running", "done", "failed", "cancelled", "interrupted")
OOM_RETRIES = 3                          # retries per job lineage after CUDA out-of-memory
OOM_RETRY_DELAYS = (120.0, 600.0, 1800.0)  # seconds before the 1st/2nd/3rd retry may start
OOM_MARGIN = 384 << 20                   # headroom over the failed run's tracked peak before a retry starts


def is_oom(exc: BaseException) -> bool:
    oom_type = getattr(torch, "OutOfMemoryError", None)
    if oom_type is not None and isinstance(exc, oom_type):
        return True
    s = str(exc).lower()
    return "out of memory" in s or "alloc failed" in s


def gpu_peak_bytes() -> int:
    """Peak allocation tracked by the caching allocator since the last reset (the job's peak plus the serving residue)."""
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        return int(torch.cuda.max_memory_allocated())
    return 0


def gpu_usable_bytes() -> float:
    """What a new job could get: free device memory plus our cached-but-unused reservation."""
    if not (torch.cuda.is_available() and torch.cuda.is_initialized()):
        return float("inf")
    free, _total = torch.cuda.mem_get_info()
    return float(free + torch.cuda.memory_reserved() - torch.cuda.memory_allocated())


@dataclass
class JobRecord:
    id: str
    argv: List[str]
    state: str = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    error: Optional[str] = None
    resumed: bool = False  # re-enqueued after an interruption: Trainer gets --resume
    retries: int = 0  # OOM retries already spent by this lineage (a retry copy carries retries + 1)
    retry_of: Optional[str] = None
    not_before: float = 0.0  # a retry waits at least until then...
    needs_bytes: int = 0  # ...and until `gpu_usable_bytes()` covers the failed run's peak + margin

    @property
    def out_dir(self) -> Optional[str]:
        if "--out" in self.argv:
            return self.argv[self.argv.index("--out") + 1]
        return None


class Ema:
    """Exponential moving average of a model's floating-point parameters, kept on the same device."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {
            k: v.detach().clone() for k, v in model.state_dict().items() if v.dtype.is_floating_point
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        sd = model.state_dict()
        for k, s in self.shadow.items():
            s.lerp_(sd[k].detach(), 1.0 - self.decay)

    @torch.no_grad()
    def state_dict(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        """The model's state with the floating entries replaced by their EMA."""
        out = {k: v.detach().clone() for k, v in model.state_dict().items()}
        out.update({k: s.clone() for k, s in self.shadow.items()})
        return out


class JobQueue:
    """Sequential training jobs driven slice-by-slice under the GPU gate."""

    def __init__(self, state_file: Path, gate: threading.Lock,
                 on_serve_sync: Optional[Callable[[dict, Dict[str, torch.Tensor], str, int], None]] = None,
                 on_finished: Optional[Callable[[JobRecord], None]] = None,
                 sync_steps: int = 100, ema_decay: float = 0.999, keep: int = 50):
        self.state_file = Path(state_file)
        self.gate = gate
        self.on_serve_sync = on_serve_sync
        self.on_finished = on_finished
        self.sync_steps = sync_steps
        self.ema_decay = ema_decay
        self.keep = keep
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._trainer: Optional[Trainer] = None
        self._thread: Optional[threading.Thread] = None
        self._shutdown = False
        self.jobs: List[JobRecord] = []
        self._load()

    # ---- persistence ---------------------------------------------------------------------
    def _load(self) -> None:
        if self.state_file.exists():
            try:
                raw = json.loads(self.state_file.read_text(encoding="utf-8"))
                self.jobs = [JobRecord(**{k: v for k, v in r.items() if k in JobRecord.__dataclass_fields__})
                             for r in raw.get("jobs", [])]
            except Exception:
                self.jobs = []
        # a job that was running when the process died resumes in front of the queue
        for r in list(self.jobs):
            if r.state == "running":
                r.state = "interrupted"
                self._insert_front(self._resume_copy(r))
        self._save()

    def _save(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        recent = self.jobs[-self.keep:]
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps({"jobs": [asdict(r) for r in recent]}, indent=1), encoding="utf-8")
        tmp.replace(self.state_file)
        self.jobs = recent

    @staticmethod
    def _resume_copy(rec: JobRecord, **fields) -> JobRecord:
        fields = {"resumed": True, "retries": rec.retries, **fields}
        nxt = JobRecord(id=secrets.token_hex(4), argv=list(rec.argv), **fields)
        if "--resume" not in nxt.argv:
            nxt.argv.append("--resume")
        return nxt

    def _insert_front(self, rec: JobRecord) -> None:
        """Ahead of every queued record, behind the running and finished ones (caller holds the lock)."""
        i = next((k for k, r in enumerate(self.jobs) if r.state == "queued"), len(self.jobs))
        self.jobs.insert(i, rec)
        self._save()

    # ---- public --------------------------------------------------------------------------
    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._worker, name="mote-jobs", daemon=True)
            self._thread.start()

    def submit(self, argv: List[str], front: bool = False) -> JobRecord:
        _argparser_for(argv).parse_args(_job_args(argv))  # reject malformed args at submit time, not hours later
        rec = JobRecord(id=secrets.token_hex(4), argv=list(argv))
        with self._lock:
            if front:
                self._insert_front(rec)
            else:
                self.jobs.append(rec)
                self._save()
        self._wake.set()
        return rec

    def cancel(self, job_id: Optional[str] = None) -> Optional[JobRecord]:
        """Cancel a queued job by id, or (with no id) stop the running one at its next step boundary."""
        with self._lock:
            if job_id is None:
                rec = next((r for r in self.jobs if r.state == "running"), None)
            else:
                rec = next((r for r in self.jobs if r.id == job_id), None)
            if rec is None:
                return None
            if rec.state == "queued":
                rec.state = "cancelled"
                rec.ended_at = time.time()
                self._save()
            elif rec.state == "running" and self._trainer is not None:
                self._trainer.request_stop("cancelled")
        return rec

    def status(self) -> dict:
        with self._lock:
            cur = next((asdict(r) for r in self.jobs if r.state == "running"), None)
            return {
                "current": cur,
                "queued": [asdict(r) for r in self.jobs if r.state == "queued"],
                "recent": [asdict(r) for r in reversed(self.jobs) if r.state in ("done", "failed", "cancelled", "interrupted")][:10],
            }

    def shutdown(self) -> None:
        self._shutdown = True
        if self._trainer is not None:
            self._trainer.request_stop("interrupted")
        self._wake.set()

    # ---- the worker ----------------------------------------------------------------------
    def _next_queued(self) -> tuple:
        """(next runnable record or None, seconds until a deferred retry may become runnable)."""
        now = time.time()
        with self._lock:
            queued = [r for r in self.jobs if r.state == "queued"]
        usable = None
        wait = 5.0
        for r in queued:
            if r.not_before > now:
                wait = min(wait, r.not_before - now)
                continue
            if r.needs_bytes:
                if usable is None:
                    usable = gpu_usable_bytes()
                if usable < r.needs_bytes:
                    continue  # waits for the GPU, the queue flows around it
            return r, 0.0
        return None, max(wait, 0.05)

    def _worker(self) -> None:
        while not self._shutdown:
            rec, wait = self._next_queued()
            if rec is None:
                self._wake.wait(timeout=wait)
                self._wake.clear()
                continue
            with self._lock:
                rec.state = "running"
                rec.started_at = time.time()
                self._save()
            try:
                self._run_job(rec)
                reason = self._trainer.stopped_reason if self._trainer else None
                with self._lock:
                    if rec.state == "running":
                        if reason == "cancelled":
                            rec.state = "cancelled"
                        elif reason == "interrupted":  # graceful shutdown: come back with --resume, first in line
                            rec.state = "interrupted"
                            self._insert_front(self._resume_copy(rec))
                        else:
                            rec.state = "done"
            except Exception as exc:
                with self._lock:
                    rec.state = "failed"
                    rec.error = traceback.format_exc(limit=8)
                    if is_oom(exc) and rec.retries < OOM_RETRIES:
                        delay = OOM_RETRY_DELAYS[min(rec.retries, len(OOM_RETRY_DELAYS) - 1)]
                        self._insert_front(self._resume_copy(
                            rec, retries=rec.retries + 1, retry_of=rec.id, not_before=time.time() + delay,
                            needs_bytes=gpu_peak_bytes() + OOM_MARGIN))
            finally:
                with self._lock:
                    rec.ended_at = time.time()
                    self._save()
                t, self._trainer = self._trainer, None
                if t is not None:
                    try:
                        t.close()
                    except Exception:
                        pass
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()  # hand the run's allocator pool back to serving
                if self.on_finished:
                    try:
                        self.on_finished(rec)
                    except Exception:
                        pass

    def _run_job(self, rec: JobRecord) -> None:
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.reset_peak_memory_stats()  # the job's peak (over the serving residue) sizes an OOM retry
        with self.gate:  # model + optimizer construction also touches the GPU
            self._trainer = t = make_trainer(list(rec.argv))
            ema = Ema(t.model, self.ema_decay)
        steps_since_sync = 0
        g = t.run()
        while True:
            with self.gate:  # one accumulation slice per acquisition: replies slot in between
                item = next(g, None)
            if item is None:
                break
            if item[0] != "step":
                continue
            ema.update(t.model)
            steps_since_sync += 1
            if self.on_serve_sync and steps_since_sync >= self.sync_steps:
                steps_since_sync = 0
                with self.gate:
                    self.on_serve_sync(t.cfg.to_dict(), ema.state_dict(t.model),
                                       f"{t.out_dir.name}/ema", t.step)
