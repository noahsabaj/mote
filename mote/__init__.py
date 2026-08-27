"""Mote — a byte-level H-Net language model (Mamba-3 encoder/decoder, Relation main
network, multi-byte prediction head) and the studio that serves it.

Working package name; rename once the product name is chosen.
"""

import subprocess as _sp
from pathlib import Path as _Path

__version__ = "0.1.0"


def _code_version() -> dict:
    """The commit this PROCESS is running, captured once at import.

    Not `git rev-parse HEAD` read at job start, which is a different and misleading number: the
    daemon runs jobs in-process on a thread (mote/serve/jobs.py) and nothing reloads modules, so a
    job executes whatever was imported when the worker started. HEAD can move underneath it — and
    on 2026-08-27 did, by fourteen hours, with a queued arm silently running stale code.

    It matters more than it sounds. `mote.cli service_run` respawns the worker whenever it exits,
    so a crash mid-queue brings it back on whatever is on disk at that moment. A 24-hour gate could
    run its control on one commit and its treatment arm on another with nothing to show for it, and
    the difference would be read as the thing under test. This is the field that makes that visible.
    """
    root = _Path(__file__).resolve().parent.parent
    out = {"version": __version__}
    try:
        r = _sp.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            out["commit"] = r.stdout.strip()[:12]
        d = _sp.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True, timeout=5)
        if d.returncode == 0:
            out["dirty"] = bool(d.stdout.strip())
    except Exception:
        pass  # a checkout without git is not a reason to fail a run
    return out


CODE_VERSION = _code_version()  # frozen at import: the code this worker actually loaded
