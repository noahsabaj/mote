"""Where Mote keeps things on disk — one owner.

`ROOT` is the checkout this code was imported from; `STATE` is its `.mote/` directory: the access
token, the studio config (`config.json`: the pinned checkpoint, device, host, port), the job queue,
the studio log and the pid files. The CLI (`mote.cli`), the server (`mote.serve.app`) and the gate
driver (`mote.eval.branch_gate`) each used to compute the root with a different `parents[...]`
depth and read `config.json` against different defaults (one said port 7860, the others 7861).
"""

from __future__ import annotations

import json
import os
import secrets
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / ".mote"
TOKEN_FILE = STATE / "token"
CONFIG_FILE = STATE / "config.json"
JOBS_FILE = STATE / "jobs.json"
LOG_FILE = STATE / "studio.log"
SUP_PID = STATE / "supervisor.pid"
SRV_PID = STATE / "server.pid"
STOP_FLAG = STATE / "stop"
RUNS = ROOT / "runs"

DEFAULT_CONFIG = {"checkpoint": "runs/pilot_sft/last.pt", "device": "cpu", "port": 7861, "host": "127.0.0.1"}


def load_config(path: Path = CONFIG_FILE) -> dict:
    """The studio config over its defaults (a missing or torn file is the defaults)."""
    cfg = dict(DEFAULT_CONFIG)
    try:
        if path.exists():
            cfg.update(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        pass
    return cfg


def save_config(cfg: dict, path: Path = CONFIG_FILE) -> None:
    """Atomic: the supervisor re-reads this file on every restart, never half-written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    tmp.replace(path)


def read_token(path: Path = TOKEN_FILE) -> Optional[str]:
    """`MOTE_TOKEN` if set, else the token file, else None (an open studio)."""
    tok = os.environ.get("MOTE_TOKEN")
    if tok:
        return tok
    return path.read_text(encoding="utf-8").strip() if path.exists() else None


def ensure_token(path: Path = TOKEN_FILE) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(secrets.token_urlsafe(24), encoding="utf-8")
    return path.read_text(encoding="utf-8").strip()


def base_url(cfg: Optional[dict] = None) -> str:
    """Where the studio answers, from its config: `http://127.0.0.1:7861` by default."""
    cfg = cfg or load_config()
    return f"http://{cfg.get('host', '127.0.0.1')}:{cfg.get('port', 7861)}"


def run_id(path: Path | str, root: Path = ROOT) -> str:
    """A checkpoint's id in the API and the pin: its path relative to the repo, forward slashes
    (`runs/t3l_dense_8e-4/last.pt`); an outside path stays as it is."""
    p = Path(path)
    try:
        return str(p.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(p)
