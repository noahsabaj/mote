"""The studio's HTTP client for everything that talks to the daemon from outside it: the CLI, the
gate drivers, the sweep scripts. One place for the token, the address and the error wording — there
were three hand-written clients (`mote.cli`, `mote.eval.branch_gate`, `scripts/gate_sweep.py`), and
one of them defaulted to the wrong port.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Optional

from .paths import base_url, read_token


class StudioError(RuntimeError):
    """The daemon refused a call (its own message) or is not there."""


def api(method: str, path: str, body: Any = None, base: Optional[str] = None, token: Optional[str] = None,
        timeout: float = 30.0) -> Any:
    """`api("GET", "/api/training/queue")`, `api("POST", "/api/training/start", {...})` → the JSON reply.

    `base` overrides the address from `.mote/config.json`; `token` the one from `MOTE_TOKEN` / `.mote/token`.
    A 4xx/5xx becomes a `StudioError` carrying the daemon's `detail` (a 400 for malformed training args, a
    404 for no such job), an unreachable studio a `StudioError` naming the address."""
    base = (base or base_url()).rstrip("/")
    tok = token if token is not None else read_token()
    headers = {"Content-Type": "application/json"}
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    req = urllib.request.Request(base + path, method=method, headers=headers,
                                 data=json.dumps(body).encode() if body is not None else None)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read()).get("detail", "")
        except Exception:
            detail = ""
        raise StudioError(f"studio refused {method} {path}: {e.code} {detail or e.reason}") from None
    except urllib.error.URLError as e:
        raise StudioError(f"no studio at {base} ({e.reason}); `mote status`") from None
