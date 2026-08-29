"""The wire shapes the studio types by hand (web/src/lib/types.ts) against what the server declares
(mote/serve/schemas.py) and what the engine actually emits: a renamed field fails here, not in a browser.

Before 2026-08-29 the two were kept in step by eye, and `/api/model` had grown two fields (`prefix_cache`,
`arena`) the studio's type never learned about."""

import re
from pathlib import Path

from mote.infer.engine import Engine
from mote.serve import schemas
from conftest import tiny_cfg, tiny_model

TS = Path(__file__).resolve().parents[1] / "web" / "src" / "lib" / "types.ts"


def ts_fields(name: str) -> set:
    """Top-level field names of `export interface <name>` (nested object literals are not walked)."""
    src = TS.read_text(encoding="utf-8")
    start = src.index(f"export interface {name} ")
    body = src[src.index("{", start) + 1:]
    depth, fields = 1, set()
    for line in body.splitlines():
        if depth == 1:
            m = re.match(r"\s*(\w+)\??:", line)
            if m:
                fields.add(m.group(1))
        depth += line.count("{") - line.count("}")
        if depth <= 0:
            break
    return fields


PAIRS = {
    "Health": schemas.HealthOut,
    "TrainingJob": schemas.TrainingJobOut,
    "JobsStatus": schemas.JobsStatusOut,
    "TrainingRun": schemas.RunOut,
    "LogPage": schemas.LogPageOut,
    "CheckpointListItem": schemas.CheckpointRowOut,
    "PrefsTableRow": schemas.PrefsTableRowOut,
    "PrefsSummary": schemas.PrefsSummaryOut,
}


def test_declared_shapes_match_the_studio_types():
    for ts_name, model in PAIRS.items():
        assert ts_fields(ts_name) == set(model.model_fields), ts_name


def test_model_info_keys_match_the_studio_type():
    """`/api/model` is the engine's `info()` plus the pin, the device and the job on the air; `unfollowed`
    rides along only on a manual load's answer."""
    cfg = tiny_cfg()
    eng = Engine.from_model(tiny_model(cfg), cfg, device="cpu")
    served = set(eng.info()) | {"challenger", "pin", "serving_device", "following"}
    assert ts_fields("ModelInfo") == served | {"unfollowed"}
