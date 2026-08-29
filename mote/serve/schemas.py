"""The HTTP contract's shapes (docs/api.md), declared once.

FastAPI validates and filters every response through these, so a field the server stops sending — or
starts sending under a new name — fails a test instead of reaching a browser as `undefined`.
`tests/test_api_contract.py` holds each of them against the studio's hand-typed `web/src/lib/types.ts`,
field for field; the two used to be kept in step by eye.

`/api/model` is not modelled: its payload is the engine's `info()` plus what the routes add, and the same
test compares its live keys with the studio's `ModelInfo` instead.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class HealthOut(BaseModel):
    ok: bool
    model_loaded: bool


class TrainingJobOut(BaseModel):
    """`mote.serve.jobs.JobRecord` on the wire (plus `waiting` on a queued job: why it is not starting)."""

    id: str
    argv: List[str]
    state: str
    created_at: float
    started_at: Optional[float]
    ended_at: Optional[float]
    error: Optional[str]
    resumed: bool
    serve: bool
    retries: int
    retry_of: Optional[str]
    not_before: float
    needs_bytes: int
    deaths: int
    start_step: int
    waiting: Optional[str] = None


class JobsStatusOut(BaseModel):
    current: Optional[TrainingJobOut]
    phase: Optional[str] = None
    queued: List[TrainingJobOut]
    recent: List[TrainingJobOut]
    halted: Optional[str] = None
    paused: Optional[str] = None


class SubmitOut(JobsStatusOut):
    submitted: str


class ReleaseOut(JobsStatusOut):
    released: str


class RunOut(BaseModel):
    id: str
    steps: int
    last_val_bpb: Optional[float]
    running: bool
    started_at: str


class LogPageOut(BaseModel):
    records: List[Dict[str, Any]]
    next: int


class CheckpointRowOut(BaseModel):
    id: str
    step: int
    val_bpb: Optional[float]
    bytes_seen: int
    file_size_bytes: int
    created_at: str
    loaded: bool
    challenger: bool


class PrefsTableRowOut(BaseModel):
    a: str
    b: str
    a_wins: int
    b_wins: int
    ties: int
    both_bad: int
    n: int


class PrefsSummaryOut(BaseModel):
    """`PrefStore.summary()`; a vote or a mark answers with the same summary plus the id it stored."""

    pairs: int
    votes: Dict[str, int]
    unrated_by_claude: int
    marks: Dict[str, int]
    table: List[PrefsTableRowOut]
    agreement: Dict[str, Optional[float]]
    rubric: Optional[str]
    pair: Optional[str] = None
    mark: Optional[str] = None
