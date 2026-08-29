"""The branch gate waits on a job's LINEAGE: a daemon restart re-enqueues the SFT job under a new id linked by
`retry_of`, and the old record never leaves `interrupted` (2026-08-29)."""

import pytest

from mote.eval import branch_gate
from mote.serve.jobs import JobQueue, JobRecord


def test_resume_copy_links_to_its_origin():
    rec = JobRecord(id="aaaa", argv=["--preset", "mote-1m", "--out", "runs/x"], serve=True)
    copy = JobQueue._resume_copy(rec)
    assert copy.retry_of == "aaaa" and copy.resumed and copy.serve and "--resume" in copy.argv
    oom = JobQueue._resume_copy(rec, retries=1, retry_of=rec.id, not_before=1.0, needs_bytes=5)
    assert oom.retry_of == "aaaa" and oom.retries == 1


def test_wait_follows_the_lineage_and_fails_closed_on_held(monkeypatch):
    calls = iter([
        {"current": {"id": "a1", "state": "running"}, "queued": [], "recent": []},
        {"current": None, "queued": [{"id": "b2", "state": "queued", "retry_of": "a1"}],
         "recent": [{"id": "a1", "state": "interrupted"}]},
        {"current": {"id": "b2", "state": "running", "retry_of": "a1"}, "queued": [], "recent": [{"id": "a1", "state": "interrupted"}]},
        {"current": None, "queued": [], "recent": [{"id": "b2", "state": "done", "retry_of": "a1"}, {"id": "a1", "state": "interrupted"}]},
    ])
    monkeypatch.setattr(branch_gate, "_api", lambda base, path, body=None: next(calls))
    monkeypatch.setattr(branch_gate.time, "sleep", lambda s: None)
    assert branch_gate.wait("http://x", "a1") == "done"

    held = iter([{"current": None, "queued": [], "recent": [{"id": "c3", "state": "held", "error": "norm collapse"}]}])
    monkeypatch.setattr(branch_gate, "_api", lambda base, path, body=None: next(held))
    with pytest.raises(RuntimeError, match="held"):
        branch_gate.wait("http://x", "c3")
