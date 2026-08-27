"""Bitwise reproducibility, and the attribution that made it available without new code.

The claim under test is not "our kernels are deterministic" — it is that the DEFAULT configuration
produces identical gradients across two same-seed runs. Measured 2026-08-27, the sources were not
where they were assumed to be: cuBLAS dominated and masked everything else, Mamba-3's kernels were
already clean, and once cuBLAS was pinned our Relation backward was the only thing left.
"""

from __future__ import annotations

import pytest
import torch

from mote import determinism

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _grads(seed=0, steps=1):
    """Gradients from a fresh model on fixed input — the whole byte path, not one kernel."""
    from mote.config import resolve_preset
    from mote.model.hnet import HNetForCausalLM
    from mote.train.train import compute_losses

    cfg = resolve_preset("mote-1m")
    cfg.mbp.enabled = False
    torch.manual_seed(seed)
    m = HNetForCausalLM(cfg).cuda()
    x = torch.randint(0, 256, (2, 256), device="cuda")
    out = None
    for _ in range(steps):
        for p in m.parameters():
            p.grad = None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, n, _, _ = compute_losses(m, x, cfg.dc.target_ratio_init, 0.0, cfg.dc.ratio_loss_weight, None)
        (loss / n).backward()
        out = {k: p.grad.detach().clone() for k, p in m.named_parameters() if p.grad is not None}
    return out, float(loss) / n


@cuda
def test_the_default_is_bitwise_reproducible():
    determinism.apply(True)
    a, la = _grads()
    b, lb = _grads()
    assert la == lb, "forward diverged, which would be a different bug entirely"
    bad = [k for k in a if not torch.equal(a[k], b[k])]
    assert not bad, f"{len(bad)} of {len(a)} gradients differ under the reproducible default: {bad[:4]}"


@cuda
def test_fast_mode_is_not_reproducible_and_that_is_the_trade():
    """If this ever passes, --fast has stopped costing anything and the default should be revisited.
    The atomics in the Relation backward are what it is buying back."""
    determinism.apply(False)
    try:
        a, la = _grads()
        b, lb = _grads()
        assert la == lb, "the forward is deterministic either way; only the backward is at issue"
        differing = sum(1 for k in a if not torch.equal(a[k], b[k]))
    finally:
        determinism.apply(True)
    assert differing > 0, "fast mode reproduced exactly — measure it again before trusting this"


@cuda
def test_the_forward_never_depended_on_the_mode():
    for repro in (True, False):
        determinism.apply(repro)
        _, l1 = _grads(seed=3)
        _, l2 = _grads(seed=3)
        assert l1 == l2, f"forward not reproducible at reproducible={repro}"
    determinism.apply(True)


def test_state_is_recorded_precisely_enough_to_compare_runs():
    """Two modes in play means a number is only comparable to another from the same mode. run.json
    carries this, so which mode a run used is recoverable rather than inferred."""
    determinism.apply(True)
    on = determinism.state()
    assert on["reproducible"] is True and on["relation_backward"] == "two-pass"
    assert on["cublas_workspace"] == determinism.WORKSPACE
    determinism.apply(False)
    off = determinism.state()
    assert off["reproducible"] is False and off["relation_backward"] == "atomics"
    determinism.apply(True)


def test_the_trainer_defaults_to_reproducible():
    from mote.train.train import build_argparser

    a = build_argparser().parse_args(["--data", "data/local_mix", "--out", "runs/_x"])
    assert a.fast is False, "reproducible is the default; --fast is the opt-out"


def test_runs_record_the_commit_the_worker_actually_loaded():
    """The daemon runs jobs in-process and never reloads modules, so a job executes whatever was
    imported when the worker started — and service_run respawns the worker on exit, so a crash
    mid-queue can bring it back on different code with nothing recording it. CODE_VERSION is frozen
    at import for exactly that reason; reading HEAD at job start would report the wrong number."""
    import mote

    cv = mote.CODE_VERSION
    assert cv["version"] == mote.__version__
    if "commit" in cv:  # absent in a checkout without git, which must not fail a run
        assert len(cv["commit"]) == 12 and all(c in "0123456789abcdef" for c in cv["commit"])
        assert isinstance(cv["dirty"], bool)
    # frozen, not re-read: a second access must not go back to git
    assert mote.CODE_VERSION is cv
