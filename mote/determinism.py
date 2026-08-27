"""Bitwise-reproducible training, and the switch that turns it off.

Measured 2026-08-27 on Mote-13M, same seed, two forward+backward passes in one process, counting
parameters whose gradients differ at all:

    DET=1, cuBLAS default                     83 / 105
    DET=1 + CUBLAS_WORKSPACE_CONFIG + strict   0 / 105
    DET=0 + CUBLAS_WORKSPACE_CONFIG + strict  54 / 105
    DET=1, no mamba kernel, cuBLAS pinned      0 / 105

Three things follow, and only the first was expected.

cuBLAS was the dominant source and was masking everything else — pin it and the picture changes
completely. Mamba-3's Triton kernels are already deterministic; disabling them changes nothing.
And with cuBLAS pinned, our own Relation backward is the ONLY remaining source, which the two-pass
kernels already close. So full reproducibility needed no new code at all: one env var and one flag
that had been sitting there since the kernel was written.

The forward was bitwise identical throughout. This is a backward-pass property.

Cost, measured interleaved on a contended card (indicative; the clean number is owed at the real
shape): about +20 % per step, of which roughly two thirds is cuBLAS and one third is the two-pass
Relation backward. That is why `--fast` exists.

CUBLAS_WORKSPACE_CONFIG is documented as being read when the cuBLAS handle is created, which would
make this a start-of-process decision. Measured, a late set is still effective — 0/105 either way —
so a job can choose its own mode. `torch.use_deterministic_algorithms` is process state, though, and
this daemon trains and serves in one process: while a job runs fast, serving is non-deterministic
too. Nothing depends on serving being reproducible, but it is why the mode is restored afterwards.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

WORKSPACE = ":4096:8"  # cuBLAS: a fixed workspace makes its reduction order stable


def apply(reproducible: bool = True) -> None:
    """Put the process into (or out of) the reproducible configuration."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = WORKSPACE  # harmless when off; needed before any strict op
    os.environ["MOTE_DETERMINISTIC_RELATION"] = "1" if reproducible else "0"
    import mote.model.flash_relation as fr

    fr.DETERMINISTIC = reproducible  # already-imported module: the env var alone is read too late
    # warn_only: an op with no deterministic implementation should be visible, not fatal — a run
    # that dies at hour six because a rarely-taken path lacks one is worse than a warning.
    torch.use_deterministic_algorithms(reproducible, warn_only=True)


def state() -> dict:
    """What a run should record about itself. Two modes in play means a number is only comparable
    to another number from the same mode, and that has to be recoverable later, not inferred."""
    import mote.model.flash_relation as fr

    return {
        "reproducible": bool(torch.are_deterministic_algorithms_enabled()),
        "relation_backward": "two-pass" if fr.DETERMINISTIC else "atomics",
        "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
