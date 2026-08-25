"""Triton's `Autotuner` keeps per-call state on `self` (`nargs`, the benchmark bookkeeping) and resets it at the
end of `run()`: two threads launching the same autotuned kernel — the training worker and a serving reply on its
own stream — race on it (`TypeError: 'NoneType' object is not a mapping` from `{**self.nargs, ...}`; hit the first
time eager serving ran beside a job, 2026-08-24 night). One process-wide re-entrant lock around `Autotuner.run`
serialises the Python side of every autotuned launch (the kernels themselves still overlap on their streams);
an autotune benchmark of a new shape holds it for its duration, which is the point."""

from __future__ import annotations

import threading

_LOCK = threading.RLock()


def install() -> bool:
    try:
        from triton.runtime.autotuner import Autotuner
    except Exception:  # no triton (CPU-only environment)
        return False
    if getattr(Autotuner, "_mote_locked", False):
        return True
    orig = Autotuner.run

    def run(self, *args, **kwargs):
        with _LOCK:
            return orig(self, *args, **kwargs)

    run._mote_orig = orig  # type: ignore[attr-defined]
    Autotuner.run = run
    Autotuner._mote_locked = True
    return True
