"""Inference: the generation engine, the per-byte decode graph, the prefix store and context folding.

Serving code that is not a server. `mote.serve` (the FastAPI app, the job queue, the preference
store, pairing) is one client of this package; the eval probes, the data builders and the RL driver
are others — before 2026-08-29 they all imported the engine out of `mote.serve`, which put the
HTTP layer under everything and closed a cycle with the trainer.
"""
