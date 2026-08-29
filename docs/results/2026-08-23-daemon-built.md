# Built 2026-08-23 — the daemon, dogfooded on the 35M the same evening

Moved verbatim from docs/shape.md on 2026-08-29 (housekeeping); what it describes is the current daemon minus the 08-24…08-28 additions recorded in shape.md § "The daemon".

The daemon exists: `mote.service` hosts training. `Trainer` (mote/train/train.py) is a generator
yielding at every accumulation slice; `JobQueue` (mote/serve/jobs.py) drives it one slice at a time
under a GPU gate the serving engine takes for whole replies — measured mid-run: **warm replies
192–402 ms while a real Muon job trained** in the same process. A sequential job queue persists to
`.mote/jobs.json` (boot re-enqueues an interrupted job with `--resume`; cancelled stays cancelled);
an **EMA shadow** (decay 0.999) follows the run and hot-swaps into the serving engine every 100 steps
(`/api/model.live = "<run>/ema@<step>"`, prefix cache cleared per swap); a finished job's final
checkpoint becomes the served model. Controls: `POST /api/training/start|stop`, `GET
/api/training/queue`, `mote train start|stop|queue`, and the Training tab. Tests: trainer-slice
equivalence, stop/resume, queue/cancel/interrupt, gate pause, EMA math, engine hot-swap.

Still deferred from the original list: online updates from votes (v1 is batch DPO), and the
process-boundary alternatives (moot — one process shipped).
