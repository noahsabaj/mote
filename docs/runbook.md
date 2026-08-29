# Runbook — running Mote on the box

The operational half of the README: what is where, how the daemon is driven, and the rules that keep a
timed run honest. The design behind each item is in `docs/shape.md`; the network side is
`docs/remote-access.md`.

## The machine

Fedora, RTX 4060 Ti 8 GB, 28-core CPU, 31 GB RAM. Repo at `/home/nsabaj/Development/mote`, venv `.venv`
(Python 3.12, torch 2.13 + cu126, triton 3.7; the Mamba-3 Triton kernels from `~/Development/mamba` at
`e9594ce`, installed with `MAMBA_SKIP_CUDA_BUILD=TRUE`). Use `.venv/bin/python` — a bare `python` is not the
venv. `data/` (the shards, ~60 GB) and `runs/` (checkpoints + logs) are gitignored; `origin` (GitHub) is the
only off-machine copy of the code.

## The daemon

One resident process trains and serves (`mote.service`, a systemd user unit; `mote service install` writes it,
`loginctl enable-linger $USER` keeps it up with nobody logged in). It listens on `127.0.0.1:7861`; Tailscale
publishes it to the phone. State lives in `.mote/`: `token` (the access token — read it only for loopback
calls, never paste it anywhere), `config.json` (the pin, device, host, port), `jobs.json` (the queue),
`studio.log`, the pid files.

```
mote status | logs | restart            # the supervisor keeps the server alive; restart = reload code + web/dist
mote build                              # svelte-check + bun build + pytest + restart + the pair link
mote train start [--front] [--serve] -- <trainer args>   # enqueue a job (args as python -m mote.train.train takes them)
mote train queue | stop [--id] | serve [--id] [--off] | release
mote engine park | restore              # move the engine off the card for a standalone measurement
mote config --checkpoint runs/x/last.pt # change the pin; `mote restart` applies it
```

The queue runs jobs **in-process on a thread**: a queued job executes whatever code the worker imported when it
started, and `run.json` records that commit. Editing the working tree does nothing until a restart — and a
crash respawns onto whatever is on disk, so **main is always a coherent commit** (housekeeping lands from a
worktree, step by step, see `docs/results/2026-08-29-housekeeping-prereg.md`). A restart interrupts the running
job at its next step boundary (checkpoint kept, resumed in front of the queue) — and every other session's
arms with it.

## Rules around a timed arm

Every queued arm is wall-clock budgeted (`--max-minutes`), so anything that takes throughput from it changes
its result:

* **No full test suite and no CPU-heavy work beside a timed arm.** A 28-thread CPU sweep cost a 2-h arm 16 % of
  its throughput (2026-08-28). CPU tests run on one pinned, niced core: `CUDA_VISIBLE_DEVICES=""
  OMP_NUM_THREADS=1 taskset -c 27 nice -n 19 .venv/bin/python -m pytest -q tests` (the trainer tests ask for
  the device that exists; the kernel and graph tests skip). GPU tests wait for an idle card; run a failing GPU
  test alone. `pgrep -ax pytest` first.
* **Read a wall-clock pair at matched steps** (`scripts/elr_report.py`), never at its final val alone: a
  confounded arm ends short of its steps.
* The GPU budget is **6.2 GB with the screen locked** (the desktop takes 1.5–1.9 GB); serving lives on the CPU
  while any job runs.
* Never vote on the live daemon's preference store from a script — it is Noah's real data.
* Never commit profiler artefacts (`.nsys-rep`, `.sqlite`, pickles): they capture the shell environment. The
  pre-commit hook (`git config core.hooksPath .githooks`) refuses them and credential-shaped strings.

## Launching the trunk

On Noah's word only. `mote train start --serve -- --preset mote-96m --data data/flagship_mix ...` with the
frozen recipe (docs/shape.md § "The model"; the lr and accumulation under review there). `--serve` puts it on
the air: its EMA answers chats every 100 steps and its final checkpoint becomes the pin. Snapshots
(`--snapshot-steps`) are the branch points for mid-training.

## Reproducibility

Runs are bitwise reproducible by default (cuBLAS pinned to a fixed workspace, the two-pass Relation backward;
`mote/determinism.py`); `--fast` gives that up for ~20 % throughput and is recorded in `run.json`. A refactor
that must not change the training step is checked by 100 matched-seed steps at the flagship shape, old code vs
new, per-step loss and the saved state dict compared bitwise.
