# Housekeeping 2026-08-29 — pre-registered (signed 00:40)

Grilled in three rounds on the evening of 2026-08-28 (four read-only surveys: flag graveyard, scripts/tests
staleness, docs drift, structural smells). What follows is the signed design; the surveys' evidence is
summarised where a decision rests on it.

## Frame

- **Software only.** The frozen recipe (2026-08-24) does not move; model ideas keep going through
  pre-registered arms.
- **Lands on main; the trunk launches on it.** The worker runs code from `58e8672` (daemon start 08-28
  18:05); everything since is dormant until the launch restart. Each finished, green step fast-forwards
  main and is pushed; in-progress work lives in a worktree (`../mote-hk`, branch `housekeeping`) so the
  daemon's checkout is always a coherent commit for a crash-respawn.
- **Bar: bitwise-identical training step.** In Tuesday's gap (after `runs/qk/on_lam1.0`, ~19:30):
  100 matched-seed steps at the flagship shape on `58e8672` (what the sweep ran) and on HEAD, per-step
  loss and the saved state dict compared bitwise. A step whose bits differ is reverted on main and
  re-landed after the trunk; the launch proceeds on the rest, on Noah's word, with `--serve`.
- **Tests beside the arms:** one pinned niced core (`taskset -c 27 nice -n 19 OMP_NUM_THREADS=1`) for the
  CPU tests after each step. Baseline before the first run: `t3l24_dense_2.5e-4` at elapsed 80 min,
  10-min mean 121.5 KB/s (min 101.0, n=223). A drop over 1 % stops the runs. GPU tests wait for the gap.
- **Compat:** `--no-mbp` stays as a hidden no-op (every queued argv carries it); old checkpoints load with
  unknown config keys ignored and `mbp.*` / `spine.*` weights dropped with one log line.

## Steps (each gated on its own)

0. **Disk** — done at signing; see `2026-08-29-housekeeping-disk.md`.
1. **Deletions.** Measured-lost knobs: JEPA (+0.017 vs control, 3.4× the gate), `--bf16-residual`
   (+0.0058), `--relation-window` / `window_chunks` (+0.018; also a branch in the flash dispatch guard),
   the attention-main control (1.1696 vs 1.1773 at 2 h — knowledge only, Relation ships), three
   consumer-less `Mamba3Cfg` fields, rlvr `--partial` (the signed doc forbids it), the `cooldown` alias.
   **Spine:** signed 08-26, no gate arm ever ran, and the 08-26 profile falsified both cost premises
   (frac +0.56 GB, expand does not fit at 16384); ~900 LOC, 663 test LOC, three scripts. **MBP head:** off at
   ≥96M and on the served 35M; ~300 LOC plus the speculative round in the engine, the `_dist` sampler,
   the `mbp`/speculative fields of `/api/model` and the WS stats, `types.ts`, Diagnostics. `MBPCfg` goes;
   the ≤35M presets lose the head. Not deleted: bounded routing + the ATDC trigger (b95d64f put them back
   deliberately), `--compile` / `--tf32` / `--aug-*` (parked with a plan), MoE / QK-Norm / feedback (live).
2. **Boundaries.** `mote/infer/{engine,graph,prefix_cache,context}` and `mote/identity.py` (kills the
   serve↔train cycle at `jobs.py:47` / `rlvr.py:60` and all 23 upward imports), `mote/paths.py` (one owner
   of `.mote/` and its defaults — reconciles the 7860/7861 disagreement), `runinfo.py` as the one
   `log.jsonl` reader (eight hand-rolled parsers today), `mote/client.py` (three HTTP clients today),
   the `latent_feedback_arms.py` flag mirror derived from the argparser, `tests/conftest.py` (26 inline
   tiny configs, 13 hand-written checkpoints), the `optimizer: None` writer/reader fix (dpo/kto → resume
   crash), `PAD_ID` for the two `258` literals, one dataclass/NamedTuple tree-walker for four.
3. **Trainer.** `config_from_args` / run-state factory / scheduler + step loop out of the 582-line class;
   the 34 overrides keep their exact order (the bitwise bar is the check).
4. **App.** A `Studio` object + `DevicePolicy` on `app.state` replace the module-level `STATE` dict
   (65 reads, 22 writes, 24 routes); `describe_checkpoint` becomes module-level; Pydantic response models
   with `types.ts` generated from the OpenAPI schema; `MockJob` gone; a `training.svelte.ts` store takes
   the queue polling out of `TrainingSheet.svelte`.
5. **Docs.** shape.md: standing rules and current state rewritten tight (decisions and numbers kept),
   readings → `docs/research`, records → `docs/results`, contradictions fixed with dated corrections
   (VOCAB 266→271, trunk 4 epochs vs 2.8 passes, three throughput figures, torch 2.12 vs 2.13,
   FROZEN-but-reopened, the retired name). `architecture.html` (orphaned, every number wrong) and
   `mote-spec.md` (byte-identical to `identity.spec_text()`, no reader) deleted. `fedora.md` archived to
   results; `context.md`'s passed gate and rejected windowed-main plan folded; `api.md` gains
   `/api/prefs/mark` and the `prefix_cache` / `arena` fields, its serving-policy paragraph becomes a
   pointer; `web/README` loses Alt+1/2/3 and gains its missing components. README rewritten for the
   public reader; operational material moves to `docs/runbook.md` + `docs/remote-access.md`; `cloud/`
   re-scoped to experimentation (shape.md fixed the roles).
6. **Memory and the launch argv** (`--no-mbp` dropped) updated.

## Gap protocol (Tuesday, after the last QK arm)

GPU tests → the bitwise check → a fixed-seed serving diff (old engine vs new, CPU) → `mote build` →
launch on Noah's word.
