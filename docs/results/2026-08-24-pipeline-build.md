# 2026-08-24 — the pipeline's machinery, built the day it was signed

The recipe itself is `docs/shape.md` § "Training pipeline: pre / mid / post". This is what exists now, what
was measured, and what is deliberately untested until a trained model exercises it.

## Built (commits 49b00d0 → 288fb6a + the rlvr commit)

| Piece | Where | Tests |
|---|---|---|
| Schedules `wsd` / `trunk` / `cooldown`, step horizon that survives resume, weights-only `snap_<step>.pt` every `--snapshot-steps` | `mote/train/train.py` | `tests/test_pipeline_stages.py` (trunk → snapshot → cooldown → stop → resume: one monotone decay, chunk target held) |
| `--mix PREFIX:SHARE:plain` — an SFT shard's bytes as plain LM data | `mote/data/loader.py` | same |
| `build_mix --list anneal --skip-after <metas>` — the ANNEAL registry + fresh documents by replaying the file-ordered HF streams past earlier builds | `mote/data/build_mix.py`, `sources.py` | same (skip counting, registry) |
| `build_local` — JSONL (`{text}` / `{messages}` with tool `parts`) → plain or SFT shards | `mote/data/build_local.py` | same |
| Sim-QA probe (held-out worlds, seeds ≥ 5 M; EM, contains, pass@k) | `mote/eval/sim_probe.py` | `tests/test_sim_probe.py` |
| Shared + per-domain val bpb of a checkpoint | `mote/eval/val_bpb.py` | dry run below |
| Branch gate: identical SFT per branch via the daemon → probes → the signed verdict rule → `docs/results` | `mote/eval/branch_gate.py` | `tests/test_branch_gate.py` (verdict rule, table, argv) |
| Tool protocol: `<|call|>` = 262, `<|result|>` = 263 (vocab 264, rows already padded), tool parts in the chat template + loss mask, engine hook (registry, result injection through `forward_from_state`, call cap, `(no such tool)` / tool errors as readable results), scripted ids for tests, graph decoder reports the stop id | `mote/tokenizer.py`, `mote/serve/engine.py`, `graph.py` | `tests/test_tool_protocol.py` (round trip, cap, stray `<|result|>`, RL ids + mask) |
| RLVR-1 environment: State2State tasks (explore k legal actions, changed facts = goal, pruned expert), action language, en/ru/ja goals and observations, `SimEnv` tool with step budget, expert traces | `mote/sim/tasks.py` | `tests/test_sim_tasks.py` (30 seeds × 3 locales reach their goals; round trips; budget) |
| RLVR-1 driver: GRPO-style, edge-of-competence groups, k3 KL to the initial policy, held-out pass@1/@k, `best.pt`, resumable, daemon job type `rlvr` | `mote/train/rlvr.py`, `mote/serve/jobs.py` | `tests/test_rlvr.py` (no-signal step, forced-signal update moves the weights, resume, dispatch) |
| Engine from a live model object (the RL policy), `want_ids` → done event carries prompt ids, reply ids, loss mask | `mote/serve/engine.py` | `tests/test_rlvr.py` |

Data: `data/sim_plain` (cooldown extra: 150 M ids of narrative + QA as plain LM), `data/sim_sft` (201 k QA
conversations with masks), `data/sim_traces` (19.6 k expert traces, 13.9 M ids; n_expert 1–5, goals 1–15
facts). Mix B (fresh flagship composition, 10 GB) and mix C (anneal, 8 GB) building on CPU (`data/*.build.log`).

## Measured

* The 35M (`overnight_sft2`) on held-out sim QA: **0 / 12 EM, 0 contains, pass@2 = 0** — the baseline the
  mid-training gate and SFT-1 are measured against (it has never seen sim data). 20 s on CPU for 36 replies.
* `val_bpb` on the 35M at seq 1024, 2 batches per slice (a code-path check, not a number to compare): shared
  1.763, backbone/code/long/math/multi 1.6–3.0.
* Expert-trace generation: 20 k tasks in 16 s; the trace shard builds in 13 s.

## Deliberately untested until a trained model exists

* The graph-decode tool path: the hook is shared, the stop-id read is plumbed (`gd.stop_id`), but no test can
  make a random model emit `<|call|>` inside a captured graph; the first SFT-1 model with protocol traces
  exercises it (the eager path is tested end to end).
* ru / ja goal sentences: built from the same tables the reviewed narratives use, but no reader has passed
  over them yet (the sim gate's reads covered narratives and QA only).
* RL at scale: the driver runs eager rollouts on the live weights (~5–10 ms/byte at the flagship); a 200-step
  run at 16 tasks × 8 rollouts is a day. The graph path for rollouts is the first speed-up once RLVR-1 has
  numbers.
