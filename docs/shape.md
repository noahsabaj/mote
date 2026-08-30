# Shape — what Mote is and how it is run

The standing rules and the current state, in one place. Dated reading notes live in `docs/research/`,
dated records and pre-registrations in `docs/results/`; the history section at the end says where each
piece that used to sit here went. Rewritten on 2026-08-29 (housekeeping, `docs/results/2026-08-29-housekeeping-prereg.md`):
every decision and number below is the one that was here, with its date; where a number had been stated two
ways the correction is dated beside it.

## Standing rules

* **One resident process owns the GPU** (2026-08-23). It trains in preemptible accumulation slices and serves
  from the same weights; no second copy of the model, no separate training job and studio server fighting
  over 8 GB. While any training job runs the engine lives on the CPU (2026-08-25); the released mode — an
  engine on the GPU beside a job — stays as a mechanism the daemon no longer runs.
* **The architecture stays**: bytes, H-Net chunking, Mamba-3 outer, Relation main, Muon. Out of scope until
  the pipeline below has run: a looped main network, test-time-training memory layers instead of a window,
  any change to the architecture's math (those are a research sweep, 2026 sources first, not a systems decision).
* **Serving contract** (2026-08-23): first byte within ~1 s while a training run is on; the run yields at
  the next slice. Measured 2026-08-23: warm replies 192–402 ms while a real Muon job trained in the same process.
* **ELR is the coordinate** (2026-08-26): any comparison that changes an optimizer, a weight decay or a
  norm-control mechanism is reported in η/‖W‖_F, not in nominal lr. Every run carries `--norm-guard`
  (default `stop`): if ‖W‖_F falls while the lr is flat — the `lr_sweep_12e-4` signature — the run stops
  gracefully and the whole queue halts until `mote train release`.
* **Model names are parameter counts** (2026-08-26): Mote-1M / 11M / 32M / 96M / 138M (`mote/config.py`;
  11M and 32M since 2026-08-29, when retiring the multi-byte head moved them from 13M and 35M). The retired
  role names `smoke` / `pilot` / `local` / `flagship` and the old sizes resolve as aliases because queued argv
  carries them. Runs go under `runs/<model>/<experiment>-<arm>`; the flat directories are dated records.
* **The chunk rate is an observable, not a setting** (2026-08-27): three trained runs measured 3.2–3.45 bytes
  a chunk; the arena and the profiler read it from the run (`mote/runinfo.py`), never from ATDC's target.
* **Runs are reproducible to an envelope, not bitwise** (corrected 2026-08-29: the 2026-08-28 rule said bitwise —
  cuBLAS is pinned and the Relation backward is two-pass, but the Mamba-3/SSD backward reduces with atomics, so
  two runs of the same code differ from step 1 in the backward — the L4 control measured `grad_norm` 1388.8065 vs
  1388.7786). The forward IS bitwise across runs and processes, and an interrupt + resume is bitwise the straight
  run (`tests/test_resume_bitwise.py`). Any trajectory comparison is read inside a ≥3-run same-code envelope;
  `--fast` widens it for ~20 % throughput and is recorded in `run.json`.
* **A timed arm is read at matched steps, and nothing heavy runs beside it** (2026-08-28): a 28-thread CPU
  sweep cost a 2-h arm 16 % of its throughput. CPU tests run on one pinned niced core; the full suite and GPU
  tests wait for an idle card (`docs/runbook.md`).
* **The 7-day trunk launches on Noah's word, with `--serve`**; screening arms never touch what is served.
* Noah reads any fit before a frozen number moves.
* **Two scales, three seeds** (2026-08-29): an architecture claim is measured at 11M with three seeds and at 32M
  with the same sign, with the plain-attention main and the pure-Mamba-3 main in the table
  (`docs/results/2026-08-29-hybrid-ladder-prereg.md`).
* **The chunk rate's controller is the decision threshold** (2026-08-29): the bounded-routing projection ranks the
  whole window and is not causal when it binds; it stays a guardrail. `--bound-floor` is a rate over the training
  window (`655ac24`).
* **Unwon options expire** (2026-08-29): an option that has not won an arm 90 days after its build date is deleted
  at the next housekeeping, its reading kept in `docs/research/`; retroactive (MoE 08-24, DPO/KTO/RLVR 08-25,
  augmentations and FIM 08-26 fall due late November 2026). A number in a comment is a dated claim housekeeping owns.
* **Prefix invariance is tested** (2026-08-29): `tests/test_prefix_invariance.py` runs the two-forward audit with a
  positive control on every arm config before it is timed.

## The model — Mote-96M, frozen 2026-08-24

95,924,732 parameters: outer width 512, 3 + 3 Mamba-3 layers, a 12-layer Relation main network at 768 / 8
heads / d_ff 2048 with Givens head mixing, tied embeddings, the residual stream in fp32, a 16384-byte window
from the first step, chunk-count bucket 64, bounded routing as the arena's guardrail. `VOCAB_SIZE` 271
(256 bytes + 15 protocol ids) padded to 272 rows, the head masking the spare row (corrected 2026-08-29: the
2026-08-24 text said 266 + six spare rows; the FIM, thinking, reversal and offset ids were added since). The
multi-byte prediction head is gone (2026-08-29): it cost 8–13 % of the step for no loss gain and was off at
every size that mattered.

**Frozen recipe** (`docs/results/2026-08-24-freeze.md`): Muon; lr 8e-4 (1.716 / 1.707 / 1.693 / 1.935 across
3e-4 / 5e-4 / 8e-4 / 12e-4, 12e-4 destabilising the router); fp32 residual (bf16 cost +0.0058 > the 0.005
gate); full-attention Relation (win128 1.379 vs winctl 1.361 at matched materialized-path steps = +0.018, 3.5×
over the gate — long-range chunk lookups are load-bearing; no window kernel gets built); EMA 0.999; α 0.1;
β₂ 0.95; batch 1 × accum 4 at 16384; fused norm (68.1 KB/s at 4.31 GB); ATDC target N 5 → 6.5 (nsweep 4 / 8 /
10: 1.732 / 1.706 / 1.691 vs control 1.693 = tie). Data: mix A = `data/flagship_mix`, 10 GB, 20 sources at
exact budgets, sim excluded, per-domain val shards in `data/flagship_val/`; `--eval-spread`.

**Under review, pre-registered** (`docs/results/2026-08-28-lr-prereg.md`): the freeze's lr gate was a
throughput confound (12e-4 led at every matched step and ran at 40 % speed), and every lr number sits at half
the trunk's bytes per step. The signed horizon fit puts the trunk's lr\* at 2.25e-4; three 24-h points
(2.5e-4 / 4e-4 / 8e-4) are running, then a re-fit with `--target-tokens 2.53e10` (accum 2 costs 11 % of
throughput: 74.8 vs 84.0 KB/s at 16384, so the trunk's constant is ~42 KB/s and the 7-day horizon ~25 GB,
which is 2.8 passes over mix A — corrected 2026-08-29: the 2026-08-24 MoE section's "~4 epochs" was at the
freeze's throughput). Launch rule: within ±30 % of 3.3e-4 and the fit's β in [−0.55, −0.32]. Muon vs Muon-SW
(1.177 vs 1.180) was reopened 2026-08-26 — the gap is inside the ELR difference between the two — and re-runs
at matched ELR (`scripts/elr_optimizer_gate.sh`; `runs/elr_gate/`). Six QK-Norm arms (`--qk-norm` with τ_s
and λ re-gated) are queued. Seed noise on final val is 0.0003 (`ab2_muon_seed7`).

**Decided against, with the numbers**: JEPA on the byte encoder (minimal / EMA / SigReg 1.1943 / 1.1985 /
1.1871 vs the Muon control's 1.1773 at 2 h — all lost); the attention-main ablation (1.1696 at 2 h against
Relation's 1.1773 — knowledge only, Relation ships by the rule above); the hyper-connection spine (signed
2026-08-26, never gated; the same day's profile falsified both cost premises — frac n=4 was +0.56 GB, not
+0, and expand n=4 does not fit at 16384 — retired 2026-08-29 with JEPA, the attention control, the windowed
Relation and the bf16 residual; `docs/research/spine-2026-08-26.md` keeps the reading).

## Training pipeline: pre / mid / post (grilled and signed 2026-08-24; amended 08-25, 08-26, 08-28)

Facts and the reasoning substrate are laid down in pretraining and mid-training (Allen-Zhu & Li 2309.14316;
Front-Loading 2510.03264; PRISM 2603.17074: mid-training restructures > 90 % of the weights, RL touches ~5 %
and only works on a mid-trained model); post-training reweights what is already reachable (RLVR raises
pass@1 while the base keeps pass@k, 2504.13837; on-policy RL is KL-minimal and forgets least, 2509.04259).
So: facts → pre/mid, format → SFT, RL last, headroom-gated.

**Pre — the trunk.** The frozen config; `--schedule trunk` = warmup (10 % of the 7-day estimate) then a
constant lr, no decay; `--eval-spread`; `--snapshot-steps` keeps a weights-only `snap_<step>.pt` about daily
— the branch points; the daemon serves the EMA. Branch trigger (automatic): the first of {24-h val-bpb gain
< 0.003, day 7, Noah's "branch"}; at day 7 with the gain still ≥ 0.003, continue on **mix B** in 1-day
resumes, day 10 at most, then branch. Mix B = fresh 10 GB of A's composition (`build_mix --list flagship
--skip-after data/flagship_mix.meta.json`: the HF streams are file-ordered and unshuffled, so skipping A's
recorded per-source bytes replays past exactly A's documents).

**Before mid — latent feedback** (signed and built 2026-08-28, `docs/results/2026-08-28-latent-feedback-prereg.md`):
three 24-h arms on the trunk snapshot — control / chunk-level / byte-level continuations on the pre mix at
the trunk's constant lr with the paper's 75 / 22 / 3 pass mixture; gate: k=1 fused-prefill val ≤ control −
0.005 and k=0 ≤ control + 0.005, a tie within 0.005 → chunk; the serving bench under Soft decoding confirms;
mid's 2×2 starts from the winner; Soft decoding by default with Fused per request; post stages k=3
throughout. Open: multi-pass memory at 16384 (profiled in the gap; fallbacks `--feedback-window 8192`, then
`--feedback-detach`), Fused prefill and the multi-turn read (built with the serving bench).

**Mid — a 2×3 over mixture and decay, ~4.5 GPU-days** (re-signed 2026-08-26,
`docs/research/midtraining-2026-08-26.md`). Two branches fork off a trunk snapshot, each at constant lr to
80 % of its horizon, snapshotting there (`--snapshot-at 0.8`), then splitting into `--schedule branch` (decay
80 → 100 % to zero), `--schedule branch --branch-decay-frac 0.3` (2608.24814 App. F.1: at a decay window of
0.1 the weight-decayed run never overtook the unregularised one, at 0.3 it did and finished lower — the window
is a variable, not a constant) and `--schedule constant` (token-matched, no decay). Control = mix B; anneal =
mix C (the ANNEAL table in `mote/data/sources.py`); the extras are identical in both — 15 % of each branch:
`data/spec_plain:0.03:plain`, `data/sim_traces:0.03:fim`, `data/sim_long_plain:0.04:plain`,
`data/sft_local:0.05:plain` — so the A/B varies the web half alone. The sim regenerates before any of this
runs (`--p-fail` swept 5 / 15 / 30 and passed with the SAME value to `mote.sim.long` — its worlds come from the one
difficulty sampler since 2026-08-29 and had no failures before; `--parallel-frac`, `--swap-frac`, retrodiction;
`build_spec_docs --typo-frac`). The three 2606.16246 augmentations (`--aug-noise` / `--aug-r2l` /
`--aug-offset`) exist and stay OFF here; they are their own two-arm comparison afterwards. The old
`cooldown` (1−√t to 0.1× over the whole branch, 55 % of peak by the first quarter — Index-1.9B 2607.09885
measured that shape as worse than not curating at all; 2605.25698 names the conflict and measures +3.27 for
fixing it) is retired; the decay sits in the last 20 % and goes to zero. Decay is an axis because 2603.16127
finds decay at any stage costs post-SFT quality while 2607.12360 finds a normalised optimizer cannot
self-anneal; the EMA already buys 0.075–0.098 bpb over the raw weights. A fifth arm — the trunk snapshot's
EMA through the same SFT — is a floor, not a contestant.

*Gate: one decider, the rest guards.* Decider `proxy_track` (`mote.eval.proxy`: mean reciprocal rank of the
expert's next byte over held-out trajectories, unweighted — chosen 2026-08-26 by measuring which of a 12-metric library
reproduces a known quality ordering; `proxy_agree` is reported, not voted — corrected here 2026-08-30, the code has
decided on `proxy_track` since 08-26; 2605.18607 — cross-entropy ranks candidates at
Spearman 0.36, trajectory proxies at 0.81). Guards, all of which must hold: shared val bpb ≤ control + 0.005
within the same decay condition, and no regression on `needle_auto`, `false_fire_rate` or `recovery_rate` beyond
the two arms' combined standard error (2026-08-29, Noah's option C: the guards were single-item vetoes at 24 / 40 / 40
items; they are 144 / 120 / 120 with a sem each, under the decider's own rule)
(`mote.eval.recovery_probe`, 2608.20314). Reading EM, sim-QA EM, chat val bpb, identity / hold / concede and
the per-domain vals are reported and do not vote (at this scale exact match sits on its noise floor). Missing
numbers fail closed. Driver `scripts/mid_2x2.sh`; the winner's final checkpoint is the base for post-training.

**Post — SFT → the preference stage → RLVR last.** The probe grew a negative class on 2026-08-25:
`false_fire_rate` over neutral prompts that must draw neither the identity card nor a pushback (`overnight_dpo2`
scored identity 0.833 and false-fire 0.90 at once). **Round A** bakes off DPO / IPO / ORPO off a shared 30-min SFT
from `t3l_dense_8e-4`, with a second SFT at `--neutral-frac 0.15` as its own arm; **Round B** runs the winner and
runner-up on the 20k sim pairs against the real gate (`docs/research/dpo-rlvr-2026-08-25.md`).

1. **SFT-1** (format): init = the base; `sft_local` + identity 5 % + sim QA ~10 %; the tool-protocol traces
   moved to mid-training (2608.20314; 2607.12463), so SFT elicits the protocol rather than teaching it;
   difficulty selection (`mote.data.select_sft`, ~4 GPU-min per 127 MB) and the mid-run re-selection
   (`--reselect-every`, the trajectory rule of `docs/research/curriculum-2026-08-28.md`) is A/B-gated before it
   is the default; search data only once the reading gate passes (`docs/search.md`). lr: T2 {1e-4, 3e-4}. After
   it: identity ≥ 5/6, chat val, reading, sim-QA, needle, pass@1 / pass@64 on the sim tasks.
2. **Preference stage** (the correctness and prefs rounds merged 2026-08-25): one run over every preference
   pair that exists — the sim's 20k verifiable pairs plus the prefs store's pairs and marks; the objective from
   the bake-off. Always re-run from SFT-1, never stack (2606.19744: stacked objectives interact
   unpredictably; re-running invalidates RLVR-1, which is redone from the new reference). Prefs pairs are
   upweighted to a fixed 20–30 % of the gradient. Gate: 0 < pass@1 < 0.5 on held-out sim QA (RLVR-1 needs
   headroom); 1 epoch, lr 5e-7, β 0.1, SFT term on; guards identity / hold / concede, false_fire_rate, chat
   val. The old prefs gate (≥ 1000 rated / ≥ 150 Noah's) is suspended pending the objective; Noah's ≥ 150 own
   judgements stand regardless (`docs/prefs.md`).
3. **RLVR-1**, multi-turn actions in the sim (household, inventory, schedule): State2State-style tasks
   (2608.04934), reward 1 iff every goal predicate holds at the end (all-or-nothing — shaping applies to the
   advantages, never the reward), step budget k+2, `<|call|>sim: …<|result|>` through the one server hook, only
   the model's bytes carry loss. Algorithm (revised 2026-08-25): z = (r − median r)/(σ+ε) (MC-GRPO 2601.22582),
   δ = 2√C·(frac − 0.5) (MDP-GRPO 2606.06058 Eq. 4), ṽ = λ±·tanh(β_PT·v) with (0.8, 1.25, 2.0), A = 0.8 z̃ +
   0.2 δ̃, asymmetric KL (`--kl` / `--kl-high`); `--anchor-alpha 0 --pt-beta 0 --baseline mean` is the
   pre-revision ablation. KL to the post-preference reference, lr ~1e-6, on-policy rollouts at T = 1, prompts at
   the edge of competence, ~200 steps, pass@k on held-out tasks; start gate pass@1 < 0.5 and pass@64 − pass@1 ≥
   0.2; guards pass@64 never below the pre-RL model, identity / hold / concede, false_fire_rate, chat val, KL
   bounded. Budget ~20 % of the rung's compute (2607.16097), never taken from the trunk; pass@8 reported beside
   EM by every gate; `mote/eval/rl_taxonomy.py` bins the policy change; the self-proposal SFT-1 arm between
   `<|think|>` / `<|end_think|>` is queued against the pruned-expert traces. Post stages run with k=3 feedback
   passes throughout.

Served = the last stage that passed its gate; every stage checkpoint sits in the studio picker. Everything
in the build order exists (`docs/results/2026-08-24-pipeline-build.md`); what remains is operational: the
pre-launch queue → "launch" → the trunk → the latent-feedback arms → the branches → the gate → SFT-1 → the
preference stage → RLVR-1.

## The daemon: train + serve in one process (built 2026-08-23 … 08-28)

`Trainer` (`mote/train/train.py`) is a generator yielding at every accumulation slice; `JobQueue`
(`mote/serve/jobs.py`) drives it one slice at a time under a GPU gate. The queue is sequential and persists
to `.mote/jobs.json`: a job that was running when the process died resumes in front once for free, a second
death before it logs a step holds it and halts the queue (`DEATHS_BEFORE_HALT`); a CUDA OOM is retried from
the front with a growing delay once free + reserved memory covers the failed run's peak plus a margin, three
times per lineage, and a retry the card could never satisfy is held with the reason; `GET
/api/training/queue` says why each queued job is waiting. An EMA shadow (0.999; zero-started and bias-corrected at every read since 2026-08-30, as is the trainer's `--eval-ema` — `hnet.ema_scale`) follows the run and is synced
into the served engine every 100 steps — only for a job **on the air**. Jobs run in-process and nothing
reloads modules: `run.json` records the commit the worker loaded.

**Serving policy** (signed 2026-08-25): the served model is the **pin** (`.mote/config.json`'s `checkpoint`)
unless a job is on the air (`mote train start … --serve`, `mote train serve`, the sheet's "Put on the air");
its EMA answers while it runs and its final checkpoint is pinned when it finishes. Screening arms never touch
what is served (a queue of 15 arms used to mean 15 silent swaps). A manual load wins: it re-pins and takes the
running job off the air with a notice. **Device**: while any job runs the engine lives on the CPU
(`Engine.moved("cpu")` — weights, prefix-store budget, tools and provenance; no arena, graphs, pool or gate;
measured: 32M 1.5 KB prefill 0.68 s then ~50 B/s, 96M ~45 B/s, a cold 4.6 KB read 10 s; continuations stay
cheap through the prefix store); when the queue drains it moves back with arena + graphs and one warm-up
(~12 s). The reply says why it waits (`{"type":"waiting","on":"prefill"|"swap"}`). Decode runs on its own
high-priority CUDA stream; the gate survives only for construction, swaps, loads, arena growth and rewarm.

**Boot without a GPU** (2026-08-28): a daemon launched `--device cuda` waits for CUDA up to 180 s before
anything in the process asks torch for it (a fresh interpreter per probe — the in-process answer is cached
for the process's life); past that it serves on the CPU with the queue paused, re-probes every 30 s and
restarts itself when the device appears. `Trainer` refuses to fall back to the CPU. Usable memory is free +
the whole reservation (the engine leaves the card when a job starts). The CPU reference path runs Mamba-3 in
windows of 256 positions with the state carried (+1.2 GB and 2.7 s for a 4096-byte flagship read, from +6.2
GB and 18.6 s); prompts are read in windows of `prefill_window` = 4096 (peak 865 → 252 MiB at 16384,
identical boundaries).

**Memory**: the ceiling is **6.2 GB with the screen locked** (the desktop takes 1.5–1.9 GB;
`docs/results/2026-08-24-evening-gates.md`; the 2026-08-28 chunk-rate sweep `scripts/gate_sweep.py` reads
against it: ≤ 4.71 GB at the frozen recipe across chunk rates 2.1–4.0). fp8 linears are parked until the 1B.
Serving allocations live in their own `torch.cuda.MemPool`; the unit sets
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## Serving root (signed and built 2026-08-24; `docs/results/2026-08-24-serving-root.md`)

From FreeToken (2608.16157; re-read 2026-08-28 — nothing else in it transfers, its expert cache is for
expert pools 100–1000× Mote's size): checkpoints of recurrent state at the semantic boundaries where a
harness edits context, kept apart from the append-only per-position cache, and every routing-dependent
decision living on the device as data inside a captured graph.

* **Arena** (`mote/model/arena.py`): the Relation {P2, Ĩ} per-chunk cache — the only state that grows with
  the context — is one static `[layers, 2, H, capacity, dh]` tensor, sized from the run's measured chunk rate,
  written in place and read as views; grows ×2 on demand.
* **Store** (`mote/infer/prefix_cache.py`): branches own arena rows as CPU pages of 256 chunks (full pages
  shared when a regenerate or an edit forks a branch) plus anchors of everything else (Mamba-3 / routing /
  dechunk states, logits, now `h_prev` / `z_prev` for Soft decoding: ~3 MB on the flagship). Anchors at card /
  prompt end / reply end; tool-result boundaries reserved for search; the card anchor pinned; LRU by branch;
  `MOTE_PREFIX_CACHE_MB` (1024). The arena stays hot between turns of one conversation. Every EMA swap drops
  the anchors and `Engine.rewarm()` re-reads the conversations of the last 10 min (≤ 3).
* **One CUDA graph per byte** (`mote/infer/graph.py`): rand (parent) → IF ¬done: nucleus sample by inverse CDF
  on the device → IF ¬done: encoder → router → IF boundary (a conditional node keyed on the device router bit,
  `CUDAGraph.begin_capture_to_if_node`, torch 2.13 — corrected 2026-08-29: the 2026-08-24 daemon text said
  `torch.cond` on 2.12) → main over the arena at a static bucket width → dechunk → decoder → head; `done`
  freezes the step exactly; the host drains rings every K=8 replays.
* **Measured** (35M, CPU, 40 turns): 92.5 % reuse, 0 moved cuts, 0/40 replies differ, warm 63 ms mean /
  531 ms worst vs 921 ms cold. The serving bench adopts worst-turn TTFT per workload as its availability
  metric beside decode rate; the CPU per-byte decode rate at the flagship shape — the desktop's state for
  weeks — has never been measured and is taken in the gap.
* Deferred: `/v1/chat/completions` fold stickiness; batched multi-stream decode for concurrent agents.

## Kernel and compile workstreams (signed 2026-08-24; `docs/results/2026-08-24-h100-probe.md`)

The H100 probe (324 KB/s at the frozen recipe = 4.8× the 4060 Ti; the 7-day trunk would cost ~135 credits)
fixed the roles: **the flagship trains locally; the H100 is for experimentation**. Two workstreams, kernel
first, then compile, each entering the running trunk by hot-swap at a daily snapshot once **exact + T1 smoke
+ ≥ 10 % faster local flagship step**; anything landing after trunk day 7 goes to the branches.
**FlashRelation v2**: head dim 96 tiled as 64+32, a one-pass FA2/3-style backward with fp32 atomic dq, the row
terms in Triton prologue/epilogue kernels, FA4's conditional rescaling and LPT order, `exp2`, fixed per-device
tile tables (`MOTE_DETERMINISTIC_RELATION=1` selects today's two-pass kernels); expected 1.7–2.2× on the kernel,
an estimated 30–40 % of the step. **Compile at root depth**: the three Triton autograd Functions as custom ops
→ compiled block stacks with dynamic T → static-shape routing/dechunk → CUDA graphs gated on measured memory;
serving stays eager. The whole-model `--compile` flag exists and was never measured; `--tf32` is a numerics
change to the fp32 residual path, screened as a 30-min pair before it is adopted within noise. FlexAttention's
FA4 backend, a CuTe DSL port, fp8, Gluon and the CUTLASS Inductor backends are rejected or H100-only.

## Mixture of experts (signed 2026-08-24; `docs/research/moe-2026-08-24.md`)

Top-k experts add parameters without activation memory, and activations are what cap the flagship at 16384
on 8 GB. Built in `mote/model/moe.py`: `MoESwiGLU` in the Relation blocks' FFN slot, stacked experts under
batched Muon, routers `lossfree` (DeepSeek-V3 bias + Moonlight gate scale + seq-level balance 1e-4) and `aux`;
three exact-equal execution paths; telemetry per layer per step; `mote.eval.moe_report`. The funnel differs
for MoE (2502.05172: the gain rides on a steeper data exponent, crossover 3.7 tokens/param): verdict = 12-h
arms at equal wall-clock. T2 on an L4 (E4 fits, E8 never); T3 read 2026-08-28 (`docs/results/2026-08-28-lr-prereg.md`):
E4 beats dense at every matched step but trails dense@4e-4 at wall-clock, so one E4 arm at 4e-4
(`runs/t3l_e4_4e-4`, 12 h) decides — pre-registered ≤ 1.0979 sends it to the flagship confirm.

## Signed, not yet run

* The three latent-feedback arms after the trunk (above); the recirculation grid on an idle queue
  (`scripts/recirc_sweep.py`; partial: three pairs at +0.02…+0.05 %, `docs/results/2026-08-28-recirc-sweep.md`);
  the QK-Norm arms; the E4 arm; the Muon vs Muon-SW pair at matched ELR.
* The `--tf32` pair, the eager / `--compile` twins, the augmentation pair, the FlashRelation v2 and compile
  workstreams.
* In the gap before the arms: GPU tests, the 100-step bitwise check of the housekeeping, the multi-pass memory
  profile, the CPU per-byte decode rate at the flagship shape.

## Open questions

* Memory split at 16384 once serving keeps its working set resident: what batch survives.
* Snapshot cadence, and what a "checkpoint" means when the weights never stop moving (votes reference
  snapshot ids).
* Memory-first training changes, evidence-gated: checkpointed or reversible outer layers, bf16/fp8 activation
  storage, fused EMA dechunk, optimizer state in pinned host memory, token-budget batching.

(Preemption granularity, serving from the EMA, online updates from votes and the process boundary were
answered by what shipped — slices, the EMA on the air, batch DPO, one process.)

## History — where the dated material went

* `docs/results/2026-08-23-fedora-day1.md` — day one on Fedora, measured, with the plan as written.
* `docs/results/2026-08-23-daemon-built.md` — the daemon as first built and dogfooded.
* `docs/results/2026-08-23-fedora-move.md` — the move runbook (was `docs/fedora.md`).
* `docs/research/reading-2026-08-23.md` — six papers of Aug 18–20 (none changed a decision; the looped-main
  arm, Forking Fast, Lévy Attention and FlashAttention-V are the keepers).
* `docs/research/recirculation-2026-08-28.md`, `docs/research/latent-feedback-2026-08-28.md` — the two readings
  behind the recirculation sweep and the latent-feedback arms.
* `docs/results/2026-08-29-housekeeping-prereg.md` — this rewrite's pre-registration.
