# Shape — one resident process that trains and serves

Root decision, 2026-08-23 (grilling): Mote's GPU is owned by **one resident process** that trains in
preemptible steps and serves from the same weights. No second copy of the model, no CPU fallback while a
training run holds the card, no separate "training job" and "studio server" fighting over 8 GB. The
studio, the prefix cache and the preference votes plug into that process. The architecture (bytes,
H-Net chunking, Mamba-3 outer, Relation main, Muon) stays.

Why: on one GPU with one user, the split into separate programs is what left the card idle while the
studio answered on the CPU at 40 B/s, and what forced training to batch 1 at 16384. The tape that backprop
needs (the activations of the byte-resolution outer layers: ~4.8 of the 6.34 GB at the flagship) is the
price of exact gradients and stays; everything around it is ours to reshape.

The design is **gated on measurements** taken on Fedora first (decided the same day): the daemon's
preemption granularity, memory split and snapshot cadence all depend on numbers we only have for WSL2.

## Serving contract (decided 2026-08-23, independent of the numbers)

**First byte within ~1 s while a training run is on; training yields.** A request preempts at the next
boundary and the run slows while the conversation is active. The daemon is designed around that boundary,
which is why deferred question 1 (step- or micro-batch-level preemption) is the first the step-time
measurement has to settle: at the flagship a full step is seconds, so the boundary will have to be a
gradient-accumulation slice, not a step.

## Day one on Fedora: measured 2026-08-23 — `docs/results/2026-08-23-fedora-day1.md`

All six ran (16384 profiles OOM beside the desktop — that itself is finding #1). Headlines: +11–12 %
step throughput vs WSL2; ~39 % of GPU time is elementwise/dtype traffic and the multi-byte head another
8–13 % with no benefit; kernel-vs-reference Δlogit 0.148 makes Input_States priority #1; the two-process
sharing baseline already meets the serving contract 40× over at 35M scale — the daemon's case is flagship
memory (one weight copy, no VRAM for two processes at 16384), not latency.

## Day one on Fedora: the measurements (as planned)

Same scripts, same configs as `docs/results/2026-08-23-chain.md`, so every number has a WSL2 twin.

1. **Step profile** — `python -m mote.train.profile_step` at: 35M local, 2048, batch 4, trained router
   (`--init-from runs/overnight/last.pt`); flagship, 16384, batch 1, `--ckpt-main`, `--chunk-bytes 6`.
   Report B/s, TFLOPS, MFU, peak GB next to 80.8 KB/s · 23.5 % and 42.4 KB/s · 43.7 % · 6.34 GB.
2. **Where the time goes** — an `nsys` trace of 20 steps at both configs: data loading, outer forward,
   main forward, backward, optimizer, launch gaps. Where the memory goes — `torch.cuda.memory._snapshot`
   at peak, grouped by layer, to confirm the activation-tape breakdown above.
3. **Step time at the flagship** — the unit of preemption a serving request would wait for.
4. **Serving on the kernels** — TTFB cold and warm, bytes/s, on the GPU with Triton (the Windows studio
   ran reference paths on the CPU: 2.1 s cold, 127 ms warm); `mote.eval.prefix_probe --device cuda`
   for the kernel-vs-reference rounding the cache now depends on.
5. **Sharing the card today** — serve while a training run is on (two processes): reply latency and the
   training slowdown. This is the baseline the resident process has to beat.
6. **Disk** — memmap read throughput of a shard on ext4 (the WSL2 shards came through 9P).

## Built 2026-08-23 (dogfooded on the 35M the same evening)

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

## Flagship config FROZEN (2026-08-24)

All gates read. **Muon** (1.177 vs Muon-SW 1.180); **lr 8e-4** (1.716/1.707/1.693/1.935 across
3e-4/5e-4/8e-4/12e-4 — clean peak, 12e-4 destabilized the router); **fp32 residual** (bf16 cost
+0.0058 > the 0.005 gate); **full-attention Relation** (win128 1.379 vs winctl 1.361 at matched
materialized-path steps = +0.018, 3.5x over the gate — long-range chunk lookups are load-bearing;
no window kernel gets built); head off; EMA 0.999; batch 1 x accum 4 @ 16384; fused norm
(68.1 KB/s @ 4.31 GB). Data: 10 GB FLAGSHIP mix + per-domain val shards (data/flagship_val/).
Launch gated on the pre-launch queue (JEPA round 1+2, attention ablation, seed-noise calibration)
and Noah's word.

## Training pipeline: pre / mid / post (grilled and signed 2026-08-24)

What each stage is, exactly, and what it may and may not do. The evidence behind the split: knowledge
and the reasoning substrate are laid down in pretraining and mid-training (Allen-Zhu & Li 2309.14316:
facts memorised without diverse forms stay unextractable "regardless of subsequent instruction
fine-tuning"; Front-Loading 2510.03264; PRISM 2603.17074: mid-training restructures > 90 % of the
weights, RL touches ~5 % and only works on a mid-trained model); post-training reweights what is already
reachable (RLVR raises pass@1 while the base keeps pass@k, 2504.13837; on-policy RL is KL-minimal and
forgets least, 2509.04259). So: facts → pre/mid, format → SFT, RL last, headroom-gated.

**Pre — the trunk.** The frozen config above; `--schedule trunk` = warmup (10 % of the 7-day estimate)
then constant lr 8e-4, no decay; `--eval-spread` (val windows over the whole mix — the default head-only
eval is the first source, see docs/results/2026-08-24-pipeline-build.md). `--snapshot-steps` keeps a weights-only `snap_<step>.pt` about daily —
the branch points. The daemon serves the EMA as today. Launch on Noah's word after the pre-launch queue.
Branch trigger (automatic): the first of {24-h val-bpb gain < 0.003, day 7, Noah's "branch"}. At day 7
with the gain still ≥ 0.003: continue on **mix B** in 1-day resumes, day 10 at most, then branch.
Data: mix A = the frozen 10 GB; **mix B** = fresh 10 GB of the same composition (`build_mix --list
flagship --skip-after data/flagship_mix.meta.json`: the HF streams are file-ordered and unshuffled, so
skipping A's recorded per-source bytes replays past exactly A's documents).

**Mid — two cooldown branches, ~1.4 GPU-days each.** `--init-from <trunk snapshot> --schedule cooldown`:
lr 8e-4 → 0.1× (inverse-sqrt) over the whole branch, no warmup, chunk target held at its final value, the
step horizon fixed at the first probe so a resume continues the decay. **Control** = mix B. **Anneal** =
**mix C** (8 GB fresh, the ANNEAL table in `mote/data/sources.py`: fineweb_edu 18 / dclm 8 / Ultra-FineWeb
rewrites 4 / fact-seeking 2 / simple-wiki 1 · SYNTH Q&A 10 / finephrase 7 / Cosmopedia 7 · finemath 15 ·
code 7 + 1 long · multilingual 6 · long documents 7, of the branch's ~93 %) plus plain-LM extras via
`--mix …:plain`: sim narrative+QA 4 % (`build_local`), chat (`sft_local`, no mask) 3 %, identity 0.2 %.
Gate: both branches get the identical 60-min SFT, then reading EM/F1, sim-QA EM (new probe),
identity/hold/concede, needle and chat val decide; guard = overall val bpb ≤ control + 0.005, per-domain
vals recorded. Anneal ships if it wins ≥ 2 of {reading, sim-QA, chat val} with no guard tripped, else
control. The winner's final checkpoint is the **flagship base**; the pair is the flagship's own
mid-training A/B (docs/results).

**Post — SFT → DPO stages → RLVR last.**
1. **SFT-1** (format): init = the flagship base; `sft_local` + identity 5 % + sim QA ~10 % + tool-protocol
   traces if built (else SFT-2 adds them); search data only once the reading gate (docs/search.md, ≥ 50 %
   EM after a small QA SFT) passes on the flagship base. lr: T2 {1e-4, 3e-4} (overnight_sft2 used AdamW
   3e-4). After it: identity ≥ 5/6, chat val, reading, sim-QA, needle, and pass@1 / pass@64 on the sim
   tasks — the RL headroom numbers.
2. **Correctness DPO**: the sim's 20k verifiable pairs; gate pass@1 > 0 on held-out sim QA; 1 epoch,
   lr 5e-7, β 0.1, SFT term on; guards identity/hold/concede + chat val (no regression).
3. **Prefs DPO**: the docs/prefs.md gate unchanged (≥ 1000 rated, ≥ 150 Noah's); skipped while unmet.
4. **RLVR-1, multi-turn actions in the sim** (household, inventory, schedule — kinship has no agent
   actions). Tasks State2State-style (2608.04934): k scripted actions from a seeded world → goal =
   predicates over the reached state (holdings, locations, bookings) rendered in the locale; reward 1 iff
   all hold at the end (fraction logged), step budget k+2, an illegal action renders as "nothing
   happened". Protocol: `<|call|>` = 262 and `<|result|>` = 263 with the tool named in the bytes
   (`<|call|>sim: take candle<|result|>…`; search is `<|call|>search: …`, docs/search.md), one server hook
   for every tool, only the model's bytes carry loss. Cold start = SFT on expert traces from per-domain
   planners. Algorithm: GRPO-style outcome reward, G = 8, group-normalised advantage, KL β to the post-DPO
   reference, lr ~1e-6, on-policy rollouts through the graph decoder at T = 1, prompts kept at the edge of
   competence (group pass rate in (0, 1)), ~200 steps, pass@k on a held-out task set every N steps. Start
   gate: pass@1 < 0.5 and pass@64 − pass@1 ≥ 0.2. Guards: pass@64 never below the pre-RL model;
   identity/hold/concede and chat val no regression; KL bounded.

Served = the last stage that passed its gate; every stage checkpoint sits in the studio picker.

Build order (all local): trainer schedules + snapshots + `:plain` mixes + the anneal/skip builder +
`build_local` (built 2026-08-24, `tests/test_pipeline_stages.py`; live at the next daemon restart at an
arm boundary) → mixes B and C (building 2026-08-24 on CPU; `data/sim_plain` + `data/sim_sft` built) →
sim-QA probe + branch funnel (built 2026-08-24: `mote/eval/sim_probe.py` held-out worlds, EM + pass@k;
`mote/eval/val_bpb.py` shared + per-domain val; `mote/eval/branch_gate.py` submits the identical SFT per
branch to the daemon, measures, applies the verdict rule, writes docs/results; the 35M scores 0 EM on
held-out sim QA — the baseline) → `<|call|>/<|result|>` ids + the shared tool hook (built 2026-08-24:
`mote/tokenizer.py`, engine hook + scripted-id tests) → env tasks, verifier, pruned experts, expert traces
(built 2026-08-24: `mote/sim/tasks.py`, `data/sim_traces` 19.6 k traces) → `mote/train/rlvr.py` as the
daemon job type `rlvr` (built 2026-08-24). Everything in the list exists; what each piece has and has not
been measured on is in `docs/results/2026-08-24-pipeline-build.md`. What remains is operational: mixes B/C
finish → the pre-launch queue → "launch" → trunk → branches → gate → SFT-1 → DPO stages → RLVR-1.

## Kernel and compile workstreams (grilled and signed 2026-08-24; numbers in docs/results/2026-08-24-h100-probe.md)

The H100 probe (324 KB/s at the frozen recipe = 4.8× the 4060 Ti, 8–13 % MFU; the 7-day trunk would cost
~135 credits) fixed the roles: **the flagship trains locally; the H100 is for experimentation** (arms,
probes, kernel tuning). FlashAttention-4 was read for what transfers: its Blackwell choreography (TMEM,
2-CTA MMA, async pipelines) does not; its algorithms do (conditional softmax rescaling, LPT block order,
software exp where MUFU binds — Hopper/Blackwell, never Ada). FlexAttention's FA4 backend runs Relation's
exact score function correctly but is sm90-only and slower than our kernel on the 4060 Ti; rejected, as is
a CuTe DSL port. Two workstreams, **kernel first, then compile**, each entering the running trunk by
hot-swap (daemon restart → auto-resume from the last checkpoint right after a daily snapshot, logged in
docs/results as a mid-run revision) once **exact + T1 smoke + ≥ 10 % faster local flagship step**; anything
landing after trunk day 7 goes to the branches and post-training.

**FlashRelation v2** (`mote/model/flash_relation.py`, exact to the materialized reference within bf16;
training + prefill only — decode T=1 uses the materialized path below `FLASH_MIN_T`): head dim 96 tiled as
64+32 (today it pads to 128: 25 % extra MMA and shared memory in every dot); a one-pass FA2/3-style
backward parallel over KV blocks with fp32 atomic `dq` (5 tile-dots instead of 7, the exp work once
instead of twice) and the PyTorch row terms (`da`, `delta`, `dλ`, self-gate corrections) moved into Triton
prologue/epilogue kernels; FA4's conditional rescaling and LPT order; `exp2`; fixed per-device tile tables
(no runtime autotune, so prefill latency stays flat — Ada measured at local queue ends, Hopper in one
autonomous ~0.3-credit H100 session once local tests pass). `MOTE_DETERMINISTIC_RELATION=1` selects
today's two-pass kernels. Gates: `tests/test_flash_relation.py` + one-pass-vs-two-pass + model-level
`USE_FLASH` on/off equality + T1. Expected 1.7–2.2× on the kernel, which is an estimated 30–40 % of the
local flagship step (extrapolated; the clean profile runs when the local queue empties).

**Compile at root depth** (the elementwise/precision glue is ~39 % of local GPU time and ~30 % + a 9.4 %
norm backward + a launch gap on the H100): register the three Triton custom autograd Functions
(flash_relation, mamba3, fused norm) as custom ops so nothing graph-breaks → compile the block stacks with
dynamic T (bucketed chunk counts; `cache_size_limit` guarded so Dynamo never falls back silently) →
routing/dechunk rewritten in static-shape ops (same chunks selected, padded to the bucket; the equality
test is the gate; the dechunk's cross-block EMA carry is already a closed form, 591e89f) → CUDA graphs
gated on measured memory beside the resident serving arena. Serving stays eager; only the trainer's
callable is compiled. The whole-model `--compile` flag exists but was never measured: eager/`--compile`
10-min twins are queued after `ab3_jepa_sig` and inform the plan.

Folded in 2026-08-24 evening after checking what torch 2.13 actually ships (each a measured flag, dropped
if it doesn't pay): Inductor **graph partition** (on by default; `reduce-overhead` splits around the chunk
layer's `.item()` and cudagraphs each partition per bucket — replaces hand-rolled bucket graphs),
`torch.compiler.nested_compile_region` on `Block.forward` (one compile shared by the 12 + 6 blocks),
`save_cache_artifacts`/`load_cache_artifacts` across daemon restarts (a hot-swap must not re-pay compile
time), compiled autograd, combo kernels, and DebugMode tensor hashing as the compile-vs-eager exactness
tool. Also measured under the kernel gate: `F.rms_norm`'s fused CUDA kernel against mamba_ssm's norm
(whose backward is 9.4 % of the H100 step), and batching same-shape Muon matrices into one `bmm` per
Newton–Schulz step (`torch.optim.Muon` is the same math as ours with no batching — nothing to gain there).
**TF32** for the fp32 residual projection is a numerics change to the frozen fp32-residual path: screened
as a 30-min arm pair (`--tf32` vs a fresh control) before launch, adopted only within noise. fp8, the
CuTeDSL/CUTLASS Inductor backends and Gluon are 1B / H100-only material.

## Daemon: serving beside training (grilled and signed 2026-08-24 evening)

Today the studio's GPU gate is one lock: a reply holds it for its whole duration and training takes it
per accumulation slice (~0.35 s at the flagship), so a chat pauses the run for the length of the reply
and a request waits up to one slice. Signed replacement: **decode runs on its own high-priority CUDA
stream concurrently with the training slice** — a decode kernel waits only for training thread-blocks to
retire, training loses ~nothing. The gate survives only for model construction, EMA swaps, checkpoint
loads, arena growth and rewarm, each a short hold at a slice boundary with the serving stream drained.
Serving allocations (arena, anchors, decode graphs) live in their own `torch.cuda.MemPool` so training's
churn cannot fragment them; the service unit sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
The decode graph's conditional nodes move to `torch.cond` capture (torch 2.12) if the per-byte replay time
holds. Gates: reply equality with today's path; under a running training slice, first byte < 1 s (the
existing contract) and p95 per-byte decode ≤ 2× idle; trainer + serving peak ≤ 7.2 GB on the flagship
preset (8.2 GB − the desktop's 0.7 − margin), measured in an idle window. Lands before launch if it passes,
else at the first swap restart (a daemon restart; the trunk auto-resumes). A green-context SM partition
(`torch.cuda.GreenContext`, works on cu126, even SM counts on Ada, no forward-progress guarantee) stays a
measured later option (`--serve-sms k`) — hard isolation at a permanent k/34 training cost.

Also settled the same evening: **fp8** linears parked until the 1B (`_scaled_mm` runs on the 4060 Ti;
tensorwise e4m3 = 3.8 % relative GEMM error; GEMMs are 13–22 % of the step at d = 768/512, where torchao's
own guidance says the cast overhead can make it a loss — the six-shape measurement lives in
docs/results); **`F.rms_norm`**'s fused kernel replaces mamba_ssm's norm if the bench wins, under the
kernel gate; CuTeDSL / CUTLASS Inductor backends and Gluon are H100-only one-flag experiments once the
compile workstream runs there.

## Serving root (grilled and signed 2026-08-24, BUILT the same day; results in docs/results/2026-08-24-serving-root.md)

Reading FreeToken (2608.16157, edge MoE serving) settled two things at once. The transferable part of
that paper is (a) checkpoints of recurrent state at the semantic boundaries where a harness edits
context, kept apart from the append-only per-position cache, and (b) every routing-dependent decision
living on the device as data inside a captured graph. Applied to Mote:

- **Arena** (`mote/model/arena.py`): the Relation {P2, I~} per-chunk cache — the only state that grows
  with the context — is one static `[layers, 2, H, capacity, dh]` tensor. Prefill and continuation
  write rows `[n, n+T)` in place and read views; `flash_relation` takes the head stride so no copy is
  made. Capacity defaults to max_seq_len/4 rows (147 MB on the flagship) and grows ×2 on demand.
- **Store** (`mote/serve/prefix_cache.py`): branches (one per linear history) own arena rows as CPU
  pages of 256 chunks — full pages are immutable and shared when a regenerate or an edit forks a
  branch — plus anchors of everything else (Mamba-3/routing/dechunk states, logits: ~3 MB on the
  flagship instead of the ~108 MB a full snapshot cost). Anchors at card / prompt end / reply end;
  tool-result boundaries are reserved for search. The card anchor is pinned; eviction drops whole
  LRU branches. **The arena stays hot** between turns of the same conversation (zero copies on
  continue; one copy up on a switch) — gated on the flagship memory measurement, fallback flag flushes.
- **One CUDA graph per byte** (`mote/serve/graph.py`): rand (parent) → IF ¬done: nucleus sample by
  inverse CDF on the device → IF ¬done: encoder → router → **IF boundary** (a conditional node keyed on
  the device router bit, `CUDAGraph.begin_capture_to_if_node`, torch 2.13) → main over the arena at a
  static bucket width, masked past S → dechunk → decoder → head. `done |= stop id | max_bytes` freezes
  the step exactly; the host drains rings every K=8 replays (one sync per 8 bytes). Graph mode is for
  models without a multi-byte head (the flagship); the 35M's speculative rounds stay eager.
  Spike facts: RNG ops are refused inside a conditional body (hence u in the parent); 0-d tensor
  indexing `.item()`s (hence scatter/index_copy); the first-capture warm-up must restore every
  buffer it touches and may only write arena rows ≥ n.
- **Swap + re-warm**: every EMA sync still drops all anchors (states depend on the weights) and then
  `Engine.rewarm()` re-reads the conversations used in the last 10 min (≤3), so the next message is
  warm; ~one prefill per branch per swap.
- **Measured (35M, CPU, 40 turns)**: 92.5 % reuse, 0 moved cuts, 0/40 replies differ, warm 63 ms mean
  / 531 ms worst (a fold) vs 921 ms cold; 3 arena rows hydrated in 40 turns. Per-byte flagship timing
  and the memory gate wait for an idle GPU (the pre-launch arms hold it).
- **Deferred, explicitly**: `/v1/chat/completions` fold stickiness (still reuses only the card after
  the window fills); batched multi-stream decode for concurrent agents.
- Superseded here: the 2026-08-24 morning plan of two host-picked graphs with host sampling. The
  toy spike had measured 10.8/23.5 µs per replay for one graph + IF node against 1.4 ms/byte for two
  host-picked graphs while a trainer shared the GPU (under the daemon's gate the trainer is paused,
  so the uncontended gap is one sync per byte).

## Questions the numbers decide (deferred, in dependency order)

1. Preemption: step-level (a reply waits one step) or micro-batch-level (gradient accumulation slices
   with serving slots between them)? Needs 3.
2. Memory split: training state (weights, master copy, momentum, tape) vs serving activations at 16384;
   what batch size survives once serving keeps its working set resident? Needs 1–2.
3. Serve from the training weights (bf16 copy the trainer already keeps) or from an EMA/snapshot? Needs 4
   (quality drift) and a val-bpb check of EMA vs raw.
4. Snapshot cadence and what a "checkpoint" means when the weights never stop moving (the challenger is a
   snapshot; votes reference snapshot ids).
5. Online updates from votes in v1, or batch DPO only (docs/prefs.md gate)? Needs the step cost and the
   disagreement data.
6. Process boundary: one Python process with a scheduler thread, or two processes with CUDA MPS / IPC
   sharing the weights. Needs 5 (the baseline) and 1.
7. Memory-first training changes, evidence-gated by 2: checkpointed or reversible outer layers, bf16/FP8
   activation storage, fused EMA dechunk, optimizer state in pinned host memory, token-budget batching.

## Reading 2026-08-23 (six papers of Aug 18-20, checked against this road before the freeze)

None changed a decision. The keepers: **2608.18222** (Think Shallow, Solve Deep) is the recipe for the
eventual looped-main arm — a weight-tied main network trained with a terminal fixed-point penalty
(headline weight 0.05, useful range 0.005–0.2, "the slowest rate that still settles"), per-step
displacement + top-8 Benettin Lyapunov exponents logged from day one, depth-safety by displacement-sum
vs decoder margin; their own scale result (the loss alone collapses Huginn-3.5B to a constant answer;
only latent anchoring survives) is why it does NOT touch the flagship. **2608.19611** (Forking Fast):
uncertainty of sampled generations converges and resampling can be smoothed cheaply — methodology for
the prefs loop and any temperature>0 probe. **2608.19171** (Lévy Attention): a mixer that emits a
closed-form trust signal for free — a future honesty-signal exploration next to Relation's
exchange-mass telemetry. **2608.18656** (FlashAttention-V): CPU-vector attention kernels — a "Mote on
edge" track. 2608.19331 / 2608.18808: theory, no bearing.

## Out of scope until the above

Looped main network, test-time-training memory layers instead of a window, any change to the
architecture's math. Those are a research sweep (2026 sources first), not a systems decision.
