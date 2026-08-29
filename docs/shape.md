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

## ELR is the coordinate every norm-control gate is read on (signed 2026-08-26)

`η^eff = η/‖W‖_F`, logged by every run (`w_norm`, `elr`, `rms_per_entry`, `rms_spread`). 2608.24814
measures that LR and parameter norm govern loss dynamics through their ratio and not independently, and
three things were then measured on Mote's own checkpoints (docs/research/elr-2026-08-26.md):

* the trunk is at the weight-decay norm equilibrium — ‖W‖ ∝ lr^0.478 across the three 12-h arms — so **lr
  and weight decay are one axis**, ELR ∝ √(lr·λ), and the 4e-4 → 16e-4 sweep spanned 2.06× in ELR, not 4×;
* entrywise RMS ≈ 0.66·√(lr/λ) and is flat across matrix shapes, which makes the relative step
  0.30·√(lr·λ) shape-independent. **That** is what transfers 512 → 768, not ELR, which falls as √(mn);
* the Muon vs Muon-SW gate that decided the freeze's optimizer ran at 0.914× ELR on one side. Muon-SW
  differs from Muon in one line — the decay — so it is a norm-control variant, and the ELR gap explains
  between 0.000 and 0.0066 bpb of a 0.00272 bpb effect. Precise (seed noise 0.00025 bpb) and
  unattributable. **Reopened**: `scripts/elr_optimizer_gate.sh` re-runs it at matched ELR, per matrix.

Two standing rules follow. Any comparison that changes an optimizer, a weight decay, or a norm-control
mechanism is reported in ELR, not in nominal lr. And every run carries `--norm-guard` (default `stop`):
if ‖W‖_F falls while the lr is flat — the `lr_sweep_12e-4` signature, whose norm ended below a 0.67×-lr
arm's — the run stops gracefully and **the whole queue halts** until `mote train release`.

## Mote-96M config FROZEN (2026-08-24)

All gates read. **Muon** (1.177 vs Muon-SW 1.180 — *reopened 2026-08-26: that gap is inside the ELR
difference between the two, see above*); **lr 8e-4** (1.716/1.707/1.693/1.935 across
3e-4/5e-4/8e-4/12e-4 — clean peak, 12e-4 destabilized the router); **fp32 residual** (bf16 cost
+0.0058 > the 0.005 gate); **full-attention Relation** (win128 1.379 vs winctl 1.361 at matched
materialized-path steps = +0.018, 3.5x over the gate — long-range chunk lookups are load-bearing;
no window kernel gets built); head off; EMA 0.999; batch 1 x accum 4 @ 16384; fused norm
(68.1 KB/s @ 4.31 GB). Data: 10 GB FLAGSHIP mix + per-domain val shards (data/flagship_val/).
Launch gated on the pre-launch queue (JEPA round 1+2, attention ablation, seed-noise calibration)
and Noah's word.

## Model names (2026-08-26)

Presets are named by their parameter count: **Mote-1M / Mote-13M / Mote-35M / Mote-96M / Mote-138M**
(`mote/config.py`). Role names rotted — `local` runs on the L4 too, and `flagship` stopped being the
flagship the moment Mote-138M existed — so `smoke`/`pilot`/`local`/`flagship` survive only as
aliases, because ten queued jobs carry them in argv. MoE variants are `Mote-<total>M-A<activated>M`.
Runs go under `runs/<model>/<experiment>-<arm>`; the existing flat directories stay as dated records.

## The hyper-connection spine (signed 2026-08-26)

One n-stream residual at **byte** resolution, seven sites: the three encoder sublayers, the whole
chunk stage, the three decoder sublayers. The main network keeps its own plain residual inside —
different width, different resolution — so the spine is 7 sites deep whatever the main stack does,
and taking Mote-96M to Mote-138M by depth does not reopen the stability question.

`H_res = J + H_disp` with `‖H_disp‖₂ ≤ 1` (sHC 2603.20896), not the Birkhoff polytope: mHC's
constraint is an affine part (which conserves the mean, and is what you want) intersected with
non-negativity (which buys the norm bound, and is what costs you — four groups measure identity
degeneration, spectral stalling and one dominant stream). `B_n ⊂ S_n`, so nothing is given up.
`--spine-project` makes every variant in the literature a control arm. `residual_proj` is subsumed
by the chunk-stage site's `H_post`, which also retires the model's only zero-norm parameter.

Cost at Mote-138M: **frac n=4** +86 K params and **+0 GB**; **expand n=4** +244 K and +0.66 GB
(~5.70 of 8.0). The risk is not memory — 7 of 42 sublayers leave the fused `rms_norm_fn` path and
carry 42 % of residual elements, which is what the fused spine kernel is for.

Gate: `scripts/spine_gate.sh`, ELR-matched on shared parameters. Full reading and the three things
measurement changed: `docs/research/spine-2026-08-26.md`.

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

**Mid — a 2x2 over mixture and decay, ~3.6 GPU-days** (re-signed 2026-08-26,
docs/research/midtraining-2026-08-26.md; the 2026 reading is tabulated there and the numbers below come
from it). Two branches fork off a trunk snapshot, each running `--init-from <snap>` at constant lr 8e-4 to
80 % of its horizon, snapshotting there (`--snapshot-at 0.8`), and then splitting:

```
                         +-- --schedule branch,   decay 80->100 %  -> *_decayed
  constant 8e-4 to 80 % -+-- --schedule branch --branch-decay-frac 0.3  -> *_decayed30
                         +-- --schedule constant, 80->100 %        -> *_constant  (token-matched)
```

**The decay window is a third arm, not a constant** (added 2026-08-26, docs/research/elr-2026-08-26.md).
2608.24814 App. F.1 held the peak lr and the budget fixed and varied only the WSD decay ratio: at 0.1 the
weight-decayed run never overtook the unregularised one, at 0.3 it did and finished lower. Same runs,
opposite verdict — because the gain is acquired early and only becomes visible once the low-ELR phase has
run long enough for the accumulated optimisation noise to be forgotten. `BRANCH_DECAY_FRAC = 0.2` sits
between the two, so the 2x2 becomes a 2x3 (~0.9 extra GPU-day) rather than resting on a window that the
one controlled experiment on the subject says can reverse the answer.

**Control** = mix B. **Anneal** = mix C (the ANNEAL table in `mote/data/sources.py`). **The extras are
identical in both** — that is the point. Until 2026-08-26 only the anneal carried sim/chat/identity, and
all three of the gate's deciders were then won by data inclusion rather than by the mixture, so the
experiment could not fail. The A/B now varies the web half alone. Extras, 15 % of each branch:
`data/spec_plain:0.03:plain` (identity as documents), `data/sim_traces:0.03:fim` (the tool protocol,
fill-in-the-middle around its own `<|call|>`/`<|result|>` boundaries), `data/sim_long_plain:0.04:plain`,
`data/sft_local:0.05:plain`. The old identity Q&A slot is gone from the mix: the card recited as an answer
is what produced `identity_recite_rate` 0.70 before DPO ever ran.

*The sim regenerates before any of this runs.* Signed 2026-08-26 after measuring that no action in any
narrative could fail, which made 99.7 % of every tool result a restatement of its call and left the
environment's three refusal strings absent from all 20,000 expert traces. `mote.sim.generate --p-fail`
(rate swept 5/15/30 against the sim probe), `--parallel-frac` and `--swap-frac` (a share of worlds in all
three locales, LINK-style lexical substitution on the English rest), plus retrodiction in all three acting
domains and `mote.sim.long` for the dependency-dense documents. `build_spec_docs --typo-frac` corrupts a
share of the identity documents, which is 2606.16246's answer to the repetition the 3 % share creates.

*Three training-time augmentations exist and are OFF here.* `--aug-noise` / `--aug-r2l` / `--aug-offset`
(2606.16246, all measured wins at 150M) are deliberately excluded from the 2x2 so its verdict is
attributable to the data changes alone; they are their own two-arm comparison afterwards, which costs
nothing because the shards are already built and they are runtime flags.

*Why the schedule changed.* The old `cooldown` decayed to 0.1x over the whole branch as `1-sqrt(t)` — a
concave curve at 55 % of peak by the first quarter, exactly when mix C's distribution shift arrives.
Index-1.9B (2607.09885 §6.4-6.5) measured that configuration as *worse than not curating at all*: the
schedule alone is worth nothing at 0.1B, what pays is decay combined with a data-quality raise, and the
combination only worked under WSD because "the cosine tail leaves too little learning rate" to adapt.
2605.25698 names the same conflict formally and measures +3.27 on a 600M dense model for fixing it. The
decay now sits in the last 20 % and goes to **zero** (`--min-lr-ratio 0`).

*Why the decay is an axis and not an assumption.* 2603.16127 finds decay at **any** stage costs post-SFT
quality at 1B and 8B, while 2607.12360 finds a normalised optimizer like Muon cannot self-anneal and needs
it for loss. Both can hold — decay lowers loss by entering a sharper minimum, and the sharpness is what
costs the SFT — and until now nothing in the pipeline could tell them apart, because both branches were
cooldowns. Mote's own EMA already buys 0.075-0.098 bpb over the raw weights at constant lr, which is most
of what a cooldown is for. The 2x2 asks the question. A fifth arm, the trunk snapshot's EMA through the
same SFT, is a **floor**: it carries none of the extras, so its capability numbers are a lower bound and
not a contestant.

**Gate: one decider, the rest guards.** Decider: `proxy_agree` from `mote.eval.proxy` — inverse-frequency
weighted top-1 byte agreement with held-out expert trajectories (2605.18607: cross-entropy ranks candidates
at Spearman 0.36, trajectory proxies at 0.81, and "a model which cannot solve a problem can still track
the CoT written by an expert"). Guards, all of which must hold: shared val bpb <= control + 0.005 **within
the same decay condition**, and no regression on `needle_auto`, `false_fire_rate` or `recovery_rate`
(`mote.eval.recovery_probe`: does the model try something else after the environment refuses — a mixture
that teaches the world but not the response to it is not ready for RLVR-1, 2608.20314). Reading EM,
sim-QA EM, chat val bpb, identity/hold/concede and the per-domain vals are reported and do not vote — at
this scale exact match sits on its noise floor (docs/search.md records a flat 0 on reading at 35M), and a
metric that cannot discriminate should not cast a ballot. Missing numbers fail closed. `needle` was
measured and then ignored by the old verdict while the same reweighting cut the long-document share
10.0 % -> 8.6 %; that share is back to 10.5 % and the guard is real. Driver: `scripts/mid_2x2.sh`. The
winner's final checkpoint is the **flagship base**.

**Post — SFT → DPO stages → RLVR last.**

*Additions signed 2026-08-25 (docs/research/dpo-rlvr-2026-08-25.md).* The probe grew a **negative class**:
`false_fire_rate` over neutral prompts that must draw neither the identity card nor a pushback template, a
shipping guard on every stage below. `overnight_dpo2` scored `identity_acc` 0.833 and `false_fire_rate` **0.90**
at the same time — the three original scores only ever reward a behaviour, so a model that recites its card at
every prompt scored full marks. Before any stage runs, **Round A** bakes off DPO / IPO / ORPO (all carrying the
new negative-class and tie pairs) off a shared 30-min SFT from `t3l_dense_8e-4`, with a second SFT at
`--neutral-frac 0.15` as its own arm — 70% of the false firing is in the SFT checkpoint before DPO ever runs, so
the SFT half is measured too. **Round B** runs the winner and runner-up on the 20k sim pairs against the real
gate. SFT-1 additionally gets difficulty selection (`mote.data.select_sft`, ~4 GPU-min on the 4060 Ti — not free; a mid-run re-selection (`--reselect-every`, trajectory rule) was built 2026-08-28 and is A/B-gated before it becomes the default — docs/research/curriculum-2026-08-28.md, which also holds RLVR-1's signed PATH-style sampler spec —
the earlier "CPU only" estimate was wrong) and On-Policy Replay between stages (`mote.data.replay`), which turns
the no-regression guards from detectors into a mechanism.
1. **SFT-1** (format): init = the flagship base; `sft_local` + identity 5 % + sim QA ~10 %. The
   tool-protocol traces moved to mid-training on 2026-08-26 (2608.20314; 2607.12463 found the
   fill-in-the-middle bias survives post-training while agentic post-training alone erodes non-agent
   ability), so SFT-1 elicits the protocol rather than teaching it. Identity likewise: the base now
   arrives having read documents *about* Mote, and the 5 % of Q&A is elicitation on top of that prior —
   2605.02087's finding is that demonstrations underspecify the generalisation, which is precisely the
   0.70 recitation rate measured on 2026-08-25. Search data only once the reading gate (docs/search.md, ≥ 50 %
   EM after a small QA SFT) passes on the flagship base. lr: T2 {1e-4, 3e-4} (overnight_sft2 used AdamW
   3e-4). After it: identity ≥ 5/6, chat val, reading, sim-QA, needle, and pass@1 / pass@64 on the sim
   tasks — the RL headroom numbers.
2. **Preference stage** (the correctness and prefs DPO rounds **merged 2026-08-25**): one run over every
   preference pair that exists at the time — the sim's 20k verifiable pairs today, plus the prefs store's
   pairs and marks whenever votes arrive. Objective decided by the Round A/B bake-off, not assumed to be DPO.

   **Always re-run from SFT-1; never stack.** A later run *replaces* the preference checkpoint rather than
   training on top of it, so there is never a sequence for a later objective to undo — which is the failure
   [2606.19744] found is unpredictable in sign (degradation, redistribution, or positive transfer, depending
   on objective relationship, signal strength and order). The price is explicit: re-running invalidates
   RLVR-1, which is redone from the new reference. Accepted.

   **Mix**: prefs pairs are upweighted to a fixed share of the gradient (start 20–30 %) rather than left at
   their proportional ~5 %, or a taste signal is simply drowned by 20k verifiable pairs that all point the
   same way. Per-kind losses and margins are logged separately so domination is visible. The share is a
   starting guess with no measurement behind it.

   **Gate**: `0 < pass@1 < 0.5` on held-out sim QA — one window, not two gates. Off the floor, because the
   stage must teach something; under 0.5, because RLVR-1's start gate needs headroom (`pass@1 < 0.5` and
   `pass@64 − pass@1 ≥ 0.2`), and a stage that saturates sim QA passes its own check while quietly making
   the next stage impossible. 1 epoch, lr 5e-7, β 0.1, SFT term on. Guards: identity/hold/concede,
   **false_fire_rate**, chat val — no regression.

   The old prefs gate (≥ 1000 rated / ≥ 150 Noah's) is **suspended** pending the objective: it was written
   for batch DPO on pairs, and a click and a comparison are not the same unit (docs/prefs.md). Noah's ≥ 150
   own judgements stand regardless.
3. **RLVR-1, multi-turn actions in the sim** (household, inventory, schedule — kinship has no agent
   actions). Tasks State2State-style (2608.04934): k scripted actions from a seeded world → goal =
   predicates over the reached state (holdings, locations, bookings) rendered in the locale; reward 1 iff
   all hold at the end (fraction logged), step budget k+2, an illegal action renders as "nothing
   happened". Protocol: `<|call|>` = 262 and `<|result|>` = 263 with the tool named in the bytes
   (`<|call|>sim: take candle<|result|>…`; search is `<|call|>search: …`, docs/search.md), one server hook
   for every tool, only the model's bytes carry loss. Cold start = SFT on expert traces from per-domain
   planners. Algorithm (**revised 2026-08-25**, docs/research/dpo-rlvr-2026-08-25.md — plain GRPO is worst at
   exactly this regime: binary reward, low dispersion, small G):

       z_i = (r_i − median r) / (σ + eps)     MC-GRPO 2601.22582. The *mean* baseline's noise flips advantage
                                              signs at small G. G+1 rollouts, the pivot at the median gets 0 and
                                              is dropped, so G still contribute — and G=2 lands within 1% of
                                              G=8, which is what makes this stage affordable on 8 GB.
       δ_i = 2·√C·(frac_i − 0.5)              MDP-GRPO 2606.06058 Eq. 4, C = len(task.goal). Defined at zero
                                              group variance, so an all-failed group teaches instead of being
                                              discarded — the "mean-centering blindness" the edge-of-competence
                                              filter does not cover.
       ṽ  = λ± · tanh(β_PT · v)               Eq. 5, (β_PT, λ₊, λ₋) = (0.8, 1.25, 2.0). Bounded and loss-averse.
       A_i = (1−α)·z̃_i + α·δ̃_i               Eq. 6, α = 0.2.
       β_i = --kl if A_i ≥ 0 else --kl-high   Eq. 8, asymmetric: move while improving, hold when regressing.

   **The reward stays all-or-nothing** — shaping applies to the advantages, not the reward, so there is nothing
   to game by satisfying the easy three predicates of four. `frac` enters only as δ_i, which tanh bounds. No sim
   change was needed. `--anchor-alpha 0 --pt-beta 0 --baseline mean` is the pre-revision algorithm, for ablation.

   Otherwise unchanged: KL β to the post-DPO reference, lr ~1e-6, on-policy rollouts through the graph decoder at
   T = 1, prompts kept at the edge of competence (group pass rate in (0, 1)), ~200 steps, pass@k on a held-out
   task set every N steps. Start gate: pass@1 < 0.5 and pass@64 − pass@1 ≥ 0.2. Guards: pass@64 never below the
   pre-RL model; identity/hold/concede, **false_fire_rate** and chat val no regression; KL bounded.

Served = the last stage that passed its gate; every stage checkpoint sits in the studio picker.

**Post-training additions signed 2026-08-24 night** (2607.16097, docs/research/pretrain-to-rl-scaling-2026-08-24.md):
*RLVR-1's budget is ~20 % of the rung's total compute* (their 50M–100M optimum, rising with scale — ~34
GPU-hours ≈ 300 eager GRPO steps at the flagship rung, or ~20 H100 credits), never taken from the trunk: the
post-RL level is exponential in the base's loss (1 % of loss ≈ 12 % of reward; a decade of RL compute buys
2–5 points). *pass@8* on the sim probe is reported beside EM by every gate (`branch_gate --k 8`; pass@64 on
demand for the RL start gate) — pass@k tracks the base, pass@1 tracks RL. `mote/eval/rl_taxonomy.py` scores the
before/after policies over each held-out state's legal actions and bins the change (ground-truth amplification /
tail discovery / top-k correction / regression / wrong-mode amplification) by expert-line length; it runs on the
SFT-1 → RLVR-1 pair and is read with the pass@64 guard (wrong-mode amplification is why pass@k stalls). An
**SFT-1 arm with self-proposal traces** — K base rollouts in the sim, merged into a prefix tree and serialised
between `<|think|>` (264) and `<|end_think|>` (265) before the verified commit — is queued against the
pruned-expert traces, gated on sim pass@8/64 (answer-only SFT killed pass@k in their runs); it needs the sim's
replay-to-step API (shared with the PIVOT item). The two ids are reserved now: `VOCAB_SIZE` 266,
`pad_vocab_to` 272 in the default config (six spare rows for later protocol ids; the head masks rows ≥ 266 to
−inf; old 264-row checkpoints keep their own config).

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
preset (8.2 GB − the desktop's 0.7 − margin), measured in an idle window — *superseded the same evening: the
desktop measures 1.5–1.9 GB (docs/results/2026-08-24-evening-gates.md), so the ceiling is **6.2 GB with the
screen locked**; the 2026-08-28 chunk-rate sweep (`scripts/gate_sweep.py`) reads against that*. Lands before launch if it passes,
else at the first swap restart (a daemon restart; the trunk auto-resumes).

**Released serving (grilled and signed 2026-08-24 night, root; built the same night).** The desktop measured
1.5–1.9 GB of the card (docs/results/2026-08-24-evening-gates.md), so the resident arena + decode graphs + MemPool
(~0.5 GB) cost training the margin it needed: eight flagship arms OOM'd in a row. Now **while any training job
runs the engine holds only its weights** — `Engine.release()` drops the arena, the captured graphs and the pool
when a job starts (`JobQueue.on_started`); a reply allocates a per-reply arena from the shared allocator, the
prefix store's CPU pages rehydrate it, decoding takes the eager per-byte path, and the memory goes back at the
end of the reply; `Engine.rearm()` + one warm-up (~12 s) restore the fast path when nothing runnable is queued
(`on_idle`). A checkpoint loaded at a job's end starts released when more work is queued. Gate restated: idle —
unchanged (first byte < 1 s, p95 ≤ 2× idle); **during a job — first byte ≤ 1 s + the prefill of the context,
per-byte ≤ 4× idle**; the 7-day trunk runs in the released mode throughout. The queue retries a CUDA OOM from the
front once free + cached memory covers the failed run's peak (jobs.py), and `mote train start --front` jumps the
queue. Hard isolation (green contexts) stays a measured later option. A green-context SM partition
(`torch.cuda.GreenContext`, works on cu126, even SM counts on Ada, no forward-progress guarantee) stays a
measured later option (`--serve-sms k`) — hard isolation at a permanent k/34 training cost.

**Boot without a GPU (grilled and signed 2026-08-28; built the same day).** A kernel update put the nvidia
module 57 s after boot; the daemon was up at 13 s. `gpu_usable_bytes()` said `inf` without CUDA, the queue
started the front job, `Trainer` fell back to the CPU on its own, and a flagship at 16384 on the CPU reference
path took 23.5 GB — the kernel OOM-killer took the whole daemon, serving included, three times in 45 s,
because the job came straight back to the front each time. Now, in dependency order: the daemon launched
`--device cuda` **waits for CUDA before anything in the process asks torch for it** (`app.py wait_for_cuda`,
up to 180 s, a fresh interpreter per probe — the in-process answer is cached for the process's life); past
that it **serves on the CPU with the queue paused**, re-probes every 30 s, and restarts itself the moment the
device appears. `Trainer` takes `--device` (default `cuda`) and **refuses to fall back**. A job whose process
dies twice before it logs a step is **held and the queue halts**, norm-guard style (`jobs.py
DEATHS_BEFORE_HALT`; one free retry covers a reboot landing right after a start). Usable memory is free + the
**whole** reservation, since the engine leaves the card when a job starts — counting only the unused cache
had parked two 6.5 GB retries behind a 1.3 GB engine for a day — and a retry the card could never satisfy is
held with the reason instead of parking. `GET /api/training/queue` says why each queued job is waiting.
`POST /api/engine/device` parks the engine on the CPU for standalone measurements (`mote engine park`).
The CPU reference path itself was O(L²) in memory — the flagship read one 4096-byte window at +6.2 GB and
18.6 s (measured 2026-08-28), the path the trunk serves from for seven days — so `Mamba3Mixer` now runs the
reference in windows of `REF_CHUNK` = 256 positions with the recurrent state carried between them (finding 8
of the serving audit; a window from the previous window's final state is the reference's own resume, exact
algebra, fp32 rounding pinned by tests, one window = the old path bit for bit): **+1.2 GB and 2.7 s** for the
same read; the 35M's 4096 read went +4.0 GB / 13.1 s → +1.8 GB / 6.1 s, the rest being the main network.

**Serving policy and device (grilled and signed 2026-08-25 after the web QA; built the same night).** Two things
the released mode left open. *Policy (root):* the served model is the **pin** (`.mote/config.json`'s `checkpoint`)
unless a job is **on the air** — `mote train start … --serve` (or `mote train serve [--id] [--off]`, the Training
sheet's "Put on the air", `POST /api/training/serve`); its EMA answers while it runs (the EMA sync is skipped for every other job) and its
final checkpoint is pinned when it finishes. Screening arms never touch what is served (a queue of 15 arms used to
mean 15 silent swaps — the night's chat answered 512 bytes of whitespace from a 30-minute checkpoint). A manual load
wins: it re-pins and takes the running job off the air for the rest of its life, with a notice. *Device (root):*
**while any job runs the engine lives on the CPU** — `Engine.moved("cpu")` carries the weights, the prefix-store
budget, the tools and the provenance across; no arena, graphs, pool or GPU gate — so training gets the whole card
and a reply never waits for it (measured: 35M 1.5 KB prefill 0.68 s then ~50 B/s; 96M ~45 B/s; a cold 4.6 KB read
10 s; continuations stay cheap through the prefix store). When the queue drains the engine moves back to the GPU with
arena + graphs and one warm-up. The reply says why it is waiting (`{"type":"waiting","on":"prefill"|"swap"}`) so the
CPU's cold-read cliff is a sentence, not a cursor. The released mode stays as a mechanism (an engine on the GPU
beside a job) but the daemon no longer runs it; **the 7-day trunk is launched with `--serve`**.

Also settled the same evening: **fp8** linears parked until the 1B (`_scaled_mm` runs on the 4060 Ti;
tensorwise e4m3 = 3.8 % relative GEMM error; GEMMs are 13–22 % of the step at d = 768/512, where torchao's
own guidance says the cast overhead can make it a loss — the six-shape measurement lives in
docs/results); **`F.rms_norm`**'s fused kernel replaces mamba_ssm's norm if the bench wins, under the
kernel gate; CuTeDSL / CUTLASS Inductor backends and Gluon are H100-only one-flag experiments once the
compile workstream runs there.

## Mixture of experts (grilled and signed 2026-08-24; numbers and the seven-paper digest in docs/research/moe-2026-08-24.md)

Root option, Noah's call: MoE enters the pre-launch funnel as its own arm family. The one property that
earns it a place on this box: top-k experts add parameters without activation memory, and activations
are what cap the flagship at 16384 on 8 GB. Built in `mote/model/moe.py` (commit da14e1e): `MoESwiGLU` in
the FFN slot of the Relation blocks — E experts as stacked tensors under batched Muon, router on AdamW fp32,
three exact-equal execution paths (dense masked for the decode graph, per-expert loop, `grouped_mm` bf16),
padded chunk rows excluded from every statistic; routers `lossfree` (DeepSeek-V3 bias + Moonlight gate
scale + seq-level balance 1e-4) and `aux` (Switch + z-loss); telemetry = load / MaxVio / top-k mass per
layer per step, expert usage per layer at eval, per-domain routing NMI/JSD via `mote.eval.moe_report`.
Relation is untouched.

**The funnel is different for MoE** — the joint scaling law (2502.05172) says its gain rides on a steeper
data exponent, so at 30 min / 2 h an MoE arm looks *worse* (+0.07…+0.17 nats) whatever the truth; the
crossover is 3.7 tokens/param. Verdict = 35M runs of 12 h at equal wall-clock (D/N ≈ 20, predicted
−0.013 bpb for the E=4-class layout, above the ±0.005 gate): dense at {lr0/2, lr0, 2·lr0} and each MoE
layout (E=4/top-2 half-size +0.7 GB; E=8/top-2 half-size +2.0 GB) at {lr0, lr0/2}, constant LR with
`--eval-ema`, so `mote.train.lr_horizon` fits ln lr* vs ln D (Kakao 2608.20061 §2.2.1) from the same runs
and extrapolates to the trunk's 75 tokens/param — the freeze's lr came from 0.2 tokens/param and cannot
have seen that slope; Noah reads the fit before any frozen number moves. 30-min flagship-preset arms keep
their role for KB/s, memory, router balance and lr stability only. Slot: after JEPA round 2, before the
confirm arm; the serving-beside-training gate is re-measured on the MoE preset. Side finding: the 7-day
trunk is ~4 epochs of the 10 GB mix (fine per 2305.16264); the day-10 extension would be 6.

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

## Reading 2026-08-28 (2608.17981, Recirculation)

Mozer et al.: a training-free recurrence for a frozen transformer — at every step leak a deep layer's
residual (source *s*, rescaled to the destination's norm) into a shallow layer *d* of the same token,
re-run *d+1…L* and overwrite that token's KV; the readout stays the first pass, later tokens see the
recirculated cache. Recurrence in depth **and** step, so the same layer can hold z(t) and z(t+1) (state
tracking no longer bounded by depth); decode ≈ free (two stacks in parallel), prefill becomes serial.
Gemma3 ppl −1…−16 % (12B −25…−35 %, a weak LM), adaptive per-dim α,β MLP −23 % ≥ full fine-tune; **other
families <0.5 %** — below our 0.005-bpb gate. It maps 1:1 onto the main network (pre-norm fp32 residual,
per-chunk {P2, I~} rows to overwrite) and chunk resolution makes the serial part 3.3× cheaper than a byte
model's, so it was measured rather than argued: `scripts/recirc_sweep.py` replicates their (s, d) grid on a
checkpoint (validated: the sequential cache path reproduces the parallel forward exactly). Partial grid on
`t3l_dense_4e-4`, α=0.1, 32 spread windows (baseline 1.15802): 1→0 +0.02 %, 2→0 +0.05 %, 3→0 +0.05 %;
4→1 −0.04 % on 4 windows. The rest runs on an idle queue — the sweep beside a timed arm cost
`elr_gate/muon_ref` 16 % of its throughput while it ran (prereg addendum). **Decisions:** training-time recurrence is out
(serial or block-serial main network vs the 47 KB/s trunk); serving-time recirculation is a post-trunk
option only if the grid finds a region above the gate (it needs a 2-row decode graph, rewriting arena rows
at layers > d, and serial prefill); the looped-main arm (2608.18222) is unchanged — the paper's cleanest
result is that looping and recirculation are different principles and looping has no training-free gain.

## Reading 2026-08-28 (2608.08888, Full-bandwidth transformer)

Wang, …, Langford (JHU/Princeton/Microsoft): widen the inter-step channel of a decoder by feeding the
previous position's **top-layer state** back into the next input through a GLU — `u_t = W_U h_{t−1} ⊙
σ(W_G e_t)` — the state on the value path, the sampled token only as the gate (an additive fusion leaves a
shortcut that ignores the state; the gate makes reading it mandatory). Two D×D matmuls per token, KV
cache and stack untouched, prefill single-pass ("Soft") or one extra fused parallel pass ("Fused").
Trained in parallel by **multi-pass teacher forcing**: pass 1 standard, pass k shifts pass k−1's top
states right by one position, fuses, re-runs the stack, NTP loss on every pass (λ = 1, no detach);
schedule 75 % 1-pass / 22 % 2-pass / 3 % 3-pass introduced mid-training from a standard checkpoint —
and the 3 % three-pass batches are load-bearing (2-pass-only diverges past its trained horizon, with them
the map is a contraction stable to 1000 passes). Prefix mixin (random plain prefix per fused pass),
depth-scaled O(1) top-state norm, RMSNorm on the fused input, tied embeddings, jitter σ = 0.02. 1B / 24
layers, Phi-4 mix, NorMuon: 200B-token FBT ≈ 400B standard on val loss and 5-shot LM-eval at 1.28×
token-equivalent compute; Soft > Standard on every generation task with the same weights (Math500
0.27 → 0.37), Fused best on code; carries through instruction tuning (GSM8K 64.5 → 67.9); shorter traces on
base models; layer-0 linear probes of global state 99.6–100 % vs chance. Caveats: 1B only, schedule
heuristic, decodability ≠ use, and Soft decoding is incompatible with exact parallel draft verification
(the fused input at t needs h_{t−1}) — their vLLM path does not speculate.

**Where it lands.** This is the trained version of what 2608.17981 hypothesised, and it fits a
from-scratch run in a way recirculation cannot: prefill and training stay parallel. Mote's byte level has
exactly the narrow channel (only the sampled byte returns to the encoder; the Mamba-3 decoder state carries
z forward but never back into the encoder or the main network), and the trunk is **data-repeating** (2.8
passes over mix A in 7 days) — the regime the paper is built for. The fusion point exists already:
`step()` holds the decoder's normalised top state `h3` before `head_logits`, tied embeddings are on, and the
decode graph can carry `h3_{t−1}` in a static buffer as their vLLM note does. Byte-level twist worth
testing first: with the fused input feeding the encoder, the **router** sees the fully processed past
when it decides boundaries. **Signed 2026-08-28 (docs/results/2026-08-28-latent-feedback-prereg.md):** not in the frozen trunk —
three 24-h arms on the trunk snapshot before mid (control / chunk-level / byte-level continuations, pre
mix, constant lr, 75/22/3 passes), gate k=1 fused-prefill val ≤ control − 0.005 and k=0 ≤ control + 0.005,
tie within 0.005 → chunk; the serving bench under Soft decoding confirms; mid's 2×2 starts from the
winner; Soft default with Fused per request; post stages k=3 throughout. **Built 2026-08-28:**
`FeedbackCfg`, `mote/model/feedback.py` (fusion, shift, prefix mixin, jitter), the feedback pass in
`HNetForCausalLM.forward` (byte: fused embeddings; chunk: fused `pad(hc)` with the encoder/routing
reused), Soft decoding in `step` / prompt reading / the decode graph (static `h_prev` / `z_prev`), the
multi-pass objective with `--feedback{,-mix,-window,-detach,-jitter}`, `--eval-feedback-passes`
(val_bpb_fb1..k), `--init-from` that tolerates the fresh fusion, `scripts/latent_feedback_arms.py`.
Open: multi-pass memory at 16384 (profile in the gap before the arms; fallbacks 8192 windows → detach);
Fused prefill and the multi-turn question (a second user turn is read plain after fused generated bytes —
a switch training never shows) are built with the serving bench. Nothing here reopens the mid 2×2 protocol.

## Out of scope until the above

Looped main network, test-time-training memory layers instead of a window, any change to the
architecture's math. Those are a research sweep (2026 sources first), not a systems decision.
