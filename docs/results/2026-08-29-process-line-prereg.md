# The process line — pre-registered (signed 2026-08-29, ~14:00 EDT)

Grilled in five rounds on the afternoon of 2026-08-29, out of the question "why pretrain for next-token
prediction when the model is for chat and agents?". The literature read that started it is summarised at
the end; the signed design comes first. Nothing here starts before the trunk is launched (Monday's gap).

## Frame

- **Goal: a recipe change for Mote.** The thesis is a search heuristic, not the deliverable. The thesis:
  NTP is the right loss on the wrong data — a corpus is the *residue* of processes (finished text with the
  drafts, tool replies, state transitions stripped out), and a model trained only on residue treats its own
  outputs as observations (Ortega et al. 2021, 2110.10819). Every 2026 win on agentic behaviour changed the
  data and kept the loss.
- **Position: after the trunk and the latent-feedback arms.** The frozen recipe does not move. Arms run at
  the 35M `local` shape (4×4 @ 2048, `data/local_mix`), **one 2×3 per candidate, sequential** (~2 GPU-days
  each), the next candidate only after the previous one has been read.
- **Adoption path: mid-training on the trunk checkpoint.** A 35M win does not enter a from-scratch recipe;
  it is applied to the finished 96M trunk as a mid-training stage against its own control (the trunk
  continued on an equal number of extra corpus bytes). That is also the first thing that would ship.

## What decides

- **Primary: a behavioural probe, tool-output prediction.** Given a REPL/shell transcript prefix, predict
  the tool's reply exactly (deterministic tools; exact match per reply, averaged over held-out instances).
- **Secondary, reported, must not lose:** lookahead/planning (path-star and Countdown-style, Bachmann &
  Nagarajan 2024, 2403.06963), state tracking (a small hidden-state world: predict the state after an action
  sequence, 2607.24720-style), open-ended diversity (Nagarajan et al. 2025, 2504.15266).
- **Probe data lives in both places.** Each family has a task train split *in the pretraining mix* (a small
  slice, byte-identical across arms — the literature's setup; this measures the objective's inductive bias
  on the task) and a held-out variant family probed zero-shot (transfer; expected near-null at 35M, reported).
  The in-mix instances decide.
- **bpb guard (the call was left to me — "No").** `val_bpb_ema` on the natural val shard must stay inside the
  control's 3-seed range. Every win in the literature (FSP, NextLat, TPT, FIM) came without a bpb loss, and
  the probe is new and ours — a trusted metric guards an untrusted one.
- **Win rule.** Primary: mean(arm) − mean(control) > 2 × pooled seed SD (3 v 3). Secondaries: not below
  control − 2 SD. bpb inside the control's range at **both** matchings below. Anything else is "no win",
  written up as such.
- **Two readings, two questions.** The arm is *designed* at matched supervised bytes (control and arm see the
  same number of loss-carrying bytes; rollout and sandbox compute are free) — that answers the thesis. The
  recipe *adopts* it only if the win also holds at matched wall-clock on the 4060 Ti, read from the control's
  log and checkpoints at the arm's wall-clock — that answers the recipe. Rollout cost is its own number.

## Candidate 1 — executed traces, then a live environment (Noah's call over my recommendation)

I recommended offline executed traces first (the thesis' own claim, zero GPU cost, a week of my time);
Noah chose the live environment. The warm start folds the offline candidate in as phase 1, so the ladder
control → +traces → +live comes out of one arm.

- **Phase 1 (first 50 % of the horizon): executed traces, offline.** Transcripts produced by *executing*
  real programs in a sandbox, never by a teacher writing the replies. Tasks come from generators I write
  (templates plus mutations of real functions from the code slice) and a few hundred hand-written seeds —
  versioned in the repo, no API spend. Process slice ≈ 8 % of pretraining bytes.
- **Phase 2 (last 50 %): live, on-policy.** The training model's own sampled actions run in the sandbox;
  the reply is appended and the loss is taken **on environment replies only** — the action bytes are
  conditioned on and never trained on (an action is an intervention, not an observation). Corpus:episode
  bytes 3:1. Rollouts reuse `mote/infer`'s Engine against the live weights; the sandbox pool is CPU.
- **Hand-over: a fixed fraction (50 %).** Preregistered, no hidden knob; the 50 % checkpoint vs the control's
  50 % checkpoint is the phase-1-only reading, free.
- **Environments: a persistent Python REPL and a shell in a scratch filesystem, interleaved** (Noah's call;
  I recommended the REPL alone first). Sandbox: `bwrap` (installed) — no network, tmpfs, rlimits; `podman`
  only if a shell task needs a fuller userland.
- **Not in this arm:** rewards, verifiers, self-imitation of actions. A success signal would make it
  RL-lite and muddy the attribution; that is a later candidate if this one reads positive.

## Later candidates (each its own 2×3, in this order unless a read changes it)

2. **NextLat-style auxiliary over the H-Net chunk latents** (2511.05963; hierarchical variant 2608.05806):
   predict the next chunk's latent — "next concept" without fixing the concept in advance. ~1 day of
   trainer work (no generic auxiliary-loss hook exists; latent feedback is the nearest machinery).
3. **Function-aware fill-in-the-middle on the code slice** (2607.12463): mask whole function bodies, PSM
   format at byte level. No new data.
4. **Live environment with a success signal** — only after candidate 1 has been read.

Not queued, with the reason: diffusion (Mote is compute-bound, not data-bound, and the outer/main stacks
are causal by design — the DLM crossover of 2511.03276 / 2507.15857 lives in the data-limited regime);
JEPA-only or fixed-unit "concept" objectives (collapse and null results: 2607.23531, 2605.15394, Mimir
2605.25263); reinforcement pre-training (2506.08007 needs a reasoner to start from).

## Build order and timeline

1. Probe suite v1, four families, validated against a control before any arm (~1 week).
2. Sandbox + trace generators + the phase-1 mix (~1 week).
3. The live loop: batched rollouts from the training weights, sandbox pool, reply-only loss mask (~2 weeks).
4. The 2×3 (~2 GPU-days) after the trunk (ends ≈ 2026-09-07) and the latent-feedback arms — ≈ early to
   mid October 2026. Read, write-up, then candidate 2.

Nothing above starts before the trunk launch; the probe suite is the first thing built.

## The reading that started it (2026-08-29, arXiv, recency-weighted)

NTP is not the relic: its supervision is free at scale, dense (one exact gradient per token vs one scalar
per RL episode) and a proper scoring rule, so it learns a simulator of whatever produced the corpus; chat
and agency are conditional slices and RL is the cheapest way to select one. Four things in the pipeline
*are* relics, each being attacked separately in 2025–26 with the loss left alone: **staging** (RL only after
SFT — Bansal et al. 2606.04272: RL applied to early from-scratch checkpoints expands the distribution; the
"RL only sharpens" effect appears only after SFT); **observation vs action** (Ortega 2110.10819; the agentic
mid-training wave: 2601.18418, 2607.12463, 2604.02345 — forward dynamics is the scalable signal —,
2608.04934, 2608.20314, 2608.26563); **teacher-forcing myopia** (2403.06963, 2504.15266; fixes: MTP
2604.11912, future-summary prediction 2510.14751, NextLat 2511.05963, HiLP 2608.05806); **the token as the
unit** (H-Net already learns the unit; fixed larger units and language-JEPA have not beaten it:
2605.25263, 2607.23531, 2602.22617 vs 2605.15394). The "reason before predicting" line (RPT 2506.08007,
RLPT 2509.19249, TPT 2509.20186) keeps the target and adds process. On the RL side, "the base model is
smarter than you think" (2510.14901, 2601.21590; pass@k inversion as overtraining, 2606.15455) says the
prior is the capability and RL is a decoder for it — which is why *what gets predicted in pretraining* is
the right question.


## Amendment 2026-08-29 (from the SSM sweep)

2607.06155 gives the architecture-level condition under which tool use adds expressivity to a finite-state
recurrent path: the tool must be unbounded and addressable and its results re-readable — then memory problems
become round-trip problems, while a bounded tool adds nothing. For the live environment this means the
transcript and every earlier tool result are exposed as files in the sandbox (re-readable), calls are cheap
enough to issue many, and a probe with n ≫ state bits (EQ_n) separates Relation-bearing from state-only
models. Tool calls do not reduce the number of Relation layers a hybrid needs for in-context multi-hop reads
(2605.16640: one attention-like layer per dependent hop).
