# Where PATH (2608.26469) lands for Mote — grilled and signed 2026-08-28

Liu & Chen, "Active Curriculum Refinement for Reinforcement Learning": treat a family of environments as an
explicit curriculum DAG (edges = one difficulty increment) and train by acquiring paths through it —
random mastery-gated paths first (advance a pointer when the policy masters its node, restart after
ε_p tries), then regret-weighted replay with successor expansion. Its load-bearing ablation is the
early stop: without it, budget goes to nodes already solved and generalization collapses.

## Measured before deciding

* RLVR-1's task hardness is **plan length `k`** (actions explored; expert length ≈ k, goal predicates ≈ k,
  call cap = k+2). The sim's `sample_difficulty` knobs (people/rooms/objects/ticks) are inert for tasks:
  expert length 1–5 at every setting, narrative ~206 B; `make_task` builds from the zero-tick world.
  `make_task(k=…)` already takes k. Held-out set: k ∈ {1:20, 2:13, 3:15, 4:5, 5:7} of 60.
* Locales differ only in narrative encoding (en 228 / ru 236 / ja 260 B); the action language is
  locale-free (76 B at k=3). Three ladders are about language competence, not plan cost.
* `rlvr.run()` today is domain randomization (round-robin cursor, uniform k) plus a post-rollout
  discard of zero-variance groups — the paper's no-early-stop failure mode, paid at 8 rollouts a task.
* No checkpoint can be measured on sim tasks yet: the tool protocol arrives in mid/SFT-1.

## Signed: RLVR-1 sampler — spec now, built with RLVR-1

* DAG: k = 1 → 8 (source 1, +1 per edge, sinks at 8), three ladders (en/ru/ja), one shared policy.
  Held-out eval unchanged (k 1–5); log per-k and per-locale pass rates and each locale's frontier k.
* PATH:Random: N = 48 pointers (16 per locale); advance at group pass rate ≥ 0.7; restart at a source
  after 20 groups without mastery; the step's 16 tasks drawn uniformly over pointers.
* PATH:Active after 30 terminated paths: weights = the group's reward std (GRPO's own statistic);
  mastered high-weight pointers spawn successor paths; replay rate p = 0.8, 0.2 fresh sources;
  lowest-weight eviction keeps N.
* Zero-variance groups are still discarded unless the anchor is on — that is about the gradient.
* A reading-depth axis (worlds with history, `long.py`'s ticks) is RLVR-2's DAG, not this one.

## Signed: SFT-1 dynamic selection — built 2026-08-28, A/B-gated before it is the default

The signed one-shot score (`select_sft`: hard-for-base × base−calibrated delta) cannot be re-run
mid-way — a window learned since has a large delta and would be kept. Mid-run the rule is about
trajectories (`select_sft.trajectory_keep`, `Trainer --reselect-every`):

* pass 0 at the start fixes τ_m = the shard's 20th-percentile loss; passes at 20/40/60/80 % of the
  budget with the live model (`--reselect-windows` samples the shard on 35M arms);
* drop **mastered** (loss < τ_m) except a **10 % floor** that stays (forgetting; the on-policy-replay
  budget); drop **stuck** (|Δloss| < 0.02 since the last pass); keep the rest; fail-safe to all.
* Scoring runs as a generator inside the job, releasing the GPU gate between batches like a training slice.

**A/B (pre-registered):** static (`--keep`) vs dynamic (`--reselect-every 0.2`) SFT-1 arms at the 35M
shape, same seed and budget, on the SFT-1 shard once `build_sft` has produced it (`data/sft_mix`; not
built yet — this A/B queues with the SFT-1 build, after the confirm arms). Dynamic becomes SFT-1's
default **iff** chat val_bpb is within 2× seed noise of static **and** identity/hold/concede show no
regression **and** the dynamic arm's final keep covers more of the shard's hard tail (windows above the
static keep's 80th-percentile base loss). Until then `--reselect-every` ships off.

## Not landing

`long.py` stays mid-training's long-range data; on-policy replay is unchanged (it is the forgetting
floor's cousin, at the stage boundary rather than inside a stage); the prefs loop has no DAG.
