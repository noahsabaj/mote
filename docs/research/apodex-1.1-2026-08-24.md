# Apodex 1.1 (technical report, Aug 2026) — what Mote's post-training takes from it

Source: https://framerusercontent.com/assets/vnib7j93v0EP1kkb4GmFU8WEU.pdf (46 pages, read in full 2026-08-24).
397B main model + 35B "mini"; architecture, pretraining data, compute and RL hyperparameters undisclosed;
"open weights" only as a link; most baselines "internally reproduced"; no ablations, no seeds. Run horizons
300–900 steps / 25–80 min. Nothing measured below 35B. Read as a recipe, not as evidence about our scale.

## The recipe in one paragraph
Task contract E = (world, initial state, objective, actions, tools, observations, budget vector, delivery
contract, terminal verifier) with the verifier hidden from the solver. Environment scaling: file / search /
code worlds generated forward from latent state or a reference program ("forward cheap, inverse expensive"),
family-specific verifiers hardened by blind-solver probes, replay manifests (seed, generator, tool and verifier
versions, action/observation log, deltas) gating which trajectories may train, failure-driven task pipeline.
Coordination scaling: lead agent, external task board re-injected after compaction, asymmetric claim-attacking
verifiers, evidence-graph synthesis. Training: SFT mixture with behaviour-validity filters (invalid tool
interaction, inconsistent state, ignored observation, incomplete delivery) and model-soup merging of domain
variants; then PIVOT-RL — hindsight localisation of the pivot decision, environment restored to that state,
localized continuations with a directional hint (never a target, absent at inference) mixed with unhinted full
tasks, asynchronous rollouts. Eval: negative credit for wrong assertions; an integrity gate that zeroes any
trajectory with a fabricated tool result or a narrated action absent from the log.

## Items folded into Mote's signed post-training (docs/shape.md "Training pipeline")
| # | item | depth | cost |
|---|---|---|---|
| 1 | **Integrity gate**: a `<\|result\|>` not produced by the harness zeroes the RLVR-1 reward and drops the trace from SFT | patch | reward/filter check |
| 2 | **PIVOT-style continuations**: first divergence from the pruned expert = pivot; replay the seeded sim to that step, start the episode from the expert prefix; mix with full episodes | partial | replay-to-step API in mote/sim |
| 3 | **Blind-solver probes** on every sim task family (later QEMU/Icarus verifiers): a random/degenerate policy that scores marks a broken verifier | partial | a script before RLVR-1 |
| 4 | Search tasks: difficulty ρ = (plausible candidates + load-bearing hops) / call budget; **negative credit for wrong assertions** in the ≥50 % EM gate | patch | task generator + gate |
| 5 | SFT-1 filters: ignored observation (next action inconsistent with the result), invalid tool interaction, incomplete delivery; model-soup merge of identity-SFT and sim-SFT as a 35M arm | patch | filter + one arm |
| 6 | Folding rule: evict tool-result bodies first, keep structure, summarise the middle only if needed (docs/context.md) | patch | ordering rule |
| 7 | Claim-level rating: Claude-as-rater and correctness-DPO checks attack one claim + evidence instead of scoring whole answers (docs/rubric.md) | patch | prompt change |
| 8 | Trajectory records carry seed + generator + verifier versions (replay gate) | patch | bookkeeping |

Irrelevant below the 1B–10B rungs: file/code worlds (33 domains, 318 occupations, 1,208 deliverable clusters),
multi-agent coordination (+3–12 points on the same policy at 35–397B), 800-step runs, asynchronous rollout
infrastructure. Tension to remember: they route verifier/solution disagreements to task construction, our
sim gate counts everything against the model — fine only because the sim verifier is exact state checking.
Related: 2608.20965 (narrow RL feedback collapses unreinforced skills; rebalancing recovers) → RLVR-1 rewards
the task families in balance and gates every checkpoint on the non-reinforced probes.
