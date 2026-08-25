# 2607.16097 — "Understanding Reasoning from Pretraining to Post-Training" (Shen, Li, Rahman, Sun, Goldblum, Telgarsky, Izmailov; NYU / UIUC / Modal, v2 2026-08-09)

Read in full 2026-08-24 (38 pages). Chess as a controlled pipeline: pretrain 5M–1B Qwen3-style models on 54B tokens
of Lichess games (81-token vocabulary, one move = 4 tokens), SFT on synthetic reasoning traces, GRPO on 156K puzzles
with an exact binary reward; 36 pretraining×RL combinations, plus a 1B OLMo-2 math run (10B–200B tokens) as the
transfer check. Models and code are public (`pavelslab-nyu/pre2post-chess`).

## Findings

1. **Joint pretraining–RL law**: `R(C_RL, N, T) = f(L_pt(N, T)) + g(N, T)·(log10 C_RL − 20)`, with
   `f(L) = 0.031 + exp(4.87 − 12.85 L)` (post-RL pass@1 at a reference RL compute, exponential in the pretraining
   loss; Spearman 0.93 → 0.99 as the RL compute grows) and `g = −0.216 + 0.0172 log10 T + 0.0098 log10 N` (reward
   gained per decade of RL compute — driven by **pretraining tokens** twice as much as by model size). LOO RMSE 0.019
   in reward; the 20M family's RL *ceiling* is also predicted by the pretraining loss (R² 0.90). Chinchilla-L
   prediction error is half the total error.
2. **Compute split**: the RL-optimal share of total compute rises with the budget — ~20 % at 50M, 28 % at 680M; the
   pretraining-token allocation stays at Chinchilla. "RL is strongly initialization-limited": starting RL from a
   weakly pretrained checkpoint does not pay back the extra RL compute.
3. **pass@1 vs pass@k**: RL raises pass@1 everywhere; pass@16 is flat or slightly worse for the larger models — more
   pretraining beats more RL for coverage.
4. **SFT data**: SFT on the bare answer improves pass@1 and *nothing else* (pass@8 ≈ pass@16: the samples lose
   diversity). SFT on the model's **own** sampled continuations (K rollouts from the pretrained model, merged into a
   prefix tree, DFS-serialised as a think trace, then the verifier-picked continuation as the committed answer,
   opponent/environment moves masked from the loss) improves every pass@k. They adopt it for all RL.
5. **What RL does**: not a uniform sharpening (power-transform fit R² ≈ 0.6, per-state exponents scattered). Per
   state, by difficulty: easy → ground-truth amplification; hard → *tail discovery* (correct move promoted from
   p < 0.05) **and** *wrong-mode amplification* (15–20 % of hard states — the wrong top-1 gets reinforced). That mix is
   why pass@k does not improve. Traces after RL: wider search, not deeper; candidate quality up; ≥4-ply lines stay
   rare. "RL improves candidate generation and selection faster than long-horizon search."
6. Chess Chinchilla fit: data exponent β = 0.68 (language ≈ 0.28) — a low-entropy structured domain wants far more
   tokens per parameter than text; the optima converge at higher compute. Their RL cost: 2000 GRPO steps ≈ 160 H200
   GPU-hours at 50M (G = 8, lr 1e-5, KL 1e-3, clip 0.2, temperature 1, no entropy or format reward).

## What Mote takes (post-training is signed; these are build-queue additions and planning numbers)

| # | item | depth | cost |
|---|---|---|---|
| 1 | **RL share is a planned number**: budget RLVR-1 at ~20 % of the rung's total compute (their 50M–100M optimum), rising with scale; at the 96M flagship's 7-day trunk that is ~34 GPU-hours ≈ 300 eager GRPO steps — the signed "start gate" already refuses a base with pass@64 ≈ 0 | patch | planning |
| 2 | **Pretraining loss is the leading indicator of RLVR-1's outcome** (`f` is exponential in L: 1 % of loss ≈ 12 % of post-RL reward; a decade of RL compute buys 2–5 points): the branch gate's val-bpb guard stays the gate that matters most for RL; never trade base loss for RL budget at this scale | patch | none |
| 3 | **Policy-change taxonomy on the sim** (`mote/eval/rl_taxonomy.py`, to build): per held-out task state, compare the SFT and RL action distributions vs the expert action — GT amplification / tail discovery / wrong-mode amplification by difficulty bin. Exact verification makes it cheap; it says where RLVR-1's gains come from and whether wrong modes are being reinforced | partial | eval script |
| 4 | **Self-proposal traces as an SFT-1 arm**: K rollouts of the base in the sim → prefix tree → serialised alternatives before the verified commit (their construction), vs the pruned-expert traces we have; gate = sim pass@8/64, not EM — needs a think delimiter (reserved id) | partial | data builder + one arm |
| 5 | pass@k (k ≥ 8) on the sim/reading probes is the *pretraining-side* metric of post-training readiness; pass@1 is RL's | patch | probe reporting |
| 6 | The RL ceiling (sigmoid asymptote) is predictable from the base loss — do not read a flat RLVR-1 curve as "RL doesn't work here" before checking the base's headroom | patch | none |

Not transferable: the chess-specific exponents and category proportions (81-token vocabulary, exact verification, no
world knowledge); 1000–5000-step RL runs; the 8×H200 rollout budget. Consistent with docs/research/apodex-1.1 (item 3
= their blind-solver probes, from the RL side) and 2608.20965 (narrow rewards collapse other skills — their
wrong-mode amplification is the same failure seen per state).
