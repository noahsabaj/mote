# Paper batch — 2026-08-23

Seventeen papers selected by the owner, read in full, plus sixteen 2026 neighbourhood papers read in
full for the questions the batch does not answer. Companion to
`docs/research/efficiency-campaign-2026-08-23.md` and
`docs/research/speculative-decoding-2026-08-23.md`; nothing load-bearing from either is repeated.

**Recency policy.** August 2026 weighted heaviest, then July, June, May, April, decreasing. Pre-2026
work appears only as labelled *background*. Every citation carries an arXiv ID or a dated URL.
Anything I could not verify from a primary source is marked **[unverified]**.

**Excluded by instruction:** arXiv 2607.27372.

**Repo state.** Written against the tree as of 2026-08-23 00:01. `morpheme/config.py`,
`serve/engine.py`, `model/mbp.py` and `train/train.py` were **modified during this session** (mtimes
00:00–00:01): `MBPCfg.accept_threshold` was removed, `position_gamma` and `transition` were added, and
`engine.py`'s speculative loop now does exact rejection-sampling verification with snapshot/replay.
Every code claim below was re-checked against that tree. `DCCfg.ratio_loss_weight` is still 0.03 and
`MBPCfg.loss_weight` is still 1.0.

## Measured baseline this report is scored against

`runs/overnight/log.jsonl`, preset `local`, 35.35M params, batch 2 × 2048 × grad-accum 8 =
32 768 bytes/step, AdamW β=(0.9, 0.95), lr 6e-4, wd 0.1, **α = 0.1** (not the `DCCfg` default 0.03),
N 5.0→6.5:

| quantity | value | source |
|---|---|---|
| throughput, mean of last 185 logged windows | **25 285 bytes/s** | `bytes_per_sec` |
| bytes per chunk, same window | **3.205** against a scheduled target of **5.09** | `bpic` / `target_ratio` |
| throughput at the start of the run | ~48–50 kB/s at `bpic` ≈ 4.6 | first log lines |
| val bits/byte | **1.453–1.489** | `val_bpb` |
| boundary/word alignment | **0.280** — 28 % of boundaries land on a separator | `boundary_on_separator_frac` |
| multi-byte head top-1 | **0.610** | `mbp_top1_acc` |
| gradient norm | **0.33–0.35**, against `--clip 1.0` | `grad_norm` |

At `bpic` 3.205 the analytic cost is ≈128.8 MFLOP/byte (my calculation, rescaling the efficiency
report's §0.1 split: the Relation linear term as 1/bpc, the pairwise term as 1/bpc²). 25 285 B/s ×
128.8 MFLOP/byte = **≈3.3 TFLOPS ≈ 7.4 % MFU** of the 44.1 TFLOPS bf16 peak in `flops.py:15`. The run
has lost about **half** its initial throughput to the router firing more often — worse than the 36 %
the efficiency report recorded at step 7 560. **The gradient clip never binds**: at 0.34 against
`--clip 1.0` it is decorative, which matters when reading the Muon papers below.

**And the α sweep has already been run, on this machine, at `pilot`.** Five 20–30-minute runs in
`runs/`:

| run | α | target N | steps | `val_bpic` | `val_bpb` | boundary/sep |
|---|---|---|---|---|---|---|
| `sweep_a0.1_n4` | 0.1 | 4.0 | 1 461 | 3.06 | 2.074 | 0.269 |
| `sweep_a0.3_n4` | 0.3 | 4.0 | 1 445 | 3.46 | 2.143 | **0.590** |
| `sweep_a0.1_n6` | 0.1 | 6.0 | 3 860 | 3.86 | **1.917** | 0.393 |
| `sweep_a0.3_n6` | 0.3 | 6.0 | 1 580 | **5.12** | 2.128 | 0.273 |
| `pilot_alpha01` | 0.1 | 6.0 (b4×a4) | 2 705 | 3.72 | 1.906 | 0.287 |

**Read this table carefully — it is the most important evidence in the report, and it is
half-spoiled.** What it establishes: α is a strong lever on compression, and **α = 0.3 at N = 6.0
lands `val_bpic` at 5.12**, i.e. the undershoot is *fixable by the loss weight alone*. What it does
**not** establish: anything about quality. These runs are **wall-clock-capped, not step-matched**, and
the step counts span 1 445–3 860 — a 2.7× spread — so the `val_bpb` column compares runs that saw
2.7× different amounts of data. Any reading of "α = 0.3 costs 0.2 bpb" from this table is unsound.
The α = 0.1 / N = 6.0 run getting 2.4× more steps than α = 0.3 / N = 6.0 *despite* the latter having
higher compression is itself unexplained and should be investigated before the table is trusted at
all. **[unverified — I did not reproduce these runs.]**

---

## (a) Executive summary

**This batch is not about Morpheme's architecture.** The seventeen papers break down as: five
grokking studies on modular arithmetic (2608.14803, 2608.07436, 2608.01833, 2607.29503, 2607.20512);
two on optimizer design (2608.19491, 2608.16760); one scaling law (2608.07222); one latent-feedback
transformer (2608.08888); one LM-head gradient analysis (2603.10145); one MoE hyperparameter transfer
(2608.20061); one industrial recommender (2608.16797); one RL convergence theory (2608.19587); one
subrecursive computability theory (2608.04871); one robot continual learning (2608.19589); one
rubric-RL reward hacking (2608.11669); one continual-learning plasticity study (2608.01475).
**There is no byte-level paper, no H-Net paper, no dynamic-chunking paper and no GPU-kernel paper in
the batch.** Read as what it is — an *optimizer and training-dynamics* batch — it is informative;
read as an answer to the owner's open questions, it is mostly silent, and §(e) is where those get
answered.

**Six things change what we should do.**

**1. The bytes/chunk undershoot is a loss-weight bug, it is documented, and we have already half-run
the fix.** The neighbourhood turned up **SOMBRERO (2601.22805, Jan 2026)**, which diagnoses exactly
this: at H-Net's default ratio-loss weight ω = 0.03 it measures empirical compression **3.95 against
a target of 5.0**, and raising ω to **1.0 gives 4.97** — target hit. **ATDC (2605.30080)**, the paper
whose schedule we implement, *exhibits the same undershoot without naming it*: BPIC **4.52 at 680M and
4.37 at 1.3B** against N_fnl = 6.5. Our own `sweep_a0.3_n6` reaches **5.12 at α = 0.3**. The single
cheapest high-value action in this entire report is a step-matched α ∈ {0.1, 0.3, 1.0} A/B at `local`.

**2. …but there is a serious counter-argument, and it should be settled before spending on the fix.**
**"Compute Optimal Tokenization" (2605.01188)**, 988 BLT models from 50M to 7B, finds an *interior*
optimal compression rate that **decreases with compute**: at 10²⁰ FLOPs the optimum is **T\* = 3.71**.
Morpheme runs three to four orders of magnitude below that. **3.2 bytes/chunk may be close to right
and the 5.0–6.5 target simply wrong.** The A/B must therefore be scored on `val_bpb` at matched
*steps*, not on whether `bpic` hits its target.

**3. The Muon question has flipped from "probably adopt" to "genuinely contested."** The efficiency
report's rank-16 rested on 2607.04033 (Muon leads at 60M/130M). Three papers here push back.
**2608.19491** tunes Muon under the same Optuna protocol as AdamW, with the *largest* per-dimension
budget of three arms and under Muon's own μP rule, 3 seeds at 67M and 370M on FineWeb-Edu — and its
Tables 2 and 8, differenced, put Muon **behind plain AdamW** by ≈0.026 nats at 67M and ≈0.071 at 370M
(my arithmetic; the paper never states it and declines to generalise). **2608.07436** finds *every one*
of nine Muon configurations groks and then loses generalization, with a mechanism that is
architectural rather than task-specific: Newton–Schulz normalises its input, so once the gradient
shrinks the Muon group's step does not (elasticity **−0.03** vs **+1.5** for the AdamW groups) and the
two groups separate at **8.0×** the rate per parameter. **2607.20512** shows the whole effect is
**orthogonalization**; the spectral scale factor is inert. All three are at ≤370M and two are at
2×10⁵ params on modular arithmetic.

**4. There is one clean memory win, validated at exactly our scale.** Adam-mini (2608.16760) keeps one
`v` scalar per Hessian block instead of per parameter: **50 % of AdamW's optimizer-state memory**,
tracking AdamW on Llama-2 from **39M to 1B** on C4, independently reproduced. ≈141 MB at `local`. Its
successor **NorMuon** — Adam-mini's row-wise `v` on Muon's orthogonalized update — is what
2608.08888's 1B run actually uses, and is reported as leading the NanoGPT speedrun.

**5. The cheapest experiment in the batch is one number: β₂.** The same thesis proves a
divergence–convergence phase transition whose β₂ threshold **rises as batch size falls**, with three
independent LLM-scale confirmations. We run β₂ = 0.95 at **32 768 bytes/step** — 16× smaller than
DeltaMomentum's 524 288-token step. The heuristic β₂ ← β₂^(B₂/B₁) points at **0.99–0.997**. One flag.

**6. Five of the seventeen are genuinely irrelevant and are recorded as such so nobody re-reads them:**
2608.04871 (subrecursive complexity hierarchies), 2608.19589 (robot VLA continual learning),
2608.19587 (natural actor-critic convergence rates), 2608.11669 (rubric-RL reward hacking), and
2608.01475 (online continual-learning plasticity). 2608.16797 (advertising CVR) carries exactly one
transferable idea and it is not free. §(f) gives the reasons.

**Standing caveat on the grokking cluster.** Every measurement in 2608.14803, 2608.07436, 2608.01833,
2607.29503 and 2607.20512 is on a 1–4-layer transformer with ~2×10⁵ parameters, full batch, on modular
arithmetic with an *interpolating* training set and a gradient that reaches 1.5×10⁻⁷. Morpheme is 35M
on a corpus it will never interpolate, with `grad_norm` steady at 0.34. **Mechanisms may transfer;
numbers do not.** Where I recommend acting on one of these it is because the mechanism is
architectural, never because the effect size means anything here.

---

## (b) Ranked to-do table

Gains are for **this machine, `local` preset**. "Effect" is on `val_bpb` at matched steps unless it
says otherwise. Items already covered by the two earlier reports are excluded.

| # | Idea | Source IDs | Expected effect here | Effort | Risk | Deciding experiment |
|---|---|---|---|---|---|---|
| 1 | **Raise the ratio-loss weight α from 0.1 toward 0.3–1.0** | 2601.22805 (2601), 2605.30080 (2605), our own `sweep_a0.3_n6` | `bpic` 3.2 → ~5. Because throughput tracks `bpic` almost exactly (efficiency §0.1), that is **≈1.4–1.6× bytes/s**, the largest single throughput item left. SOMBRERO measures ω 0.03→1.0 taking compression 3.95→4.97 at 1B/839B bytes | **1 flag** — `--ratio-weight` already exists | **medium**: the quality cost is unmeasured, and item 2 says it may be a real loss | Three `local` runs, α ∈ {0.1, 0.3, 1.0}, **step-matched** (fix `--max-steps`, not `--max-minutes`), same seed and data order. Score `val_bpb`, `bytes_per_sec`, `val_bpic`, `boundary_on_separator_frac`. ~12 h total |
| 2 | **Settle whether 3.2 bytes/chunk is a bug at all** | 2605.01188 (2605) | None directly — it decides whether #1 is a win or a self-inflicted quality loss. T\* = 3.71 at 10²⁰ FLOPs and falling with compute; we are 3–4 orders below | folded into #1 | none | Same runs as #1. If `val_bpb` at α = 0.3 is ≥ α = 0.1 at matched steps, **the undershoot is not a bug** — retarget `DCCfg` to 3.5–4.0 and stop fighting the router |
| 3 | **β₂ 0.95 → 0.99 / 0.997** | 2608.16760 (2608) | Unknown but plausibly the largest per-minute-of-effort item in the batch. The threshold is batch-size-dependent and our batch is small | **15 min + 2 runs** | none | `local`, step-matched, β₂ ∈ {0.95, 0.99, 0.997}, `val_bpb`. Remark 4 warns larger is not monotone, so this is an A/B not a substitution |
| 4 | **Adam-mini** (one `v` per row; per head for `w1`/`w2`) | 2608.16760 (2608) | **≈141 MB** freed at `local` (AdamW `m`+`v` fp32 ≈283 MB). Composes with micro-batch 2→4, which efficiency ranks #3 | 0.5–1 d | low | Same A/B as #3 plus `torch.cuda.max_memory_allocated()`. Relation's `w1/w2/wi/wo` are already separate `[d,d]` matrices, so no surgery |
| 5 | **Raise `mbp.n_candidates` 3 → 7** now that exact verification has landed in `engine.py` | 2608.15454 (2608) | **+29 % to +37 % decode throughput, output identical by construction** (their measurement, 373M, B200, batch 1) | **1 config field** | low — under rejection sampling the distribution is exact at any n | Serve one `local` checkpoint, sweep `n_candidates` ∈ {3, 5, 7}, read `bytes_per_sec` and `mbp_accept_rate` from the engine's own `stats()`. Watch the `PAUSE_BELOW` break-even guard fire |
| 6 | **Score every Muon A/B on wall-clock, and log per-group applied-update RMS** | 2608.07436, 2607.20512, 2608.19491 (2608/2607) | Prevents adopting a 2.2×-in-steps result that is 1.28× in seconds — 2608.07436's own headline shrinks exactly that much. The RMS log is the elasticity diagnostic reduced to two `.norm()` calls | 1–2 h | none | Add `muon_upd_rms` / `adam_upd_rms` to the JSONL. If the ratio drifts up over a run, the split-optimizer concern is live here; if flat, close it |
| 7 | **Byte-level evaluation hygiene**: strict UTF-8 DFA validity; drop HumanEval/MBPP/BBH; report BPB on byte-matched windows | 2606.14122, 2605.12928 (2606/2605) | No throughput effect. Makes "honest evaluation of byte-level models" a solved problem instead of an open one | 3–4 h | none | Add a DFA validity check to `train.py::evaluate`. 2606.14122 measures validity converging at **4.2B tokens vs perplexity at 2.1B — a 2× lag** — so BPB alone will say "converged" while output is still undecodable |
| 8 | **Data budget: ~60 bytes per parameter** | 2605.01188 (2605) | Sets the flagship budget: 105M → **≈6.3 GB of bytes**; 35M → 2.1 GB; 12.7M → 760 MB | 0 (a planning number) | low | Compare against what `data/local_mix.meta.json` actually holds. If we are far under, more data beats every architectural change on this list |
| 9 | **LM-head kernel diagnostic** — measure and publish that the gradient bottleneck does not exist here | 2603.10145 (2603) | Zero on throughput. One verifiable number for the honest-evaluation argument: 95–99 % of the logit gradient is destroyed in tokenizer LMs; here `ker(Wᵀ) = {0}` exactly at `local`/`flagship` | 1 h, read-only | none | 40-line QR script; report `‖p_ker(∇_L L)‖/‖∇_L L‖` at `pilot` and `local`. Expect ≈0 |
| 10 | **Relation vs attention: run the A/B we have never run** | 2608.20172 (2608), 2608.02032 (2608) | Possibly **+15–24 % throughput** if Relation loses. FlashRelation is measured at only **0.764–0.849× of PyTorch FlashAttention**, buying ΔNLL −0.015 at 30M where the per-seed SD is 0.0136 — *larger than the effect* | 1–2 d | medium | Swap `FullRelation` for plain MHA behind a config flag, `local`, **≥3 seeds**, step-matched. DART (2608.02032) is the better-evidenced alternative at 130M |
| 11 | **Head-wise partition of `w1`/`w2` for Muon / Adam-mini** | 2608.16760 (2608) | Small. `w1`/`w2` produce `P1`/`P2` whose only use is a per-head score — structurally the Query/Key case the thesis says to split by head | 2 h on top of #4 | low | Fold into #4's A/B |
| 12 | **L-shape scaling profile toward 1B** | 2608.07222 (2608) | Recovers a full-grid-quality scaling law from the cheap edges only: Farseer 5.1e21 vs 5.0e22 FLOPs at better accuracy than full-grid Chinchilla | 2–4 h script + 12–20 tiny runs | **medium-high** | **Gated on #1/#2.** Every fitted grid in that paper starts at ≥57M; and while `bpic` drifts during training, compute-per-byte is not constant across runs, so the law would be fit to a moving architecture |
| 13 | **Shape-gated kernel dispatch for FlashRelation** | 2608.12700 (2608) | A shape-gated selector gave **2.174× geomean** over a serial default in that paper. We have dynamic chunk shapes — the exact case | 0.5–1 d | low | Benchmark `flash_relation` across the chunk-count buckets that actually occur; dispatch per bucket |
| 14 | **Latent feedback on the Relation stack** | 2608.08888 (2608) | Per-*token* gains bought with 1.28–1.5× extra training FLOPs and k× activation memory. On a card already at 5.9/8 GB and 7.4 % MFU this is the wrong currency | 1–2 d | **high** | `pilot`, baseline vs 2-pass, **equal wall-clock**. If it does not win in seconds on one 4060 Ti it is dead |
| 15 | **DeltaMomentum** | 2608.19491 (2608) | 46 % fewer steps at 67M, 22 % at 370M — but 11–20 % more FLOPs and per-layer hooks on a launch-bound GPU | 1–2 d | medium | **Do not implement first.** Run the paper's §4.2.3 diagnostic (per-layer input-covariance condition number, effective rank, top-1 variance fraction) on a `local` checkpoint. If our feature spectra are already well conditioned, the mechanism has nothing to fix — 2 h vs 2 d |

---

## (c) The papers

Ordered by how much they bear on Morpheme, not by ID.

### C1. 2608.16760 (2608) — *On the Principles Behind Neural Network Optimizers*
Yushun Zhang (CUHK-Shenzhen), PhD thesis, INFORMS Dantzig Dissertation Competition version.
`arXiv:2608.16760v1 [cs.LG] 17 Aug 2026`. **The constituent papers are NeurIPS 2022 / NeurIPS 2024 /
ICLR 2025 — this is a 2026 packaging of older work, and I flag that against the recency policy.**

**What it is.** Four connected results. (i) Adam's divergence debate dissolves once you notice Reddi
et al. pick (β₁, β₂) *before* the problem while practitioners pick them *after*: with the problem
fixed there is a phase transition in the (β₁, β₂) plane, and Adam provably converges above a
**batch-size-dependent** threshold `β₂ ≥ γ₁(n) = 1 − O((1−β₁ⁿ)/n⁵)`, with `n` the number of
mini-batches. (ii) Transformer Hessians become **near-block-diagonal with strongly heterogeneous
block spectra** during training — one block per *row* of a weight matrix, one per *head* for
Query/Key — and that structure, not the loss, is why a diagonal preconditioner works. (iii) The
structure comes from consecutive multiplication of large matrix variables, provable by random matrix
theory. (iv) Therefore keep one `v` per Hessian block rather than per parameter: **Adam-mini**.

**Results.** Adam-mini pre-trains Llama-2 at **39M, 67M, 102M, 162M, 271M and 1B** on C4 at
Chinchilla budgets and "closely tracks AdamW across both compute and model scales" at **50 % less
optimizer-state memory**, independently reproduced by a Stanford benchmark ("closely tracks…
sometimes even performs better"). Theorem 3.2 gives Adam `Õ(max_l κ_l)` against GD's `Ω(κ)` on
block-diagonal quadratics, with a 1 000-trial simulation putting `r ≤ 1000` w.p. ≥ 2/3, hence ~5×
faster than GD at κ = 5000. JS distances between blockwise Hessian spectra are much smaller in
ResNet-18 than in BERT or GPT2-nano — the proposed explanation for why Adam ≈ SGD on CNNs. The Muon
connection is exact algebra:
`vec(W_{k+1}) = vec(W_k) − η (I_m ⊗ (M_kᵀM_k)^{−1/2}) vec(M_k)`, i.e. **Muon *is* a block-diagonal
preconditioner with one block per row**, matching the Hessian structure — but applying the *same*
preconditioner to every row, which **NorMuon** fixes with per-neuron rates at no extra memory.

**Evidence vs claim.** *Evidence:* the 39M→1B Adam-mini curves; the Hessian visualisations; the JS
distances; the independent reproduction. *Claim:* the β₂ threshold is explicitly **"not claimed
tight"**; Theorem 3.2 needs β₂ = 1 and a block-diagonal quadratic; whether CNNs even have the block
structure is admitted **untested** ("why Adam lacks advantage on CNNs remains largely open"). The
DeepSeek adoption of head-wise Muon rests on a recommendation letter and the Kimi K3 adoption on a
blog post — **[unverified from a primary technical source]**, as is the NanoGPT-speedrun leadership
claim. This is a thesis's own framing of its own line of work.

**Relevance — ADOPT (β₂ A/B), then TEST (Adam-mini). The highest-value paper in the batch.**

*β₂ first.* `build_optimizer` passes `betas=(0.9, 0.95)` and our step sees 32 768 bytes. The thesis
quotes three independent LLM-scale confirmations (Zhang 2024a: "larger β₂ … substantially improves
small batch size training"; Porian 2024: "essential at lower batch sizes"; Srećković 2025) and the
practitioner heuristic **β₂ ← β₂^(B₂/B₁)**, which against DeltaMomentum's 524 288-token step maps
0.95 → 0.95^(1/16) ≈ 0.997. Remark 4 warns larger is *not* monotonically better.
*File:* `train.py` — expose `--betas`. *Effort:* 15 min + 2 runs. *Risk:* none.

*Adam-mini.* ≈141 MB freed at `local`. *File:* new `adam_mini.py` + `build_optimizer`. *Effort:*
0.5–1 d. *Risk:* low — Relation's `w1/w2/wi/wo` are separate `[d,d]` matrices (`relation.py:93-96`),
so "partition by row" needs no surgery. Mamba-3's row-stacked `in_proj` should be partitioned by
sub-projection, which is the same reasoning `split_muon_params` already applies to it.

*Head-wise partition.* `w1`/`w2` produce `P1`/`P2` whose only use is `U_ij = p1_i·p2_j/√d_h`
**per head** — structurally the Q/K case. Two lines in `split_muon_params`.

---

### C2. 2608.19491 (2608) — *DeltaMomentum: A Key-Value based Anisotropic Momentum Update via Delta Rule*
Euijin Hong, Guannan Qu (Carnegie Mellon ECE). `arXiv:2608.19491v1 [cs.LG] 19 Aug 2026`.

**What it is.** A linear layer's per-sample gradient is the outer product `g = δxᵀ`, so the momentum
buffer is an associative memory keyed by the input activation. Replace the EMA
`M ← βM + (1−β)g` with the Widrow–Hoff delta rule `M ← M(βI − ηxxᵀ) + ηδxᵀ`: the forgetting factor
becomes *data-dependent*, erasing and rewriting directions currently being queried while leaving
unqueried ones to decay at β. The deployed batch form is `M ← βM + η(G − M Σ̂)` on row-normalised
keys; the fixed point `Ĝ(μI + Σ̂)⁻¹` with `μ = (1−β)/η` is a Tikhonov-regularised Wiener predictor —
exactly K-FAC's input-side Kronecker factor, without inverting anything.

**Results.** Llama-2-style 24-layer decoder, d_head 64, SwiGLU 2.67, untied embeddings; 10BT
FineWeb-Edu, V = 32 000, seq 2048, **524 288 tokens/step**, bf16, cosine + 500-step warmup, H200s,
matched-validation-loss levels from step 2 000, Optuna/TPE on the 67M proxy then μP transfer.

| scale (seeds, tokens) | mean loss gap vs AdamW | mean step reduction | max step reduction |
|---|---|---|---|
| 67M (3 seeds, 10BT) | 0.0936 ± 0.0150 nats | 39.25 ± 4.55 % | **46.39 ± 4.32 %** |
| 370M (3 seeds, 10BT) | 0.0383 ± 0.0028 | 17.61 ± 0.95 % | 22.12 ± 0.80 % |
| 1B (1 seed, 20BT, Chinchilla) | 0.0245 | 13.43 % | 19.37 % |

Cost: counted per-step FLOP overhead **11.2 % / 17.4 % / 20.3 %** at 67M / 370M / 1B; measured step
time **1.177× / 1.153×** on one H200; FLOP crossing at 420 PFLOPs of a 4.0 EFLOP horizon at 67M.
Zero persistent optimizer-state overhead. CIFAR-10: ViT-Tiny 0.079 ± 0.013 nats (25.1 %), ResNet-18
0.041 ± 0.006 (11.1 %), 3 seeds each.

**The number that matters here.** Appendix O reports a tuned **Muon** arm, 3 seeds at 67M and 370M,
same sampler and pruner, tuned under Muon's *own* μP rule (α = Θ(n^−1/2) with the √(m/n) prefactor)
with a decoupled auxiliary learning rate, and given the **largest** per-dimension budget of the three
arms (38 full-trial equivalents vs 33 AdamW / 32 DeltaAdamW). DeltaAdamW beats it by
**0.120 ± 0.008 nats at 67M** and **0.109 ± 0.003 at 370M**.

> **Differencing Table 2 and Table 8 — same window, same seeds, same protocol — puts tuned Muon
> behind plain AdamW by ≈0.026 nats at 67M and ≈0.071 nats at 370M.** This is *my* arithmetic; the
> paper never states it and writes "We make no general claim about Muon from Table 8." At 67M the
> implied deficit is within roughly 1.5 combined seed-sd — directional only. At 370M it is an order
> of magnitude larger than the reported sds. Muon was not run at 1B.

**Evidence vs claim.** *Evidence:* the AdamW comparison at 67M/370M (3 seeds, window fixed before the
numbers were computed, shared init and data order); the FLOP and wall-clock accounting; three
mechanistic diagnostics all moving as predicted. *Claim:* 1B is **single-seed** and the paper calls
it a scale check. The theory assumes quasi-static parameters, and the rate corollary needs linear
regression plus an underdamping band — Lemma 3.9 extends the determinant argument past the squared
loss, but a smaller determinant implies a smaller spectral radius only when underdamped. Wall-clock
uses a "research-quality, non-fused" implementation.

**Relevance — TEST, but not first.** *Where:* `train.py::build_optimizer`. *Expected effect:* the
step saving is largest at the smallest tested scale (46 % at 67M) and our presets are *below* 67M, so
the direction is favourable — but it needs forward/backward hooks capturing `x̂` and `δ` per linear
layer, and most of Morpheme's linear layers are inside Mamba-3's fused Triton path where those are not
exposed. Realistically it reaches the Relation stack and the MBP head — **77 % of `local`'s
parameters**, so not a small fraction. *Effort:* 1–2 d. *Risk:* medium — 11–20 % more FLOPs at 7.4 %
MFU, plus per-layer Python work of exactly the kind efficiency §0.2 identified as the bottleneck; the
measured 1.15–1.18× step time is an optimistic transfer to a launch-bound card. *Deciding experiment:*
run §4.2.3's diagnostic first (2 h) rather than the implementation (2 d). Two references worth chasing:
**Newton-Muon, arXiv 2604.01472** and **DoPr, arXiv 2606.06418** — **[unverified, cited only]**.

---

### C3. 2608.07436 (2608) — *Post-Grokking Collapse at the Representation–Readout Interface in Muon-Trained Transformers*
Janati (Columbia DSI), El Maghraoui (Columbia CS), Kanavalau (Stanford), Belfatmi (CentraleSupélec).
`arXiv:2608.07436v1 [cs.AI] 7 Aug 2026`. Code at `github.com/Na00s/muon-grokking`.

**What it is.** Under the standard split routing (Muon on hidden matrices, AdamW on embeddings and
the unembedding), a transformer on `(a+b) mod 113` groks faster and then *loses generalization*,
repeatedly, over hundreds of thousands of steps. The residual stream has no privileged basis, so
`(Rh, W_U R⁻¹)` computes the same function for any invertible `R`; once the training set is solved
the loss stops selecting a member of that family, and the two optimizers respond to the residual
gradient differently. Newton–Schulz normalises its input before orthogonalizing, so Muon's step does
not shrink as the gradient does; AdamW's does. The readout stops being able to decode the
representation it grew alongside.

**Results.** Depth-1/2/4 decoder-only transformer, d_model 128, 4 heads, MLP 512, seq len 3, **no
normalisation layers**, 226 048 params at depth 1; `p = 113`, 30 % train, full batch; MPS float32 for
sweeps, CPU for the deterministic five-seed replication.
* **Speed:** Muon 9/9 configurations grok (mean 13 011 steps) vs AdamW 7/11 (mean 30 486). Depth-2:
  2 300 vs 5 100 steps = **2.22× in steps, 1.28× in elapsed time** — a Muon step costs **1.75–1.80×**
  an AdamW step (57.81 vs 33.06 ms at depth 2; 110.87 vs 61.68 at depth 4). Depth-4: Muon 52 600
  steps / 5 856 s; AdamW never sustains the threshold in 300 000 steps / 19 768 s.
* **Instability:** **none of the nine Muon configurations is strictly stable**, and sub-threshold
  counts have *no monotone relationship* to lr or wd (56, 107, 2, 145, 592, 5, 130, 473, 265). AdamW
  is not exempt — 4 of 7 grokking configurations collapse, severity rising monotonically with lr from
  0 at 1e-3 to **918 at 1e-2**. Across five seeds Muon records **137–321** sub-threshold evaluations
  on all five (minima 16.27–76.05 %); the selected AdamW baseline records 1–2 on four of five (minimum
  27.59 %). *The optimizers differ in severity by two orders of magnitude, not in whether it happens.*
* **Mechanism**, over the 691 steps before one collapse: training loss **1.5×10⁻⁷**, hidden gradient
  ~10⁻⁶, non-hidden ~10⁻⁵. Log–log elasticity of applied step on gradient norm: **−0.026 (R² 0.84)**
  for Muon vs **+1.51 (R² 0.98)** and **+1.47 (R² 0.95)** for embeddings and readout. Net
  per-parameter displacement **0.073 vs 0.0091 — a factor of 8.0**. On the hidden group the
  gradient-driven step (0.1499) and weight decay (0.1475) oppose to within 1.6 %, and the *residual*
  grows 64 % (0.0462 → 0.0759) while the orthogonalized step stays flat.
* **Fix:** freezing embeddings + unembedding after the circuit forms ("Stable Muon") gives **0
  sub-threshold evaluations across 451 400 post-grokking steps, 4 519 evaluations, five runs**, and 0
  on all five paired seeds where the unfrozen arm records 137–321. Freezing the unembedding *alone* is
  not enough (18 of 936 evaluations below 95 %). Per-step cost *falls* after the freeze.
* **Representation:** Muon spreads power over **326.09 effective conjugate pairs** (of 6 384) vs
  AdamW's **4.95**, leaving the task-aligned Fourier family holding 28.0 % of non-constant power
  against AdamW's 91.0 %. Ablating normalisation + orthogonalization collapses this to 4.11 — *past*
  AdamW's — and all six ablated runs end in a **non-finite loss**.
* **The diagnostic warning:** across a collapse where test accuracy falls 100 % → 19.04 %, the set of
  dominant frequencies is **unchanged (Jaccard 1.0000)** and their power distribution keeps
  **cosine similarity 0.9899**. Spectral progress measures report an intact circuit at the step the
  model stops computing the task.

**Evidence vs claim.** *Evidence:* the sweeps, the five-seed deterministic replication, matched
branches from bit-identical in-memory states, the elasticity regressions, the freeze interventions,
and Fourier interventions across 98 checkpoints. Unusually careful. *Claim:* the matched-collapse
analysis, cross-readout substitutions and layerwise causal analysis are **single-seed**. One
architecture, one task family, ~2×10⁵ params, at a loss of 1.5×10⁻⁷.

**Relevance — TEST the diagnostic, do not adopt the fix.** Morpheme has the same split
(`split_muon_params` puts hidden 2-D matrices on Muon; `embeddings`/`lm_head`, which are **tied**, and
everything with `ndim != 2` on AdamW). Three things break the analogy, all in our favour:
1. **Our gradient never vanishes** — `grad_norm` 0.33–0.35 on a corpus we cannot interpolate. The
   elasticity mechanism needs a loss that has gone flat.
2. **Our embedding and readout are the same tensor** (`tie_embeddings=True`, `hnet.py:75`). The writer
   and reader cannot drift apart the way two independent matrices can.
3. **We have RMSNorm everywhere.** The paper is explicit that its model has *no* normalisation layers
   precisely so the symmetry is full `GL(d)`; RMSNorm restricts it to `O(d)` and a learned gain
   restricts it further.
*What to do:* nothing to the model. **Add the instrument** — log per-group applied-update RMS beside
`grad_norm`. 1 h, no risk. If the Muon/AdamW displacement ratio drifts upward, the concern is live and
late-training freezing of the tied embedding becomes worth an A/B; if flat, close the question.
*The number to carry over regardless:* **a Muon step costs 1.75–1.80× an AdamW step** here, and the
headline 2.22× shrinks to 1.28× in seconds. **Any Muon A/B must be scored on wall-clock.**

---

### C4. 2607.20512 (2607) — *The Active Ingredient in Muon's Grokking*
Yufeng Wang, independent researcher. `arXiv:2607.20512v1 [cs.LG] 6 Jul 2026`. Code released.

**What it is.** A 2×2 ablation of Muon's two components — the Newton–Schulz orthogonalization and the
per-matrix scale `γ = √max(1, n_out/n_in)` — under two metrics (first threshold crossing and
*sustained* generalization), because the first can invert the ranking of the second.

**Results.** One-layer 4-head transformer, d_model 128, mod-97 addition (subtraction and
multiplication as checks), 40 % train, ≥5 seeds/cell, Holm–Bonferroni corrected, 258 runs / 3.1×10¹⁶
FLOPs, bit-for-bit deterministic per seed.

| variant | ortho | spectral | first-cross | stable-grok | FLOPs |
|---|---|---|---|---|---|
| M0 full Muon | 5 | yes | 1856 ± 118 | 1856 (stable) | 2.96e13 |
| M1 orthogonalize-only | 5 | no | 1744 ± 113 | 1744 (stable) | 2.78e13 |
| M2 spectral-only | 0 | yes | 3088 ± 1107 | 4932 (unstable) | 4.69e13 |
| M3 AdamW | — | — | 2372 ± 360 | 4488 (unstable) | 3.60e13 |

M1 vs M0 p = 0.21 (indistinguishable); M1 vs M3 p = 0.010 (primary, survives Holm); M2 vs M3 p = 0.25
with σ = 1107. Newton–Schulz sweep ns = 0/1/3/5 → first-cross 3088/1544/2000/1856, stable-grok
4932/3136/2000/1856: **ns = 1 is fastest to touch and worst to settle.** "Lean1" (ns=1, no spectral)
reaches 0.95 in 1395 ± 196 steps (20 % sooner than Muon, p = 0.005) then collapses on **6/8 seeds**.
Mechanism: orthogonalizing variants grok at max singular value 5.5–6.2 vs AdamW's 15.9 (**~3× lower
spectral norm**) with a near-uniform embedding Fourier spectrum; an embedding-movement control
(relative ‖ΔE‖ ≈ 0.994–0.998 for *every* optimizer) rules out "Muon just moves the embedding less".

**Evidence vs claim.** *Evidence:* the ablation, the lr sweep (3e-4, 1e-3, 3e-3), the operation
robustness check, the movement control, the determinism. *Claim:* §5.3's account of why ns=1
destabilises is labelled a hypothesis; the Fourier claim uses "a coarse proxy" by the author's own
admission; one architecture, one scale, one task family, and the at-scale behaviour "where
orthogonalization is not free" is untested.

**Relevance — ADOPT one comment, and take one do-not-do.**
* **Do NOT remove our scale factor.** `muon.py`'s
  `upd * (rms_scale * max(p.shape[0], p.shape[1]) ** 0.5)` is **not** the factor this paper ablates.
  Ours is Liu et al. 2025's RMS match (`0.2·√max(A,B)`) whose purpose is to let Muon reuse AdamW's lr
  and wd — exactly what `build_optimizer` relies on when it hands both optimizers the same `lr`.
  Removing it would silently change the effective learning rate. Add a code comment saying so; this is
  a near-miss a fast read would get wrong.
* **Keep `ns_steps = 5`.** The clearest actionable finding is negative: fewer Newton–Schulz iterations
  trade stability for speed, and five is the rate-robust operating point. `muon.py` already defaults
  to 5. This closes off an "optimisation" that would have looked attractive on a launch-bound GPU.
* *Effort:* 15 min. *Risk:* none. Together with C3 (which cites this paper as [17]) the two give the
  sharpest current picture: orthogonalization produces a **lower-norm, more widely distributed**
  solution, and that same distribution is what drifts when the loss goes flat.

---

### C5. 2608.07222 (2608) — *Skaling: Chinchilla's Exponents Meet Kaplan's Coupling*
Videau, Youbi-Idrissi, Lopez-Paz, Ahuja (FAIR at Meta). `arXiv:2608.07222v1 [cs.CL] 7 Aug 2026`.

**What it is.** Replaces `L = A/N^α + B/D^β + E` with **`L = (A/N^α + B/D^β)^k + E`** — one extra
outer coupling exponent; k = 1 recovers Chinchilla. The motivation is measured: two independent
mesh-free estimators (moving least squares and a Gaussian process) over ~400 existing runs show the
mixed derivative `∂²L/∂N∂D` is non-zero and **negative** everywhere, which any additive law forces to
be identically zero. Appendix A.2 shows an additive interaction term cannot fix it without predicting
loss *increasing* with model size. Compute-optimal allocation keeps its closed form.

**Results.** Fitted on Farseer (404 configs, 100M–6.4B, 1B–512B tokens) and their own SK-Grid (134
configs, 134M–4.9B), 5-fold CV, log-space Huber + L-BFGS-B with 2 000 basin-hopping restarts.

| grid | law | R² | interp | Ext-N | Ext-D | Far |
|---|---|---|---|---|---|---|
| Farseer full | Chinchilla | 0.995 | 0.77 ± 0.04 | 1.48 ± 0.03 | 1.98 ± 0.08 | 2.46 ± 0.19 |
| | **Skaling** | **0.998** | **0.41 ± 0.05** | **0.47 ± 0.03** | **0.88 ± 0.06** | **2.31 ± 0.18** |
| Farseer **L-shape** (10× cheaper) | Chinchilla | 0.954 | 2.51 ± 0.07 | 4.32 ± 0.13 | 3.29 ± 0.11 | 9.82 ± 0.48 |
| | **Skaling** | **0.995** | **0.85 ± 0.10** | **0.89 ± 0.23** | **1.35 ± 0.20** | **1.51 ± 0.67** |
| SK-Grid **L-shape** | Chinchilla | 0.955 | 2.19 ± 0.10 | 6.09 ± 0.24 | 3.63 ± 0.13 | 14.63 ± 0.39 |
| | **Skaling** | **0.998** | **0.33 ± 0.03** | **0.77 ± 0.44** | **0.55 ± 0.08** | **1.15 ± 0.53** |

The **L-shape** strategy trains only the cheap grid edges — sweep D at the smallest models, sweep N at
the shortest horizons — at ~10× less fitting compute, and still beats a full-grid Chinchilla fit.
Fitted k: 0.41 ± 0.01 (Farseer full), 0.31 ± 0.02 (SK-Grid). On 112 held-out highest-compute Farseer
runs Skaling's pooled MAPE is **0.60 ± 0.27 (R² 0.98)** against Chinchilla's **2.34 ± 1.11 (R² 0.74)**
— a 3.9× reduction. Allocation: empirical `D*/N* ∝ C^m` gives m ≈ −0.14/−0.15, close to Skaling's
analytic −0.11 and **opposite in sign** to Chinchilla's +0.03.

**Evidence vs claim.** *Evidence:* four datasets (three external), 5-fold CV with reported std, two
independent derivative estimators, one fitting procedure held constant across laws, and a
parameter-count control (the 9-param Farseer law loses to the 6-param Skaling law). *Claim:* the 5
folds are resamples of the (N, D) grid, **not training seeds** — per-configuration training noise is
never quantified. The authors are honest where it costs them: E ≈ 0 on Farseer should *not* be read as
a vanishing loss floor; allocation direction is dataset-specific; Skaling ≈ Chinchilla whenever
k ≈ 0.77–0.90 (which is what they measure on Farseer-code and the Besiroglu data, where **Skaling is
worse on Ext-N**); and §E.2 concedes the fitted k is a property of (architecture, data, HP recipe),
not a constant of nature. v1 sloppiness: §4.1 says 134 configs / 15 sizes while Table 7 lists 125 / 14;
§3.1 says "100-fold" where §4.2 and §A.3 say ~10×. **Scale floor: the smallest model in any fitted grid
is 57M.** Nothing here is evidence about the sub-100M regime, which is where Morpheme lives.

**Relevance — ADOPT the L-shape strategy for the path to 1B, but gate it.** *Where:* new code, not an
edit — a ladder of widths below `pilot` in `config.py`, exact N and D logged per run in `train.py`, and
care with `flops.py`, whose per-stage accounting will **not** equal the paper's `C = 6ND`. Fit on
Morpheme's own N and D; do not mix conventions. *Effort:* 2–4 h script + 12–20 tiny runs (2–5 days
wall-clock at `pilot`). *Risk: medium-high, for three specific reasons.* (1) Every fitted grid starts
at ≥57M; our ladder sits below that, where E and k are least identified. (2) The measured coupling
depends on the HP recipe, and we use hand-set LRs, so the fitted surface will partly encode HP
mistuning. (3) **`bpic` drifts during training, so compute-per-byte is not constant across runs** —
every scaling law assumes it is. **Fix item #1 before fitting anything, or the law is fit to a moving
architecture.** *Deciding experiment:* run 3 sizes × 3 budgets at `pilot` and check whether plain
Chinchilla already fits within ~1 % MAPE. If it does, k ≈ 1 and Skaling buys nothing — exactly what
the paper found on two of its four datasets. Also adopt the **dominated-pair fit** (App. F) as the
default fitting mode, since E will be badly identified at our scale.

---

### C6. 2603.10145 (2603) — *Lost in Backpropagation: The LM Head is a Gradient Bottleneck*
Nathan Godey, Yoav Artzi (Cornell). `arXiv:2603.10145v2 [cs.CL] 10 Jul 2026`, COLM 2026.
**Oldest item in the batch** — original submission March 2026, five months before the rest.

**What it is.** The logit gradient `∇_L L = diag(f)(P_θ − Ñ)` is near-full-rank (V−1) by a
Gershgorin / graph-Laplacian argument, but reaches the backbone only through `W_θᵀ`, so everything in
`ker(W_θᵀ)` is annihilated. Worse, the realizable logit *update* has rank ≤ 2D (one rank-D term from
updating `W`, one from updating `H`), so Eckart–Young gives a strictly positive residual whenever the
gradient's rank exceeds 2D. The architecture-independent generalisation: for **any** head
`g_θ(H) = L`, `∇_H L = ∇_L L · J_g(H)` with `rank(J_g) ≤ D`.

**Results.** On 10 000 shuffled FineWeb documents across GPT-2, Pythia, Llama-3, OLMo-2 and Qwen-3-Base
at D/V ∈ [0.02, 0.10]: **95–99 % of the logit-gradient Frobenius norm falls in the null space**, with
surviving-gradient cosine ≈0.1–0.3 (the paper is internally inconsistent: 0.1–0.2 in §2.4, 0.1–0.3 in
§3.3). OLMo2-1B over ~4 000B tokens: lost fraction flat at 98.900–99.05 %. Controlled 2B pretraining
(6 layers, d_m 4096, factored head `W = A·B`, D swept 32→4096, 11B FineWeb-Edu tokens, V = 49 152,
B200s, ~760 GPU-hours, **one run per D, no seeds**): D = 4096 reaches D = 32's *final* loss within
700M tokens — a **×16 convergence speedup** — with downstream weighted average rising monotonically
41.37 → 49.41, D = 2048 → 4096 worth only **+0.55**. All three proposed mitigations (orthogonality
regularisation, an alignment auxiliary loss, feedback alignment) **made convergence slower, with no
numbers reported.**

**Evidence vs claim.** *Evidence:* the 95–99 % measurement across five model families — cheap to
replicate and the strongest thing here; the near-full-rank logit gradients on the Pile; the monotone
D→quality ordering. *Claim:* the ×16 is a single unreplicated run per arm read off a figure, with
total parameters varying 1.8–2.0B across arms; causality is inferred, not isolated — varying head rank
D also varies the factored head's expressivity and no ablation separates them; §2.4 concedes the
theory "establishes the inevitability of a compression … but not its severity." Code unreleased.
**No experiment anywhere varies V at fixed D on real data**, so this paper does *not* say small
vocabularies are better.

**Relevance — TEST (a one-hour diagnostic), change no code. Evidential value only.**
The pathology **does not exist here, by construction**:
* `VOCAB_SIZE = 262`, `pad_vocab_to = 264`, `lm_head` tied to `embeddings`; `d_model_outer` is
  256 / 384 / 512, so **D/V = 0.97 / 1.45 / 1.94** against the paper's measured range of 0.02–0.10.
  We are 10–100× outside the regime where 95–99 % loss was observed.
* The rank bound is **vacuous at every preset**: it bites only when `rank(∇_L L) > 2D`, and here
  `2D = 512 / 768 / 1024` while `rank(∇_L L) ≤ V−1 = 263`.
* At `local` and `flagship`, `ker(W_θᵀ) = {0}` **exactly** (D0 > V ⇒ full row rank). Only `pilot` has a
  kernel, at most 8-dimensional out of 264, upper-bounding the destroyed fraction at ≈**17 %** under an
  isotropic-gradient assumption. *(That 17 % is a derivation from the paper's construction, not a
  number it reports.)*
* Same reasoning covers `mbp.py`: `LCAHead` runs at width D0 and feeds the shared `lm_head`, so Eq. (10)
  applies but is non-binding.
*Experiment:* ~40 lines beside `train/report.py` — `torch.linalg.qr(lm_head.weight)` (264×D0), take
the trailing `V − D0` columns of Q, report `‖p_ker(∇_L L)‖_F/‖∇_L L‖_F` and the surviving cosine at
`pilot` and `local`. *Effort:* 1 h. *Risk:* none. *Payoff:* one verifiable number for the honest-
evaluation question. **Do not overclaim the ×16** — the paper varies D at fixed V and never the
reverse on real data, and is silent on the sequence-length cost bytes pay. **Do not extend it to the
router** — dynamic chunking compresses along sequence length, not vocabulary rank. One follow-up worth
a look: **arXiv 2605.06997** *Echo: KV-Cache-Free Associative Recall with Spectral Koopman Operators*
— **[unverified, title only]**.

---

### C7. 2608.08888 (2608) — *Full-bandwidth transformer*
Wang (JHU), Cai (Princeton), Zhan, Dong, Fan, de Rosa, Pearce, Langford (Microsoft).
`arXiv:2608.08888v1 [cs.AI] 9 Aug 2026`.

**What it is.** At each decode step the input is a gated fusion of the sampled token embedding with the
*previous step's top-layer hidden state*: `e_t ⊗ h_{t−1} = (W_U h_{t−1}) ⊙ σ(W_G e_t)`. The asymmetry
is load-bearing — the hidden state is on the value path and the token is only a multiplicative gate, so
the model *cannot* learn to ignore the state (an additive fusion could). Everything else is unchanged;
cost is two D×D matmuls per token (<1 %). Training uses a Jacobi-style **k-pass "temporal
parallelism"**: pass k shifts pass k−1's states right by one and re-runs the stack in parallel over all
positions, with the NTP loss applied to *every* pass and gradients **not** detached, so memory grows
with k.

**Results.** 1B decoder, 24 layers, d_model 1536, GQA 16/8, 8192 context, tied 100 352-token
embedding, Phi-4 data, **NorMuon** for matrix params + Adam for the rest, WSD. The four runs and the
authors' own token-equivalent accounting: 10B tokens → 40B equivalent; 100B → 150B; 200B → 256B;
400B → 512B. Post-instruction-tuning (Table 1), FB-400B FUSED vs Standard-400B / Standard-1T:
GSM8K **71.80** / 68.39 / 70.13; MATH-500 **48.40** / 46.40 / 47.40; HumanEval 47.60 / 44.85 /
**50.01**; MBPP 41.70 / 40.28 / **41.93**. Stability ablation (Fig. 3), the most interesting result: a
75 %-one-pass / 25 %-two-pass mixture **diverges** past its trained depth, while adding just **3 %
three-pass batches** makes the map a contraction — validation loss flat through 30 feedback steps and
stable to **1 000** passes (Fig. 10). Layer-0 linear probes reach **99.6 %** on completion tracking and
**100 %** on delayed memory under one-step recurrent prefill, versus near-chance under standard prefill.

**Evidence vs claim.** *Evidence:* four real 1B runs up to 400B tokens; Tables 1–2; the stability and
prefill-scaling figures; the probes. *Claim:* **single seed throughout** — no run-to-run variance for
any headline number, and Fig. 4's error bars are per-task standard errors, not seeds. The Table 2
comparators were **lifted from a third paper**, different data and harness. Decoding temperature was
grid-searched *separately per method* on coding tasks. The abstract's "match transformers trained with
roughly 1.5× more tokens" is a per-*token* comparison bought with 1.28–1.5× extra training FLOPs — the
paper reports this honestly in its run table. And the authors correctly note that "improved decodability
does not by itself imply improved output": the 99.6 % probe shows information is *present*, not used.

**Relevance — TEST last, and only at equal wall-clock.** Three separable pieces.
*(a) Latent feedback on the Relation stack.* Ports structurally — Relation is attention-like and
KV-cached — but Morpheme's encoder/decoder are Mamba-3 and **already** carry recurrent state across
bytes, so the frozen-state problem is partly absent at the byte level. The only novel place is
chunk-rate: feed the Relation stack's top-layer chunk output into the next chunk's input. *Files:*
`hnet.py` (input construction to `main_network`), `relation.py`, a new `RelationCfg` field, and a
k-pass loop in `train.py`. **Expected effect on the metric that matters here: negative.** k passes cost
~k× forward+backward and, with no detach, k× activation memory; `local` already peaks at 5.9 GB on an
8 GB card. *Effort:* 1–2 d. *Risk:* high. *Deciding experiment:* `pilot`, baseline vs 2-pass, **equal
wall-clock**. If it does not win in seconds on one 4060 Ti it is dead for this project.
*(b) The stability recipe — free.* Weight tying: already done. The 3 %-three-pass contraction trick is
only meaningful with (a). Depth scaling so `‖h^L‖ ~ O(1)` is worth auditing in `norm.py`/`blocks.py`
for the 12-layer `flagship` regardless.
*(c) The drafter clarification — worth stating so it is not misread.* Appendix D notes the difference
from EAGLE/MTP: "our model feeds its own state back into the same model to define the actual next-token
distribution." **Latent feedback is not a drafter and cannot be bolted on under `verify_draft`'s
rejection sampling** — it changes the target distribution itself. Where it *is* decision-relevant: if
the MBP head fails to justify its 33 % of compute, latent-feedback training is a competing way to spend
that budget. Combining them is explicitly **untested** (their footnote 1).
*2026 papers it names that matter here:* **2607.15178** (T²MLR), **2605.26797** (Latent Recurrent
Transformer — reports the scaling behaviour this paper lacks), **2606.18206** (fixed-point reasoners,
source of the depth-scaling stability result), **2602.08984** (Next Concept Prediction — a direct
competitor to the MBP head for "how to spend the extra 33 %"), **2606.03938** (q0, NanoGPT slow-run:
deep ensemble + distillation under fixed data, squarely the single-GPU regime). **[all unverified —
cited, not read.]**

---

### C8. 2608.20061 (2608) — *Let's Scale Step by Step: Compute-Efficient Hyperparameter Transfer for Large-Scale Mixture-of-Experts*
Kim, Lee, Bak, Park, Kim (Kakao Corp. / Upstage AI). COLM 2026. `arXiv:2608.20061`.

**What it is.** A two-step recipe for picking one number — the peak learning rate — for a very large
MoE run without sweeping it there. Step 1: a μP formulation for MoE with Multi-head Latent Attention
under **Muon**, classifying parameters as vector-like (μP init only) or matrix-like (init + lr
scaling by `fan_in_base/fan_in`), scaling width and expert count together while holding depth, active
experts and expert intermediate dimension fixed. Step 2: a token-horizon law — fit a parabola of
validation loss against log lr at several token budgets, take each vertex, then regress
`log η* = β log B + γ` in log-log space and extrapolate.

**Results.** Under SP the optimal lr shifts across width; under μP it transfers from a 0.6B-total /
0.3B-active proxy to 2×, 4× and 8× width (30.7B total / 3.6B active), trained on 1.3B tokens with lr
swept over {1,3,6}e-4, {1,3,4,6}e-3, {1,3}e-2. The token-horizon regression over budgets 255B–502B on
a 10.8B/3.3B proxy achieves **R² = 0.95** and predicts **η\* = 3.85 × 10⁻⁴ for 10T tokens**; the
155B-total / 17B-active model trained on 10T tokens at that rate shows a **loss trajectory with no
spikes**. Total proxy cost is **1/98th** of the target run; the 1-D token sweep costs 64.8 ZFLOPs
against 305.1 for a 2-D sweep. Two engineering details that transfer: proxy runs are terminated in the
**stable** phase of WSD with **EMA (α = 0.6, updated every ~2B tokens)** standing in for a decay phase,
giving many budget points from one run; and Muon's lr is RMS-matched to AdamW by `0.2·√max(A,B)` — the
same factor `muon.py` implements.

**Evidence vs claim.** *Evidence:* the SP-vs-μP transfer figure across four widths; the R² = 0.95 fit;
the spike-free 10T trajectory; the compute accounting. *Claim:* the headline validation is a **single
production run** whose optimality was never checked against alternatives — the paper says so plainly:
"conducting exhaustive full-scale sweeps to definitively verify the optimality of the predicted
learning rate is computationally infeasible." A stable loss curve is consistent with a good lr and also
with a merely safe one. The sparsity axis is admitted to be confounded with width. No seeds anywhere.

**Relevance — IGNORE the MoE content, TEST two by-products.** Morpheme is dense, has no experts, no
MLA, and one GPU; nothing about expert routing or 10T horizons applies. Two things do:
1. **EMA-in-the-stable-phase as a cheap proxy for a decayed checkpoint.** `train.py` uses a WSD
   schedule with a 20 % decay tail; every hyperparameter comparison currently pays for that tail. If
   EMA of the weights during the stable phase approximates the decayed model, an α or β₂ A/B could
   read many budget points off one run. *Effort:* 3–4 h. *Risk:* low. *Deciding experiment:* one
   `local` run, compare EMA-at-step-k against a genuinely decayed checkpoint at the same k.
2. **The parabola-vertex protocol** — fit `L = a(log η)² + b log η + c` and take `η* = exp(−b/2a)` —
   is a better way to read our lr sweeps than picking the best of three. *Effort:* 30 min.
Neither needs the paper's MoE machinery. Its μP content is not usable: μP transfer is a *width* rule,
and Morpheme's presets change width, depth, layer counts and `max_seq_len` together.

---

### C9. 2608.16797 (2608) — *UniDot: A Unified Network for Sequence Modeling and Feature Interaction in Large-scale Recommendation*
Lin, Sun, Zhang, Xiong, Ji, Chen, Bu (Meta). `arXiv:2608.16797v1 [cs.IR] 17 Aug 2026`.
KDD Cup 2026 Tencent Uni-Rec Challenge, Industrial track.

**What it is.** A post-click conversion-prediction architecture built on the observation that the
factorization-machine inner product and attention's query·key score are the same primitive. One
stackable macro-block runs a token-mixing bus (Wukong blocks) and a sequence-retrieval bus (item tokens
cross-attending behavioural histories) **in parallel**, exchanging state each layer through an
MLP-Mixer fuser with zero-initialised side projections, while an "FM Highway" routes explicit
per-layer dot products *around* the residual stack straight to the classifier.

**Results.** Runner-up at **0.83217 AUC** against a winner at 0.83254 (gap 0.037 %). Incremental table:
baseline 0.81398 → +UniDot **+1.102 %** (the largest single jump) → … → +EMA weights +0.099 → width
64→96→128 +0.031/+0.019 → multi-path DML +0.085/+0.068 → all-data retrain +0.021; total **+1.818 %**.
Ablations (tiny mode, d = 64): removing the FM Highway costs **−0.127 %** (the most of anything);
removing cross-bus dots −0.087 %; removing sequence cross-attention only −0.053 %; **removing the
FuseFFN bus fusion *raises* AUC by +0.022 %** while worsening LogLoss — the authors read their own
fuser as under-designed. Scaling: doubling embedding width gave **no win**; widening the dense path
64→128 gave +0.050 % cumulative; **a second identical path under mutual distillation gave +0.135 % at
d = 64 — more than the entire width sweep** — at 2× dense parameters and FLOPs but unchanged embedding
tables. Optimizer: dual sparse/dense **Adagrad + Muon**.

**Evidence vs claim.** *Evidence:* a leaderboard placement on a held-out test set, a 15-row incremental
ablation and a 12-row component ablation. *Claim:* every number is one run on one competition dataset;
the "tiny mode" ablations use an in-distribution held-out split so "only relative gaps matter" (their
words); the mixer slot-swap comparisons against TokenMixer and UniMixer are **not matched end-to-end
retrains** and the authors say so.

**Relevance — IGNORE the architecture. One idea is worth naming and it is not free.** Advertising CVR
prediction over sparse ID features has essentially nothing in common with byte-level language
modelling: no vocabulary, no autoregression, no sequence generation, no KV cache. The one transferable
result is **multi-path mutual learning**: two identical paths over one shared embedding table, each
regularised toward the other's stop-gradiented prediction, beating the entire width sweep. Morpheme has
a structurally similar shared trunk — the tied `embeddings`/`lm_head` — and already trains two heads
(next-byte and MBP) on the same targets from different information. Their mutual term
`D(sg[p_i], p_n)` is one line. **But it costs 2× dense compute in training**, which on a card at 7.4 %
MFU is the wrong direction, and their +0.135 % is an AUC delta on a competition leaderboard, not a
language-model loss. *Verdict:* record it, do not schedule it. If anything, the **more** relevant
observation is the one the paper stumbles over: their strongest single component (the FM Highway) works
by routing an explicit low-order signal *around* the residual stack, and Morpheme already does the
analogous thing with `residual_proj` in `hnet.py`. No change.

---

### C10. 2608.01833 (2608) — *Tunneling the Loss Landscape: Bypassing Memorization with Monte Carlo Parameter Swapping*
Chan, Zhang, Shang, Zhang, Yang (City University of Hong Kong / UPenn / Air Liquide).
`arXiv:2608.01833v1 [cond-mat.dis-nn] 3 Aug 2026`. AAAI 2027 format.

**What it is.** Characterises grokking as glass-like kinetic arrest using three observables:
parameter mobility (net displacement per window), replica correlation (Pearson correlation between
independently perturbed trajectories branched from a common state), and trajectory fractal dimension
`D_f = 2/α` from the scaling of squared displacement against arc length. Then proposes **SAM-Swap** —
when `D_f` falls below 1.1, randomly exchange a fraction `r` of parameter *values within a layer*,
rejecting any swap that raises the loss above its initial value.

**Results.** One-layer transformer on `(x² + y) mod 67`, 30 % held out, full-batch Adam/AdamW without
dropout, averaged over **20 runs**, 10 000 epochs. Parameter mobility drops from ~1 to ~10⁻² by
t ≈ 10² and toward ~10⁻⁴ by the end. Replica correlation shows two-step relaxation at `t_w = 0` and
aging (suppressed decay at `t_w = 10², 10³`). `D_f` collapses to ≈1 after memorisation and rises again
at generalization. SAM-Swap at `r = 10⁻²` generalizes at **≈650 epochs against ≈3 000 for AdamW** —
a ~4.6× reduction. Large additive Gaussian noise (σ = 10⁻²) produces a similar acceleration.

**Evidence vs claim.** *Evidence:* 20-run averages for the three observables; the swap-ratio and
threshold sensitivity checks. *Claim:* the framework is descriptive and the paper's own conclusion is
that "accelerated generalization is consistently associated with random exploration" — i.e. **the swap
is not shown to beat plain large Gaussian noise**, which is the obvious cheaper baseline. Every result
is on one algorithmic task with a 2×10⁵-parameter model. The `D_f` estimator is explicitly
window-dependent (20 epochs; longer windows up to 500 "obscure early dynamics") and the paper flags
this as its main limitation. Whether any of it generalizes beyond modular arithmetic is listed as open.

**Relevance — IGNORE, with one small salvage.** Morpheme has no memorisation plateau to escape: it
trains on a corpus far larger than its capacity, `val_bpb` falls monotonically, and there is no
kinetic arrest to tunnel out of. Injecting parameter swaps into a 35M model mid-run would be pure
risk with no identified problem to solve. The salvage is diagnostic and cheap: **replica correlation**
is a well-defined way to ask whether two runs branched from one checkpoint under different data order
end up in the same place — which is exactly the question behind "does this α change actually change
the model, or just the seed noise?" Two forks from one checkpoint and a Pearson correlation of the
flattened parameters is ~20 lines. *Effort:* 1 h. *Risk:* none. *Verdict:* **not scheduled** — the
step-matched A/B in item #1 answers the same question more directly by measuring the thing we care
about (`val_bpb`) instead of a proxy.

---

### C11. 2607.29503 (2607) — *The Grokked Illusion: True Equilibrium Mitigates Catastrophic Forgetting*
Zhang, Chan, Shang, Zhang, Yang (same group as C10).
`arXiv:2607.29503v1 [cs.LG] 31 Jul 2026`.

**What it is.** Takes two networks that both reach 100 % test accuracy on `x + y mod 67` under an
identical weight-norm constraint `‖w‖ = 30` — one trained by AdamW, one *sampled* from the Boltzmann
entropy landscape by Wang-Landau Molecular Dynamics — then forces both to additionally memorise 500
new noisy samples and measures how much of the original task survives.

**Results.** Single-layer transformer, d_model 128, d_mlp 512, 4 heads, ~2.15×10⁵ params, 2 244
training samples; 1.2×10⁸ WLMD epochs to convergence; 10 seeds per condition. After memorising random
noise to >99.8 % training accuracy: **AdamW falls from 100 % to below 75 %** on the original task while
**WLMD holds ≈95 %**. The gap narrows as the injected noise becomes structurally similar: for
`x² + y mod 37`, 84 % vs >98 %; for `x + y mod 37`, ≈97 % vs ≈100 %. Mechanism: effective rank
`ER(W) = exp(−Σ σ̃_i log σ̃_i)` is far higher for WLMD before injection (W_Q 96.8 vs 2.2; W_in 116.1 vs
48.1; W_out 116.6 vs 33.7) and stays higher after, despite a *larger* absolute drop. Cosine similarity
of the flattened parameters before/after is **lower** for WLMD (e.g. W_in 0.857 vs 0.996) — it moves
further and survives better.

**Evidence vs claim.** *Evidence:* 10 seeds per condition; the effective-rank table; the monotone
structural-similarity trend. *Claim:* WLMD is a sampling procedure requiring 1.2×10⁸ epochs on a
2×10⁵-parameter model — it is not an optimizer and the paper does not claim it is; the one attempt at
turning it into one (WanD) is described as having "training instability". Whether the high-entropy
advantage survives at LLM scale is explicitly listed as untested, and the causal direction between
effective rank and robustness is called correlational.

**Relevance — IGNORE as a method; ONE metric is worth adopting, and cheaply.** WLMD is unreachable
here. But **effective rank of the weight matrices is a two-line diagnostic** (`torch.linalg.svdvals`
then the entropy exponential) and it is the same quantity 2608.07436 measures as "spectral dispersion"
and finds differs by two orders of magnitude between Muon and AdamW. That gives a *shared* observable
across two papers in this batch, computable on any checkpoint we already have:
* *Where:* `morpheme/train/report.py`. *Effort:* 30 min. *Risk:* none.
* *What it decides:* whether Morpheme's Muon and AdamW runs produce measurably different weight spectra
  at 35M, which is the cheapest available probe of whether the Muon literature's mechanism is even
  operating at our scale. If the effective ranks are indistinguishable, most of the grokking cluster's
  relevance to us evaporates and we can say so with a number.
The paper's other lesson is worth stating without adopting anything: **matched final accuracy does not
imply matched robustness.** If Morpheme ever does sequential SFT off `data/build_sft.py`, "the SFT
model still scores the same `val_bpb`" will not be sufficient evidence that nothing was lost.

---

### C12. 2608.14803 (2608) — *Is Grokking a Loss of Normal Hyperbolicity of the Interpolation Manifold?*
Suvinava Basak (TU Braunschweig). `arXiv:2608.14803v1 [cs.LG] 14 Aug 2026`. SKILL 2026 (GI-Edition).

**What it is.** A single, clean, negative result. If the post-memorisation phase of grokking is a
fast–slow system with the interpolation manifold as the slow manifold, then a sharp transition might be
a *loss of normal hyperbolicity* — a normal restoring direction going flat. The proposed diagnostic is
the smallest nonzero singular value `σ⁺_min(J)` of the residual Jacobian, which for the squared loss
equals the slowest normal restoring rate. A dip at the transition supports the bifurcation hypothesis;
a value bounded away from zero refutes it.

**Results.** Two-layer ReLU network, width 96, modular addition mod 11, 70/30 split, squared loss,
full batch, Kaiming ×3.5 init, AdamW (lr 3e-3, decoupled wd 2.0), 35 000 steps, snapshots every 500,
**five seeds**. Median `σ⁺_min(J)` by phase: pre-memorisation ≈0.010 (min 0.0096), memorisation plateau
0.128 (min 0.045), **transition 0.220 (min 0.202)**, post-transition 0.182. The diagnostic is near zero
**only before memorisation** — unrelated to grokking — and is at its *largest* during the transition.
The six smallest singular values form a tight cluster showing the same thing, so it is not an artefact
of looking only at the minimum, and no seed shows a dip.

**Evidence vs claim.** *Evidence:* the five-seed replication and the six-smallest-singular-value check.
*Claim:* the author is exemplary about scope — AdamW is used where the theory is stated for gradient
flow; the transition is "moderately gradual rather than maximally sharp"; a subspace-local collapse
confined to test-relevant directions could hide beneath a global measure; and a discrete trajectory is
not the gradient-flow limit. The paper explicitly frames the result as "constraining, not refuting".
A secondary observation is reported honestly: the parameter norm does **not** decrease through the
transition here, which the author attributes to AdamW's ℓ∞-type implicit bias rather than to the
manifold.

**Relevance — IGNORE. There is no action.** Morpheme uses cross-entropy, not squared loss, so the
interpolation manifold is not even a well-defined object; it never interpolates; and computing the SVD
of a residual Jacobian for a 35M-parameter model over a byte corpus is not tractable. Nothing in
`morpheme/` changes and no experiment would decide anything. **The reason it earns its place in this
report is methodological, not technical**: it is a negative result, published as one, with the
limitations enumerated and the deciding follow-up experiments named. That is the standard the α A/B in
item #1 should meet — a pre-registered diagnostic, multiple seeds, and a written-down account of what
would falsify the conclusion.

---

### C13. 2608.19587 (2608) — *Unregularized Convergence of Single-Loop, Entropy-Regularized Natural Actor-Critic*
Zhiqiang Tan (Rutgers Statistics). `arXiv:2608.19587v1 [cs.LG] 20 Aug 2026`. ~33 000 lines.

**What it is.** Finite-sample convergence theory for an entropy-regularized Natural Actor-Critic on
infinite-horizon discounted MDPs with log-linear policies. *Single-loop* = one critic SGD step per
actor update. *Uncentered critic* = regress onto raw features rather than action-centred score
features, so the moment matrix `Σ̄_unc ⪰ κI` stays invertible as the policy goes deterministic and the
Fisher matrix degenerates. The headline is an **Exponential Translation** (Thm 3.1): under a Minimal
Action Gap `Δ > 0`, the unregularized gap is bounded by the regularized one plus
`C·exp(−Δ/2τ)`, replacing the usual `O(τ)` penalty and allowing τ to be annealed only
*logarithmically*.

**Results.** Stochastic regime: `Õ(1/T) + Õ(ε_app)` for both average and last iterate. Deterministic
regime: `Õ(T^{−2/3})` average and `Õ(T^{−1/3})` last (the latter slower purely because it routes
through Pinsker). Tabular corollary: the same rates with `ε_∞ = 0` and **no concentrability assumption**,
needing only the Minimal Action Gap. Every fixed-τ rate blows up as τ→0 (`τ^{−2}`, `τ^{−5/3}`,
`τ^{−4/3}`), and approximation bias always enters as `ε_app/τ`. **There are no numerical experiments —
zero. No figures, no code, no toy MDP.** The two tables compare other papers' theoretical rates.

**Evidence vs claim.** *Proved,* under fourteen assumptions, the load-bearing ones being a strictly
positive action gap with a *unique* optimal action at every state (which the paper concedes "may be
violated in non-tabular MDPs"), τ-independent concentrability constants, and conditionally independent
sampling — **Markovian sampling and TD bootstrapping are explicitly out of scope**. The paper's most
candid moment is Lemma 7.2: in the stochastic regime the assumptions are *mutually incompatible* in a
specific way that forces `ε_app ≥ C·τ`, so **exact realizability is impossible** and the headline
`Õ(1/T)` holds only over a constrained window. Nothing is validated empirically.
*(Reading disclosure: the delegated read covered the entire main body, Appendices A–C in full, and two
load-bearing proofs — Theorem 3.1 and Theorem 10.3 — line by line; the ~14 000 lines of routine
algebra in Appendix E and the re-derivations in Appendix D were indexed and skimmed rather than
verified.)*

**Relevance — IGNORE. Genuinely zero.** Morpheme is supervised density estimation: next-byte
cross-entropy against fixed targets plus SFT. There is no MDP, policy, value function, critic, reward,
discount factor, or exploration anywhere in the stack. Two near-misses worth naming so they are not
mistaken for transfer: (i) "entropy regularization" here is a term in a *trajectory* objective, not the
softmax sampling temperature in `engine.py` — the τ-annealing schedule says nothing about decode
temperature; (ii) the uncentered-features trick is specific to compatible function approximation in
NPG, where the centred features *are* the score function. The one genuinely general fact — in a
two-timescale system, balance SGD noise `O(α²)` against target drift `O(η²/α)` to get `α ∝ η^{2/3}` —
does not apply, because Morpheme has no separately-stepped tracking subsystem: the router, main network
and decoder train end-to-end under one loss, and the dechunk EMA is a fixed-coefficient smoother inside
the forward pass, not an SGD process with a schedule.

---

### C14. 2608.04871 (2608) — *Step Recursion: A Three-Parameter Refinement of the Grzegorczyk Hierarchy*
Kirill Osipov, independent researcher. `arXiv:2608.04871v1 [cs.LO] 5 Aug 2026`.

**What it is.** Replaces the predecessor in bounded primitive recursion with a generalized inverse
`ρ_φ(y) = min{z : φ(z) ≥ y}`, so a recursion on `y` visits `y, ρ(y), ρ²(y), …, 0`. With `φ(x) = x²+2`
a recursion from 38 visits 38, 6, 2, 0 — three updates instead of thirty-eight. Three parameters index
the resulting classes `H^m_{n,l}`. It is entirely a set of theorems about which class contains which.

**Results.** **Zero measurements.** Neither of its two tables contains a number. The results are a
complete containment criterion for `n, n′ ≥ 2` (Thm 56), a saturation theorem (`H^m_{n,l} = E_m ⟺
m ≥ n+1`), a reverse-divisibility structure in which strides are ordered by divisibility rather than
magnitude (stride 6 is simulable by stride 3, but 2 and 3 are incomparable), `H²_{1,l} ⊊ FP` strictly,
and a conditional corollary that `H²_{1,l} = E₂ ⟹ P = NP`. Remark 70 is an unusually candid list of
what remains open.

**Evidence vs claim.** Everything is a proof, so the axis is proved-vs-conditional. The classification
is claimed complete for `n, n′ ≥ 2` and proved by three independent components. Blunt: single author,
independent researcher, v1 preprint, no stated peer review, no machine-checked formalization —
correctness rests on a reader verifying ~50 pages of hand proofs. The P=NP corollary is conditional and
the author flags it twice as not used in the main theorem.

**Relevance — IGNORE. Genuinely irrelevant.** No neural network, no gradient, no kernel, no throughput,
no hardware. Not one line of `morpheme/` would change and no experiment would decide anything, because
the paper makes no claim about anything Morpheme does. The only surface collision is vocabulary —
"chunking", "stride", "schedule", "descent" — and it is coincidental: the strides are iterates of
Ackermann-style generators over the naturals, with nothing to do with bytes per chunk or sequence
segmentation. Its one 2026 citation (Curzi & Das, *Cyclic implicit complexity*, ACM TOCL 27) is
likewise irrelevant.

---

### C15. 2608.19589 (2608) — *OrthoSkillVLA: Continual Skill Learning via Gradient-Informed Skill Subspace Adaptation*
Wang, Fang, Shi, Zhou (Southeast University). `arXiv:2608.19589v1 [cs.RO] 20 Aug 2026`.

**What it is.** Continual learning for flow-matching vision-language-action policies. Three mechanisms:
project each new skill's accumulated gradient off the stored orthonormal basis of prior skills and SVD
the residual to extend the basis; a LoRA whose down-projection `A` is *initialised from* the top-r left
singular vectors of that projected gradient and then **frozen**, so every update stays orthogonal to
prior skills; and per-module energy budgets (`ε_VLM = 0.99` loose, `ε_Head = 0.9999` tight) with the
velocity decoder excluded from orthogonal fine-tuning in favour of per-skill additive experts selected
by a training-free projection router.

**Results.** X-VLA (~0.9B) on a self-reorganized LIBERO-100 skill-incremental split, 50 rollouts/task,
averaged over **3 skill orderings**. Final success rate: SeqLoRA 32.44 ± 3.31, IncLoRA 34.11 ± 1.70,
EWC 25.17 ± 4.38, OLoRA 30.50 ± 4.34, KeepLoRA 56.61 ± 7.22, **OrthoSkillVLA 83.50 ± 1.42**. The 2×2
ablation is clean and superadditive: MoE decoder alone +11.33, split thresholds alone +3.83, both
+26.89. Rank sweep: r = 32 → 79.50, r = 64 → 83.50, r = 96 → 74.06 (best FWT, worst final SR). The
training-free router costs only **2.7 points** against oracle skill identity. Real-world: 69/80 vs
KeepLoRA's 57/80.

**Evidence vs claim.** *Evidence:* the main table with per-ordering std, the 2×2 ablation, both
threshold sweeps, the rank sweep, the router-vs-oracle gap. *Claim:* "3 orderings" is **not 3 seeds** —
initialization and data order are not independently varied. Every baseline was re-implemented by these
authors on a benchmark split they invented, so no number is comparable to a published figure. The
real-world result is a single unrepeated run. Tables 4/5 report the chosen configuration at 84.83 %
while Table 1 reports 83.50 ± 1.42 and the discrepancy is never reconciled. "Minimal inference latency
and memory footprint" is supported only by a parameter-count fraction — **nothing is ever timed.**

**Relevance — IGNORE.** Morpheme is single-objective pretraining from scratch: no skill sequence, no
catastrophic-forgetting problem, no LoRA, no frozen pretrained backbone, no VLM, no action space. The
central mechanism presupposes exactly what Morpheme does not have. One overlap deserves naming so it is
dismissed accurately rather than by accident: the **training-free projection router** (argmax over
`‖x·U_i‖₂`, 91.5–98.9 % accurate) is a cheap alternative to a learned router, and Morpheme *has* a
router in `dc.py`. But they answer different questions — Morpheme's decides *where a boundary falls in
a byte stream*, an unsupervised per-position binary decision with no discrete class inventory and no
per-class gradient basis to project against. There is no formulation of "argmax over K stored bases"
that answers "is byte t a boundary", so this does not even qualify as a test candidate for the
bytes/chunk undershoot.

---

### C16. 2608.11669 (2608) — *Rubric Dropout: A Simple Way to Mitigate Reward Hacking in Rubric-as-Reward RL*
Yang, Guo, Tyagi, Zhang, Dumitru, Hou, He, Zhang, Liu (Scale AI / Arizona / UT Dallas).
`arXiv:2608.11669v1 [cs.LG] 12 Aug 2026`. Self-labelled **"Work in Progress"**.

**What it is.** Two things: a measurement protocol (grade an out-of-distribution eval set with both the
training judge and a stronger cross-family judge, and read *divergence* of the two curves as the
reward-hacking signal — a constant judge bias shifts a curve but cannot make gold fall while proxy
rises), and a one-line intervention (drop a random fraction `f` of the rubric's positive-weight
criteria per step, one mask per rollout group seeded by `SHA256(instance_id, step)` so GRPO's
group-standardized advantages stay comparable and the normalizer cancels).

**Results.** Qwen3-8B / 4B, GRPO, 16 rollouts, lr 1e-6, 600-step horizon, window 400–600, proxy
gpt-4o-mini, gold claude-sonnet-4-6. The phenomenon: base 8B Medical gold **peaks at 31.2 % at step
~240** then declines while proxy climbs to 72 %; Science gold falls **21.5 points** from its peak.
Dropout at f = 50 % lifts gold by **+2.0** (Medical) and **+7.0** (Science) at 8B, winning at
**11/11 matched checkpoints on both pairs**. Per-criterion at step 600: gold pass 48.7 → 52.3 with
overclaim 41.5 → 37.9 at matched proxy pass rates. Best-checkpoint gold is essentially **tied across
all arms (30.6–31.5 vs base 31.2)** — the entire effect is post-peak decay, not peak capability.

**Evidence vs claim.** *Evidence:* the divergence phenomenon on two pairs at two sizes; the 11/11
matched-checkpoint result with both hacking measures moving together; the criterion-level decomposition
at matched proxy pass rates. *Claim:* **every configuration is one seed** — stated plainly
("preemptible-only compute ruled out seed replication"), and the Medical margins sit inside a sweep
whose own arms are non-monotone. The mechanism is unresolved and the paper says so: anti-co-adaptation
and plain implicit regularization "both stories predict the same figures", and the decisive frontier
test **overlapped and is reported as "(not shown)"**. The POW3R baseline is a modified single-run port.

**Relevance — IGNORE.** No RL, no policy gradient, no judge, no reward model, no rubric, no
post-training alignment stage. Every moving part requires machinery Morpheme does not have. One thread
is worth naming only so it can be dismissed properly: the paper is a clean Goodhart case study, and
Morpheme *does* have a proxy being satisfied in letter rather than spirit — the ratio loss, whose
target the router undershoots. But the *method* does not transfer: there are no discrete criteria to
drop, no group-relative normalization for a mask to cancel out of, and no independent "gold"
measurement of chunk quality. The right tools for that problem are in §(e) items 1–3, not here.

---

### C17. 2608.01475 (2608) — *Plasticity of Growing and Elastic Neural Networks in Online Continual Learning*
Jeong Min Kong (UCLA ECE), Richard S. Sutton (Alberta / Amii).
`arXiv:2608.01475v1 [cs.LG] 2 Aug 2026`.

**What it is.** Four CasCor-descended constructive MLP variants under *online* (one sample at a time)
continual supervised learning: SGN adds a ReLU unit every K samples and freezes the previous unit's
incoming weights; AGN is the same without freezing; AEN adds pruning of *estimated*-dead units at each
task boundary (estimated from a random 5 % sample, so live units get pruned too). Plain SGD, lr 0.001,
backprop after every sample.

**Results.** **There are no tables in this paper — zero.** All 28 figures are plots and the text quotes
no numeric accuracy or dormancy value. The only numbers stated are hyperparameters: online permuted
MNIST and permuted FashionMNIST, N = 10 000 and 40 000 samples/task, K ∈ {6 500, 4 000, 3 000, 24 000,
12 000}, an F-FCNN baseline with 3 hidden layers of width 200. **No seed count, no error bars, no task
count.** Qualitatively: F-FCNN and SGN lose plasticity; AGN holds accuracy flat *despite* rising
dormancy; AEN holds accuracy with dormancy near 0 % and network size converging to a compact value.

**Evidence vs claim.** Every claim is measurement-backed but only *visually* — a reader cannot extract
a single number to compare against. Single-seed must be assumed since no variance is reported anywhere.
Two central mechanisms are explicitly speculative ("We speculate that the decline eventually plateaus";
"We hypothesize that this distinct behavior results from…") and neither is tested. No wall-clock or
memory measurement despite the paper arguing AGN's cost grows.

**Relevance — IGNORE.** Morpheme is single-objective pretraining on a fixed byte corpus; loss of
plasticity across a sequence of permuted tasks is not a failure mode it encounters. The one
superficially transferable idea — dormancy % as a health metric — is defined for ReLU units that output
exactly 0, and Morpheme uses SwiGLU (`relation.py`) and Mamba-3 gating, where "dead unit" has no
equivalent definition. No file changes; no experiment worth running.

---

## (d) Synthesis

### D1. What reinforces what

**Three papers converge on the same statement about Muon, from three directions, and it is not the
statement the efficiency report assumed.** 2607.20512 isolates *orthogonalization* as the active
ingredient and shows it produces a **lower-norm, more widely distributed** solution (max singular value
5.5–6.2 vs AdamW's 15.9, near-uniform Fourier spectrum). 2608.07436 measures the same dispersion in the
basis the model computes in (**326 effective conjugate pairs vs AdamW's 4.95**) and then shows that the
*same property* is what makes the solution drift when the loss flattens. 2608.16760 explains why
orthogonalization works at all — Muon is a block-diagonal preconditioner with one block per row, which
matches the Hessian structure — **and identifies its specific defect**: it applies the same
preconditioner to every row, ignoring neuron-wise heterogeneity, which is exactly what NorMuon fixes.
Then 2608.19491 adds the empirical sting: a fairly tuned Muon at 67M and 370M sits *behind* AdamW under
a matched protocol.

The coherent picture: **Muon's benefit and Muon's fragility are the same mechanism**, and the 2026
frontier has already moved past plain Muon to per-row-normalized variants. If we adopt anything from
this family it should be **NorMuon-shaped** (Adam-mini's row-wise `v` on an orthogonalized update),
not plain Muon — and 2608.08888's 1B run using NorMuon for matrix parameters is independent corroboration
that this is where practice has gone.

**Two papers converge on "the ratio loss weight is the lever".** SOMBRERO (2601.22805) measures
ω 0.03 → 1.0 taking compression 3.95 → 4.97 at 1B. ATDC (2605.30080), whose schedule we implement,
exhibits BPIC 4.37–4.52 against N_fnl = 6.5 at 680M–1.3B **and never mentions it**. Our own
`sweep_a0.3_n6` reaches 5.12. Three independent observations of the same lever at three scales.

**Two papers converge on "measure in the right currency".** 2608.07436's 2.22×-in-steps becomes
1.28×-in-seconds. 2608.08888's "1.5× more tokens" is bought with 1.28–1.5× more FLOPs. Both papers are
honest about it in their own tables and misleading in their own abstracts. On a wall-clock-bound
single-GPU project, **every A/B in item table §(b) must be scored in seconds or at matched steps, never
in "steps to reach X".**

### D2. What contradicts what

**The undershoot: bug or feature?** SOMBRERO (2601) and ATDC (2605) treat higher compression as the
goal. **"Compute Optimal Tokenization" (2605.01188)** — 988 models, 50M–7B, compression as an explicit
scaling variable — finds an interior optimum that **falls with compute**: T\* = 3.71 at 10²⁰ FLOPs.
Morpheme trains at ~10¹⁷–10¹⁸ FLOPs. These cannot both be right for us, and the disagreement is
resolvable by exactly one experiment: the step-matched α A/B. **This is the single most valuable
experiment named anywhere in this report**, because it decides whether we spend effort *raising*
compression or *lowering the target*.

**Muon at 60–130M: 2607.04033 vs 2608.19491.** The efficiency report's rank-16 cited 2607.04033 ("Muon
leads at 60M and 130M and halves optimizer state"). 2608.19491's differenced tables put Muon behind
AdamW at 67M and 370M. Both are 2026, both tune carefully, both use multiple seeds. I cannot adjudicate
from here — the protocols differ (24 optimizers broadly tuned vs 3 arms deeply tuned; different data,
different architectures) — but **the efficiency report's confidence in Muon should be downgraded from
"adopt after fixing the param split" to "A/B on wall-clock, with NorMuon as the variant most likely to
win"**. The half-the-optimizer-state claim survives either way and is better served by Adam-mini, which
has 39M→1B evidence at exactly our scale.

**"More adaptation capacity is better" is contradicted twice.** 2608.19589's rank sweep (r = 96 gives
the best forward transfer and the *worst* final accuracy) and 2603.10145's D sweep (monotone gains, but
D = 2048 → 4096 worth only +0.55 with no seeds) point in opposite directions on whether more rank helps.
Neither is about Morpheme, and the honest reading is that the question is unsettled and cheap to
sidestep: we should not add capacity to any module without a step-matched A/B.

**The grokking cluster contradicts itself on whether the geometry is special.** 2608.14803 finds *no*
loss of normal hyperbolicity (σ⁺_min is largest during the transition). 2608.01833 finds textbook
glassy signatures (mobility collapse, two-step relaxation, aging, `D_f` → 1). Both are on modular
arithmetic; both are careful; they measure different things and reach compatible-but-differently-framed
conclusions. For our purposes the contradiction is instructive rather than actionable: **it shows how
unsettled the mechanism is even in the cleanest possible setting**, which is a good reason not to import
any of it into a 35M byte model.

### D3. What contradicts our current design choices

1. **`DCCfg.ratio_loss_weight = 0.03` is the number SOMBRERO names as the bug.** The `local` run already
   uses 0.1 and still undershoots by 37 % — *worse* than SOMBRERO's 21 % at 0.03. That difference is
   itself a warning: SOMBRERO ran at 1B params on 839B bytes with 16 384-byte sequences; we run 35M at
   2 048. **The ω fix is a hypothesis to test here, not a guaranteed transfer.**
2. **`betas=(0.9, 0.95)` at 32 768 bytes/step is on the wrong side of a theorem.** 2608.16760's
   threshold rises as batch size falls, and three independent LLM-scale reports confirm it empirically.
3. **`--clip 1.0` is inert.** `grad_norm` is 0.33–0.35. It is not doing anything, and it will not save
   us from an optimizer instability. Worth knowing before reading any Muon-stability result as
   reassurance.
4. **`mbp.n_candidates = 3` now leaves measured throughput on the table.** The τ-threshold rule is
   *gone* — `accept_threshold` was removed from `MBPCfg` during this session and `engine.py` implements
   exact rejection sampling (`verify_draft`, snapshot/replay, a `PAUSE_BELOW` break-even guard). With
   verification exact, 2608.15454 measures **+29–37 % throughput at n = 7–8 with output identical by
   construction**, and under the *old* τ rule throughput fell monotonically with n. The blocker is
   removed; the knob has not been turned. Two new fields landed alongside it —
   `mbp.position_gamma = 0.0` and `mbp.transition = False`, the DFlash/DSpark position weighting and
   Markov head from the speculative-decoding report — both still at their no-op defaults, and
   `mbp.loss_weight` is still 1.0, which that report argues against. Those are its recommendations to
   close out, not this one's.
5. **`boundary_on_separator_frac = 0.280` is a weak number and we now have external context for it.**
   2608.17325 measures H-Net morphological alignment F1 **below 0.1 in all 18 languages tested**, and
   SOMBRERO measures H-Net's "boundary enrichment" at **1.19 against a null of 1.0**, with boundaries
   landing on the *first character of a word* rather than on whitespace. Our 0.28 is consistent with the
   published behaviour of the architecture, not with a bug in our implementation — but the `sweep_a0.3_n4`
   run reaching **0.590** shows the metric is movable by α and N, and that combination has never been
   tried at `local`.
6. **The flagship's `max_seq_len = 4096` interacts badly with an unfixed `bpic`.** At `bpic` 3.2 a
   4096-byte window produces ~1 280 chunk positions, and Relation's pairwise term is O(T²). Item #1
   should land before the flagship is sized.

### D4. Ranked, with the deciding experiment

The table in §(b) is the ranked list. The three that matter most, restated as questions:

1. **Is 3.2 bytes/chunk a bug?** — three `local` runs at α ∈ {0.1, 0.3, 1.0}, **step-matched**, scored
   on `val_bpb`. Everything else about the router waits on this answer.
2. **Is β₂ = 0.95 wrong at 32 768 bytes/step?** — three runs, one flag, 15 minutes of work.
3. **Does Muon actually win here, in seconds?** — the A/B we have never run, with the per-group
   update-RMS log from 2608.07436 attached so the answer comes with a mechanism.

---

## (e) The neighbourhood — 2026 papers that answer what the batch does not

Sixteen papers read in full. Ranked by relevance. Three of these (2608.20172, 2608.15454, 2605.30080)
were covered in the earlier reports and were re-read here for a specific new purpose, noted in each.

**E1. 2601.22805 (2601) — SOMBRERO: Measuring and Steering Boundary Placement in End-to-End Hierarchical Sequence Models.**
Introduces *boundary enrichment* `B` — the ratio of mean next-byte surprisal at chunk starts to mean
surprisal overall — as a router-agnostic boundary-quality metric, and fixes H-Net's chunker by moving
confidence-weighted smoothing from realized chunks to the byte level, swapping the cosine router for a
sigmoid projection, and adding a confidence-alignment boundary loss.
**The number:** at ω = 0.03, `C_emp = 3.95` against `C_tar = 5.0`; at **ω = 1.0, `C_emp` = 4.97** —
measured at 1B params, 839B bytes, 16 384-byte sequences, mixed EN/DE/code/math. H-Net's `B` is only
**1.19** against a null of 1.0, and its boundaries land on the *first character of a word*, spending
backbone compute predicting the second byte; SOMBRERO's land on whitespace at `B = 3.035`, BPB
0.6701 → 0.6568. **ADOPT the ω sweep (item #1). TEST byte-level smoothing (−0.013 BPB alone) and the
CAB loss. IGNORE the sigmoid-for-cosine swap** — it was a small regression on its own (0.6578 vs
0.6571) and only pays after byte-level smoothing. **January 2026 — the oldest neighbourhood item, and
flagged as such against the recency policy; it survives because nothing newer addresses the question.**

**E2. 2605.01188 (2605) — Compute Optimal Tokenization.** 988 BLT models, 50M–7B, compression rate
T ∈ {1,2,4,6,8,12} as an explicit scaling variable. **The numbers:** compute-optimal data is
**≈60 bytes per parameter** (α = 0.465, β = 0.471), and the optimal latent compression rate at
C = 10²⁰ FLOPs is **T\* = 3.71** (entropy patching), *falling* at higher budgets. **This is the
strongest argument that our "undershoot" may not be a bug**, and it gives the data budget directly:
105M → ~6.3 GB of bytes, 35M → 2.1 GB, 12.7M → 760 MB. ADOPT both the budgeting rule and the reframing.

**E3. 2605.30080 (2605) — ATDC.** *(Re-read: the efficiency report used it for the schedule; here for
the undershoot.)* With `N_init = 5.0 → N_fnl = 6.5` and α = 0.03, the 680M model measures **BPIC 4.52**
and the 1.3B model **4.37** — a 30 %+ undershoot the authors never acknowledge, reporting BPIC as a
"diagnostic" rather than a target. BPB improves only 0.005–0.006 over fixed-N (0.778 vs 0.783 at 680M);
the robustness gains under HellaSwag perturbation are larger. **ADOPT** BPIC-vs-N as a first-class
logged metric (we already log it — promote it in `report.py`). **TEST** whether the ATDC *schedule*
earns its keep once α is fixed: on this evidence the schedule buys ~0.005 BPB and the loss weight buys
the compression.

**E4. 2608.20172 (2608) — Ask Self, Ask Others: Relation Is All You Need.** *(Re-read for the honest
read of the evidence, not the kernel structure.)* **The number:** **FlashRelation reaches only
0.764–0.849× of PyTorch FlashAttention throughput** at 10M/30M/100M on an RTX 5090 in BF16 — a 15–24 %
throughput tax — to buy ΔNLL of −0.0412 / −0.0151 / −0.0310. The 30M win is **2 of 3 seeds with
per-seed SD (0.0136) larger than the effect (0.0151)**; seed 42 loses outright. Per-seed PIQA at 100M
fully overlaps (MHA 0.586–0.605, MHR 0.589–0.604). The ablations show the count calibration `λ_ℓ`
(+0.0506 if removed) and multi-head (+0.0507) carry nearly all the gain, while Givens head-mixing
contributes **+0.0032**. **TEST, do not assume** (item #10): in a 7.4 % MFU regime a 15–24 % throughput
loss for a sub-seed-noise NLL gain is not obviously worth it. And `RelationCfg.givens = True` is buying
+0.0032 for real dispatch cost — the efficiency report's rank-4 already flags `_givens` as one of the
largest remaining per-layer op costs.

**E5. 2605.12928 (2605) — The Efficiency Gap in Byte Modeling.** Compute-matched isoFLOP, 48M–1.23B,
byte vs Llama-2 BPE, on SlimPajama, reported in BPB. **The number:** at BPB = 1.0 the AR byte model
needs **7.9× the FLOPs of BPE** (2.3e20 vs 2.9e19); the gap narrows to 2.3× at BPB = 0.8 with
extrapolated parity at **~1.3e22 FLOPs**. Appendix D holds context fixed and byte still loses, so it is
not just sequence length. **The negative result matters most: HumanEval, MBPP and BBH were all
near-chance and were omitted.** ADOPT the benchmark exclusion list and the expectation-setting — at
~10¹⁷–10¹⁸ FLOPs we are 3–4 orders below byte/BPE parity, and any honest write-up must say so.

**E6. 2608.15454 (2608) — Dynamic Multi-Byte Prediction With Hierarchical Language Models.** *(Re-read
for the n knob.)* **The number:** under speculative-decoding verification, raising n from 3 to 7–8 gives
**+29 % to +37 % throughput with output identical by construction**; under probability-threshold
acceptance throughput *falls* monotonically with n (210 → 164 bytes/s on DailySum) and quality drifts.
`engine.py` now implements exact verification, so **item #5 is a config change away from a measured
30 %+ decode win.** Also worth carrying: MLP-MBP achieves *higher* acceptance (62–65 %) yet worse
downstream — confidence is not correctness. Caveat: single scale (373M), single B200, batch 1, and the
backbone uses a hinge boundary loss, not H-Net's ratio loss.

**E7. 2608.17325 (2608) — What Tokens are Learned when Tokenization is Optimized Jointly with Language
Modeling?** 14 fixed tokenizers vs H-Nets and subword-segmental LMs across 18 typologically diverse
languages. **The number: H-Net morphological alignment F1 is below 0.1 in all 18 languages**, against
0.26–0.65 for jointly-optimized SSLMs and comparable-or-higher for plain BPE/ULM; H-Net mean token
length reaches **17–18.5 characters** in non-Latin scripts. **Direct answer to the boundary-quality
question, and the answer is no** — H-Net boundaries do not recover morphemes; they optimize byte-level
compression. **If Morpheme's story rests on "learned boundaries recover morphemes", this falsifies it.**
Caveat heavily: their H-Nets are ~3M params on ~250k sentences, far below our smallest preset.

**E8. 2608.02032 (2608) — DART: Decoded Attention over Recurrent States.** Keeps Mamba-2's chunked scan
but retains each chunk's state contribution ΔH as a memory and does FlashAttention-style attention over
those chunk memories, added back as a zero-initialized gated residual. **The number:** at **130M** on
Pile/100B, +2.9 % parameters moves PPL 12.36 → 12.25 and zero-shot average 39.07 → **40.06**, with SWDE
extraction 17.1 → 34.9; ablating the branch at eval collapses MQAR from 95.94 % to **0.11 %**. The
closest 2026 sub-200M head-to-head between a recurrent backbone and an attention-style retrieval
mechanism, and it reuses the Mamba-2 chunked scan, so it grafts onto our existing machinery. **TEST as
an alternative to Relation** — at <200M it is a cheaper and better-evidenced bet. Caveat: DART's chunks
are fixed-size (S = 256); pairing with learned variable chunks is unexplored.

**E9. 2604.27263 (2604) — Decoupling the Benefits of Subword Tokenization via Byte-level Simulation.**
Trains a 1.7B byte-level Llama-3 and simulates each hypothesized benefit of subword tokenization in
isolation. **The number:** artificially compressing 4:1 to match a BPE model's isoFLOP sample throughput
**for only the first 50k of 100k steps** produces the single largest recovered gain — the treated model
crosses the baseline shortly after reverting and holds the slope. Two direct implications: the dominant
benefit of compression is **throughput, not semantics**, which supports going after the throughput loss
aggressively; and **start-of-subword boundaries work as a removable inductive bias while
end-of-subword boundaries do not**, with next-subword-as-output-unit *worse* than next-byte — a caution
flag for how the MBP head's target is defined.

**E10. 2606.14122 (2606) — Beyond Perplexity: UTF-8 Validity in Byte-aware Language Models.** 355M model,
80B tokens, EN/JA/KO/ZH, 420 checkpoints evaluated with a UTF-8 DFA. **The number: UTF-8 validity
converges at ~4.2B training tokens while perplexity stabilizes at ~2.1B — a 2× lag**; final *strict*
binary validity is only 50.47 % on common characters and 30.24 % on rare ones despite 95–96 %
partial-credit validity. A byte model can look converged on BPB and still emit undecodable output.
**ADOPT the strict DFA check** as a standing eval (item #7). Honest limitation: their model is
byte-*fallback* over a subword vocab, so absolute rates will not transfer — the convergence-lag *shape*
is the transferable finding.

**E11. 2608.12700 (2608) — A Contract-Grade Verifier for LLM-Generated GPU Kernels.** *(The efficiency
report listed this as metadata-only; read in full here.)* **The number: 39.5 % of 2 638 already-accepted
machine-generated kernels are broken beyond any tolerance argument, 62.1 % carry a violation.** The
Blackwell content does not apply to Ada. What transfers is the **killed-levers ledger** — a two-level
scan gave 1.11× (below their 1.3× gate), stage-B GEMM fusion was a measured negative, activation
checkpointing for the backward gave *zero* speedup and was reverted — and the finding that a
**shape-gated selector gave 2.174× geomean** over a serial default. **ADOPT** the shape-gated dispatch
idea (item #13): we have dynamic chunk shapes, which is exactly the case, and `flash_relation.py`
currently uses one fixed `BLOCK_M`/`BLOCK_N` choice. Also adopt the tolerance-free correctness gates
(NaN propagation, determinism, shape polymorphism, aliasing) for `tests/test_flash_relation.py`.

**E12. 2604.17861 (2604) — GPUOS: A GPU Operating System Primitive for Transparent Operation Fusion.**
One long-lived persistent kernel polls a device-visible ring buffer and dispatches through a versioned
device-function-pointer table, replacing 3–7 µs host launches with <100 ns device calls. **The number:
11.3× on an RTX 5090** for launch-overhead-dominated micro-batched elementwise workloads (15.3× on
H100), per-op latency 8 µs → 3.1 µs; 8.7× on attention decode. The only 2026 paper with a measured
**consumer-GPU** number for the exact pathology behind our 7.4 % MFU, and it is explicit that CUDA
Graphs fall back to eager under shape polymorphism — our situation. **TEST with low expectations:** it
is inference-focused with no training-backward numbers, and the efficiency report's §0.2 should be
re-run first to confirm launch overhead still dominates after this week's fixes.

**Read but not ranked** (each read in full, each dismissed for a stated reason):
* **2606.18246 (2606) — Variable-Width Transformers.** ×-shaped width schedule, ~3 % perplexity gain at
  200M–2B with 22 % fewer FLOPs. **Ignore:** the authors' own limitations say heterogeneous per-layer
  widths demand per-shape kernels and *add* kernel launches. We already have dynamic shapes at 7.4 %
  MFU; this makes that strictly worse.
* **2607.16117 (2607) — Rate–Utility Frontiers for Language Encodings.** The cleanest 2026 methodology
  for a fair byte-vs-token comparison (parallel sentences, shared bottleneck swept 256→1) at 5.6–13.8M
  params. Bytes win cross-lingual alignment (R@1 0.45–0.47 vs 0.27–0.34) at 3.0–4.86× the FLOPs.
  Encoder-only and non-generative, so it does not transfer — but the *control design* is the right
  template for an honest-evaluation section.
* **2607.07706 (2607) — The Key to Going Linear.** Proves softmax attention implements key-dependent
  rank-1 orthogonal projections, explaining why delta-style updates beat gated accumulation. Excellent
  analysis, but it is post-hoc linearization of frozen 0.6B–32B backbones with **no from-scratch <200M
  results**, so it cannot arbitrate Relation against alternatives at our scale.
* **2608.06398 (2608) — EntropyMoE.** *(Already in the efficiency report.)* Re-confirmed here for its
  negative: the sparse implementation is **2.0× slower than dense BLT** in wall-clock.

### Gaps that are real, not unsearched

These were searched for and do not exist. They are worth writing down because they mean *we* would be
generating the first data point, which is both an opportunity and a reason to be careful.

* **No 2026 paper takes the ratio-loss target-compression undershoot as its primary subject.** SOMBRERO
  fixes it in two sentences of experimental setup; ATDC *exhibits* it and never mentions it. **There is
  no ablation anywhere of ω/α against measured bytes/chunk.** Item #1 would be that ablation.
* **No boundary-quality measurement exists in the 12M–200M band.** SOMBRERO measures at 0.98B and 2.19B;
  2608.17325 measures ~3M H-Nets on 250k sentences. The gap is exactly where Morpheme lives.
* **No 2026 paper replaces the cosine router with something evaluated head-to-head from scratch and
  wins.** The only attempt is a slight regression on its own.
* **No H-Net replication study of any kind in 2026.** Every 2026 H-Net number is from a paper proposing
  a modification or from an author with a competing architecture.
* **No 2026 byte-level generative model reports BPB *and* downstream at 35–100M params.** The measured
  points are 130M, 180M, 355M, 373M, 680M+, 1B+, and 5.6–13.8M encoder-only. **The 12–105M generative
  band is genuinely unmeasured in 2026.**
* **Nobody has published the positive list of benchmarks that are above chance for a byte model under
  200M.** 2605.12928 names three that are not and proposes no replacements.
* **Zero replications, critiques or citing work for Relation (2608.20172)** — it was posted 2026-08-20,
  three days before this sweep. Nothing can exist yet. **The entire evidence base for our main network
  is the authors' own nine paired seed comparisons**, of which the 30M scale wins two of three with a
  per-seed SD larger than the effect.
* **No from-scratch <200M head-to-head between Relation and the gated-delta / linear-attention family.**
  DART at 130M is the nearest point and tests a different mechanism against a different baseline.

---

## (f) Sounds good but not for us

| Idea | Source | Why not |
|---|---|---|
| **DeltaMomentum as a default optimizer** | 2608.19491 | 46 % fewer *steps* at 67M is the right direction, but it needs per-layer forward/backward hooks on a card whose bottleneck is per-layer Python dispatch, and 11–20 % more FLOPs at 7.4 % MFU. Most of our linear layers are inside Mamba-3's fused Triton path where `x̂` and `δ` are not exposed. Run the diagnostic (2 h), not the implementation (2 d). |
| **Latent feedback / k-pass training** | 2608.08888 | Gains are per-*token*, bought with 1.28–1.5× training FLOPs and k× activation memory, on a card at 5.9/8 GB. Single seed throughout. And it changes the target distribution, so it cannot compose with `verify_draft`. |
| **Lean-Muon (ns = 1, drop spectral scaling)** | 2607.20512 | 20 % faster to first crossing and it collapses on 6/8 seeds. Also, our scale factor is Liu et al.'s RMS match, not the factor the paper ablates — removing it would silently change the effective learning rate. Keep ns = 5. |
| **"Stable Muon" freezing of embeddings/readout** | 2608.07436 | The mechanism needs a vanished gradient (1.5×10⁻⁷), independent writer and reader, and no normalisation layers. We have `grad_norm` 0.34, a **tied** embedding/head, and RMSNorm everywhere. Log the diagnostic; do not freeze anything. |
| **SAM-Swap parameter swapping** | 2608.01833 | Designed to escape a memorisation plateau we do not have. The paper's own comparison shows large Gaussian noise achieves the same acceleration, so the swap is not even the identified mechanism. |
| **WLMD / high-entropy sampling** | 2607.29503 | 1.2×10⁸ sampling epochs on a 2×10⁵-parameter model. Not an optimizer, and the paper does not claim it is. Take the effective-rank *metric* (30 min) and leave the method. |
| **σ⁺_min(J) as a training diagnostic** | 2608.14803 | Defined for squared loss on an interpolation manifold. We use cross-entropy and never interpolate; the SVD of a residual Jacobian at 35M over a byte corpus is not tractable. |
| **Multi-path mutual learning** | 2608.16797 | +0.135 % AUC for 2× dense compute. On a card at 7.4 % MFU, doubling compute to chase a leaderboard-sized delta is the wrong trade. Record it; do not schedule it. |
| **μP width transfer** | 2608.20061 | μP is a *width* rule. Our presets change width, depth, layer counts and `max_seq_len` together, so there is no width axis to transfer along. Take the EMA-as-decay-proxy trick and the parabola-vertex protocol instead. |
| **Variable-width transformers** | 2606.18246 | 22 % fewer FLOPs at loss-matched scaling, but the authors' own limitations say it demands per-shape kernels and *adds* launches. We are launch-bound with dynamic shapes already. |
| **Sparse expert routing over byte patches** | 2608.06398 | Best BPB among matched baselines and **2.0× slower than dense in wall-clock**. |
| **Rubric dropout / RL reward-hacking mitigation** | 2608.11669 | No RL, no judge, no rubric, no post-training stage. The Goodhart analogy to the ratio loss is real and the method still does not transfer. |
| **Orthogonal-subspace continual learning** | 2608.19589 | No skill sequence, no LoRA, no frozen backbone. Its training-free router answers "which of K skills", not "is byte t a boundary". |
| **Growing/elastic networks, dormancy metrics** | 2608.01475 | No continual-task sequence. Dormancy is defined for ReLU units that output exactly 0; we use SwiGLU and Mamba-3 gating. |
| **Entropy-regularized natural actor-critic** | 2608.19587 | No MDP, policy, critic, or reward anywhere in the stack. The τ schedule is not a decode temperature. |
| **Step-recursion complexity classes** | 2608.04871 | No neural network, no gradient, no measurement. The vocabulary overlap ("chunking", "stride") is coincidental. |

---

## (g) Papers read

### The batch (17 of 17, all read in full)

| ID | Mo | Title | One line |
|---|---|---|---|
| 2608.19491 | 2608 | DeltaMomentum | Delta-rule momentum buffer; 46 % fewer steps at 67M; its Muon arm lands behind AdamW |
| 2608.16760 | 2608 | On the Principles Behind NN Optimizers | Adam's β₂ threshold scales with batch size; Hessian block structure; Adam-mini at 50 % memory, 39M→1B |
| 2608.07436 | 2608 | Post-Grokking Collapse (Muon) | Every Muon config groks then collapses; Newton–Schulz decouples step size from gradient; freeze fixes it |
| 2608.07222 | 2608 | Skaling | One coupling exponent on Chinchilla; L-shape edge sampling recovers the law at 10× less compute |
| 2608.08888 | 2608 | Full-bandwidth transformer | Gated hidden-state feedback into the input; 3 % three-pass batches turn divergence into a contraction |
| 2608.20061 | 2608 | Let's Scale Step by Step | μP + token-horizon regression predicts a 10T-token LR from 1/98th the compute |
| 2608.16797 | 2608 | UniDot | Dual-bus recommender; FM Highway; mutual learning beats the whole width sweep |
| 2608.01833 | 2608 | Tunneling the Loss Landscape | Grokking as glassy kinetic arrest; parameter swaps cut delay 3 000 → 650 epochs |
| 2608.14803 | 2608 | Is Grokking a Loss of Normal Hyperbolicity? | Negative result: σ⁺_min is *largest* at the transition, five seeds |
| 2608.19587 | 2608 | Single-Loop Entropy-Regularized NAC | RL convergence rates; no experiments at all |
| 2608.11669 | 2608 | Rubric Dropout | Proxy/gold judge divergence as a hacking signal; drop rubric criteria; single seed |
| 2608.19589 | 2608 | OrthoSkillVLA | Orthogonal LoRA subspaces for robot skill sequences |
| 2608.04871 | 2608 | Step Recursion | Subrecursive complexity hierarchies; zero measurements |
| 2608.01475 | 2608 | Plasticity of Growing/Elastic Networks | Constructive MLPs on permuted MNIST; no tables anywhere |
| 2607.29503 | 2607 | The Grokked Illusion | Equal test accuracy, unequal robustness; effective rank is the mechanism |
| 2607.20512 | 2607 | The Active Ingredient in Muon's Grokking | Orthogonalization is the active ingredient; spectral scaling is inert; ns=1 is fast and fragile |
| 2603.10145 | 2603 | Lost in Backpropagation | 95–99 % of the logit gradient dies in `ker(Wᵀ)` — a pathology our D/V ratio avoids entirely |

**Batch by month: 2608 — 14, 2607 — 2, 2603 — 1. Total 17.**

### The neighbourhood (16, all read in full)

| ID | Mo | Title | One line |
|---|---|---|---|
| 2608.20172 | 2608 | Ask Self, Ask Others: Relation Is All You Need | Our main network; FlashRelation at 0.764–0.849× FlashAttention; the 30M win is under seed noise |
| 2608.17325 | 2608 | What Tokens are Learned when Tokenization is Optimized Jointly | H-Net morphological alignment F1 < 0.1 in all 18 languages |
| 2608.15454 | 2608 | Dynamic Multi-Byte Prediction (LCA) | Our MBP head; n 3→7–8 gives +29–37 % under exact verification |
| 2608.02032 | 2608 | DART | Attention over retained Mamba-2 chunk states; beats Mamba-2 at 130M |
| 2608.12700 | 2608 | Contract-Grade Kernel Verifier | 39.5 % of accepted generated kernels are broken; shape-gated dispatch gave 2.174× |
| 2608.06398 | 2608 | EntropyMoE | Best BPB among matched baselines, 2.0× slower than dense in wall-clock |
| 2607.16117 | 2607 | Rate–Utility Frontiers for Language Encodings | The cleanest fair byte-vs-token control design; encoder-only |
| 2607.07706 | 2607 | The Key to Going Linear | Why delta-style updates beat gated accumulation; no from-scratch <200M results |
| 2606.14122 | 2606 | Beyond Perplexity: UTF-8 Validity | Validity converges at 4.2B tokens vs perplexity at 2.1B — a 2× lag |
| 2606.18246 | 2606 | Variable-Width Transformers | 22 % fewer FLOPs, but needs per-shape kernels and adds launches |
| 2605.30080 | 2605 | ATDC | Our ratio schedule; BPIC 4.37–4.52 against N_fnl 6.5, never acknowledged |
| 2605.01188 | 2605 | Compute Optimal Tokenization | T\* = 3.71 at 10²⁰ FLOPs and falling; ≈60 bytes per parameter |
| 2605.12928 | 2605 | The Efficiency Gap in Byte Modeling | 7.9× the FLOPs at BPB 1.0; HumanEval/MBPP/BBH are near-chance and were dropped |
| 2604.27263 | 2604 | Decoupling the Benefits of Subword Tokenization | Throughput is the dominant benefit; start-boundaries work, end-boundaries do not |
| 2604.17861 | 2604 | GPUOS | 11.3× on an RTX 5090 for launch-bound work; CUDA Graphs fall back under shape polymorphism |
| 2601.22805 | 2601 | SOMBRERO | ω 0.03 → 1.0 takes compression 3.95 → 4.97; boundary enrichment B = 1.19 for H-Net |

**Neighbourhood by month: 2608 — 6, 2607 — 2, 2606 — 2, 2605 — 3, 2604 — 2, 2601 — 1. Total 16.**

### Combined

**2608 — 20 · 2607 — 4 · 2606 — 2 · 2605 — 3 · 2604 — 2 · 2603 — 1 · 2601 — 1. Total 33 papers read
in full.**

Three neighbourhood papers (2608.20172, 2608.15454, 2605.30080) also appear in the earlier reports and
were re-read here for a specific new purpose, noted at each entry. Everything else is new to this
campaign.

**Cited but not read — marked [unverified] at each point of use:** 2604.01472 (Newton-Muon),
2606.06418 (DoPr), 2605.06997 (Echo), 2607.15178 (T²MLR), 2605.26797 (Latent Recurrent Transformer),
2606.18206 (fixed-point reasoners), 2602.08984 (Next Concept Prediction), 2606.03938 (q0),
2606.31779, 2603.03818, 2602.00722, 2601.09512, 2603.17850, 2602.22818, 2601.05242, 2605.12474,
2605.20164, 2606.04923.
