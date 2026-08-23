# Multi-byte prediction, speculative decoding and acceptance rules — Morpheme, 2026-08-23

Scope: the MBP head (`morpheme/model/mbp.py`), its use at decode time (`morpheme/model/hnet.py`,
`morpheme/serve/engine.py`), and everything in the 2026 literature that bears on whether it should
exist and how it should be verified. Companion to `docs/research/efficiency-campaign-2026-08-23.md`
(§3.3, §5.1, §5.2 are checked here and partly revised).

**Recency policy.** Sources are weighted Aug 2026 → Apr 2026 descending. Pre-2026 work appears only
where labelled *(background)* and where nothing from 2026 supersedes it. Every claim carries an
arXiv ID or a dated URL. Claims I could not verify are marked **[unverified]**.

**Two corrections to the brief, up front.**

1. **arXiv 2607.05147 is not called "DeepSpec".** It is **DSpark: Confidence-Scheduled Speculative
   Decoding with Semi-Autoregressive Generation** (Cheng, Yu, Shao, Li, Xiong *et al.*, Peking
   University + DeepSeek-AI, 6 Jul 2026, cs.AI). **DeepSpec** is the *training repository* they
   open-sourced alongside it (`github.com/deepseek-ai/DeepSpec`, MIT, built on SGLang's SpecForge);
   it contains three algorithms — Eagle3, DFlash and DSpark. Both are covered in §C.
2. **The repo moved under me during this session, in the direction this report was going to
   recommend.** Commit **`db77440`** — *"serve: exact speculative verification (rejection sampling,
   snapshot + replay), continue-from-state forward, auto-pause below break-even; UI drops the
   threshold control"* — landed while I was reading, and implements ranks 1 and 2 of my original
   table. Everything below describes the code **as of `db77440`**. The τ-threshold critique that
   dominated the first draft of this report is now historical; the live question has moved to the
   **cost structure of the round**, which is §A0.

---

## A. Executive answer (one page)

### A0. The finding that matters now: the round costs two target passes, not one

`db77440` got the hard part right. `engine.py::verify_draft` is textbook Leviathan/Chen — accept with
`min(1, p(x)/q(x))`, correct from `norm(max(0, p−q))`, with the residual-underflow guard that byte
granularity needs. `_dist` applies temperature and top-p **identically to draft and target**, which is
the single easiest way to silently break losslessness and is handled. The draft is *sampled* from `q`,
not argmaxed. `target_logits = [logits] + [lg_seq[0, m] …]` correctly reuses the next-byte
distribution already in scope for draft position 0. The output is now distributed exactly as the
next-byte head would have produced it. None of that needs changing.

What needs changing is what happens **after a rejection**:

```python
state = snapshot
if n_acc > 0:
    self.model.forward_from_state(torch.tensor([xs[:n_acc]]), state)   # pass 2: replay
...
lg_next, routing, is_b, mbp = self.model.step(torch.tensor([[fix]]), state)  # pass 3: consume fix
```

A rejected round with `n_acc ≥ 1` costs **three** target passes (verify, replay, fix-step) for
`n_acc + 1` bytes. Let `T` = one target pass and `c ≈ 0.2·T` the draft. Writing `α` for per-position
acceptance and `k = n_candidates`, and noting the loop emits no bonus byte on full acceptance:

```
E[bytes] = 1 + α + α² + … + α^(k−1) = (1 − α^k)/(1 − α)
E[cost]  = 1.2·α^k  +  2.2·(1 − α)  +  3.2·(α − α^k)      = 2.2 + α − 2α^k     ← as landed
```

| variant | E[cost]/T | break-even α (k=3) | α=0.6, k=3 | α=0.6, k=6 | α=0.7, k=6 |
|---|---|---|---|---|---|
| **(i) as landed** — replay + separate fix step | `2.2 + α − 2α^k` | **α ≈ 0.705** | **0.83×** | 0.88× | 1.10× |
| **(ii) merge replay and fix into one call** (one line) | `2.2 − α^k` | **α ≈ 0.605** | 0.99× | 1.11× | 1.41× |
| **(iii) + fix-as-anchor and no replay** (§D1) | `1.2` | **α ≈ 0.17** | **1.63×** | **1.99×** | **2.45×** |

**The measured aggregate `mbp_top1_acc` on the 35M `local` run is 0.605** (`runs/overnight/log.jsonl`
— higher than the ~0.50 in the brief). That is *exactly* variant (ii)'s break-even and well below
variant (i)'s. So as the code stands, speculation is **net-negative at the acceptance we actually
have**, which is presumably why `db77440` also added the `PAUSE_AFTER=48, PAUSE_BELOW=0.15` guard —
a sound safety net, but its threshold (15 %) is set far below the real break-even (70.5 %), so it
will not fire on a merely mediocre head; it only catches a broken one.

Two fixes, in order of ratio-to-effort:

* **(ii), one line.** Replace the replay + `step(fix)` pair with a single
  `forward_from_state([*xs[:n_acc], fix])`. One pass instead of two. 0.83× → 0.99× at α = 0.6.
* **(iii), the real one.** Make the corrected byte the **anchor of the next round's verification pass**
  instead of a separate step, and recover the post-prefix state without re-running it. This is
  exactly DSpark's round structure (Fig. 1: the target emits `D`, the drafter proposes `EFGH` *from*
  `D`, the target verifies; "we use the terms anchor token and bonus token interchangeably"). It
  removes the extra pass entirely and takes cost per round to a flat `1.2·T`. **0.83× → 1.63×.**

Everything else in this report is worth less than that 2× swing.

### A1. Q1 — the 2026 landscape, in one paragraph each

**MTP / multi-token heads.** 2026 sharpens rather than overturns Gloeckle's scale story, and it
names the mechanism. **AdaMTP (2608.00434)** measures standard fixed-horizon MTP *below* plain
next-token prediction on the average of eight benchmarks for all three backbones it tests
(Llama-3.1-8B 35.90 vs 36.32; Qwen2.5-7B 61.60 vs 62.92; Gemma3-12B 44.99 vs 45.72), and shows
accuracy falling monotonically with head count (GSM8K 11.60 at n=2 → 9.68 at n=6, against 11.30 for
NTP). Its diagnosis is *representation interference*: forcing auxiliary heads to predict across
high-entropy semantic boundaries injects conflicting gradients into the shared trunk. Its fix is to
segment on entropy and mask the loss for predictions that cross a boundary — and it uses
**λ = 0.1**, not 1.0. **HiLP (2608.05806)** prices the training cost at 1B: MTP runs at 49 % of NTP
throughput (126 278 → 61 789 tok/s/GPU) for +0.44 HumanEval pass@1. **Kirchenbauer *et al.*
(2602.06019)** show offline multi-token cross-entropy cannot represent the joint distribution at
all, and their ablations favour hard argmax teacher targets, randomised k, causal masking inside the
MTP region, and — against the usual advice — **no auxiliary next-token loss on prefix tokens**.

**Draft models vs self-drafting heads.** The 2026 answer is that self-drafting wins at small target
scale, for a reason that matters to us. **"Speculative Decoding: Performance or Illusion?"
(2601.11580)** measures in production vLLM that a separate draft model's forward-time ratio `c` is
12.5 % of the target at 70B but ~37.5 % at 8B, so draft-model speculation "consistently
underperforms EAGLE-3 and, in some cases, even the training-free n-gram method" on Qwen3-8B. At
12–100M the ratio would be worse still. A separate byte drafter for Morpheme is a dead end (§E).

**Feature-level drafting (EAGLE-3 style) and KV injection.** **DFlash (2602.06036)** is the 2026
state of the art on *how* to condition a drafter: concatenate hidden states from ~5 target layers,
project once (`H_ctx = RMSNorm(W_c[H^{l1};…;H^{lm}])`), and **inject that into the K and V of every
draft layer** rather than only at the input. Its own ablation (Table 9) shows KV injection beats
EAGLE-3-style input fusion at equal depth on every task. It also finds acceptance keeps improving
with draft depth while *speedup* peaks at 5 layers, and that a drafter trained at block size 16
generalises down to 8 but not up.

**Block verification / tree drafts.** Trees are a low-batch-only win and 2026 says so twice.
2601.11580 (Fig. 2): EAGLE-3 tree k=21 beats chain k=3 at batch 1 (1.85× vs 1.65×) but falls **below
1×** at batch 64 on every workload, because the acceptance *rate* collapses 0.415 → 0.095 while
verification dominates. **Bole (2608.01651)** is the counter-case for hybrids and is discussed in
§A3. For batch-1 local serving a chain is the robust choice.

**Self-speculation / layer skip / early exit.** Present in 2026 (**S²-MoE 2608.15018**,
**2607.27735**) but aimed at MoE and at 7B+ dense targets; nothing in it transfers to a 4-Mamba-layer
outer network. **[listing-only — abstracts read, full text not.]**

**SSM / hybrid / Mamba speculation.** The richest 2026 vein, and it answers Q3.
**SpecLA (2607.16673)** states the problem exactly as we have it: "rejected candidates cannot be
removed by truncating recurrent state", and evaluates the three options — full-state snapshotting
("excessive memory consumption"), token replay ("heavy redundant computation" — **what `db77440`
currently does**), and its own **accepted-factor buffering**, which logs the compact per-token
factors verification already produced and reconstructs only the accepted state. It measures
factor-buffer reuse at **2.74–4.28×** lower accepted-state recovery latency, and 1.42–1.70×
end-to-end on a GDN-1.3B target. **Bole (2608.01651)** does the same for gated-delta trees with an
exact closed form, cutting transient state memory **82–99×** and tree-verification time 3.4–7.7×.
**DeltaLog (2608.15533)** generalises the trick to ordinary decoding: a dense base state plus a
bounded log of compact updates, merged every M steps; 1.19–1.86× kernel, 1.05–1.20× end-to-end.
**ReplaySSM (Tri Dao, tridao.me/blog/2026/replayssm/, Jun 2026)** states it for speculative decode:
cache recent *inputs*, "there is no full state to restore for each rejected token, and no per-token
state copy to keep in memory for recovery"; rollback becomes a ring-buffer pointer move. Reported
1.87–1.96× for speculative decoding.

**Byte-level / patch-level speculation.** Three 2026 papers, and they agree.
**FastBLT (2605.08044, FAIR)** adds three modes to BLT: BLT-D (block diffusion in the local decoder),
**BLT-S** (the local decoder drafts *past its normal patch boundary*, the full model verifies in one
forward — "reduces the number of expensive encoder/global calls while preserving the output of
standard autoregressive BLT decoding"), and BLT-DV. BLT-S achieves **up to 77 % lower memory
bandwidth with no loss in task performance**. **2608.15454** (our head's source paper) is covered in
§A2. **EntropyMoE (2608.06398)** confirms the byte-patch stack is still the live byte-level design
point at 1B (BPB 0.8351 vs dense BLT 0.8442) but does no speculation.

**Acceptance rules.** **"Revisiting Lossy Verification in Speculative Decoding" (2607.26627)** proves
all published lossy rules fall into two classes: *truncation-based* (Medusa typical acceptance,
SpecCascade) and *collaborative* (Leviathan lenience, CoS). For truncation-based rules the induced
law is `q(x)/Z_Θ(q)` — **the renormalised draft, not the target** — and the cost grows with task
difficulty and explodes under multi-candidate verification: under EAGLE-3 the gap to the matched
baseline widens from −0.32 to **−6.32 points** for typical acceptance (−8.8 on INCLUDE), enough to
fall below the EAGLE-3 baseline on all four benchmarks. Its constructive finding: for collaborative
rules the entire benefit of lenience comes from **capping draft overshoot at `p(x)/ℓ`**, not from
adaptive interpolation (Table 1: overshoot-ceiling holds Pass@1 at 75.1–75.9 across λ; adaptive
interpolation collapses to 50.3–66.1). **The old τ = 0.9 rule was a truncation rule with
`A_Θ = {x : q(x) ≥ τ}` — gated on the *draft's* probability, so 2607.26627 applied a fortiori.
`db77440` removed it and the UI control. Nothing to do here any more; keep the exactness.**

### A2. Q2 — is the LCA head the right draft for this architecture?

**Yes on architecture; no on the training weight; and it is missing the one cheap thing that would
raise α.**

*Architecture: keep.* Three 2026 results converge on LCA-MBP's design.
(i) 2608.15454 itself: at 373M, LCA-MBP sits on the Pareto front on 3 of 4 downstream tasks, and its
MLP-MBP baseline (Medusa/Gloeckle independent heads on a shared state) reaches **higher** byte
acceptance (58–65 % vs LCA's 46–52 %) while **losing** on downstream quality — confidently wrong.
(ii) DSpark §4.3.2 reaches the same conclusion from the other direction: a *lightweight sequential
correction* over a parallel backbone is worth more than depth — a **2-layer DSpark beats a 5-layer
DFlash on every domain** — and the semi-autoregressive gain over pure-parallel grows with block size
(+16 % at γ=7 → +30 % at γ=15 on math). (iii) **AdaMTP (2608.00434)** is the strongest and least
obvious support. Its entire contribution is that fixed-horizon MTP hurts because the horizon crosses
high-entropy semantic boundaries, and its fix is to segment on entropy and mask supervision that
crosses a segment. **Our router already produces exactly that segmentation, and the LCA mask already
confines the head to its own chunk plus the previous one.** Morpheme's head is structurally AdaMTP's
fix, obtained free from the H-Net. That materially weakens efficiency-report §3.3's use of the
MTP-hurts-at-small-scale literature: every 2026 paper that measures MTP hurting measures the
*fixed-horizon* variant AdaMTP indicts.

*The gap nobody has noticed: **at draft time our head is a pure parallel drafter**.* At training time
`lca_mask` is causal within the chunk, so byte at offset *m* attends to the real bytes at offsets
< *m*. At inference, `_speculate` builds all *n* slots from
`build_inputs(last_chunk_z.expand(n), last_chunk_start_state.expand(n), offsets 1..n)` — identical
inputs differing only by the position embedding. **The draft positions carry no information about
the bytes drafted beside them.** They attend to each other, but only to placeholders. So the head is
trained teacher-forced and drafts marginally: exactly the train/inference mismatch AngelSpec §2
describes ("the original MTP block is trained for one teacher-forced application with ground-truth
tokens … whereas multi-step inference repeatedly applies the same block to its own predicted
tokens") and exactly the multimodal collision DSpark §3.1 illustrates — "of course"/"no problem"
becoming "of problem". It is the mechanical reason positional acceptance decays, and it is the
largest untapped source of α.

**The fix is unusually cheap at byte granularity.** DSpark's Markov head is a first-order transition
bias `B(x_{k−1}, ·)` on the logits, low-rank-factorised as `W₁W₂` with `r = 256` *only because their
vocabulary is ~150 000*. Ours is **264**. A full-rank `V × V` byte transition matrix is
**264 × 264 = 69 696 parameters — 0.067 % of the flagship** — and at inference it is one row lookup
added to a 264-vector. DSpark measures the sequential loop adding **0.2–1.3 %** to round latency at
γ = 4–16 for up to +30 % accepted length. This is the highest ratio-to-effort item in the whole
report after §A0. (§D3.)

*Training weight: change.* `cfg.mbp.loss_weight = 1.0` is the one number 2026 is clear about.
AdaMTP uses **λ = 0.1**; DSpark weights CE at **α_ce = 0.1** and puts 0.9 on distribution matching;
DFlash and DSpark both apply an **exponentially decaying position weight `w_k = exp(−(k−1)/γ)`**.
`train/train.py::compute_losses` applies a flat auxiliary cross-entropy at 1.0 with no position
weighting — the configuration 2602.06019's ablations argue against.

**In fairness, `loss_weight = 1.0` is not an oversight: it is 2608.15454's own setting** (§4.3, "we
set λ₀ = 1, λ₁ = 1 and λ₂ = 10", raised to λ₁ = 2 for SFT). So this puts the source paper against
AdaMTP and DSpark. Tie-breakers: (i) 2608.15454 never ablates λ₁ and never reports a no-MBP quality
baseline at its own scale, whereas AdaMTP measures the λ-sensitive degradation across three backbones
and eight benchmarks; (ii) 2608.15454 is at 373M and *re-splits* its decoder (Table 2: FxT's 4
next-byte layers become 2 next-byte + 2 multi-byte) rather than adding a head on top of a full
decoder, as this repo does; (iii) we are 4–10× below its scale. If the A/B says otherwise, believe
the A/B.

*Better drafts from 2026?* Ranked honestly:
- **Drafting the whole chunk from the chunk vector** — already what we do, and right. `build_inputs`
  keys off `z_k` + `W_r·x̂_{start_k}` + offset, i.e. the chunk vector *is* the draft's context.
- **A DSpark-style sequential correction** — see above. Direct 2026 evidence, ~70K params.
- **DFlash-style KV injection** — replace the *additive* input with target features injected into
  every LCA layer's K and V. Measured to beat input fusion at equal depth (2602.06036 Table 9).
- **A tiny separate byte drafter** — ruled out by 2601.11580's `c`-ratio result and by 2608.12703's
  cost boundary: a full recognizer as drafter had near-perfect acceptance and **0.59–0.70×** speed.

*Scale.* There is **no 2026 evidence at 35M–100M for a byte-level hierarchical MTP head.** 2608.15454
tests **one scale, 373M**, and says so: "verifying that the gains persist at larger scales remains
future work". Anyone claiming to know what this head does at 100M is guessing; the A/B is the only
evidence that will exist.

### A3. Q3 — verification for a recurrent target

The engine now does **(a) sequential verify + replay**. The three designs, with what each is worth:

**(a) Snapshot + replay — what `db77440` does.** `clone_state()` before the round;
`forward_from_state()` over the k draft bytes as one pass; on rejection restore and re-run over the
accepted prefix. Correct, tested (`tests/test_forward_from_state.py` asserts equivalence to k
sequential `step` calls, including when a chunk boundary falls inside the segment). Costs the extra
pass quantified in §A0.

**(b) Per-position state checkpoints.** Cheap for us and expensive for everyone else, and that
asymmetry is the point. SpecLA and Bole build elaborate machinery because their states are 2–4 MiB
*per layer* at tree widths of 32–100 and batch 8–16 — Bole Fig. 1(b) measures **36 GiB** of snapshots
for eight 32-node requests. Ours, at batch 1:

| preset | d_model_outer | d_inner = 2·D0 | nheads = d_inner/64 | `ssm` per layer | Mamba layers | total SSM state |
|---|---|---|---|---|---|---|
| `local` (35M) | 384 | 768 | 12 | 12·64·64·4 B = **192 KiB** | 4 | **768 KiB** |
| `flagship` (100M) | 512 | 1024 | 16 | 16·64·64·4 B = **256 KiB** | 6 | **1.5 MiB** |

(`Mamba3Mixer.nheads = int(expand·d_model)//headdim`, `expand=2, headdim=64`; `Mamba3State.ssm` is
`[B, nheads, headdim, d_state]` fp32, `d_state=64`.) `clone_state` is therefore already affordable —
`db77440` is right to use it. The problem is not memory; **the problem is that a snapshot alone gives
you the state *before* the round, not after the accepted prefix**, which is why a replay pass is
still needed.

**(c) Accepted-factor recovery — now worth building, for latency not memory.** My first read filed
this under "not now, our state is small". That was the wrong axis: it is what removes the replay pass
and unlocks the 1.63× of §A0 row (iii). For Mamba-3's diagonal decay the algebra is elementary —
`mamba3.py::step` already computes `alpha, beta, gamma, k_rot, v` per position, so

```
S_j = (Π_{t≤j} α_t)·S_0 + Σ_{t≤j} (Π_{u=t+1..j} α_u)·(β_t·v_{t−1}k_{t−1}ᵀ + γ_t·v_t k_tᵀ)
```

is a prefix scan over ≤ k ≤ 8 terms — elementwise work, no extra model pass. Have
`forward_from_state` optionally return those per-position factors, and rejection becomes an index.
The `RoutingState` is just `last_hidden_state` (available per position from the same forward), the
`DeChunkState` EMA is a scalar recurrence (prefix-scannable, and `dc.py::_ema_chunked` already takes
an `init` carry after the same commit), and the Relation `{P2, Ĩ}` cache is append-only — **rejecting
it is a slice, not a copy** (record `S = cache[0].shape[2]` before the round, truncate to
`[:, :, :S]`). That last point also fixes the growing-cache obstacle the efficiency report flags for
CUDA graphs (§5.4).

**Break-even, our numbers.** On a launch-bound GPU (efficiency report §0.2 measures 8–15 % MFU,
~900 dispatches/forward at B=1; 2604.10597 measures 48.6× from launch count alone under WSL2), a
k-byte `forward_from_state` issues nearly the same *number* of kernels as a 1-byte `step`, so
**T_verify(k) ≈ T for k ≲ 16**. That property is the entire reason speculation can pay here. See the
§A0 table for the resulting break-even α of 0.705 / 0.605 / 0.17 across the three designs. On CPU
(i7-14700F, AVX2 only — efficiency report §5.3) the same structure holds with `c ≈ 0.10` instead of
0.20 because the head is 10.5 % of params by bandwidth, giving break-even α ≈ 0.69 / 0.59 / 0.14 and
a *higher* ceiling, since a k-byte forward streams the weights once regardless of k.

**Ceiling.** With the extra pass removed, speedup → `E[bytes]/1.2`, which at α = 0.6 saturates near
**2.1×** as k grows and at α = 0.7 near **2.8×**. With the extra pass retained it saturates at
**1.14×** at α = 0.6 — the whole design lives or dies on that pass. 2608.15454's own limitation
section agrees on direction: the horizon "cost disappears under speculative decoding verification
(Figure 8), where n can be increased without changing model output", and its Fig. 8 measures
**1.42–1.74×** at equal quality. Under the old τ rule throughput *fell* with n; under verification it
*rises*. **Fix the round first, then raise `n_candidates`.**

**Measured acceptance, and the caveat that matters.** `runs/overnight/log.jsonl` (35M `local`)
reaches **`mbp_top1_acc` = 0.605**; `pilot_sft` 0.540, `pilot_1h` 0.487, `sweep_a0.1_n4` 0.472,
`sweep_a0.3_n6` 0.362. **That is an aggregate over all in-chunk offsets, not the positional profile
`α₁,α₂,α₃` that actually sets `E[bytes]`.** Every 2026 paper reporting the profile finds steep decay:
**AngelSpec (2607.25852)** Table 2 measures 0.799 / 0.518 / 0.266 for a naively reused MTP head,
lifted to 0.814 / 0.653 / 0.524 by training-time test; **2601.11580** measures GLM-4.5-Air's shipped
MTP head at 0.92 → 0.68 → 0.38. A profile like (0.70, 0.45, 0.28) with the same 0.6-ish mean gives
`E[bytes] = 1 + 0.70 + 0.315 = 2.02` against 1.96 for uniform 0.6 — similar in aggregate, but the
*shape* is what decides whether raising `n` helps. **Add a per-offset breakdown to
`train.py::evaluate` before trusting any speedup estimate** (§D6).

### A4. Q4 — keep / modify / drop for the 100M flagship

**MODIFY, with a hard quality gate.** In order:

1. **Fix the round (§A0/§D1).** Merge replay+fix (one line, 0.83× → 0.99×), then make the corrected
   byte the anchor of the next verification pass and recover the post-prefix state from factors
   rather than by replay (→ 1.63× at today's α). Without this the head has no inference
   justification and the pause guard will keep switching it off.
2. **Add the byte transition head (§D3).** ~70K parameters, direct DSpark evidence, and it targets
   the one thing with real headroom — α. It is also what makes raising `n_candidates` worthwhile.
3. **`cfg.mbp.loss_weight = 1.0 → 0.3`** plus DFlash's position weighting (§D4), and add DSpark's
   TV-distance term if step 2's A/B is encouraging.
4. **Let the queued A/B decide keep-vs-drop at the reduced weight**, on the stated rule.

**What would change this to DROP.** If, at equal bytes and after the full cooldown, the no-MBP arm's
`val_bpb` is better by more than **two seed-sigma** — measure the sigma, do not assume a threshold —
drop the head from `flagship` and reclaim 24.5 % of training FLOPs. At 100M that is the largest
single compute lever in the model, and the speed argument does not rescue it: 2608.15454's own Fig. 8
shows an *external* verifier recovers the speed without the head being load-bearing for quality.

**What would change this to KEEP-as-is.** If the MBP arm *wins* on bpb at equal bytes, that is a
byte-level datapoint below 300M contradicting Gloeckle and confirming the AdaMTP mechanism, it is
publishable, and `loss_weight` should be swept upward rather than down. It is also the only world in
which the explorative-MBP proposal (§A5) becomes interesting.

**A design risk in the queued A/B.** Neither `runs/ab_nombp_2048` nor `runs/ab_adamw_2048` exists yet,
so this is still fixable. The names imply the arms differ in **two** things — MBP on/off *and*
optimizer (Muon vs AdamW, cf. commit `4b53299` "Muon-SW option"). If so the comparison answers
neither question. `train.py --no-mbp` sets `cfg.mbp.enabled = False`; make that the *only* difference
and name the arms accordingly. Second: the brief says "equal bytes", efficiency report §3.3 said
"equal wall-clock". These answer different questions and can disagree, because MBP-off trains
~1.4–1.5× more bytes per hour. **Log both curves** and state which one the decision used.

### A5. Explorative Modeling as an MBP head — evaluated

**Proposal** (from the coordinator, after 2607.27372, Gladstone/Ji/Du, 29 Jul 2026): give the MBP
head K learned latent embeddings; run K head forwards per chunk over the shared backbone; train only
the candidate closest to the data (Forward XM / best-of-K); at decode sample one latent, draft the
chunk under it, and verify exactly with `q` = the latent-conditioned per-position distribution.

**What the paper actually says, verified.** XM factors the *training loop* rather than the generation
procedure: "at each training step, the model explores K possible matches between what it generates
and the data, and trains on the closest", raising *generative expressivity* — "the number of distinct
modes a generative model's training objective allows it to capture". For discrete models it is
implemented exactly as proposed: "For XMDLMs, we instead learn K discrete latent embeddings: each
candidate samples one of these embeddings at random for every masked position, and the best-of-K
selection is over the resulting latent-conditioned predictions." Cost: "each additional explored
candidate in Forward XM adds only a forward pass … XM-K costs (K+2)/3 standard steps in the
FLOP-efficient mode, plus one more forward pass in the memory-saving mode". Inference is unchanged:
"Exploration happens entirely during training, so inference is unchanged." Gains grow with scale
(7 %→36 % as data grows, 13 %→23 % as models grow); recommended sweep K ∈ {1,2,3,5}, then {8,12}.

**Verdict: mechanically sound, exactly on-target for the diagnosed problem, and still ranked below
the sequential head — for a reason that is specific to speculation.**

*In its favour.* (1) **Exactness survives.** Conditioned on a sampled latent ℓ the head's
per-position output is a plain softmax, so `q(x) = p_head(x | ℓ, z_k, offset)` is an exact
evaluation. Leviathan's proof requires only that `x ~ q` and that the *same* `q` is evaluated — not
that `q` be the marginal over ℓ. So the DSpark §6 requirement ("the drafter must provide exact
per-token probabilities … most techniques cannot, due to iterative refinement, latent
marginalization, or global normalization") is met, unlike CRF-NAT or CTC drafters. This is a real
and non-obvious point in the proposal's favour. (2) It attacks precisely the defect diagnosed in §A2
— that our draft slots are conditionally independent — and it does so *without* a sequential loop, so
all n bytes stay parallel. (3) It is architecturally trivial: one `nn.Embedding(K, D0)` added inside
`build_inputs`, alongside the existing `pos` term.

*Against, and this is decisive.* **Best-of-K optimises the wrong quantity for a drafter.** Acceptance
is exactly `1 − ½‖q − p‖₁` (Leviathan; DSpark Eq. 8 uses it as the supervision target). What
maximises acceptance is *TV proximity to the target's distribution* — a calibration objective. What
best-of-K maximises is *mode commitment* — a sample-quality objective. These pull in opposite
directions: a draft that confidently commits to the wrong mode has **larger** TV distance than a
hedged one, and its `q(x)` is **large** for a byte the target dislikes, which is precisely the
*overshoot* regime 2607.26627 identifies as the driver of low-quality generation and which its
overshoot-ceiling result is built to suppress. 2607.27372 makes the tension explicit itself, in
another context: "likelihood has long been observed to correlate poorly with sample quality …
likelihood measures how well a density is fit while generative expressivity determines how many modes
that density can hold." Acceptance is a likelihood-ratio quantity. Mode-committing improves the
metric that does not govern acceptance and degrades the one that does. 2608.15454's MLP-MBP result is
the empirical shadow of this: the *more confident* head (58–65 % acceptance) was the worse one.

*Cost, priced against our budget.* The head is **24.5 % of flagship training FLOPs, 33 % at `local`**.
On 8 GB the memory-saving mode is forced (K candidates' activations will not fit), which is K no-grad
forwards + 1 forward + 1 backward = `2NK + 6N` against `6N`, i.e. the head's cost **doubles at K=3**.
Total training cost: **+24.5 % at flagship, +33 % at `local`**. (FLOP-efficient mode would be +16 %
and +22 % but needs the activation memory.) We are currently debating whether to *delete* this head to
reclaim compute; adding a quarter to the training bill for the axis with no supporting evidence is the
wrong direction.

*Evidence.* I queried 2607.27372 directly for speculative decoding, draft heads, MTP heads and
acceptance rates and **found none** — it evaluates images, video, and language modelling at DiT
scales. **There is no 2026 evidence for best-of-K on a draft head or an MTP head.** The coordinator
cites a §6 claim that MTP targets are more multimodal and so "give exploration more to offer"; the
pages returned to me were 1, 2, 4, 6, 7, 8, 28, 35, 36 and did not include §6, so I record that as
**[unverified in my read]**. I have no reason to doubt it, and I note it cuts both ways: it is an
argument that XM would improve the head's *modelling*, not its *acceptance*.

*Do they compose?* Yes, mechanically — the latent is an additive input, the Markov bias an additive
logit; nothing conflicts, and you could run both. But they are substitutes for the same defect, one
has direct 2026 measurement on drafters (DSpark: 2-layer + sequential beats 5-layer parallel; +16 %
to +30 % accepted length; 0.2–1.3 % latency) at ~70K parameters and no training-cost increase, and the
other has none at +24.5 % training cost. Do the cheap one first and measure.

*The one condition that flips this.* If the A/B shows the MBP objective **helps `val_bpb`** — i.e.
the head is earning its keep as a *training signal* rather than as a drafter — then the relevant axis
is modelling quality, 2607.27372's scale argument applies, and exploration becomes worth a
K ∈ {2,3} sweep. In that world the right place to spend it is **the next-byte objective / the model
as a whole**, not a draft head whose job is TV-matching. That is XM's own framing: a third
*pretraining* axis. Filed as table row 12.

---

## B. Ranked table of applicable techniques

Effort: S = under an hour, M = a day, L = a week+. Risk is to numerics/quality, not schedule.
Rows 0a/0b record what `db77440` already did, so the table reads as a plan and not as a wish.

| # | Technique | 2026 evidence | Expected effect here | Effort | Risk |
|---|---|---|---|---|---|
| 0a | ~~Use `forward_from_state` so k drafted bytes cost one target pass~~ | — | **DONE in `db77440`** | — | — |
| 0b | ~~Replace τ=0.9 with exact rejection sampling~~ | 2607.26627 | **DONE in `db77440`**; output is now exactly the target's law | — | — |
| 1 | **Merge the rejection replay and the fix step into one `forward_from_state`** | Structural; DSpark's round has one target pass | **0.83× → 0.99×** at α=0.6, k=3. One line | S | None |
| 2 | **Fix-as-anchor + accepted-factor state recovery** — corrected byte becomes position 0 of the next verification pass; no replay | DSpark Fig. 1 anchor/bonus structure; SpecLA §5 factor buffering (2.74–4.28× recovery); Bole §IV-D; ReplaySSM | **0.83× → 1.63×** at today's α; **→ 1.99×** at k=6. The largest single lever in this report | M | Medium — needs an equivalence test against `step` |
| 3 | **Byte transition head** — full-rank `V×V = 264×264` first-order logit bias on the draft, giving the parallel draft an intra-chunk dependency | DSpark §3.1 Markov head (they low-rank it only because V≈150k); §4.3.2 "a little autoregression goes a long way"; AngelSpec Table 4 replicates | Raises α, the only quantity with real headroom. **70K params (0.067 % of flagship)**, 0.2–1.3 % round latency | S | Low |
| 4 | **`mbp.loss_weight` 1.0 → 0.3** + `exp(−(m−1)/n)` position weighting | AdaMTP 2608.00434 (λ=0.1); DSpark Eq. 9–12 (α_ce=0.1); DFlash Eq. 4 | Targets the representation interference AdaMTP measures | S | Medium — changes the A/B's meaning; freeze before the run |
| 5 | **Per-offset acceptance telemetry** (`mbp_top1_acc@1..n`) | Every 2026 paper reports positionwise α; profiles decay steeply | You cannot estimate speedup from the aggregate. Prerequisite for #6 | S | None |
| 6 | **Raise `n_candidates`** once #1–#3 land | 2608.15454 (−22 % from n=3→7 under threshold, +29–37 % under verification); DSpark (gain grows with γ) | Free once the round is fixed; k≈6 is the natural target at bpic 3.3 | S | Low |
| 7 | **Recalibrate the `PAUSE_BELOW` guard** from 0.15 to the model's actual break-even (0.605 today, 0.17 after #2) | §A0 arithmetic | Today the guard sits 4.7× below break-even, so it never fires when it should | S | None |
| 8 | **TV-distance loss** `‖p_mbp − p_nbp‖₁` alongside CE | DSpark Eq. 10, α_tv = 0.9; acceptance ≡ `1 − ½‖q−p‖₁` | Trains the head for the thing verification measures. Both distributions are already computed | M | Medium — new term, needs its own A/B |
| 9 | **DFlash-style KV injection** into every LCA layer | 2602.06036 Table 9 | Higher α at equal head size | M | Medium — invalidates checkpoints |
| 10 | Confidence head + draft early-stop | DSpark §3.2.1; 2608.14787 (9.6–13.5 % fewer verifier calls) | Small at batch 1 — DSpark says the payoff is high-concurrency. Build only if the studio batches | M | Low |
| 11 | Tree / multi-candidate drafts | 2601.11580 Fig. 2 (tree < 1× at batch 64); Bole 2608.01651 | Batch-1-only, and reintroduces the state-divergence problem §A3 just avoided | L | High |
| 12 | **Explorative (best-of-K) training** — K latent embeddings, train the closest candidate | 2607.27372 (7→36 % gains that grow with scale) — but **no evidence on draft heads or acceptance** | **Defer.** Optimises mode commitment, not TV distance (§A5); +24.5 % flagship training cost. Revisit only if the A/B shows the MBP objective helps `val_bpb`, and then apply it to the next-byte objective, not the draft head | L | High — pays the axis with no drafter evidence |
| 13 | `torch.inference_mode()` on the decode path; hoist the per-byte `torch.tensor([[cur_offset]])` | Efficiency report §5.3 | Free; removes a per-byte host-to-device sync | S | None |

---

## C. Deep dive: DSpark (arXiv 2607.05147) and the DeepSpec repository

### C1. What they are

**DSpark** is a speculative-decoding *method* from Peking University and DeepSeek-AI (34 authors,
6 Jul 2026, cs.AI; 586 votes / 9 626 views on alphaXiv — the most-read speculative-decoding paper of
2026). **DeepSpec** (`github.com/deepseek-ai/DeepSpec`, MIT) is the *training and evaluation
repository* released alongside it. DeepSpec is not itself an algorithm; it implements three —
**Eagle3**, **DFlash** and **DSpark** — on a common stack adapted from SGLang's SpecForge
(Apache-2.0; see the repo's `NOTICE`). It ships checkpoints for Qwen3-{4B,8B,14B} and Gemma-4-12B for
all three, and its `README` warns the data pipeline needs "roughly 38 TB" of target cache for the
default Qwen3-4B setting.

### C2. The draft mechanism — semi-autoregressive

DSpark's thesis is that the two drafter families each sacrifice one term of
`L = (T_draft + T_verify)/τ`: autoregressive drafters get high τ but pay `T_draft ∝ γ`; parallel
drafters collapse `T_draft` to one pass but sacrifice τ because each position marginalises over all
possible predecessors — "of course"/"no problem" becomes "of problem". DSpark keeps the heavy backbone
parallel (it *is* DFlash: 5 layers, KV-injected target features) and adds a **lightweight sequential
head** supplying a prefix-dependent logit bias:

```
P(X | x₀) = Π_k p_k(x_k | x₀, x_<k),
p_k(v | x₀, x_<k) = softmax_v( U_k(v) + B_k(x₀, x_<k, v) )
```

`U_k` is the parallel backbone's base logit. Two instantiations, both in
`deepspec/modeling/dspark/markov_head.py`:

* **Markov head** (default): `B(x_{k−1}, ·) = W₁[x_{k−1}]·W₂`, `W₁ ∈ R^{V×r}, W₂ ∈ R^{r×V}`,
  **r = 256** — a low-rank first-order transition matrix. **At our V = 264 the low-rank
  factorisation is unnecessary; the exact `V×V` matrix is 70K parameters.**
* **RNN head**: a GRU-style gated state over the whole in-block prefix (`_rnn_step`). The paper's own
  finding is that it "provides only marginal additional gains over the Markov head, mainly at longer
  proposal lengths", so the Markov head is the default.

**The crucial design property (§6):** "DSpark circumvents these limitations by keeping the sequential
correction *local*, so per-token probabilities remain exact softmax evaluations." CRF-NAT and CTC
drafters cannot be verified losslessly because global normalisation / latent marginalisation prevents
exact per-token `q(x)`. **A drafter that cannot report exact `q(x)` cannot be verified losslessly.**
Our LCA head can — it ends in a plain softmax over the shared `lm_head`. So would a latent-conditioned
explorative head (§A5).

### C3. The acceptance rule

DSpark uses "standard speculative decoding (Chen et al. 2023; Leviathan et al. 2023)" at temperature
1.0. `deepspec/eval/base_evaluator.py::verify_draft_tokens`:

```python
accept_prob        = torch.clamp(selected_target_probs / selected_draft_probs.clamp_min(1e-8), max=1.0)
accept_mask        = (torch.rand_like(accept_prob) < accept_prob).to(torch.int64)
accept_prefix_mask = accept_mask.cumprod(dim=1)          # left-to-right; first rejection truncates
accepted           = int(accept_prefix_mask.sum(dim=1)[0])
next_token = sample_residual(target_probs[:, accepted, :], proposal.draft_probs[:, accepted, :])
```

with `sample_residual` doing `residual = clamp(p − q, min=0); residual /= residual.sum()` and falling
back to `p` when the residual mass underflows below `1e-8` — a guard that matters at byte
granularity, where a confident head can put ~1.0 on one byte. **`engine.py::verify_draft` now matches
this, including the guard.**

**Confidence-scheduled verification** is a *second*, orthogonal mechanism and is **not** an acceptance
relaxation. A confidence head `c_k = σ(wᵀ[h_k; W₁[x_{k−1}]])` predicts per-position survival,
supervised against the analytic acceptance rate `c*_k = 1 − ½‖p^d_k − p^t_k‖₁` (Eq. 8) and calibrated
by **Sequential Temperature Scaling** (left-to-right 1-D grid search minimising ECE of the cumulative
product). A hardware-aware scheduler then prunes the draft *before* verifying. DSpark is careful
about why this stays lossless: admission decisions must satisfy the **non-anticipating property**, so
the scheduler early-stops the greedy search the moment throughput drops, which "isolates the admission
event from future tokens, ensuring exact target-distribution recovery" (counterexample for the naive
global-sort version in their Appendix A). **Anyone adding confidence-based pruning must reproduce
that argument or losslessness is gone.**

*Note for our engine:* the existing draft-truncation `if b in STOP_IDS: break` is safe under this
criterion, because shortening the proposal only means the next byte comes from the target directly —
it changes efficiency, not the emitted law.

### C4. Training recipe

Target frozen. The draft shares the target's embedding and LM head, **both frozen**; only the backbone
drafter, sequential block and confidence head update. Anchor positions sampled randomly per sequence
to form γ-token blocks. Three losses, all position-weighted by **`w_k = exp(−(k−1)/γ)`** (from
DFlash):

```
L_ce   = −Σ w_k log p^d_k(x*_k)                      α_ce   = 0.1
L_tv   =  Σ w_k ‖p^d_k − p^t_k‖₁                     α_tv   = 0.9
L_conf = −Σ w_k [c*_k log c_k + (1−c*_k) log(1−c_k)] α_conf = 1.0
```

The 0.1/0.9 split has an exact rationale: per-step acceptance **equals** `1 − ½‖p^d − p^t‖₁`, so
minimising TV distance directly maximises expected acceptance, whereas cross-entropy against ground
truth optimises the wrong objective. Data: Open-PerfectBlend prompts only, responses **regenerated by
each target model**; 10 epochs; 5 draft layers (1 for Eagle3); block size 7.

### C5. Reported results

* **Offline** (Qwen3-4B/8B/14B, T=1.0, chain drafting): accepted length improves over Eagle3 by
  **30.9 / 26.7 / 30.0 %** and over DFlash by **16.3 / 18.4 / 18.3 %**.
* **Depth** (§4.3.2): a **2-layer DSpark beats a 5-layer DFlash on all domains**; steepest marginal
  gain is 1 → 2 layers.
* **Block size**: gain over DFlash grows with γ — +16/15/18 % (math/code/chat) at γ=7 → **+30/26/22 %**
  at γ=15.
* **Sequential-loop latency**: at batch 128, scaling draft length 4 → 16 adds **0.2–1.3 %** to
  full-round latency for up to +30 % accepted length.
* **Production** (DeepSeek-V4 serving, live traffic, vs the MTP-1 baseline): per-user generation
  **+60–85 %** (V4-Flash), **+57–78 %** (V4-Pro) at matched aggregate throughput.
* **Quality**: not reported as a delta, and correctly so — the acceptance rule is exact, so the output
  is distributed identically to the target. **That is the point:** the entire speedup is obtained at
  zero quality cost.
* **Independent corroboration**: AngelSpec (2607.25852) retrains DSpark on Qwen3-8B and measures mean
  accepted length **5.32** vs DFlash 4.57 vs MTP 3.24; its own DFly edges it to 5.41.

### C6. Which parts apply to a byte-level recurrent model

**Applies directly:**
1. **The exact acceptance rule and `sample_residual`.** Architecture-independent. **Already ported.**
2. **The anchor/bonus round structure** (Fig. 1). This is table row 2 and the 1.63× of §A0.
3. **The Markov transition head**, un-factorised at V=264. Table row 3.
4. **`α_tv = 0.9` distribution matching.** Our head and next-byte head share `lm_head`, so
   `‖p_mbp − p_nbp‖₁` is computable in the forward that already produces both. Table row 8.
5. **`w_k = exp(−(k−1)/γ)`** over the in-chunk offset. Table row 4.
6. **"A little autoregression goes a long way"** — support for `mbp.n_layers = 2`.
7. **The exact-`q(x)` requirement.** Keep the head ending in a plain softmax.

**Applies with modification:**
8. **The confidence head.** Cheap and its supervision target is free during training, but DSpark says
   the payoff is high-concurrency; at batch 1 there is nothing to schedule.

**Does not apply:**
9. **The hardware-aware prefix scheduler.** It maximises `U(B,ρ)/C(B,ρ)` over a batch. `B = 1`.
10. **KV injection as literally written.** Eq. 2–3 concatenate target features into the key/value
    sequence dimension of an attention drafter. Our outer network is Mamba-3; there is no KV to inject
    into. The *idea* transfers (row 9), the mechanism does not.
11. **Frozen-target post-hoc training.** DSpark retrofits onto a shipped model. Our head is co-trained,
    which is strictly better for acceptance (2601.11580: co-trained MTP heads "achieve substantially
    higher token-acceptance rates" than post-hoc fine-tuned ones) and is why our A/B is about
    *quality*, not acceptance.

---

## D. Code-level guidance

### D1. `serve/engine.py::_generate` — fix the round (ranks 1–2)

**Rank 1, one line.** Replace

```python
state = snapshot
if n_acc > 0:
    self.model.forward_from_state(torch.tensor([xs[:n_acc]], device=self.device), state)
...
lg_next, routing, is_b, mbp = self.model.step(torch.tensor([[fix]], device=self.device), state)
```

with a single call over `xs[:n_acc] + [fix]`, taking `logits` from its last row and the boundary flags
from its returned `bm_seq`/`bp_seq`. Saves one target pass on **every** rejected round
(78 % of rounds at α = 0.6).

**Rank 2, the real fix.** Two changes together:

* *Fix-as-anchor.* Do not consume the correction with its own pass. Carry it as `pending` and make the
  next round's verification input `[*pending, *draft]`, reading the correction's own successor logits
  from row 0 of that pass. This is DSpark's Fig. 1 exactly. The draft for the next round can be built
  *before* the state advances past `fix`, because `LCAHead.build_inputs` depends only on
  `(last_chunk_z, last_chunk_start_state, offset)` — and if `fix` is mid-chunk, neither of the first
  two changes. Only the attention *keys* (`cur_chunk_inputs`) depend on the accepted bytes; refresh
  them on the next pass.
* *No replay.* Get the state after the accepted prefix from the factors the verification pass already
  computed, per §A3(c). Add an optional `return_factors=True` to `forward_from_state` returning
  per-position `(alpha, beta, gamma, k_rot, v)` for each Mamba layer plus per-position `h` for the
  routing state, and reconstruct `S_j` by prefix scan. Truncate the Relation cache by slice.

Guard rails for both:
* **Do not `argmax` the draft** — exact verification needs `x ~ q` and the sampled `q(x)`. The current
  code already samples; keep it.
* `forward_from_state` mutates `state`, so any `clone_state` must precede it.
* The window can cross a chunk boundary; `forward_from_state` handles it and
  `test_forward_from_state.py` asserts at least one seed exercises that path. Keep the assertion, and
  extend it to the factor-recovery path.
* Emit `t_ms` per **round**, not per byte, or the studio's timing panel is a fiction. (`db77440`
  already divides by accepted count — reasonable, but label it.)

### D2. Recalibrate the pause guard (rank 7)

`PAUSE_AFTER, PAUSE_BELOW = 48, 0.15` fires only below 15 % acceptance, while real break-even is
**70.5 %** as landed and **60.5 %** after rank 1. Either raise `PAUSE_BELOW` to a computed break-even,
or — better — replace the acceptance heuristic with a measured one: track
`bytes_emitted / target_passes` over a window and pause when it drops below 1.0. That is the quantity
that actually decides, it needs no constant, and it stays correct as ranks 1–3 change the cost model.
`stats()` already carries `spec_rounds`, `spec_fixes`, `spec_replays`; add `target_passes`.

### D3. The byte transition head (rank 3)

In `mbp.py`, add `self.trans = nn.Embedding(V, V)` (264×264, zero-init so it starts as a no-op) and,
in `_speculate`, sample left-to-right over the n slots adding `self.trans(prev_byte)` to each logit
row before the softmax. The head's forward stays one parallel pass; only the n cheap 264-vector
lookups are sequential. Train it by unrolling the same bias over the teacher-forced in-chunk bytes,
which the LCA mask already exposes. Zero-init means the A/B against no-transition is a clean ablation.
Report `mbp_top1_acc@1..n` with and without.

### D4. Training loss (rank 4)

`config.py::MBPCfg.loss_weight = 1.0` → **0.3**. In `train.py::compute_losses` the MBP term is a flat
`_masked_ce_sum`; add position weighting over the in-chunk offset, already returned as `out.offset`:

```python
w = torch.exp(-(out.offset.float()) / max(cfg.mbp.n_candidates, 1))   # DFlash Eq. 4
```

Offset 0 is predicted by the next-byte head and carries no MBP loss anyway, so this concentrates
supervision on offsets 1..n — the only ones speculation uses.

### D5. Dead configuration to remove

`config.py::MBPCfg.accept_threshold` is still present (line 56) but `db77440` removed the only
consumer. Delete it, or it will be reintroduced by someone reading the config as documentation.

### D6. Telemetry (rank 5)

`train.py::evaluate` computes `mbp_top1_acc` as a single aggregate over `targets.numel()`. Replace
with a per-offset vector — `out.offset` is in `HNetOutput`, so it is a `scatter_add` over
`offset.clamp(max=n)`. Log `mbp_top1_acc@1..n`. Without it neither the A/B verdict nor any speedup
estimate in §A0 is checkable.

### D7. Two small things on the same hot loop

* `hnet.py::step` still builds `torch.tensor([[state.cur_offset]], device=h.device)` per byte — a
  host-to-device sync on CUDA. Preallocate a 1×1 buffer and `fill_`. (`forward_from_state` builds its
  offsets on device; `step` remains the non-speculative path.)
* `@torch.no_grad()` → `torch.inference_mode()` on `step`, `prefill`, `forward_from_state` and
  `engine._generate`.

---

## E. Sounds good but will not help here

1. **A separate small byte drafter.** 2601.11580 measures the draft/target forward ratio at 12.5 % for
   a 70B target but ~37.5 % for 8B, and concludes draft-model SD "consistently underperforms EAGLE-3
   and, in some cases, even the training-free n-gram method" on Qwen3-8B. At 35–100M it is worse.
   2608.12703's cost boundary is the same lesson: a full recognizer as drafter had near-perfect
   acceptance and **0.59–0.70×** the speed of no speculation at all.
2. **Tree / multi-candidate drafts.** 2601.11580 Fig. 2: EAGLE-3 tree k=21 gains at batch 1 but is
   **below 1×** at batch 64 on every workload (acceptance rate 0.415 → 0.095). And it reintroduces
   exactly the recurrent-state branching that §A3 avoids, which is why Bole needs a closed-form solver
   and an 82–99× memory reduction to make it viable at all.
3. **Full SpecLA/Bole/DeltaLog machinery.** Their designs exist because a GDN state is 2 MiB *per
   layer* at tree widths of 32–100 and batch 16 (Bole Fig. 1b: 36 GiB). Ours is 1.5 MiB total.
   Borrow the **factor-recovery idea** (§A3(c), table row 2) for its latency, not the memory
   machinery.
4. **Windowed-MTP (2607.21535).** Important — a built-in MTP head's full-context attention can make
   deep speculation *net-negative* at 1M tokens (0.80× on Qwen-122B code QA) — but its premise is
   `S = 10⁶`. Our `max_seq_len` is 2048–4096 and the LCA mask is **already** a window (own chunk +
   previous, ~7 bytes). We have the fix by construction.
5. **Diffusion drafters (DFlash 2602.06036, xPress 2608.02438, DARTree 2608.13524, DBLAST
   2608.05448).** The point of a parallel/diffusion drafter is that `T_draft` becomes independent of
   block size, which pays at γ = 8–16. Our γ is bounded by the chunk (bpic ≈ 3.3), so `T_draft ∝ γ`
   is already negligible. *(DARTree, DBLAST, xPress: listing-only.)*
6. **Confidence-scheduled verification / D-cut / verifier skipping** (DSpark §3.2, AngelSpec
   2607.25852, 2608.14787). All optimise verification budget **across concurrent requests**.
   `engine.py` holds a `threading.Lock` and serves one generation at a time. AdaMTP says the same of
   its adaptive-horizon mode: the saving "yields little speedup for single-sample inference, which is
   memory-bandwidth bound."
7. **Tuning the head for acceptance rate.** 2608.15454's MLP-MBP baseline reaches **58–65 %**
   acceptance against LCA's **46–52 %** and loses on downstream quality at comparable throughput.
   `mbp_top1_acc` is a diagnostic; optimising it pushes toward the worse architecture. The correct
   training target is TV distance to the *next-byte head* (DSpark Eq. 10), not to ground truth.
8. **Typical acceptance (Medusa) as a middle ground.** 2607.26627 Table 3: under EAGLE-3 it falls
   below the plain baseline on all four benchmarks and loses **8.8 points** on INCLUDE. Of all lossy
   rules it is the one 2026 most specifically indicts. `db77440` was right to delete the threshold.
9. **Explorative / best-of-K training on the draft head** (2607.27372). Full evaluation in §A5.
   Short version: it optimises mode commitment, but acceptance is `1 − ½‖q − p‖₁`, a calibration
   quantity; a confidently-wrong committed draft has *worse* TV distance and manufactures exactly the
   `q > p` overshoot that 2607.26627 identifies as the driver of low-quality generation. Costs
   **+24.5 % of flagship training FLOPs** (memory-saving mode, K=3, on a head that is 24.5 % of the
   budget), against ~70K parameters and no training-cost increase for the DSpark sequential head that
   fixes the same defect with direct drafter evidence. **No 2026 paper applies best-of-K to a draft or
   MTP head.** Revisit only under the §A5 condition, and then on the next-byte objective.
10. **Expecting a 6.4× byte-level speedup.** Gloeckle's *(background, pre-2026)* §3.3 8-byte result is
    at **7B**. 2608.15454, the only byte-level MBP measurement in 2026, reports **1.42–1.74×** at 373M
    with verification. Budget for that range.
11. **Deciding the A/B from a partial loss curve.** Restated from efficiency report §3.3: 2509.02046
    *(background)* found optimizer/architecture rankings flip during LR decay. Run the full cooldown.

---

## F. 2026 papers read

Read in full or substantially (full text via PDF query, or repository source). 19 arXiv papers + 1
dated blog post.

| ID | Month | Title | Relevance in one line |
|---|---|---|---|
| 2601.11580 (v2 Mar) | Jan | Speculative Decoding: Performance or Illusion? | Production vLLM study: verification dominates, draft-model SD fails at small targets (`c`≈37.5 % at 8B), trees die above batch 16 |
| 2602.06019 (v2 Apr) | Feb | Multi-Token Prediction via Self-Distillation | Offline multi-token CE cannot represent the joint; ablations favour hard targets, randomised k, **no** auxiliary NTP loss |
| 2602.06036 (v2 May) | Feb | DFlash: Block Diffusion for Flash Speculative Decoding | KV injection into every draft layer beats input fusion; `w_k = exp(−(k−1)/γ)` loss weighting; 5 layers is the speedup optimum |
| 2605.08044 | May | Fast Byte Latent Transformer (FAIR) | BLT-S: the local decoder drafts past its patch boundary and the full model verifies — 77 % less bandwidth, no quality loss. Closest byte-level analogue to what we should build |
| tridao.me/blog/2026/replayssm/ | Jun | ReplaySSM (Tri Dao) | Cache inputs not states; rollback is a ring-buffer pointer move; 1.87–1.96× for speculative decode |
| 2607.05147 | Jul | **DSpark** (DeepSeek-AI + PKU) — the must-read | Semi-autoregressive draft (parallel backbone + low-rank Markov head), exact rejection sampling, confidence-scheduled verification; +60–85 % per-user in DeepSeek-V4 production |
| 2607.16673 | Jul | SpecLA: Speculative Decoding for Linear-Attention Models | Names the three state-recovery options (snapshot / replay / accepted factors) and prices them; 1.42–1.70× on GDN-1.3B |
| 2607.21535 | Jul | Windowed-MTP (NVIDIA) | A built-in MTP head's full-context attention makes deep speculation net-negative at 1M; a draft-only window is lossless by construction |
| 2607.25852 (v2) | Jul | AngelSpec / DFly (Tencent) | Workload heterogeneity; positionwise acceptance 0.799/0.518/0.266 → 0.814/0.653/0.524 with TTT; independently replicates DSpark at accepted length 5.32 |
| 2607.26627 | Jul | Revisiting Lossy Verification in Speculative Decoding | **The acceptance-rule paper.** Truncation rules emit the renormalised *draft*; −6.32 pt under EAGLE-3; overshoot ceiling is the only relaxation that survives |
| 2607.27372 | Jul | Explorative Modeling (UIUC + Harvard) | Best-of-K as a third pretraining axis; K learned discrete latents for XMDLM; XM-K costs (K+2)/3 standard steps; inference unchanged. Evaluated for our head in §A5 |
| 2608.00434 | Aug | AdaMTP: An Adaptive Training Paradigm for MTP | Fixed-horizon MTP is *below* NTP on all three backbones; mechanism is cross-boundary gradient interference; fix = entropy segmentation + masked loss, λ = 0.1 |
| 2608.01651 | Aug | Bole: Efficient Tree Speculation for Hybrid-Attention LMs | Exact closed form for tree verification over gated-delta recurrence; 82–99× less transient state; quantifies why snapshots are unaffordable at scale (and affordable at ours) |
| 2608.03447 (v2) | Aug | Approximate Speculative Decoding | Budgeted longest-prefix greedy verification with a request-level regret ledger; +3.05–15.26 %, explicitly *not* output-preserving |
| 2608.05806 | Aug | Hierarchical Latent Prediction (HiLP, MSR) | Prices MTP training: 49 % of NTP throughput at 1B for +0.44 HumanEval; latent auxiliary objectives dominate and cost nothing at inference |
| 2608.06398 | Jul/Aug | EntropyMoE: Entropy-Aware Sparse Expert Routing for Tokenizer-Free LLMs | Byte-patch stack at 1B; BPB 0.8351 vs dense BLT 0.8442; confirms byte-patch is a live 2026 design point, does no speculation |
| 2608.12703 | Aug | Alignment Drift in Single-Model Speculative Decoding for ASR | Restart-vs-continuation acceptance split; a full-capability drafter reaches near-perfect acceptance at **0.59–0.70×** speed — the cost boundary for self-drafting |
| 2608.14787 | Aug | From Positionwise Confidence to Prefix Scheduling: Verifier Skipping | Better positionwise predictors ≠ better schedulers (skips need *contiguous* prefixes); fewer verifier calls ≠ higher throughput |
| 2608.15454 | Aug | Dynamic Multi-Byte Prediction With Hierarchical LMs (LCA-MBP) | **Our head's source paper.** 373M only; acceptance 46–52 %; MLP-MBP is confidently wrong at 58–65 %; Fig. 8 shows external verification gives 1.42–1.74× at equal quality |
| 2608.15533 | Aug | DeltaLog: Deferred Materialization of Recurrent States | Dense base + bounded compact-update log, merge every M steps; 1.19–1.86× kernel, 1.05–1.20× end-to-end |

**Count by month (papers read):** Jan 1 · Feb 2 · Mar 0 · Apr 0 · May 1 · Jun 0 (+1 blog) · Jul 6 ·
Aug 9. **Total 19 arXiv + 1 blog = 20 sources.**

### F2. Surfaced but read at listing level only (title + abstract; **not** used for any load-bearing claim)

2601.07353 Talon · 2601.18902 Flatter Tokens · 2601.19278 DART (diffusion) · 2602.01274 PACER ·
2604.09557 Speed-Bench · 2604.12247 SpecBound · 2605.08632 PARD-2 · 2606.00487 TAPS ·
2606.02091 DFlare · 2607.22022 HEMERA · 2607.27735 A Sparse Glimpse of the Whole ·
2608.02032 DART (recurrent) · 2608.02438 xPress · 2608.02954 LowRank-SSM · 2608.02989 AcceptMoE ·
2608.05448 DBLAST · 2608.11231 LinearKV · 2608.12435 MARCH · 2608.13524 DARTree · 2608.15018 S²-MoE.

### F3. Background (pre-2026), cited only where 2026 does not supersede

2211.17192 Leviathan (speculative sampling; the correctness result is a theorem and stands) ·
2302.01318 Chen (same rule) · 2401.10774 Medusa (typical acceptance — now indicted by 2607.26627) ·
2402.05109 Hydra · 2404.19737 Gloeckle (MTP scale curve; 8-byte 6.4× at 7B) ·
2507.07955 H-Net dynamic chunking · 2505.14969 STree · Snakes and Ladders (NeurIPS-ENLSP 2024).

### F4. Claims I could not verify

* Efficiency report §5.1 quotes 2608.15454 for "at n=6 the mean accepted count is 3.05 with 15 % of
  steps accepting the full window, but no candidate is ever accepted at the first decoding step."
  My PDF queries returned pp. 1–6, 9–10, 12 and did not surface that passage. **[unverified —
  re-check against the paper's Appendix B before acting on the "skip the first step" advice.]**
* The coordinator's citation of 2607.27372 §6 (that MTP targets are more multimodal and so "give
  exploration more to offer") is consistent with the paper's framing, but §6 was not among the pages
  returned to me (1, 2, 4, 6, 7, 8, 28, 35, 36). **[unverified in my read.]** Note it argues XM would
  improve the head's *modelling*, which is a different axis from acceptance — see §A5.
* ReplaySSM is a blog post, not peer-reviewed; its numbers (1.43–1.48× standard, 1.87–1.96×
  speculative, 3.0–3.3× concurrency) are the author's and unreplicated. The *mechanism* is
  independently corroborated by SpecLA §5 and DeltaLog §3, which is what §A3 leans on.
* The `c ≈ 0.2·T` draft-cost constant throughout §A0/§A3 is my estimate from the dispatch counts in
  efficiency report §0.2 (mbp_head 98 of 2380 aten dispatches on the CPU reference path), not a
  measurement. **Measure it** — time `_speculate` and `step` under `torch.cuda.Event` before trusting
  the absolute speedups. The qualitative conclusions (as-landed is net-negative at α = 0.6; removing
  the extra pass is worth ~2×) are robust to `c` anywhere in 0.05–0.4; the absolute numbers are not.
* The `T_verify(k) ≈ T` assumption underpinning every speedup figure follows from the launch-bound
  diagnosis in efficiency report §0.2 but has not been measured on this model. **Time
  `forward_from_state` at k = 1, 3, 6, 12** — it is a five-minute experiment and it validates or
  invalidates the entire §A0 table.
