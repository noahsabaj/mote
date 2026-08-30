# The hybrid ladder — Relation × Mamba-3 main networks, pre-registered 2026-08-29

Signed after a five-round grilling on 2026-08-29 (Noah's answers are the decisions below; the evidence is the
2026 SSM sweep in § Literature). Nothing here changes the trunk that launches Monday 2026-08-31.

## Frame

* **Question.** Does a main network that mixes Relation layers with Mamba-3 layers beat the all-Relation main at
  matched parameters and bytes — and where do the Relation layers belong? Staged: the claim on a ladder first,
  then the trunk's context (32–64k bytes on 8 GB) is built on the winner. A hybrid trunk is a future pretrain;
  no distillation of the running trunk.
* **Identity kept.** Bytes, H-Net chunking, Relation, the 4060 Ti as the primary card, the resident daemon.
* **Rungs.** 11M (`mote-11m`, 6 main layers) at 2048 bytes, 3 seeds, 1 GB of bytes per run; the top three arms at
  32M (`mote-32m`, 8 main layers) at 8192 bytes, 1 seed, 3 GB. The 96M shape is where the ratio axis is fully
  expressible (12 layers); below it the 3:1 and 5:1 arms both round to two Relation layers.
* **Matching.** Matched parameters and bytes; wall-clock reported. `scripts/ladder_arms.py` solves the main FFN
  width per arm by building the model and counting (e.g. 11M 3:1 at expand 2: d_ff 768 → 513, 10,868,394 vs the
  control's 10,870,110; 96M: 2048 → 1544). Positions scale by Relation count and depth fraction.
* **Where it runs.** Local, after the signed pipeline (latent-feedback arms → mid 2×3) and the chunk-rate A/B;
  ≈2.4 GPU-days staged. One ≈$2 pilot on the org account's free credits this week (below).
* **Defaults in every hybrid** (Round 5): Mamba-3 at main-layer 0 (the first recurrent layer is uniquely critical,
  2603.22473); ≥2 Relation layers (one attention-like layer per dependent hop — the tool-result read is two,
  2605.16640); an element-wise sigmoid output gate on Relation (`--relation-out-gate`, the one stability lever with
  data against the pre-attention activation spikes, 2608.12149); RMSNorm on the Mamba-3 scan output before its gate
  and out-projection (`--mamba-out-norm`, 2603.15569's pre-gate norm, "crucial for long-context extrapolation" in
  hybrids). Parallel arms share the p2/value-side projections with separate p1 and a short conv on the shared path
  (2605.28769 Table 4: shared K/V free, shared Q +0.17 ppl).

## Arms

Sequential (`--main-pattern`, 12-layer positions in brackets; `scripts/ladder_arms.py` prints the 6/8-layer forms):

| arm | Relation layers | note |
|---|---|---|
| R | all | control — today's main |
| M | none (2 R minimum waived: pure Mamba-3 main is a control) | is Relation load-bearing at all |
| A | all, plain attention (re-added; deleted 08-29) | the baseline in the table |
| 1:1 | [1,3,5,7,9,11] | dense end of the curve |
| 3:1 uniform | [3,7,11] | Kimi / Relation-paper interleave; the placement control |
| 3:1 evidence | [1,6,9] | early + mid + 75 % — the placement three lines converge on |
| 3:1 mid | [5,6,7] | adjacency |
| 3:1 late | [9,10,11] | the falsifier — every paper says it loses |
| 5:1 | [5,11] | the minimum (2 R) |
| A3+M9 | attention at [1,6,9] | "Relation composes with SSMs like attention does" needs this comparator |
| 3:1 expand 1 | [1,6,9], Mamba-3 expand 1 | lighter Mamba-3, wider FFN at the same count |
| deeper | 16 blocks, expand 1, matched | does the hybrid want depth |

Parallel per layer (a new block; built after the trunk week): Hymba normalized-mean fusion; Falcon-H1 channel
split; zero-init gated residual `y = mamba(x) + g(x)·relation(x)` (2608.02032's rule — the fused layer starts as
pure Mamba-3). New families (built after the trunk week): **Raven** slot-memory mixer with a NoPE-Relation partner
(`--no-rope`; 2607.25357 — the only hard extrapolation numbers, and it collapses beside RoPE); **SISA** score-bias
Relation (2606.02332; a cheap falsifier — it lost to Mamba-3 at 369M); **banded decay init** for Mamba-3 (Harmonic
2606.24650's one solid number; outer fast ≈ chunk size, main slow); **GDN[−1,1] partner** with a parity/S3
state-tracking probe (2606.12364; bpb will not separate linear partners — ≤0.05 nats within the family). Linear
Relation (the paper's own 3 Full + 9 Linear) needs a compute path decision: its retention is CHANNEL-wise (α ∈ (0,1]^{d_h}, KDA-style), and `mamba_ssm`'s SSD chunk-scan takes `A: (nheads)` — scalar-per-head decay only (verified 2026-08-30 against the checkout; the 08-30 note claiming SSD reuse was wrong) — so the paper-faithful form needs a GLA-class chunked scan (pure-torch like the dechunk EMA's, or the `fla` kernel), while a scalar-α variant rides the SSD kernel at the cost of the paper's retention class. No Mamba-2 layer exists in Mote either way.

## What decides

* **Decider:** a 2-hop tool-result probe (find the latest `<|result|>`, then read a field in it) — new, in the probe
  suite. Attention count moved retrieval and not PPL in every 2026 study (8 attention layers vs 1 bought 0.04 PPL
  at 340M), so ratio and placement are decided on the probe.
* **Guards, all must hold:** val bpb ≤ control + the control's 3-seed range at matched bytes; the 1-hop needle probe
  no worse; parity/S3 for the partner arms; the prefix-invariance audit (`tests/test_prefix_invariance.py`,
  positive control included) passes on every arm's config before the arm is timed.
* **Logged per arm:** per-layer output/residual norm ratio (2603.22473's r = 0.55 predictor of ablation damage),
  chunk-0 max |h| per layer (sink–spike alignment), bytes/chunk and boundary-on-separator fraction (already in the
  eval record).
* **Win rule:** > 2× the pooled 3-seed SD at 11M and the same sign at 32M. A 32M-only result is a lead, not a claim.

## The pilot (this week, ≈$2)

`scripts/cloud_arms.py submit ladder-pilot` behind session 1 and the lr arms (the waiter submits it): the 11M
control and the 3:1 evidence-placed hybrid (`MMMRRM`, d_ff 513, expand 2, gate + norm), one seed, 1 GB each, on
one L4. Reads: the per-arm cost, bpb at matched bytes, val_bpic, boundary-on-separator, and the eval record's
throughput — the first hybrid number, not a verdict.

## Lessons adopted with it (standing, `docs/shape.md`)

1. Every architecture claim is measured at two scales with three seeds at the smaller.
2. The chunk rate's controller is the **decision threshold**, never the projection: `project_boundaries` ranks the
   whole window, so when a bound binds a boundary can depend on later bytes (the audit demonstrates it). The
   chunk-rate A/B (after the signed pipeline): threshold arms calibrated to 5 and 6.5 B/chunk vs control at 32M,
   matched wall-clock, decode-identical by construction.
3. Built-but-unwon options expire 90 days after their build date, swept at each housekeeping, retroactively.
4. A number in a comment is a dated claim; housekeeping owns it.
5. The next trunk trains ≥1 epoch on A ∪ B ∪ C (28 GB, on disk); a fresh mix D is built only for a second trunk.

## Found on the way (2026-08-29)

* `--bound-floor` was an absolute count: never binding in training at 16384, every byte a boundary on served
  continuations — fixed as a rate (`655ac24`); launch-blocking, now a gap-protocol check.
* The dechunk EMA's block carry rounded through the block's own total decay (`s_incl − logd`), so a change after
  position t moved z̄_t by ~1e-5 — exact value causal, computed value not; now an exclusive cumsum. The trunk's
  forward differs from `58e8672` by that roundoff (measured in the gap protocol, expected ≤1e-6 relative).

## Literature (all read in full unless marked; 2026, after Mamba-3)

Placement/components: 2606.30562 FlashMorph (converted Qwen3: kept layers {0–1, 11–16, 21}/28, never the last
three; ratio 6:1 < 3:1 < 1:1); 2603.22473 (Qwen3.5-0.8B/Falcon-H1-0.5B ablation: recurrent path carries
likelihood — PPL 7.6 → 268k without GDN vs 625 without attention; attention layer 15/24 most sensitive; layer-0
GDN critical); 2603.20997 (one pairwise layer locates, more are needed to use; partial pairwise coverage is
all-or-nothing); 2608.12149 (spikes before every attention layer, ~10× larger at 3:1 than 12:1; single attention
layer at 4/24 useless for retrieval, 12 and 20 both work; output gating cuts magnitude). Fusion: 2606.02332 SISA;
2605.28769 Oryx (shared K/V, separate Q, conv+gate); 2608.02032 DART (zero-init gated residual; independent
readouts never learn). Memory: 2607.25357 Raven; 2608.12435 MARCH; 2608.17896 dynamic compression (synthetic;
read-twice beats it); 2606.24650 Harmonic (banded init; its 1B eval violates the entropy floor). Bytes: 2608.17325
(H-Net boundaries byte-denominated, whitespace-ish in Latin script ≈4–6 B/chunk, morpheme-blind; no evidence on
forcing a rate); 2608.14691 (complex substrate = Mamba-3's rotation vs a hobbled Mamba-1; not an arm); 2608.03599
(boundary-divergence 1−F1 as a free metric). Linear partners/theory: 2606.12364 (Mamba-2 ≈ GDN; GDN[−1,1] fixes
tracking, costs code); 2607.07953 (KDA > DeltaNet ≳ GDN > GDN-2 by ≤0.05 nats; hybrid −0.017…−0.076 vs pure);
2607.06155 (unbounded re-readable tools lift the fixed-state ceiling; hops unchanged); 2605.16640 (≥1 attention
layer per dependent hop; scratchpad does not rescue a fixed state); 2608.22876 (the audit; two gates). Swept by
title/abstract only: 211 further papers, of which Dion3 2608.11612, CODA 2605.19269 and the kernel verifier
2608.12700 go to the throughput line, Attention Amnesia 2606.11052 to post-training, asymmetric paging
2605.22416 to serving.

## Amendment 2026-08-30 (grilled and signed): Linear Relation, the second partner family

Grilled in three rounds on 2026-08-30 out of the source paper's own hybrid (2608.20172 §5.2/A.4, read in
full today); Mamba-3 stays the primary partner. Nothing above moves.

1. **Operator: paper-faithful Linear Relation.** Channel-wise KDA-style retention α ∈ (0,1]^{d_h}
   (`α = exp(−exp(A_log) ⊙ softplus(W↑W↓x̄ + dt_bias))`), L2-normalized RoPE'd relation coordinates,
   the Self relation S_t = σ(p₁ᵀp₂ / τ_S√d_h) gating both the read `E_t = S_t (C_t⁻)ᵀ p̂₁` and the write
   `C_t = C_t⁻ + S_t p̂₂ Ĩᵀ`, strictly-historical read, `Y = Ĩ + E` with no out-norm and no gate. The
   pre-gate-norm lever is the NAMED FALLBACK if an LR arm spikes — applied and recorded, never silent.
2. **Arms: paired minimal.** `LLLRRL` (the evidence twin of `MMMRRM`) and `LLRLLR` (the uniform twin of
   `MMRMMR`), 3 seeds each at 11M / 2048 / 1 GB (~11 h of local GPU inside the ladder's signed slot);
   d_ff solved by building and counting, as every arm. Winners compete for the 32M top-three on equal
   footing with the Mamba-3 arms. No other LR variants (no NoPE twin, no ratio sweep) in this round.
3. **Layer 0 is L.** The signed layer-0 rule's rationale (2608.12149: the first layer wants the linear
   operator) generalizes to "a hybrid starts with its linear partner"; `main_pattern` gains the letter
   `L` under the same invariants (≥ 2 R, layer 0 non-R).
4. **Pre-registered falsifier:** Linear Relation's α ∈ (0,1] retention class (no rotational/negative
   eigenvalues) FAILS the parity/S3 state-tracking probe where Mamba-3's rotational state passes. If LR
   passes, the expressivity-class story is wrong and is retired with numbers. Decider and guards are
   otherwise the ladder's own (2-hop tool-result probe decides; bpb inside the control's 3-seed range;
   wall-clock reported; the prefix-invariance audit before any arm is timed).
5. **Compute path:** `flash-linear-attention`'s GLA chunked scan, version-pinned, optional import in the
   house pattern; the pure-torch chunked scan (the dechunk-EMA pattern) is written FIRST as the
   correctness oracle and the CPU/fallback path. S_t folds into q and k; a diagonal-inclusive kernel gets
   the current-token term subtracted exactly. The SSD chunk scan cannot run this operator (`A: (nheads)`
   — scalar-per-head decay; verified 2026-08-30). If fla is incompatible with torch 2.13 / triton 3.7.1,
   the arms run on the torch scan and the kernel question comes back with numbers.
6. **Timing:** the layer, scan, decode path and tests are CPU-side work in the trunk week; the arms run in
   the ladder's slot after the signed pipeline. The 3:1 ratio/placement questions stay with the existing
   arms (3:1 evidence-placed is the prior; the ladder decides).
