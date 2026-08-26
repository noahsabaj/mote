# 2026-08-26 — the mid-training stage, against the 2026 literature

Sweep of 2026 mid-training work, prompted by the same question that turned up the post-training problems
the day before: *how much of this stage is carrying assumptions nobody has re-checked?* Enough that the
gate could not fail.

The design as it stood was signed 2026-08-24: a trunk snapshot, two cooldown branches (control = mix B,
anneal = mix C plus the sim/chat/identity extras), `lr 8e-4 -> 0.1x` as `1-sqrt(t)` over the whole branch,
and a verdict that shipped the anneal on **≥ 2 of {reading EM, sim-QA EM, chat val bpb}** with a val-bpb
guard.

## What the sweep found

| paper | what it says | what it does to Mote |
|---|---|---|
| [2603.16127] | Warmup-Stable-**Only** beats every decay schedule *after SFT*, at 1B and 8B. "Introducing decay at any stage reduces SFT performance." Decay wins on the pretrain metric and loses on the shipped one; the mechanism is that decay lands in sharper minima. | The trunk is already WSO, which this vindicates. The branch's decay is now an axis of the experiment rather than an assumption. |
| [2607.09885] Index-1.9B §6.4–6.5 | At 0.1B / 1T tokens, cosine, linear and WSD converge to the same loss *and* the same benchmark scores — the schedule alone is worth nothing. What pays is decay **combined with** a data-quality raise, and only under WSD: **cosine + curated data scored below plain cosine**, because "the cosine tail leaves too little learning rate" to adapt to the shift. | The direct hit. `1-sqrt(t)` is concave: it was at **55 % of peak by the first quarter** of the branch, which is when mix C's shift arrives. Retired for constant-80 %-then-decay. |
| [2605.25698] | Names the conflict formally — under a decay the model meets its best data exactly when its learning intensity is weakest — and measures **+3.27** on a 600M dense model for fixing it (Drop-Stable-Rampup). | Same conclusion, from theory. Mote is already at minimum batch (1 x accum 4), so the batch-drop half is unavailable; the schedule half is what changed. |
| [2607.12360] | Cooldown helps *exactly* when the optimizer's step does not shrink with the gradient. SGD self-anneals; sign/normalised methods cannot. Ships an online diagnostic: regress per-microbatch gradient variance on ‖ĝ‖². | Cuts the other way. Muon is in the normalised class (orthogonalised updates keep unit scale; the paper analyses signSGD/normalised-SGD/Adam, not Muon by name), so for *loss* it wants the decay. Both can be true: decay lowers loss by entering a sharper minimum, and the sharpness is what costs the SFT. |
| [2602.03702] | 150M/300M to 32x Chinchilla: **constant LR + EMA ≈ well-tuned cosine**, horizon-free. Their EMA half-life is time-scaled, `τ_t = 2^(−f/t)` with f = 25 → 4 % of elapsed steps, best in every experiment. | Mote's fixed `--eval-ema 0.9999` gives a 3.6 % half-life at 190k steps — right for a 12-hour run and wrong everywhere else. Measured below. |
| [2601.09000] | The sharpness increase during cooldown is architecture-universal (a 160M LM and a 334K CNN show the same river-valley geometry and the same PCA structure). | No exemption for the Mamba/H-Net stack. |
| [2603.17074] PRISM | Already cited in shape.md. What was not: short-context mid-training **destroys long context** (RULER@128k 59.09 → **6.46**), and a 15 % base + 85 % mid linear merge recovers most of it. Token budget saturates at 15–27B for a 3B model. | `needle` was measured, rendered, and then ignored by the verdict — while the reweighting cut the long-document share. |
| [2605.18607] | Cross-entropy ranks candidates at Spearman **0.36**; token-level proxies over expert trajectories reach **0.81**. On DataDecide (25 corpora — the same problem the branch gate has) frequency-weighted top-5 accuracy hits >0.85 decision accuracy at **1e-5 of target compute**. "A model which cannot solve a problem can still track the CoT written by an expert." | The gate's new decider. |
| [2605.02087] MSM, [2607.26654] | Alignment fine-tuning on demonstrations generalises shallowly. Training on synthetic **documents about the model** first shapes how the later demonstrations generalise (agentic misalignment 54 % → 7 %). Content presence matters more than structure; no capability cost; survives benign fine-tuning. | Identity moves from SFT Q&A into the mid mix as documents. |
| [2608.20314] MidTool, [2607.12463] | Tool use "benefits from dedicated mid-training rather than being left entirely to post-training". The agent's action → observation → continuation loop is isomorphic to a function call site; fill-in-the-middle over it gives +2.8–5.4 SWE-Bench, and the bias *survives* post-training while agentic post-training alone erodes non-agent ability. | The tool protocol moves to mid, with FIM. |
| [2608.05141] OctoLong | Existing long corpora are long but locally coherent, and therefore scarce in long-distance dependencies. Swapping ~12 % of a context-extension mixture for dependency-dense material moved long-range retrieval and state tracking. | The reason the long-document share goes back up and `sim_long` exists. |

## The finding that was not in a paper

**All three of the gate's deciders favoured the anneal by construction, and it needed only two.**

`branch_gate.py` shipped the anneal on ≥ 2 of `reading_em`, `sim_em`, `chat_val_bpb`. The anneal branch
was mix C **plus** `sim 4 % / chat 3 % / identity 0.2 %`; the control was mix B with none of them.

* `chat_val_bpb` — the anneal trained on chat bytes, the control did not.
* `sim_em` — held out by *seed*, but the anneal trained on 4 % sim: same generator, same domains, same
  format. The control had never seen the format at all.
* `reading_em` — mix C raises SYNTH Q&A 8.0 → 10.8 % and fact-seeking, the closest thing in the mix to
  the probe's shape.

The stated question was "does quality annealing produce a better base". The question actually asked was
"does training on X help X". Two hypotheses — reweight the web mixture, and add the local extras — were
welded together and only one of them was named.

## Measurements taken here

**The EMA is already doing the cooldown's job, and nothing tested that.** Both branches were cooldowns,
so no arm ever asked whether the decay was worth 2.8 GPU-days.

```
run                  steps      raw     ema     gain   halflife/steps
t3l_dense_4e-4      189,958   1.1029  1.0276  0.0753       0.04
t3l_dense_8e-4      186,144   1.1265  1.0370  0.0894       0.04
t3l_dense_16e-4     162,000   1.1776  1.0798  0.0978       0.04
smoke_moe_lf            400   2.7465  2.9517 -0.2053      17.33
```

The trunk EMA is worth 0.075–0.098 bpb at zero extra compute, and `train.py:394` already calls it "the
decayed-quality stand-in for constant-LR runs". At a fixed β = 0.9999 the half-life is 3.6 % of elapsed
steps at 190k — within rounding of [2602.03702]'s optimal 4 % — and 17x the entire run at 400 steps,
where the EMA is a **0.205 bpb penalty**. Horizon-free `τ_t = 2^(−25/t)` is the fix; deferred until the
current queue drains so the three in-flight T3 arms stay comparable.

**The long-document share.** FLAGSHIP 10.0 % → ANNEAL 8.6 %, a 14 % relative cut in the same direction as
PRISM's collapse. Raised to 10.5 % (`finewiki_long` 3→4, `gutenberg` 2→3) and asserted in
`test_pipeline_stages.py`.

**The long-range dependency source.** `mote.sim` already had the property OctoLong builds a language
server to get — an answer settled by an event thousands of bytes earlier — but only at the tick counts it
was generating. Measured over 400 traces:

| | doc p50 | dependency p50 | > 1 KB | > 4 KB |
|---|---|---|---|---|
| ordinary sim (ticks 4–18) | 461 B | 139 B | **0 %** | 0 % |
| long sim (ticks 60–220) | 2,803 B | 273 B | 22.8 % | 2.3 % |
| long sim, `--min-bytes 4000` | 6,031 B | 622 B | **41.0 %** | **24.9 %** |

The ordinary generator produces *no* dependency past 1 KB. Without this there is nothing in Mote's own
data that teaches a 16384-byte window anything it could not learn at 512.

**The decider's weighting, which the first attempt got wrong.** Entropy weighting looked like the natural
byte-level port of [2605.18607] — most byte positions are trivially predictable, so uniform weighting
spends its mass where no checkpoint can fail. It does not work. Scored against three checkpoints whose
quality order is known from 12-hour runs (val_bpb_ema 1.0276 / 1.0370 / 1.0800), at n = 120 held-out sim
trajectories:

```
metric                      4e-4      8e-4     16e-4  order  spread/sem
recip_rank_uniform        0.4655    0.4476    0.4180   YES      2.3x   <- the decider
top2                      0.4528    0.4226    0.3890   YES      2.6x
recip_rank (inv-freq)     0.4626    0.4454    0.4163   YES      2.2x
agree_freq                0.4231    0.3735    0.3673   YES      2.1x
agree_uniform             0.3436    0.3277    0.3236   YES      0.9x
agree (inv-freq top-1)    0.3407    0.3259    0.3219   YES      0.8x
recip_rank_entropy        0.3492    0.3873    0.3308   no       1.2x
agree_entropy             0.2235    0.2581    0.2271   no       0.2x
ce                        2.9538    3.1715    3.6565   YES      5.9x
```

Every entropy-weighted cell fails; every model-independent one succeeds. Entropy is the *candidate's own*
uncertainty — a worse checkpoint is more uncertain everywhere, so weighting by it re-normalises away
exactly the difference being measured. That is why the paper's winning cell is **frequency**-weighted: a
property of the expert's text rather than of the candidate.

The decider is `recip_rank_uniform`, not the marginally better `top2` (2.6x vs 2.3x, a difference well
inside the noise): a threshold metric has a knob tuned to this vocabulary, and reciprocal rank does not
saturate at any vocabulary size. It is also *unweighted*, which is the opposite of the first two attempts
— rank is already graded at every position, so the weighting that compensates for a binary top-5 hit at
100k ids has nothing left to do at 269.

`ce` separating these three best of all is not a contradiction. 2605.18607's 0.36 is for ranking across
model families; these three differ only in learning rate. The gate's comparison is the harder one — two
branches trained on *different mixtures*, whose val distributions differ so val bpb stops being comparable
between them — which is why `ce` stays in the table and does not decide.

**The verdict now has to clear its own noise.** At n=120 the best-to-worst gap above is 2.3 standard
errors, so a branch difference inside one combined sem has decided nothing; `verdict()` requires
`|delta| > sqrt(sem_c^2 + sem_a^2)` and prints the figure. Without it a coin flip could ship a 1.4-GPU-day
branch and the table would not show that it had. The gate runs 120 sim + 100 reading trajectories, so the
sem there is roughly 0.015 and a real difference needs about 0.021.

## What changed

* **Schedule** — `cooldown` (`1-sqrt(t)` to 0.1x over 100 %) retired. `branch` is constant to 80 % then
  linear to `--min-lr-ratio` (default **0**); `constant` is the same run without the decay.
  `--snapshot-at 0.8` marks the fork.
* **The experiment** — a 2x2 (mixture x decay) from two branches plus two 20 % forks, token-matched, with
  the extras identical in every arm. ~3.6 GPU-days against 2.88 for the old 1x2.
* **The gate** — one decider (`proxy_agree`) and three guards (val bpb ≤ control + 0.005 within decay
  condition, `needle_auto` and `false_fire_rate` no-regression). Exact match stays in the table and out of
  the verdict. Missing numbers fail closed.
* **Identity** — `mote.serve.identity.SPEC_SECTIONS` states each choice with its reason;
  `mote.data.build_spec_docs` turns it into documents *about* Mote for the mid mix. The SFT Q&A share and
  the negative class both stay, as elicitation and as a shipping guard.
* **Tools** — `data/sim_traces` enters the mid mix under `:fim`, permuted around its own
  `<|call|>`/`<|result|>` boundaries. Three sentinel ids at 266–268 (VOCAB_SIZE 266 → 269, still inside
  the 272-row padding, so no existing checkpoint needs surgery).
* **Long context** — ANNEAL long docs 8.6 % → 10.5 %, plus `mote.sim.long` in the extras.
* **Extras budget** — 15 % of each branch: spec docs 3, tool traces 3, sim long 4, chat 5.

## Open, and flagged

**The extras budget is more aggressive than the papers it is calibrated on.** 3 % of an 8 GB branch is
240 MB of identity documents. The generator produces 400k byte-distinct documents (332 MB), but the
*content* behind them is 7 spec sections plus ~31 claims and ~28 illustrations — roughly 12 KB of distinct
prose, so 3 % is a ~20,000x repetition of the information. [2607.26654] expanded Anthropic's Constitution
by a comparable ratio but gave it **0.33 %** of the mixture, not 3 %, and [2605.02364] (InfoLaw) is
specifically about upweighted high-quality data turning into repetition and costing performance. The
share is built as signed and is one number in `scripts/mid_2x2.sh`; 1 % would be closer to the evidence,
and the freed 2 % would go back to the web half.

**Two bugs this build caught, both of the same shape.** `--min-lr-ratio` was first given a global default
of 0, which silently changed every `wsd` lab arm from 0.1x — including three already queued, whose
controls all ran at 0.1x. It defaults per-schedule now. And the three FIM sentinel ids at 266-268 are
inside the padded embedding but *outside* the head's mask on a new-config model, so a checkpoint trained
on them could have emitted the literal text `<|fim_middle|>` to a user; they are in `STOP_IDS` now. Both
are the same failure: a vocabulary or schedule constant that several stages read, changed in one place.

**PRISM's merge is not implemented.** 15 % base + 85 % mid recovered most of the long-context loss at 8B.
Mote's answer to the same risk is preventative (long docs restored, `needle_auto` as a guard) rather than
corrective. If the guard trips, the merge is the next thing to try.

**The decay diagnostic is not implemented.** [2607.12360]'s ρ = σ₀²/(σ₁²‖∇f‖²) is estimable online by
regressing per-microbatch gradient variance on ‖ĝ‖², and Mote already computes 4 micro-batch gradients per
step, so it is nearly free. It would predict the 2x2's decay answer instead of measuring it — worth doing
if the 2x2 comes back ambiguous.
