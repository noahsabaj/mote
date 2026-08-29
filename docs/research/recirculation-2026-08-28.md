# Reading 2026-08-28 — 2608.17981, Recirculation

Moved verbatim from docs/shape.md on 2026-08-29 (housekeeping). The measurement it led to is docs/results/2026-08-28-recirc-sweep.md.

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
