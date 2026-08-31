# The CoT line — pre-registered (grilled and signed 2026-08-31)

Three grilling rounds out of the four-agent sweep (`docs/research/cot-2026-08-31.md`; ids there). Noah's
frame: no reasoning-effort knob — the system sizes its own thinking (adaptive reasoning), the same law as
reply sizing (shape.md standing rule, 2026-08-31), applied to compute.

## Signed

1. **Scope: the full line, staged.** Explicit interleaved CoT (traces → SFT-1 → RLVR-1) + an explicit
   compute allocator + — last, evidence-gated — a learned latent/explicit switch (TARPO-style, 2606.05859).
   No user knob; the API keeps an optional effort override; the Studio renders think spans **collapsed**
   (tap to expand); latent k appears in telemetry only.
2. **Traces, two sources.** (a) Programmatic: `mote/sim/tasks.py` expert plans rendered as interleaved
   spans — think → `<|call|>` → `<|result|>` → sub-answer → think … — steps verifier-gated, byte-exact,
   no spend. (b) Claude-distilled chat-domain traces, admitted only on **grader pass + format** (rubric:
   steps relevant, no hedging loops, interleaved, sized to the question — reply sizing applies INSIDE
   thinking). The copy-shortcut confound (2605.22870) is why nothing unverified and ungraded enters SFT.
3. **Entry: SFT-1's existing hook grows.** Think-traces join SFT-1's mix (~10 % initial share, tuned at
   build); the queued self-proposal arm (shape.md § post) becomes its A/B. Then RLVR-1's action space
   gains the think-tags with the interleaved recipe: **conditional intermediate reward** (gated on final
   correctness + format + rising accuracy — ungated IR hurts, 2505.19640), sim intermediate steps verified,
   chat structure-only; **length cost on successful episodes only** (a continuous penalty on failed
   rollouts structurally collapses — ACOER 2606.22716) plus rollout editing (2606.17890); QLPO
   distributional resampling (2607.21793) is the named fallback if reward-side cost is unstable at 96M.
   **Hard order: SFT-CoT before any reasoning RL** — the 135M RLVR failure (2606.22189) is the reason.
   Turn-level credit from group statistics (HiDiffTIR 2608.21863 / FACTOR 2608.07118) when the tool loop
   enters. Task mix must span difficulty (the Countdown collapse, 2608.20256) — checkable up front from
   the sim's plan-length labels.
4. **Allocator: computed, not emergent** (2608.22347; gate failures 2607.20519). Engine-side:
   `k* = argmax_k stake·p̂(success|k) − λk` over per-k success curves calibrated on sim difficulty labels
   (enumerating k per task and checking the verifier is free — SLPO's supervision, 2607.19691, works at
   124M); entropy/margin readouts are the cheap baseline; controls both k (0..3) and think-on/off; stake
   constant 1 until a reason exists. **Never gated on self-agreement** (agreement is conviction, not
   correctness). A trained first-byte mode token (2608.20256's collapse kit) follows ONLY if the code
   allocator shows headroom a learned policy could close. Before serving any k above trained: the settling
   check (per-pass hidden-state displacement, 2608.18222).
5. **T²MLR arm.** The latent-feedback experiment gains a fourth arm — the previous position's MIDDLE-layer
   state fused into an EARLIER layer (2607.15178's winning placement) beside the signed top→input arms —
   +24 h on the trunk snapshot; mid slips one day. (Amendment in
   `docs/results/2026-08-28-latent-feedback-prereg.md`.)
6. **Timing.** Trace generators (sim think-rendering; the distillation + grading harness) and the
   allocator scaffolding are CPU-side trunk-week work beside the Linear Relation build; the arms and
   stages run in their signed slots; nothing displaces the queue or the launch.

## Also standing

Monitorability is a cost of compression (2607.09786) — the probes read these traces, so length pressure
is tuned against hint retention, not bytes alone. A too-shallow think segment can score below no-thinking
(DeepTool 2605.29568): budgets are tuned, never assumed. First-of-kind: nothing interleaved below 0.6B,
nothing byte-level at any scale — Mote is the first data point either way.
