# 2026-08-24 evening — gates, probes and two regressions found by them

Everything below ran on the 4060 Ti (daemon `mote.service`) or on Lightning jobs from the `vitruvian` studio env,
between 19:10 and 20:10 EDT. Commits: `da14e1e` (MoE), `d0f3f08`, `e1f55c1` (kernel constexpr), `3317c4a`
(serving pool scope), `4ca6677` (graph-capture telemetry guard).

## FlashRelation v2 (docs/shape.md § "Kernel and compile workstreams")

| gate | result |
|---|---|
| exactness (tests/test_flash_relation.py + test_relation_paths.py on the GPU) | **38/38 pass** — locally and on an L4 |
| microbench v2 / v1, fwd+bwd, five shapes | **1.53–1.97×** (fwd 0.87–2.59×; deterministic two-pass fallback 0.81–1.20×) |
| step level at the 30-min flagship-preset shape (4096, accum 8, `--ckpt-main`) | nsweep_8 on v2 **49.4 KB/s** vs lr_sweep_8e-4 on v1 **49.1 KB/s** = 1.006× — the kernel is a small share of the step at 4096 |
| step level at the frozen 16384 recipe | pending: `t2_ctl_30m_v2` (queued tonight) vs the freeze's 68.1 KB/s; the ≥10 % hot-swap rule is decided there |

**Bug found by the L4 probe** (`e1f55c1`): the kernels read module-level Python floats (`LOG2E`, `LN2`,
`RESCALE_THRESH`); Triton 3.7 rejects that at compile time ("Cannot access global variable … from within
@jit'ed function") while the CPU interpreter (`TRITON_INTERPRET=1`, our whole test loop while the GPU was busy)
accepts it — v2 had never compiled on a GPU before 19:19. The daemon restarted on v2 at 19:10 crash-looped on
it until the fix (three restarts, one interrupted arm auto-re-queued). Lesson recorded: an interpreter-green
kernel is not a compiled kernel; the GPU test run is part of the gate, not optional.

## Serving beside training (docs/shape.md § "Daemon")

**Regression found by the queue** (`3317c4a`): the serving `MemPool` (28d33ae) wrapped *every* serving
allocation, including prefill/rewarm transients; a MemPool keeps its high-water mark cached for itself, so an
evening of EMA syncs left 2.9–5.4 GB of idle reservation in the daemon. Five queued arms (nsweep_10, nsweep_4,
ab3_jepa_sig, compile twins) OOM'd at their first forward at 19:46–19:48, and the v2 step gate's own runs
OOM'd at 19:51. Fix: the pool holds only the arena (construction and growth) and the decode graphs; transients
go back to the shared allocator. Daemon restarted 19:53 at ~0.5 GB; the five arms re-queued behind nsweep_10.

**Second bug** (`4ca6677`), found by the serving-gate bench on the flagship preset: `Mamba3Mixer.step` copies
telemetry to the host (`.tolist()`) unless the graph path has installed `telemetry_dev`, which `warmup()`'s
first capture does not — "Cannot copy between CPU and CUDA tensors during CUDA graph capture". The 35M never
hit it because its multi-byte head disables graph decode. The host copy is now skipped while capturing. The
gate bench (`bench_serve_beside_training.py`: gated vs stream, idle vs under a training slice, peak memory) is
re-armed for the next idle queue.

## fp8 `_scaled_mm` and RMSNorm on the 4060 Ti (`bench_fp8_norm.py`)

fp8 GEMM alone 1.8–1.9× bf16 (74–78 vs 38–40 TFLOPS) but **0.20–0.73× once the casts are counted** at five of
six shapes (only 768→4096 wins, 1.40×) — fp8 needs fused casts (compile) and stays parked for the 1B.

| shape | mamba_ssm fused add+norm fwd / fwd+bwd | `F.rms_norm` + add | torch compiled |
|---|---|---|---|
| 16384×512 (byte level) | **425 / 1600 µs** | 1091 / 3180 | 475 / 1633 |
| 2730×768 (main) | 70 / 360 | **54 / 311** | 63 / 324 |
| 65536×512 | **1727 / 6496** | 4524 / 13175 | 1645 / 7464 |

Verdict: the fused kernel stays for the byte-level norms; `F.rms_norm` is 1.16× on the main-network norms ≈
1.4 % of the step, below the 10 % swap rule — not adopted. Inductor matches the fused kernel at 16384×512
(evidence for the compile workstream that the norm needs no custom kernel there).

## MoE T1 (docs/research/moe-2026-08-24.md)

`smoke_moe_lf` / `smoke_moe_aux` (smoke preset, 4 experts top-2, 400 steps): both routers train on the GPU with
the `grouped_mm` bf16 path; loads 0.24–0.26 per expert, MaxVio 0.04–0.05 (lossfree) / 0.07–0.09 (aux), val bpb
2.746 / 2.828 vs 2.840 for the dense v2 smoke run of the gate script (not a verdict — 400 steps); 403–448 KB/s;
`val_bpb_ema` logged.

## Lightning probes (signed $1.20; spent $0.45)

| machine | 35M lab config | flagship preset | notes |
|---|---|---|---|
| **L4** (GCP on-demand $0.48) | **74.7 KB/s**, 2.0 GB | 41.5 median / 79 max KB/s, 7.3 GB | kernel tests 38/38; 0.75× the 4060 Ti (91–105 / 68 KB/s) |
| RTX PRO 6000 Blackwell (AWS interruptible $2.11) | — | — | "no kernel image is available": torch cu126 has no sm_120 — needs a cu128+ env |
| A100-40 (Lambda) | — | — | job submission refused (`ApiException`) |

Consequence for the five 12-h verdict arms: on L4s, 12 h ≈ 9 local hours (D/N ≈ 15, predicted MoE signal
≈ −0.010 bpb, above the ±0.005 gate) for ≈ $29; D/N ≈ 20 needs 16 h ≈ $38. Noah decides.

## Other findings

- Dynamo cannot trace the Triton kernels inside the custom ops (`triton_kernel_wrap`: "`_semantic` argument
  must be provided outside of JIT functions" / CompilationError at 136:13 and 175:13) — the compile
  workstream's first item; the eager path is unaffected.
- nsweep_8's throughput probe (first 40 s) overlapped a 40-s GPU test run; its wall-clock schedule is
  unaffected (WSD follows wall-clock), its `total_steps` estimate is not meaningful. nsweep_8 val 1.706 vs the
  control's 1.693 (chunk target 8 → 10.4 vs 5 → 6.5); nsweep_4/10 re-run tonight.
- The 7-day trunk is ~4 epochs of the 10 GB mix; the day-10 extension would be 6 (2305.16264: fine at 4).
