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

## H100 tile session (20:06–20:13, interruptible, ≈ $0.45; the consented Hopper session)

Tests 38/38 on the H100. Tile sweep at the flagship shape (B, 8 heads, 2730 chunks, d 96):

| kernel | Hopper fallback (64,64,4,2) | best | gain |
|---|---|---|---|
| fwd, B=1 | 0.143 ms | **(64,128,4,2) 0.132 ms** | −8 % |
| fwd, B=4 | 0.434 ms | (64,128,4,2) 0.378 ms | −13 % |
| bwd, B=1 | 0.358 ms | **(128,64,8,2) 0.352 ms** | −2 % (128-wide N tiles run out of shared memory) |

`_TILES` now carries these Hopper entries. v2 vs v1 on the H100 with the fallback tiles: fwd+bwd 1.17–1.60× but
**fwd 0.45–0.81×** at four of five shapes — a structural effect (the 64+32 head split costs Hopper's wide tensor
cores two dots instead of one padded 128), not a tile effect; irrelevant for the local card, where v2's fwd is
1.4–2.6× v1, and a note for any future Hopper training (v1-style padding or a Gluon/wgmma port there).

**The per-kernel table the probe lost** (profile b4, 16384, `--trace`; 579 KB/s, MFU 11.8 %, 80 % busy, 5322
launches/step): elementwise/copy **31.2 %**, Mamba-3 21.1 % (`mamba3_siso_fwd` 9.3 + `_bwd_dqkv` 8.7),
"other" 18.7 %, GEMM 14.0 %, layernorm 12.3 % (`_layer_norm_bwd_kernel` alone 10.2 %, 90 calls),
FlashRelation v2 12.7 % (`_bwd_kernel` 8.1, `_fwd_kernel` 4.6), `CatArrayBatchedCopy` 4.2 % (122 calls),
`angle_dt_bwd` 2.5 %. The fusion/launch bucket (elementwise + copies + cat ≈ 35 %) is the compile
workstream's target; the norm backward is the one custom kernel with real headroom (8× off its bandwidth floor).

## 20:20–21:30 — the desktop's share of the card, and what the queue does about it now

`nsweep_4` died at its Muon step at 20:2x: "Tried to allocate 72 MiB … 115 MiB free"; the daemon held 5.41 GB
(4.42 allocated + 0.44 in the serving pools + 0.26 reserved-unallocated + ~0.3 CUDA context). The rest of the
8188 MiB card was the desktop: plasmashell 276 MB, plasma-keyboard 261, kwin 195, Discord's GPU process 300, and
the lock screen (`kscreenlocker_greet`) 431 MB while the screen is locked — 1.46 GB plus ~0.4 GB of driver
overhead. **The daemon gets ≈ 6.2 GB while the screen is locked, ≈ 6.6 GB unlocked**, not the 7.2 GB the
serving gate assumed. Every flagship-preset arm queued behind it then OOM'd within a minute (20:37–20:40):
`compile_twin_eager`, `t2_ctl_30m_v2`, `t2_tf32_30m` and the six MoE T2 arms; `compile_twin_compile` failed on
the known Dynamo assertion instead (the whole-model `--compile` flag is dead until the custom-op work). The
35M arms (4.9 GB) fit; the flagship recipe at 16384 needs ≈ 5.2–5.5 GB in the daemon (4.31 standalone + the
serving residue + context) — a ~1 GB margin at 6.2; the E4 MoE flagship (+0.68 GB) is marginal and the E8
(+2.0 GB, 7.4 GB) does not fit on this card beside the desktop at all: its flagship-preset numbers need an L4 or
a lighter desktop. Discord's hardware acceleration is the one avoidable 300 MB; the lock screen's 431 MB is
transient.

Muon's batched Newton–Schulz (da14e1e) also raised the flagship step's peak by ~0.4 GB: the 12 × [768, 4096]
fp32 SwiGLU group is concatenated, converted and iterated as one 151 MB tensor with group-sized temporaries.
Capped at 32 MiB per stacked tensor (the 48 Relation projections stay batched in 4 launches).

An operator error on top: `mote train stop --id X` put `--id` into the CLI's REMAINDER and cancelled the
*running* arm (`ab3_jepa_sig`, 10 min in, checkpoint kept) — five times. Options after the action are parsed now;
`mote train --id X stop` always worked. `nsweep_10` finished before all this: 1.691 vs the control's 1.693
(−0.002, inside noise → no re-target of the frozen 5→6.5); `nsweep_8` 1.706; `nsweep_4` still owed.

What the daemon does now (d9692a0 and the follow-up commit): a CUDA OOM leaves the record `failed` and queues
a `--resume` copy **in front** with a 2/10/30-min delay that only starts once free + cached GPU memory covers
the failed run's tracked peak + 384 MiB — a structurally too-big job waits visibly while the queue flows around
it (three retries per lineage); `mote train start --front` jumps the queue; a resumed run continues its wall
clock (`elapsed_sec` in the checkpoint; older checkpoints read the log's last `elapsed_min`) so `--max-minutes`
and the WSD progress no longer restart; a fresh start into a used directory rotates the old log aside and
`lr_horizon.read_run` reads only the last fresh start. The restart at 20:52 that made this live exposed the last
gap: systemd's SIGTERM ended the process in under a second — no shutdown hook, so the running T3 arm
(13 min in, no checkpoint yet) restarted from step 0. `app.main` now calls `jobs.shutdown()` + `join` after
uvicorn returns (the job checkpoints at its next step boundary, skips the final eval, re-enqueues in front),
the unit carries `TimeoutStopSec=180`, and the second restart waits for the arm's next checkpoint.

Queue after the fixes: `t3l_dense_4e-4` (running since 20:52) → `ab3_jepa_sig` (resume) → `nsweep_4` → the
other four T3 arms → the nine flagship arms re-queued (they retry-wait for memory if the desktop is still heavy).

## 22:00–23:00 — grilled and signed: the desktop, the E8 arms, released serving, the think ids

Noah's calls (four rounds): Discord's hardware acceleration off and Plasma's virtual keyboard off — done from
the shell at his request (Discord killed, `enableHardwareAcceleration: false` written to its settings for the next
launch; kwin respawns `plasma-keyboard`, but the fresh process idles at 2 MiB — the 261 MB was its rendered scene;
the proper toggle is System Settings → Keyboard → Virtual Keyboard → None). **Desktop share now ≈ 0.56 GB, daemon
ceiling ≈ 7.3 GB.** The seven flagship-preset T2 arms (six MoE + a dense control on the same machine) run as four
on-demand L4 jobs (`mote-t2-{0..3}-08242225`, ≈ $1.9, account A; `peak_gb` logged per record since `b5cfe21`).

**Released serving (root, Noah's pick over the renumbered gate).** While a training job runs the engine keeps only
its weights: `Engine.release()` on `JobQueue.on_started` drops the arena, the captured graphs and the MemPool; a
reply allocates a per-reply arena from the shared allocator, the prefix store's CPU pages rehydrate it, decoding
takes the eager path, the memory goes back after the reply; `Engine.rearm()` on `on_idle` (nothing runnable
queued) restores the resident arena + graphs with one warm-up. A checkpoint loaded at a job's end starts released
when more work is queued; the server boots released when a job is about to start. Gate restated in docs/shape.md
(during a job: first byte ≤ 1 s + prefill, per-byte ≤ 4× idle; idle unchanged). The bench
(`bench_serve_beside_training.py`) still needs an idle window to put numbers on the released mode.

**Think ids reserved.** `<|think|>` = 264, `<|end_think|>` = 265, `VOCAB_SIZE` 266, `pad_vocab_to` 272 (six spare
rows); `HNetForCausalLM.head_logits` masks rows ≥ 266 to −inf on every path (forward, prefill, continuation, step,
the multi-byte head, the decode graph) so a padding row is never produced; old 264-row checkpoints load through
their own saved config. The rows moved the tiny test models' random init: one RL test now ends rollouts on EOS only.

Also built: `mote/eval/rl_taxonomy.py` (Table 14 of 2607.16097 over each held-out state's legal actions; tested on
tiny checkpoints), `branch_gate --k 8` by default (pass@8 beside EM), the RLVR-1 budget rule and the self-proposal
SFT arm in docs/shape.md.

## Other findings

- Dynamo cannot trace the Triton kernels inside the custom ops (`triton_kernel_wrap`: "`_semantic` argument
  must be provided outside of JIT functions" / CompilationError at 136:13 and 175:13) — the compile
  workstream's first item; the eager path is unaffected.
- nsweep_8's throughput probe (first 40 s) overlapped a 40-s GPU test run; its wall-clock schedule is
  unaffected (WSD follows wall-clock), its `total_steps` estimate is not meaningful. nsweep_8 val 1.706 vs the
  control's 1.693 (chunk target 8 → 10.4 vs 5 → 6.5); nsweep_4/10 re-run tonight.
- The 7-day trunk is ~4 epochs of the 10 GB mix; the day-10 extension would be 6 (2305.16264: fine at 4).
