# Shape — one resident process that trains and serves

Root decision, 2026-08-23 (grilling): Mote's GPU is owned by **one resident process** that trains in
preemptible steps and serves from the same weights. No second copy of the model, no CPU fallback while a
training run holds the card, no separate "training job" and "studio server" fighting over 8 GB. The
studio, the prefix cache and the preference votes plug into that process. The architecture (bytes,
H-Net chunking, Mamba-3 outer, Relation main, Muon) stays.

Why: on one GPU with one user, the split into separate programs is what left the card idle while the
studio answered on the CPU at 40 B/s, and what forced training to batch 1 at 16384. The tape that backprop
needs (the activations of the byte-resolution outer layers: ~4.8 of the 6.34 GB at the flagship) is the
price of exact gradients and stays; everything around it is ours to reshape.

The design is **gated on measurements** taken on Fedora first (decided the same day): the daemon's
preemption granularity, memory split and snapshot cadence all depend on numbers we only have for WSL2.

## Serving contract (decided 2026-08-23, independent of the numbers)

**First byte within ~1 s while a training run is on; training yields.** A request preempts at the next
boundary and the run slows while the conversation is active. The daemon is designed around that boundary,
which is why deferred question 1 (step- or micro-batch-level preemption) is the first the step-time
measurement has to settle: at the flagship a full step is seconds, so the boundary will have to be a
gradient-accumulation slice, not a step.

## Day one on Fedora: the measurements

Same scripts, same configs as `docs/results/2026-08-23-chain.md`, so every number has a WSL2 twin.

1. **Step profile** — `python -m mote.train.profile_step` at: 35M local, 2048, batch 4, trained router
   (`--init-from runs/overnight/last.pt`); flagship, 16384, batch 1, `--ckpt-main`, `--chunk-bytes 6`.
   Report B/s, TFLOPS, MFU, peak GB next to 80.8 KB/s · 23.5 % and 42.4 KB/s · 43.7 % · 6.34 GB.
2. **Where the time goes** — an `nsys` trace of 20 steps at both configs: data loading, outer forward,
   main forward, backward, optimizer, launch gaps. Where the memory goes — `torch.cuda.memory._snapshot`
   at peak, grouped by layer, to confirm the activation-tape breakdown above.
3. **Step time at the flagship** — the unit of preemption a serving request would wait for.
4. **Serving on the kernels** — TTFB cold and warm, bytes/s, on the GPU with Triton (the Windows studio
   ran reference paths on the CPU: 2.1 s cold, 127 ms warm); `mote.eval.prefix_probe --device cuda`
   for the kernel-vs-reference rounding the cache now depends on.
5. **Sharing the card today** — serve while a training run is on (two processes): reply latency and the
   training slowdown. This is the baseline the resident process has to beat.
6. **Disk** — memmap read throughput of a shard on ext4 (the WSL2 shards came through 9P).

## Questions the numbers decide (deferred, in dependency order)

1. Preemption: step-level (a reply waits one step) or micro-batch-level (gradient accumulation slices
   with serving slots between them)? Needs 3.
2. Memory split: training state (weights, master copy, momentum, tape) vs serving activations at 16384;
   what batch size survives once serving keeps its working set resident? Needs 1–2.
3. Serve from the training weights (bf16 copy the trainer already keeps) or from an EMA/snapshot? Needs 4
   (quality drift) and a val-bpb check of EMA vs raw.
4. Snapshot cadence and what a "checkpoint" means when the weights never stop moving (the challenger is a
   snapshot; votes reference snapshot ids).
5. Online updates from votes in v1, or batch DPO only (docs/prefs.md gate)? Needs the step cost and the
   disagreement data.
6. Process boundary: one Python process with a scheduler thread, or two processes with CUDA MPS / IPC
   sharing the weights. Needs 5 (the baseline) and 1.
7. Memory-first training changes, evidence-gated by 2: checkpointed or reversible outer layers, bf16/FP8
   activation storage, fused EMA dechunk, optimizer state in pinned host memory, token-budget batching.

## Out of scope until the above

Looped main network, test-time-training memory layers instead of a window, any change to the
architecture's math. Those are a research sweep (2026 sources first), not a systems decision.
