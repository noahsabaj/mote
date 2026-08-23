# Efficiency campaign — Morpheme on one RTX 4060 Ti (8 GB, WSL2)

Research report, 2026-08-23. Target: raise training throughput and MFU for the byte-level H-Net in
this repo, on this machine, without changing numerics (bf16 + fp32 accumulation stays).

**Repo state this was written against:** commit `cf65c84` ("speed: FlashRelation Triton kernel,
chunk-count bucketing, activation checkpointing, Muon option, MFU logging, step profiler"). That
commit landed *during* this campaign; everything below is scored against the post-`cf65c84` tree, so
"already-planned" items that are now *implemented* are called out as such and are not re-derived.

Anything I could not verify from a primary source is marked **[unverified]**.

**How to read the evidence.** Sources are ranked by how directly they bear on *this* machine:

1. **Measurements taken here, 2026-08-23** — run logs in `runs/`, parameter/FLOP counts computed
   against the real model, `aten` dispatch counts, `nvidia-smi` and driver-attribute probes,
   filesystem benchmarks. Strongest, and most of §0, §2, §4.3 and §4.5 rest on it.
2. **Source code and 2026 documentation** — PyTorch 2.13.0 and `mamba_ssm` source read directly,
   PyTorch 2.11–2.13 release notes and devlogs (all 2026), Triton 3.7 source, 2026 GitHub issues,
   NVIDIA/Microsoft docs current as of 2026-08.
3. **2026 papers** — listed in §9 with month and relevance.
4. **Pre-2026 papers** — used only where nothing from 2026 supersedes them, and labelled
   *background (pre-2026)* at the point of use. Several of these are theorems or measured scaling
   fits that do not decay; those are called out as such.

---

## 0. The measured baseline, and where the time actually goes

### 0.1 What the run logs say

From `runs/overnight/log.jsonl` (preset `local`, 35.35M params, batch 2 × 2048 × grad-accum 8 =
32 768 bytes/step, α = 0.1, N 5.0→6.5) — throughput tracks the achieved compression almost exactly:

| `bpic` (bytes/chunk) | chunk count M = 2048/bpic | bytes/s |
|---|---|---|
| 4.60 | 445 | 54 078 |
| 4.17 | 491 | 51 173 |
| 3.73 | 549 | 46 375 |
| 3.47 | 590 | 43 047 |
| 3.39 | 604 | 34 834 |
| 3.33 | 615 | 34 737 |

A **36 % throughput loss over the run**, caused by the router firing more often as it converges. The
model does not reach its target ratio: final `val_bpic` 3.27 against a target of 5.0–6.5.

Analytic FLOPs from `morpheme/train/flops.py` at the converged `bpic = 3.3`, computed against the
real parameter counts (my calculation, `local` preset):

```
126.8 MFLOP/byte total
   byte-level modules (enc+dec+routing+residual+lm_head)   20.7 %
   Relation main network, linear part                      39.1 %
   Relation pairwise (P1·P2ᵀ and F·Ĩ)                       7.3 %
   MBP head, linear part                                   18.0 %
   MBP head, DENSE L² attention                            14.9 %   ← see §2.2
```

37 kB/s × 126.8 MFLOP/byte = **4.7 TFLOPS = 10.6 % MFU** against the 44.1 TFLOPS bf16 peak in
`flops.py:15`. A run sampled live today at step 7 560 was at **~1.125 s/step = 28.5 kB/s** — 3.6
TFLOPS, **8.2 % MFU**. Both bracket the ~9 % in the brief; the spread is `bpic`, not noise.

Parameter split (`local`, 35.35M): Relation main network 77.1 %, MBP head 10.5 %, encoder 5.4 %,
decoder 5.4 %.

### 0.2 Is this workload compute-bound, bandwidth-bound, or launch-bound?

Launch-bound, and it is not close.

* **Power.** 91 W of a 160 W limit, SM clock 2730 of 3105 MHz (`nvidia-smi`, sampled while a run was
  in progress). Not power- or thermally-throttled. `utilization.gpu` reads 89 %, but that counter is
  the fraction of time *any* kernel was resident, not how well the SMs were fed — 89 % util at 10 %
  MFU is the classic signature of many tiny kernels.
* **Bandwidth.** At `bpic` 3.3 the Relation score tensors are the largest traffic: ~46 MB per layer
  per micro-batch read/written a few times, ~1–2 GB per micro-batch total. A micro-batch takes
  ~97 ms, and 288 GB/s × 0.097 s = 28 GB of *available* traffic. We are using well under 10 % of it.
* **Op count (measured).** I counted `aten` dispatches through a `TorchDispatchMode` on the real
  model (`local` widths, B=1, L=192, CPU so the reference Mamba path is used):

  ```
  2380 aten dispatches in ONE forward
     main_network   1230 (51.7 %)   ← 8 Relation blocks; FullRelation.forward alone = 128 ops
     encoder         424            ← CPU reference path; the CUDA fused kernel is far fewer
     decoder         424
     mbp_head         98
     dechunk_layer    76
     routing_module   33
     chunk_layer      24
     lm_head           8
  RMSNorm(prenorm, residual) = 9 ops each
  lca_mask alone            = 26 ops, and 8.4 MB of bool mask at B=2, L=2048
  ```

  On CUDA post-`cf65c84` (FlashRelation collapses the score chain; `mamba3_siso_combined` collapses
  the scan) I estimate **~900 dispatches per forward at B=1**, so ~2 700 fwd+bwd, ×2 batch ×8
  grad-accum ≈ **14 000 kernel launches per optimizer step**. At WSL2 launch latencies that is tens
  of milliseconds of pure submission cost per step even with perfect overlap — and §2.1 shows the
  overlap is currently destroyed.

**Conclusion: optimise launches and synchronisation first, FLOPs second.** Every recommendation below
is ranked on that basis. Three 2026 results say this is the expected regime, not a misconfiguration:

* [**2604.13327**](https://arxiv.org/abs/2604.13327) *(2604, Event Tensor, MLSys 2026)* gives the unit
  cost: *"Each kernel launch typically incurs 5–10 μs of latency, while the fastest kernels may
  complete in 2 μs."* That is the constant in the arithmetic above.
* [**2604.10597**](https://arxiv.org/abs/2604.10597) *(2604, COREY)* benchmarks the real `mamba_ssm`
  selective scan on an **RTX 3070 under WSL2 with Triton 3.6 and PyTorch 2.11** — nearly this
  environment — and finds latency falls monotonically with chunk size purely because fewer launches
  are issued (L=4096: chunk 32 → 128 calls → 6.32 ms; chunk 512 → 8 calls → 0.75 ms). Its conclusion:
  *"the performance bottleneck is Python-side kernel-dispatch overhead rather than numerical
  complexity of the scan."* A companion profile on an RTX 4050 under WSL2 measures a per-timestep loop
  (4 096 calls) at 14.96 ms against 9 calls at 0.308 ms — **48.6×, attributed entirely to launch
  overhead.** *(Caveat: the paper self-describes as "Concept & Feasibility"; the chunk-size/launch-count
  measurements are solid, its entropy-scheduling contribution is not end-to-end validated.)*
* [**2608.08961**](https://arxiv.org/abs/2608.08961) *(2608)* measures **8–15 % GPU utilisation for
  memory-bound 1B LLM training against 96–99 % for compute-bound vision models** on an A100. Low
  utilisation on small-LM training is a documented 2026 phenomenon. It also reports that gradient
  checkpointing costs **20–60 % runtime** (relevant to `--ckpt-main`), and that gradient accumulation
  sits on the Pareto frontier for every architecture tested while adding no GPU memory.

### 0.3 Honest anchor: what is achievable

The H-Net authors state plainly (2507.07955 §5) that their implementation *"may be approximately up
to 2× slower than an isotropic model during training"*, and attribute it to dynamic shapes and
dynamic memory. None of the five source papers report MFU at all. A byte-level H-Net at 10 % MFU on
a consumer card is a *bad but not pathological* starting point; the realistic target here is
**25–35 % MFU**, not 50 %.

---

## 1. Ranked recommendations

Gain estimates are for **this machine at the `local` preset, batch 2 × 2048 × accum 8**, and are
multiplicative on wall-clock bytes/s unless stated. Where I could not measure, the estimate is
labelled as such — run §8 before believing it.

| # | Idea | Expected gain here (reasoning) | Effort | Numerics risk | Evidence | Status |
|---|---|---|---|---|---|---|
| 0a | `export TRITON_CACHE_AUTOTUNING=1` | **kills a ~30 s autotune sweep per process, and per train↔generate transition** — measured 3.05 s → 0.32 s cross-process on a standalone kernel. The cache has never been used here (0 `.autotune.json` files across 820 entries). | 1 min | none | §4.1; `triton/runtime/autotuner.py:39,170-210`, `knobs.py:375` | **new** |
| 0b | Probe CUDA device attribute **110**; if 0, retry `expandable_segments:True` | reopens the single best fix for continuously-varying allocation sizes | 1 min | none | §4.9; [pytorch#192330](https://github.com/pytorch/pytorch/issues/192330) (2026-08-06, open, lists 2.13.0+cu126) | **new** |
| 0c | If 110 = 1: `PYTORCH_ALLOC_CONF=roundup_power2_divisions:8,per_process_memory_fraction:0.9,garbage_collection_threshold:0.8` | fragmentation from ever-changing block sizes; note GC is a **no-op** unless the memory fraction is < 1.0 | 15 min | none | §4.9; `c10/core/AllocatorConfig.cpp`, `CUDACachingAllocator.cpp` | **new** |
| 0d | `torch._inductor.config.combo_kernels = True` (+ `benchmark_combo_kernel = True`) on the compiled leaves | **PyTorch 2.10's horizontal fusion of *independent* small ops — shipped Jan 2026 as "reduced kernel launch overhead", off by default.** Aimed at exactly the ~21 600-launch profile in §0.2; merges kernels rather than just cheapening their submission | 1 h | none (fusion only) | §4.10; [PyTorch 2.10 release blog](https://pytorch.org/blog/pytorch-2-10-release-blog/), `torch/_inductor/config.py` | **new** |
| 1 | Remove the ~40 GPU→CPU syncs per optimizer step | **1.1–1.5×, ceiling ~2× [estimate — M1 decides]** — 5 `.item()` calls per micro-batch × 8 accum; each drains the queue so backward launches never overlap forward execution. Bounded above by the CPU-dispatch share of the step; see §2.1 for the arithmetic. | 1–2 h | **none** | `train/train.py` `compute_losses` (`ce.item()`, `lr_.item()`, `ce_m.item()`), `model/dc.py` `bytes_per_chunk` (2 × `.item()`) | **new** |
| 2 | MBP head: exact block-local attention instead of a dense `[B,L,L]` mask | **1.10–1.25×** — removes 14.9 % of analytic FLOPs, 8.4 MB mask + ~34 MB of bool temporaries per micro-batch, and the slow masked-SDPA path. Waste factor measured at **310×** at `bpic` 3.3. | 0.5–1 d | **none** (mask is identical) | `model/mbp.py` `lca_mask`, `train/flops.py:47`; my verification in §2.2 | refinement (FlexAttention was planned; the *exact* structure and a no-FlexAttention recipe are new) |
| 3 | Raise micro-batch 2 → 4 now that FlashRelation + `--ckpt-main` cut Relation memory | **1.10–1.30× [estimate]** — halves Python/launch overhead per step. Previously blocked by the fp32 T² score tensors, which `cf65c84` removed. | 0.5 h (just try it) | none | §2.3 memory table; README's "batch 4 collapsed to 13 kB/s" | **new** (the blocker was just removed) |
| 4 | Cache RoPE tables; rewrite `_givens` without `clone()` + index-scatter | **1.03–1.08×** — these are now the *largest remaining* per-layer op cost in the main network, because FlashRelation fused everything else. ~110 of ~430 main-network dispatches. | 1–2 h | none (identical arithmetic) | `model/relation.py` `_rope_cos_sin` / `_apply_rope` / `_givens`, called per layer per forward | **new** |
| 5 | Fix `profile_step.py` to match the trainer before trusting it | measurement validity, not speed | 15 min | n/a | `train/profile_step.py:54` builds the model with `dtype=bfloat16` (trainer uses fp32 params + autocast); `--grad-accum` defaults to 1 (trainer uses 8) | **new** |
| 6 | ~~Re-enable the SSD kernel for the dechunk EMA~~ → **leave `use_kernel = False`; fix the stale comment instead** | **withdrawn.** The EMA runs at `dstate=1`, while the SSD kernels tile `dstate` with `BLOCK_SIZE_K ∈ {32,64}` — **≥97 % of every dot tile would be mask**. The chunked-torch scan is the right tool for a scalar-decay recurrence | 15 min (comment) | none | §2.8; `model/dc.py:144` and its (incorrect) rationale | **new** (reverses my own earlier recommendation) |
| 6b | **`D=self.D if torch.is_grad_enabled() else self.D.detach()`** in `Mamba3Mixer.forward` | frees **~100 MB** during every eval on a card at 6.7/8.2 GB: passing a raw `Parameter` makes `needs_backward` true even under `no_grad`, so eval allocates every backward-only buffer and saves 27 tensors | 15 min | **none** | §2.8b; `model/mamba3.py:168`, `mamba3_siso_combined.py:82` | **new** |
| 7 | Raise `_ema_chunked`'s block size C from 64 to 256 | ~0.5–1 % | 15 min | none (fp32, different summation order only) | `model/dc.py` `_ema_chunked`, `for b in range(nb)` Python loop | **new** |
| 8 | Write checkpoints to ext4, not `/mnt/d` | ~0.5–1.3 % | 15 min | none | `runs/overnight/last.pt` is **424 MB**, written every `--ckpt-minutes 10` across the 9p/DrvFs boundary | **new** |
| 9 | Data pipeline: pinned staging buffer + prefetch thread; ship `int32` not `int64` | ~1–3 % [estimate] | 2–3 h | none | `data/loader.py` `sample_batch` returns a **non-pinned** tensor, so `.to(device, non_blocking=True)` in `train_step` is silently synchronous | **new** |
| 10 | **Fix the grad-accum loss denominator for SFT** (mean-of-means bug) | correctness, not speed — silently up-weights short micro-batches | 1 h | **fixes** a numerics bug | `_masked_ce` normalises per micro-batch by `w.sum()`, then `train_step` divides by `grad_accum`; see §3.1 | **new** |
| 11 | Log McCandlish `B_simple` from the 8 micro-batch gradients | tells you empirically whether accum 8 is right; 2 extra `.norm()` | 2 h | none | McCandlish et al. [1812.06162](https://arxiv.org/abs/1812.06162) App. A; see §3.2 | **new** |
| 12 | Pin M completely (boundary budget) → zero syncs, whole-micro-batch CUDA graph | **1.15–1.4× on top of #1 [estimate]** | 2–4 d | **low–med** (exact whenever the router does not over-fire; drops lowest-probability boundaries when it does) | §2.6; builds on `ChunkLayer(bucket=…)` from `cf65c84` | refinement (goes past "bucket to multiples of 64") |
| 13 | `torch.compile` the *shape-stable leaves*, never the whole model | see §2.7 | 0.5–1 d | none | `train.py` `--compile` wraps the whole model, which recompiles on every new M | refinement |
| 14 | **A/B the MBP head off entirely** (`mbp.enabled = False`) | **1.4–1.5× if you drop it** — it is 33 % of the FLOP budget, and the MTP literature predicts a *quality regression* at 35M | 1 h | n/a (an experiment) | §3.3; Gloeckle [2404.19737](https://arxiv.org/abs/2404.19737) Fig. 3 | **new** |
| 15 | 4096-context A/B — **only after #2** | — | — | n/a | MBP dense attention is O(L²); at L=4096 it grows from 14.9 % to ~26 % of FLOPs | refinement (sequencing) |
| 16 | **Fix `split_muon_params`: take Mamba `in_proj` off Muon**, then A/B | 2026 reverses my first read. Muon **wins at 60–130M** and *halves* optimizer-state memory ([2607.04033](https://arxiv.org/abs/2607.04033)), but on an SSM the gain is localised to `out_proj`; Muon on the 8-way-stacked `in_proj` is *worse than AdamW* ([2608.03941](https://arxiv.org/abs/2608.03941)) | 2 h | none | §3.4; `train/muon.py::split_muon_params`, `model/mamba3.py:109-110` | **new** (supersedes my earlier "likely negative") |
| 16b | Muon weight decay `λ·η_t` → `λ·η_t²/η_max`; Gram Newton–Schulz | **+21.7 % at 72M**, the nearest published scale ([2607.23777](https://arxiv.org/abs/2607.23777)); Gram-NS is 1.5–2× cheaper with *mathematically identical* output ([2608.11612](https://arxiv.org/abs/2608.11612)) | 3 h | none for Gram-NS (exact); the wd change is a recipe change | §3.4 | **new** |
| 17 | Serving: switch the MBP head to speculative *verification* acceptance | **1.4–1.7×** decode, and it becomes distribution-preserving | 1 d | **fixes** a silent distribution shift | §5.1; LCA 2608.15454 Fig. 8; [2607.26627](https://arxiv.org/abs/2607.26627); `serve/engine.py` uses the threshold rule (`pm >= accept_threshold`) | refinement |
| 18 | Serving: restructure the decode cache for **input replay** before adding rejection | prerequisite for #17 — SSM state rollback is O(T) without it | 1–2 d | none | §5.2; [ReplaySSM (Tri Dao, 2026)](https://tridao.me/blog/2026/replayssm/) | **new** |

Rows 0a–0d cost minutes to an hour and change what the rest of the list is worth — do them first;
0d in particular is a one-line config change that PyTorch shipped in 2026 specifically for this
failure mode. Rows 1–5 are this week's work; taken together they plausibly move ~9–10 % MFU to
**~15–20 %**, and none of them touches numerics. Rows 6b and 10 are bug fixes, not optimisations —
do them because they are wrong, not because they are fast. Row 14 is the largest single compute lever
in the model and is a one-hour experiment. Rows 12 and 13 are the multi-day items and should wait
until §8 has told you whether the step is host-bound or device-bound.

Two rows reverse advice I gave earlier in this campaign, and are marked as such: **row 6** (do *not*
re-enable the SSD kernel for the dechunk EMA — §4.2) and **row 16** (Muon is more promising than I
first judged, but the parameter split is wrong — §3.4).

---

## 2. Per-idea detail

### 2.1 Kill the per-micro-batch synchronisations (rank 1)

**What is there now.** `morpheme/train/train.py::compute_losses` ends with:

```python
stats = {"ce": ce.item(), "ratio": lr_.item(), "bpic": bytes_per_chunk(out.routing.boundary_mask, mask)}
...
    stats["ce_mbp"] = ce_m.item()
```

and `morpheme/model/dc.py::bytes_per_chunk` is:

```python
nb = (boundary_mask & mask).sum().item()
return float(mask.sum().item()) / max(nb, 1)
```

That is **five `cudaStreamSynchronize`-equivalents per micro-batch**, and `train_step` calls
`compute_losses` inside the `for _ in range(args.grad_accum)` loop — **40 hard syncs per optimizer
step** at `--grad-accum 8`.

**Why it costs so much here.** `ce.item()` executes immediately after the forward is *enqueued*. The
CPU then blocks until the entire forward has *executed*, and only then begins submitting the ~1 800
backward kernels. On a workload where CPU submission time is comparable to GPU execution time
(§0.2), this converts an overlapped pipeline into a strictly serial one, twice per micro-batch.

The PyTorch team put the second-order cost well in a devlog dated **2026-08-11**
([*Host-to-device syncs are bad too*](https://docs.pytorch.org/devlogs/eager/2026-08-11-hidden-h2d-sync/)):
a sync costs the host wait, which is visible in profiles and minor, and then **"the bubble
afterwards"** — *"from the instant the sync returns, the host has no lead, so GPU idle time is
directly exposed to Python overhead."* On 34 SMs with a 35M model the kernels are short and the
launch lead is thin, so **one `.item()` is not one unit of cost — it is one full drain of the launch
queue**, after which every Python and dispatcher microsecond becomes GPU idle. There are five per
micro-batch.

The same devlog names hidden H2D syncs worth grepping for: `torch.tensor([...], device="cuda")`
(*"Three numbers through `torch.full` cost nothing. Three numbers through `torch.tensor` drain your
entire launch queue."*) and list/tuple indexing like `t[:, (0,2,3)]`. This repo has one in the decode
path — `hnet.py::step` builds `torch.tensor([[state.cur_offset]], device=h.device)` **every byte**
(see §5.3).

**The fix.** Keep the statistics as 0-dim CUDA tensors and only materialise them on logging steps.

```python
# dc.py — return a tensor, not a float
def bytes_per_chunk(boundary_mask, mask):
    return mask.sum() / (boundary_mask & mask).sum().clamp(min=1)

# train.py::compute_losses — no .item() anywhere
stats = {"ce": ce.detach(), "ratio": lr_.detach(),
         "bpic": bytes_per_chunk(out.routing.boundary_mask, mask)}
if out.mbp_logits is not None and mbp_weight > 0:
    stats["ce_mbp"] = ce_m.detach()

# train.py::train_step — accumulate on the GPU
for k, v in stats.items():
    agg[k] = agg[k] + v / args.grad_accum if k in agg else v / args.grad_accum
gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)   # keep the tensor
opt.step()
agg["grad_norm"] = gnorm
return agg          # dict of CUDA tensors

# main loop — one sync per *logging* interval, not per micro-batch
if step % args.log_every == 0:
    stats = {k: float(v) for k, v in stats.items()}
    ...
```

Nothing else in the loop depends on a device value: `progress()`, `wsd_lr` and `atdc_target_ratio`
are all driven by `time.time()`, and `--max-steps 0` makes the schedule wall-clock-based. So the loop
can run entirely async between logging points.

**How big is this, honestly?** Bounded above by the CPU's share of the step, so let me do the
arithmetic rather than assert a number. ~900 `aten` dispatches per forward (§0.2) plus ~1 800 in
backward ≈ 2 700 per micro-batch, ×8 = **~21 600 per optimizer step**. At an eager per-dispatch cost
of 5–15 µs (Python frame + dispatcher + launch, at the high end under WSL2's VMBUS submission path)
that is **110–320 ms of host work** against a measured step of ~0.8–1.1 s.

If the host work were perfectly overlapped, the step would cost `max(CPU, GPU)`. The syncs force it
toward `CPU + GPU` in eight places, because after every drain the queue is empty and the GPU can only
proceed as fast as the CPU refills it. So the recoverable time is **at most** that 110–320 ms →
**1.1–1.5×**, with ~2× only if the profiler shows CPU total ≈ CUDA total. That is exactly what M1
measures, and it is why M1 comes before any of this. I would rather under-promise here: the reason
this is still rank 1 is not the size of the number, it is that the number is free — two hours, zero
numerics risk, and it makes every subsequent measurement trustworthy by removing the thing that most
distorts a profile.

**What remains after this.** One unavoidable sync per micro-batch: `M = int(num.max())` in
`ChunkLayer.forward` — a data-dependent shape. That drops the count from 40 to 8, and each one drains
much less (it fires right after the encoder rather than after the full forward). Rank 11 removes even
that.

**Verify with** `torch.cuda.set_sync_debug_mode("warn")` around one step; every remaining sync prints
a stack trace. (`torch.cuda.set_sync_debug_mode`, PyTorch docs.)

### 2.2 The multi-byte head is doing ~310× more attention than its mask allows (rank 2)

`morpheme/model/mbp.py::lca_mask` builds a `[B, L, L]` boolean mask over **all 2 047 bytes**, and
`_MHA.forward` passes it to `F.scaled_dot_product_attention` as an explicit `attn_mask`. Two
consequences:

1. `L² = 4.19 M` score pairs per head per layer are computed, of which the mask keeps ~7 (at
   `bpic` 3.3) — **310× waste**. In `flops.py` this is the `12.0 * seq_len * D0 * cfg.mbp.n_layers`
   term, **14.9 % of the analytic FLOP budget** at the `local` preset (the file's own comment already
   concedes "dense mask; block-sparse would be less").
2. An explicit `attn_mask` **provably** forces SDPA off the FlashAttention backend. Confirmed in
   source at `aten/src/ATen/native/transformers/sdp_utils_cpp.h` (v2.13.0):
   `check_for_attn_mask` returns false with *"Flash Attention does not support non-null attn_mask"*,
   and it is listed in `can_use_flash_attention`'s `general_constraints`. So the head silently runs
   on the memory-efficient or math backend. The mask itself is 8.4 MB at B=2, L=2048, with ~34 MB of
   bool temporaries (`causal`, `same`, `prev`, `allowed`, `eye`) rebuilt every micro-batch, 8× per
   step.

**I verified the mask's exact structure** (200 random boundary patterns, `L=512`): `lca_mask` is
*exactly* the contiguous causal window

```
allowed(i) = { j : start_of_chunk(chunk_id(i) − 1) ≤ j ≤ i }
```

with measured widths: mean 6.1, max 18 at boundary rate 0.30 (`bpic` 3.3); mean ~12, max ~74 at
`bpic` 6.0. It is a **variable-width causal sliding window**, nothing more. This matches the LCA
paper's own description (2608.15454 Fig. 2: *"a query at position i in segment s_i may attend to
itself, to earlier positions within its own segment, and to all bytes in the immediately preceding
segment"*).

**Two implementations, both exact.**

*(a) Blocked local attention — pure PyTorch, no FlexAttention.* Split the byte axis into blocks of
`W`; query block `b` attends only to key blocks `b−1` and `b`. Correct whenever the true window
never exceeds `W + 1`.

```python
# mbp.py::_MHA.forward, W = 64 at bpic ≥ 3, W = 128 for headroom at bpic ≥ 6
q = q.view(B, H, L // W, W, dh)
kv = torch.stack([shift_prev(k), k], dim=?)      # keys from blocks b-1 and b  -> [B,H,L//W,2W,dh]
y  = F.scaled_dot_product_attention(q, k2, v2, attn_mask=local_mask)  # [B,H,L//W,W,2W]
```

The mask becomes `[B, L//W, W, 2W]` = **524 KB** instead of 8.4 MB, and score work drops by
`L / 2W` = **16× at W=64**. Add a guard: if `max_chunk_len > W` in a batch, fall back to the dense
path (or clamp `W` upward), so the result stays exact.

*(b) FlexAttention.* `mask_mod(b, h, q, kv) = (kv <= q) & (chunk_id[b, kv] >= chunk_id[b, q] - 1)`
with `chunk_id` captured, then `create_block_mask(...)`. Exact, and the block mask skips
`14/16` of key blocks at `BLOCK=128`, `30/32` at `BLOCK=64`. Four things make this much more
attractive than it looks, and one that could sink it:

* **Dynamic shapes are a non-issue here.** Seq len is fixed at 2048 and the micro-batch at 2; only
  the BlockMask *contents* vary with the router. `_dense_to_ordered` argsorts the full last dim, so
  `kv_num_blocks` is always `[B,H,16]` and `kv_indices` always `[B,H,16,16]` — **shape-invariant in
  M**. The FlexAttention FAQ says it directly: *"Even changing the block-sparsity doesn't require a
  recompile. However, if the block-sparsity changes, we do need to recompute the BlockMask."* So
  `torch.compile(flex_attention, dynamic=False)`, rebuild the BlockMask once per step, thread it
  through both layers → **zero recompiles**. And 2048 = 16 × 128 → `IS_DIVISIBLE=True`, the fast path.
* `create_block_mask` cost is small here — it materialises a dense `(B,H,2048,2048)` bool (4 MB) then
  reduces. Build it **once at the top of the model** and pass it down, per the official guidance.
* **It must be compiled.** Uncompiled, `flex_attention` still routes through Dynamo with the *eager*
  backend, Inductor never runs, and it falls back to `sdpa_dense` → `math_attention`, which
  materialises the full `B×H×2048×2048` **fp32** score matrix. That is strictly worse than today.
* `create_block_mask(_compile=True)` is **deprecated**; use `torch.compile(create_block_mask)`.
* Gradients: a learnable tensor captured by `score_mod` gets gradients (since 2.6), but a tensor
  captured by **`mask_mod` may not require grad** — `raise AssertionError("Captured buffers from mask
  mod that require grad are not supported.")`. **`chunk_id` must be `.detach()`ed** before entering
  `mask_mod`. It is integer and detached already in practice, but make it explicit.

> ⚠️ **The one that could sink it: head dim — and it bites exactly the preset you train.** `LCAHead`
> is built with `d_model = D0` (`hnet.py:73`), so the MBP head's head_dim is `D0 / mbp.n_heads`:
> **`pilot` 256/4 = 64 ✓, `flagship` 512/8 = 64 ✓, but `local` 384/4 = 96 ✗.** FlexAttention pads 96
> up to 128 (`set_head_dim_values` rounds up and masks), and the default config tables only have keys
> 64/128/256. On Ada the per-block shared-memory ceiling is
> **99 KB** (101 376 B) versus H100's 227 KB, and PyTorch has **no shared-memory-based config pruning
> on the flex path** — it will emit a config that does not fit and fail at launch.
> [pytorch#133254](https://github.com/pytorch/pytorch/issues/133254) (open) reports exactly this on an
> **RTX 4090, bf16, S=2048, D=128, backward**: `out of resource: shared memory, Required: 131074,
> Hardware limit: 101376`, and notes it does **not** occur at D=64. There is no sm_89-specific forward
> config table; capability (8,9) falls into the `>= (8,0)` branch and inherits A100's 164 KB budget.
> **Set `local`'s `mbp.n_heads = 6` (384/6 = head_dim 64) before trying FlexAttention there**, or
> override `kernel_options={"fwd_BLOCK_M": 64, "fwd_BLOCK_N": 64, "fwd_num_stages": 1, "bwd_BLOCK_M1":
> 32, "bwd_BLOCK_N1": 32, "bwd_BLOCK_M2": 32, "bwd_BLOCK_N2": 32}` (constraints: `BLOCK_N1 % BLOCK_M1
> == 0`, `BLOCK_M2 % BLOCK_N2 == 0`). The Relation main network is already safe at head_dim 64 across
> all three presets (384/8, 512/8, 768/8 → 48, 64, 96 — ⚠️ `pilot` is 48 and `flagship` is 96, so if
> you ever route Relation through FlexAttention rather than the hand-written kernel, the same check
> applies there).

Do **(a)** first: it is an afternoon, needs no new PyTorch machinery, has no head-dim hazard, and
captures most of the win. Reach for **(b)** if you also want it for the Relation mixer, or if the
blocked-local guard (max chunk length > W) fires too often.

**While on the subject of Ada's shared memory — check `flash_relation` at the flagship width before
the H100 run.** `_tile()` in `morpheme/model/flash_relation.py` picks `BLOCK_D = next_pow2(D)` and
`BLOCK_M = BLOCK_N = 64 if BLOCK_D * elem_size <= 256 else 32`, with the comment that 64×64 tiles fit
Ada's 99 KB "up to 128-wide bf16 heads". At `local` (D = 64 → BLOCK_D 64) that is ~25 KB of tiles and
it demonstrably works. At `flagship` (Relation d_model 768 / 8 heads = **D 96 → BLOCK_D 128**) the
three staged tiles come to ~49 KB before `num_stages=2` double-buffering, which is closer to the
ceiling than anything currently exercised. On H100 (227 KB) it is moot; **on Ada it is untested**.
Cheap guard: assert the computed tile footprint against
`torch.cuda.get_device_properties(0).shared_memory_per_block_optin` at kernel-launch time, so it fails
loudly rather than as a Triton `OutOfResources` mid-run.

### 2.3 Micro-batch 2 → 4 (rank 3)

The reason batch 4 previously "collapsed to 13 kB/s" (README) was the materialised Relation score
path. My accounting of the pre-`cf65c84` code — `u` (fp32), the `torch.where` output (fp32), the
`masked_fill` output (fp32), `flow` (fp32), `flow.to(bf16)` = **18 bytes per T² element live, 10
bytes saved for backward**, per layer, ×8 layers:

| B | `bpic` | M | saved for backward | peak transient |
|---|---|---|---|---|
| 2 | 3.3 | 620 | 0.46 GiB | 0.82 GiB |
| 4 | 3.3 | 620 | 0.92 GiB | 1.65 GiB |
| 4 | 2.0 (early training) | 1024 | 2.50 GiB | 4.50 GiB |

The 4.5 GiB row is what killed batch 4: at initialisation the router fires on ~50 % of bytes, so the
transient peak is ~5.5× the converged value, on a card with ~7.3 GiB usable.

`cf65c84` removed that entirely — `flash_relation` never materialises T², and `--ckpt-main`
recomputes the Relation blocks. **Batch 4 (or 8) should now be re-tested**, and it is the single
cheapest experiment on this list: it halves the number of micro-batches, hence halves Python and
launch overhead per optimizer step, on a launch-bound workload. Keep the effective batch constant
(`--batch-size 4 --grad-accum 4`) so the recipe is unchanged, and compare bytes/s.

### 2.4 RoPE and Givens are now the main network's biggest remaining overhead (rank 4)

`FullRelation.forward` is 128 `aten` ops in the materialised path (measured). FlashRelation removes
the score chain but **not** the code above it, which still runs per layer, per forward:

```python
pos = torch.arange(S, S + T, device=x.device)
cos, sin = _rope_cos_sin(pos, dh, self.rope_theta, x.dtype)   # arange, pow, div, outer, cat, cos, sin, to
p1 = _apply_rope(p1, cos, sin)                                 # mul, chunk, neg, cat, mul, add
p2 = _apply_rope(p2, cos, sin)
info = self._givens(info)                                      # cos, sin, 2 index-gathers, clone, 2 index-scatters
```

`_rope_cos_sin` recomputes an identical table **8 times per forward** (once per layer), and `_givens`
allocates a full `[B,H,T,dh]` copy via `info.clone()` then writes it back with advanced indexing —
a scatter kernel, the slowest way to express what is a fixed pairwise rotation.

**Fixes, both bit-identical in arithmetic:**

* Cache the table. Register `cos`/`sin` up to `max_seq_len` as a non-persistent buffer on the
  *stack* (not per layer) and slice `[:T]`, or memoise on `(T, dtype, device)`. Saves ~8 ops × 8
  layers.
* Rewrite `_givens` as a view. For even layers the pairing is `(0,1),(2,3),…`, so
  `info.view(B, H//2, 2, T, dh)` and one `torch.stack` of the two rotated halves does it with no
  clone and no scatter. For odd layers the pairing is `(1,2),(3,4),…,(H−1,0)` — a rotation of the
  head axis — so `torch.roll(info, -1, dim=1)`, the same even-pairing view, then `torch.roll` back.
  Saves ~6 ops × 8 layers and one full activation copy per layer.

Together these remove roughly 110 of the ~430 dispatches the main network still issues per forward.

### 2.5 `profile_step.py` does not currently measure the trainer (rank 5)

Two discrepancies that will make the profiler lie:

* `profile_step.py:54` — `HNetForCausalLM(cfg, device=device, dtype=torch.bfloat16 …)` creates
  **bf16 parameters**. `train.py` creates fp32 parameters and relies on `torch.autocast`. The
  profiler therefore measures a model with half the parameter memory, half the optimizer-state
  memory, no autocast casts in the graph, and different kernel selection. It will over-report speed
  and under-report memory.
* `--grad-accum` defaults to `1`; the trainer runs `8`. The per-micro-batch sync amplification in
  §2.1 is invisible at accum 1.

Fix both (`dtype=None` + wrap in `autocast` exactly as `train_step` does; default `--grad-accum 8`)
*before* using it to score any of the changes below. Also add `torch.cuda.set_sync_debug_mode` and a
count of CUDA launches (`prof.key_averages()` row count, or the `cudaLaunchKernel` CPU-side total)
to the printed summary — launch count is the metric that matters most for this model.

### 2.6 Pinning M completely: zero syncs and a whole-micro-batch CUDA graph (rank 11)

`cf65c84` added `chunk_bucket = 64`, which rounds M up to a multiple of 64. That is the right first
step and it is bit-neutral for the reasons the docstring gives (the main network is causal, the
padded tail holds non-boundary bytes at positions the dechunk gather never reads). It stabilises the
Triton specialisation of `flash_relation` too: `TQ`/`TK` are runtime ints, and Triton specialises
integer arguments on divisibility-by-16, so a multiple of 64 always lands in the same specialisation
bucket.

It does **not** remove the `int(num.max())` sync, and it does not give constant shapes — only a small
ladder of them. To get both:

**Boundary budget.** Choose a constant `M_fix` (e.g. `round64(L / N_min)` with `N_min` the smallest
compression you will tolerate). On the GPU, if more than `M_fix` boundaries fire, keep the `M_fix`
highest-probability ones; if fewer, the tail is padding as today. Everything stays on-device — no
`.item()`, no `int()`, no data-dependent shape.

* **Exact** whenever `num.max() ≤ M_fix`, which after warm-up is essentially always (measured
  `bpic` 3.3 → M ≈ 615; `M_fix = 768` leaves 25 % headroom).
* At initialisation the router fires on ~50 % of bytes and the budget *will* bind — which is
  arguably a feature (it caps the worst-case memory that currently forces batch 2) but it is a
  genuine change to the first few hundred steps. Gate it with a flag and A/B the loss curve.
* Once shapes are constant, the whole fwd+bwd of one micro-batch becomes CUDA-graph-capturable.

**How to capture this with grad accumulation — the officially blessed answer, and one real trap.**
Do *not* hand-capture the whole step and replay it 8×. The PyTorch notes are explicit that a captured
backward *"refills static `.grad` tensors in place"* — i.e. **overwrite, not accumulate**. Replaying
eight times would silently leave you with only the eighth micro-batch's gradient.

Use `torch.cuda.make_graphed_callables` instead, which the docs recommend for exactly this shape of
problem: *"If some of your network is unsafe to capture (e.g., due to dynamic control flow, **dynamic
shapes**, CPU syncs, or essential CPU-side logic), you can run the unsafe part(s) eagerly and use
`torch.cuda.make_graphed_callables` to graph only the capture-safe part(s)."* Its captured backward
calls `torch.autograd.grad(..., only_inputs=True)` and threads parameters through as autograd-Function
inputs, so parameter gradients are *returned* and accumulated by ordinary eager `AccumulateGrad`
outside the graph. Grad accumulation therefore works normally, and the optimizer step is outside by
construction.

Two constraints to plan for: `autocast` must run with `cache_enabled=False` (it hard-errors
otherwise), and modules must have no forward hooks registered at capture time — which conflicts with
`profile_step.py`'s `register_forward_pre_hook` instrumentation, so those are mutually exclusive.

> ⚠️ **Verify this before trusting any number.** Whether `AccumulateGrad` *steals* the graph's static
> buffer on the first accumulation into a `None` `.grad` is undocumented either way. If it does, the
> next replay overwrites in place and accumulation silently becomes overwrite:
> ```python
> opt.zero_grad(set_to_none=True)
> loss(model(x)).backward(); g1 = p.grad.clone()
> loss(model(x)).backward(); g2 = p.grad.clone()
> assert torch.allclose(g2, 2*g1), "accumulation clobbered by static buffer"
> ```

A pragmatic middle path that needs none of rank 12: **graph only the encoder and decoder**, which
already see fixed `2 × 2048` shapes regardless of the router. Pass them to `make_graphed_callables`
as a tuple in run order so they share one memory pool, and leave the M-dependent trunk eager. NVIDIA's
own CUDA-Graph guidance names this pattern for structurally identical architectures — *"since the
expert routing is inherently dynamic, graph the non-MoE layers and keep MoE layers in eager mode"* —
and a learned chunk router is the same problem. Note that §4.3 says the encoder/decoder are the
24-CTA occupancy-starved part, so they are also where launch overhead hurts most.

The alternative the H-Net authors chose is worth stating for contrast: 2507.07955 §5 says they
*"[handle] variable sequence lengths within a mini-batch using specialized kernels provided by Dao
(2024) and Dao and Gu (2024)"* — i.e. **varlen/`cu_seqlens` packing, not padding**. Packing wastes
less (no `B × (M_bucket − num_b)` padding) but keeps shapes variable. On a *launch-bound* consumer
card, shape stability is worth more than the padding waste; on the H100 flagship run the trade
reverses, and `cu_seqlens` support in `flash_relation` would be the better investment there.

### 2.7 `torch.compile`: compile the leaves, not the graph (rank 12)

`train.py` currently does `fwd = torch.compile(model) if args.compile else model`. With M changing
every step this will recompile until it hits `torch._dynamo.config.cache_size_limit` and then fall
back to eager — worse than not compiling. That is presumably why `--compile` is not the default.

The productive pattern for this model is the opposite: **leave the graph alone and compile the
shape-stable elementwise chains**, whose only varying dimension is the leading one:

* `morpheme/model/norm.py::RMSNorm.forward` — 9 ops each, ~24 calls per forward, all fp32 and
  memory-bound. The single best compile target in the repo.
* `morpheme/model/relation.py::SwiGLU.forward` and `_apply_rope`.
* `morpheme/model/mamba3.py::heavy_tail_activation` and the `_preprocess` split/softplus chain.
* `hnet.py`'s `h2 = (z.float() * ste_ones(...) + residual).to(h.dtype)`.

Each is `@torch.compile(dynamic=True)` on a free function (or a compiled submodule), with `D` static
and the leading dims dynamic, so each compiles **once** for the whole run. Detailed flag guidance in
§5.

### 2.8 Dechunk EMA (rank 7; my earlier rank-6 advice is withdrawn — see §4.2)

`DeChunkLayer.use_kernel` is hard-coded `False` with the comment *"the Triton SSD kernel re-autotunes
for every new chunk count."* Mid-campaign I recommended re-enabling it, on the strength of ATDC
2605.30080 §III.C.2 — *"can be efficiently computed via parallel scan kernels by reformulating the
recurrence as a linear SSM"* (a statement H-Net itself never makes; I checked). **I withdraw that.**

Two reasons, both in §4.2. The comment's stated mechanism does not exist — nothing autotunes on chunk
count — so the premise is wrong but the conclusion is right for a different reason: `dc.py:192` calls
`mamba_chunk_scan_combined` with **`dstate = 1`**, because the EMA is a scalar-decay recurrence, while
the SSD kernels tile the state dimension with `BLOCK_SIZE_K ∈ {32, 64}`. **At `dstate = 1`, ≥97 % of
every dot tile is mask.** ATDC's claim is about parallel scans in general, not about a
one-dimensional state. Keep `use_kernel = False`; the only change worth making is to correct the
comment so the next person does not re-litigate it. (Secondary caveat if it is ever revisited: the
kernel path casts to bf16 while `_ema_chunked` works in fp32, so it would be an A/B, not a
bit-neutral swap.)

What survives is the cheap part. `_ema_chunked` runs a Python `for b in range(nb)` loop over `M/C` blocks with `C = 64`:
10 iterations at M=640, 16 at M=1024, each issuing ~3 ops plus autograd nodes. Raising `C` to 256
cuts that to 3 blocks. The cost is `O(M·C·D)`, so the transfer matrix grows from 0.2 MB to ~1.6 MB
and its matmul from ~63 MFLOP to ~300 MFLOP per micro-batch — at 44 TFLOPS that is under 0.01 ms,
i.e. free, in exchange for ~28 fewer dispatches and 7 fewer autograd nodes. Same mathematics,
different summation order (so bit-identical only up to fp32 associativity).

### 2.8b The cost of evaluation is small — with one avoidable exception

`--eval-every 500 --eval-batches 8` costs 8 forward passes against 500 steps × 8 micro-batches ×
(fwd + ~2× bwd), i.e. roughly **0.07 % of the run**. Not worth optimising, and the ~80 `.item()`
calls inside `evaluate` are harmless at that frequency.

The exception is `chunk_sample`, which runs a forward on a **110-byte** sequence. That is a length
nothing else in the run uses, so it enters a different Triton *specialisation* class (110 is not
divisible by 16, unlike 2048 — see §4.2) and, via `blocks.py:44`, a different autotune key
(`RETURN_FINAL_STATES=True`). One 27-config sweep ≈ 3.4 s plus a specialisation recompile, on the
first eval. Cheap fixes, pick one: pad the probe text to 2048 with `<|pad|>` and slice the boundary
mask back, or run it once at the end of training. Rank 0a makes it a one-off across all future runs.

> ★ **A real memory bug found while checking this, worth ~100 MB per eval (rank 6b).** I first assumed
> `evaluate`'s `@torch.no_grad()` flips `STORE_SSM_STATES_ADT_OUTV` and forces a second autotune
> sweep. **It does not** — and the reason is worse than a sweep.
> `mamba3_siso_combined.py:82` computes `needs_backward = any(ctx.needs_input_grad)`, and
> `morpheme/model/mamba3.py:168` passes `D=self.D`, a raw `nn.Parameter`. Measured on torch 2.13.0:
> under `torch.no_grad()` with a Parameter input, `ctx.needs_input_grad = (False, True)`, so
> **`needs_backward` stays `True` during evaluation.**
>
> The key does not change (no extra sweep), but every eval forward allocates and fills the entire set
> of backward-only buffers — `SSM_States`, `Out_v`, `DA_CS`, `Q_store`, `K_store`, `QK`, `Scale`,
> `Gamma` — and `save_for_backward`s 27 tensors that are then discarded. Estimated ≈25 MB per layer at
> B=2, L=2048, so **~100 MB across the four outer layers**, on a card measured at 6.7/8.2 GB. The same
> applies to `prefill` in the serving engine.
>
> ```python
> # morpheme/model/mamba3.py, in the use_kernel branch
> D=self.D if torch.is_grad_enabled() else self.D.detach(),
> ```
>
> `D.detach()` yields `ctx.needs_input_grad = (False, False)` (measured). Upstream's own
> `Mamba3.forward` has the identical pattern — *unverified whether upstream considers it a bug.* The
> ≈25 MB/layer figure is computed from tensor shapes, not profiled.

### 2.9 Checkpoints and the 9p boundary (rank 8)

`runs/overnight/last.pt` is **424 MB** (35.35M params × 4 B fp32 + 8 B of Adam moments), written by
`save_checkpoint` every `--ckpt-minutes 10`. `runs/` lives on `D:\`, so from WSL2 that write crosses
the 9p/DrvFs boundary while the training loop is blocked inside `torch.save`. Write to
`~/runs/<name>` on the ext4 VHD and rsync to `/mnt/d` afterwards (or on completion). The training
data is already on ext4 (`"data": "/home/noahs/data/local_mix"`), so this is the only remaining
cross-filesystem hot path.

Two cheap extras: `save_checkpoint` serialises `gen.get_state().tolist()` into JSON-able Python (a
5 056-element list) on every save — keep it as a tensor in the checkpoint dict instead; and consider
`--ckpt-minutes 20` for short local runs.

### 2.10 Data pipeline (rank 9)

`data/loader.py::sample_batch` builds the batch on the main thread inside the grad-accum loop, from a
`np.memmap`, with `np.stack` + `torch.from_numpy` — and the result is **not pinned**, which makes
`.to(device, non_blocking=True)` in `train_step` silently synchronous. Also `_window` does
`.astype(np.int64)`, doubling the host→device bytes for indices that `nn.Embedding` accepts as
`int32`.

Minimal fix that keeps the loader's semantics: a single background thread producing
`(ids, mask)` into a small queue of **pre-allocated pinned** buffers, `copy_` into the pinned buffer,
then `.to(device, non_blocking=True)` on a side stream. Keep the pinned tensor alive until the copy
is known complete (the standard non-blocking-H2D lifetime rule). At 32 KB per micro-batch the
bandwidth is irrelevant — the win is purely removing a synchronous host-side step from the critical
path 8 times per optimizer step.

---

## 3. Training recipe and data efficiency

The kernel work in §2 buys wall-clock. This section is about whether the wall-clock is being spent
on the right thing. Two of these are the highest-value items in the whole report and neither is a
speed change.

> **Recency note.** This is the section where the evidence is oldest, and it is worth being explicit
> about why. §3.1 (the loss-denominator bug) rests on reading this repo's code, not on a paper.
> §3.2's estimator is a 2018 derivation that this repo can *run on its own gradients* — the point is
> the measurement, not the citation. The remaining claims lean on 2024–2025 optimizer and schedule
> work, each flagged **background (pre-2026)** at the point of use, with §9 recording what 2026 work
> was found and what was not. Where a 2026 result supersedes an older one, the older one is dropped
> rather than kept alongside.

### 3.1 The gradient-accumulation loss denominator is wrong for SFT (rank 10)

`morpheme/train/train.py::_masked_ce` normalises **per micro-batch**:

```python
per = F.cross_entropy(..., reduction="none")
w = loss_mask.reshape(-1).float()
return (per * w).sum() / w.sum().clamp(min=1.0)
```

and `train_step` then does `(loss / args.grad_accum).backward()`. That is a **mean of means**:

```
computed:  (1/K) · Σ_k [ Σ_{i∈k} ℓ_i / n_k ]     correct:  Σ_k Σ_{i∈k} ℓ_i  /  Σ_k n_k
```

The two agree **only when every `n_k` is equal**. For pretraining (`loss_mask is None`) they are,
so pretraining is unaffected. For **SFT they are not** — `n_k` is the number of assistant bytes in
micro-batch `k`, which varies a lot window to window. The effect is that micro-batches with *few*
assistant bytes get up-weighted, which is exactly backwards.

This is the same bug that was found and patched across the ecosystem in late 2024
([Unsloth writeup](https://unsloth.ai/blog/gradient), [HF fix](https://huggingface.co/blog/gradient_accumulation)).

**Fix, which needs no pre-pass over the labels** (gradients are linear in the loss, so you can
normalise once at the end):

```python
# in train_step
total_w = torch.zeros((), device=device)
for _ in range(args.grad_accum):
    ...
    loss_sum, w_sum, stats = compute_losses_sum(...)   # reduction="sum", plus the mask weight
    loss_sum.backward()                                 # UNNORMALISED
    total_w += w_sum
for p in model.parameters():
    if p.grad is not None:
        p.grad /= total_w                               # exact, single normalisation
gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)   # AFTER normalisation
```

Two follow-ons: **log the loss with the same denominator** (otherwise the BPB curve is subtly wrong
even once the gradients are right), and **clip after normalising**, not between micro-batches. The
`p.grad /= total_w` loop is ~70 extra tiny kernels — use `torch._foreach_div_` on the grad list.

Note this composes with §2.1: `total_w` stays a device tensor, so no sync is introduced.

### 3.2 Is grad-accum 8 the right batch? Measure it, don't guess (rank 11)

**2026 first, because it changes the model.** [**2607.01487**](https://arxiv.org/abs/2607.01487)
*(2607, "How to Allocate Your Tokens?")* re-derives critical batch size with a **three-term law**
`L(N,M,K) = E + A/N^α + B/M^β + C/K^γ`, splitting the token budget into batch size M and steps K,
fitted on 246 runs. Its critique of McCandlish is structural and worth knowing before you lean on the
older fits: the McCandlish form `(K/K_min − 1)(D/D_min − 1) = 1` **implies an optimal batch size of
one**, which contradicts the AdamW empirics where the optimum grows with token budget. The three-term
law reproduces the critical-batch phenomenon *and* admits a non-trivial optimum:

| source | fitted optimal batch |
|---|---|
| **2607.01487 (2026)** | **M\* = 0.667 · D^0.566** |
| Li et al. Step-Law (pre-2026) | M\* = 0.58 · D^0.571 |
| DeepSeek-AI (pre-2026) | M\* = 0.086 · D^0.688 |

The 2026 re-derivation lands on ≈0.57 and **agrees with Step-Law while contradicting the DeepSeek fit**
(0.688) that I used below — so prefer the ≈0.57 exponent. It also reproduces the structural result
that the optimum scales with *data* size and essentially not with model size. Its practically useful
output is an **ε-suboptimal interval** (≤5 % of compute wasted) that is roughly **4× wide** and
consistent across 0.05B–1B. Two caveats before using it: the law is underparameterised and loses
accuracy at the edges of the fitted range, and its smallest fitted D is 2×10⁹ tokens — two orders of
magnitude above this repo's regime, so applying it downward is an extrapolation the paper explicitly
warns against.

A second 2026 result removes a shortcut you might otherwise take:
[**2607.27731**](https://arxiv.org/abs/2607.27731) *(2607)* runs a dedicated controlled test titled
*"The η/B ratio does not determine training dynamics"* and rejects it — **you cannot trade LR schedule
for batch schedule at constant η/B**. That supersedes the linear-scaling rule as a design principle.
The same paper derives a closed-form optimal batch *schedule* `B_t ∝ η_t / √(∫₀^T η · ∫_t^T η)`
depending only on the LR schedule's shape, worth 6–15 % compute efficiency, and reports that the
advantage is **inversely proportional to base batch size** — so a small batch like this one should
benefit most. But its Figure 12 shows the advantage **vanishes under purely bf16 training** (+0.1 %)
while fp32 gives −2.6 % and mixed fp32/bf16 gives −2.5 %. This repo is the mixed case (fp32 master
weights under bf16 autocast), so it should land near −2.5 % — close enough to the failure mode that it
warrants a measurement rather than an assumption.

**No 2026 paper re-measures the gradient noise scale itself at small model scale.** The nearest datum
is 2607.12360's log-log slope of 0.56 for gradient variance against ‖∇f‖ — neither the 1.0 a purely
multiplicative model predicts nor the 0.0 a purely additive (McCandlish) model predicts — but that is
on a 2-layer MLP, not an LM. Which is the argument for measuring it yourself, below.

For completeness, the pre-2026 fits evaluated at this model's compute budget (~2×10⁹ bytes in 12 h at
25–50 % MFU) put the optimal/critical batch at **1.3–2.9 ×10⁵ token-equivalents**:

* DeepSeek LLM [2401.02954](https://arxiv.org/abs/2401.02954): `B_opt = 0.2920·C^0.3271`. At
  C = 4.2×10¹⁷ → **B_opt ≈ 170 000 tokens**. (I checked this fit reproduces their own 7B run:
  predicted 9.2M vs actual 9.44M tokens, and η 4.25e-4 vs actual 4.2e-4.)
* Zhang et al. [2410.21676](https://arxiv.org/abs/2410.21676): `B* = 93.20·N^0.47` seqs → at
  N = 35M, ~254 000 tokens. Their structural result matters more than the constant: **CBS scales
  with data size (exp 0.47), essentially not with model size (exp 0.087)**.
* McCandlish et al. [1812.06162](https://arxiv.org/abs/1812.06162): `B_noise` ≈ 150 000 (run-average
  for LM).

At 32 768 bytes ≈ 7 300 token-equivalents, this run is **5–10× below** those numbers. So
`--grad-accum 8` is *not* wasteful in the "past diminishing returns" sense — the opposite.

**The catch, and it is the one that matters here.** Those are *run-averaged* figures.
McCandlish's own Table shows `B_noise` growing ~150× within a single run (Billion Word: 700 → 100 000).
A 1–12 h run at 35M never leaves the early regime, where the critical batch may be ~1–10k
token-equivalents — in which case accum 8 is costing 1.7–8× in data efficiency during the *entire*
run. There is also a direct contrary recommendation in the literature:
[2507.07101](https://arxiv.org/abs/2507.07101) says *"we recommend against gradient accumulation
unless training on multiple devices"*, and small batches with more steps give equal or better
per-FLOP performance.

**Resolve it empirically — it is nearly free, because the 8 micro-batch gradients already exist.**
McCandlish's two-batch estimator (App. A):

```
|G|²  = (B_big·|G_big|² − B_small·|G_small|²) / (B_big − B_small)
S     = (|G_small|² − |G_big|²) / (1/B_small − 1/B_big)
B_simple = S / |G|²        # EWMA this; the per-step ratio is badly biased
```

Take `|G_small|²` from the first micro-batch's gradient norm and `|G_big|²` from the accumulated
one — two extra `.norm()` calls per step, both device-side. Log `B_simple` next to `bytes_per_sec`
and set accumulation from it, ramping 2 → 4 → 8 as it climbs (modded-nanogpt does exactly this,
[PR #163](https://github.com/KellerJordan/modded-nanogpt/pull/163)).

If you do reduce accumulation, **raise β₂**: the token half-life of the second moment scales with
batch, so `β₂* = β₂^(B*/B)`. Going 32 768 → 4 096 bytes at β₂ = 0.95 wants β₂ ≈ 0.95^(1/8) ≈ **0.994**.

### 3.3 The multi-byte head is 33 % of the compute budget, and the MTP literature says it hurts at this scale (rank 14)

From §0.1, at the `local` preset the MBP head is **18.0 % (linear) + 14.9 % (dense attention) = 33 %
of the analytic FLOP budget**, for 10.5 % of the parameters. It is worse at the smaller preset —
at `pilot` (12.66M, the one the α sweeps used) it is **19.5 % + 22.0 % = 41.5 %**, because the head's
cost is set by `D0` and sequence length rather than by the main network's width. At `flagship` it
falls to 24.5 %. So: **two-fifths of the pilot's compute goes to the auxiliary head.** That is a lot
to spend on an auxiliary objective, and the evidence about that objective at this scale is not
encouraging:

* Gloeckle et al. [2404.19737](https://arxiv.org/abs/2404.19737) *(background, pre-2026)* Fig. 3, at
  300M / 600M / 1.3B / 3B / 6.7B / 13B: *"Multi-token prediction models are worse than the baseline for small model sizes, but
  outperform the baseline at scale."* MTP **underperforms at 300M** and the crossover sits between
  **1.3B and 3B**. This model is ~10× below the smallest size at which MTP was measured to hurt.
* The counter-evidence is real but far away: the same paper's §3.3 byte-level result (**7B**, n=8,
  +67 % MBPP) and the LCA paper itself (**373M**, and it *re-splits* the decoder rather than adding a
  head — 2608.15454 Table 2 shows FxT's 4 next-byte decoder layers become 2 next-byte + 2 multi-byte).
  This repo *adds* a 2-layer head on top of a full 2-layer decoder, at `loss_weight = 1.0`.
* **2026 does not overturn the scale picture — it sharpens it, and prices the training cost.**
  [**2608.05806**](https://arxiv.org/abs/2608.05806) *(2608, HiLP)* measures MTP at 1B: quality
  +0.44 HumanEval pass@1 over next-token (8.77 → 9.21), and **training throughput at 49 % of the
  next-token baseline** (126 278 → ~61 900 tok/s per GPU), because each future-token head must be
  evaluated against the LM logits every step. Latent-space prediction dominates token-MTP on *both*
  axes there — and its whole apparatus is dropped at inference, giving zero added latency. A 51 %
  training-throughput cost at 1B for +0.44 points is roughly consistent with the 33–41 % FLOP share
  measured here.
* **A 2026 result says the objective as written may not be learnable.**
  [**2602.06019**](https://arxiv.org/abs/2602.06019) *(2602)* shows offline multi-token cross-entropy
  over k ground-truth targets **cannot represent the joint distribution**: trained on "the zookeeper
  fed the panda bamboo" and "…the lion meat", independent sampling feeds the lion bamboo half the
  time. Their ablations favour hard (argmax) teacher targets, randomised k, causal masking across the
  MTP region, and — contradicting the usual advice — **no auxiliary next-token loss on prefix
  tokens**. `compute_losses` currently does exactly the thing they argue against: a plain auxiliary
  cross-entropy at `loss_weight = 1.0` alongside the next-byte loss.

Also worth noting: the runs already show `mbp_top1_acc` around 0.47–0.55, and `val_bpb` is the
metric that matters.

**This is a one-hour experiment and it is the biggest compute lever in the model.** Build a config
with `mbp.enabled = False`, run it at equal wall-clock against the current `local` preset, and
compare `val_bpb`. Three outcomes, all useful:

* MBP-off wins on `val_bpb` at equal wall-clock → drop it from the flagship, take the 1.4–1.5×.
* MBP-on wins → you have a byte-level datapoint below 300M that contradicts Gloeckle, which is worth
  writing down, and §2.2 becomes more valuable (you keep the head but pay 14.9 % less for it).
* A wash → keep it for the inference speculation (§5.1) and fix §2.2 to make it cheap.

Do **not** decide this from a partial loss curve: [2509.02046](https://arxiv.org/abs/2509.02046)
found optimizer/architecture rankings flip during LR decay, so run the full cooldown before judging.

### 3.4 Muon: 2026 reverses my first read — and finds a defect in the parameter split (rank 16)

`cf65c84` added `train/muon.py`. My initial assessment, based on 2025 work, was that Muon would
disappoint at this batch size. **The 2026 literature says close to the opposite**, and it also
identifies a concrete defect in how this repo assigns parameters to Muon. Four results, in order of
how much they change what you should do.

**(1) The parameter split is wrong for an SSM — and this is the highest-value item in §3.**

`split_muon_params` in `train/muon.py` sends **every** 2-D parameter except the embeddings to Muon.
For a Mamba model that is measurably the wrong split.
[**2608.03941**](https://arxiv.org/abs/2608.03941) *(2608, "Muon Meets Mamba")* runs Mamba-2 at 130M
— the closest published scale *and* architecture to this repo — and finds Muon's benefit is
**localised to the output projection**:

| Muon applied to | vs AdamW (OpenWebText, 1B tokens) |
|---|---|
| `out_proj` only | **−0.116 nats** (best) |
| `in_proj` only | worse than AdamW |
| both | worse than `out_proj` alone (3.128 vs 3.118) |

The mechanism is structural and transfers verbatim to `morpheme/model/mamba3.py`: `in_proj`
**row-stacks functionally distinct sub-projections**, and one global Newton–Schulz pass is ill-matched
to that. This repo's `Mamba3Mixer.in_proj` stacks **eight** of them —
`[z, x, B, C, dd_dt, dd_A, trap, angles]` (`mamba3.py:109-110`) — which is *more* heterogeneous than
the five in Mamba-2. Per-slice Newton–Schulz on `in_proj` recovers 0.024 nats over the merged version,
and a leave-one-out shows the data channel carries the whole effect while the gating channel is free
to drop. Everything else — `A_log`/`D`/`dt_bias`, RMSNorm, embeddings, LM head — stays on AdamW.
Corroborated independently by NVIDIA ([**2607.20548**](https://arxiv.org/abs/2607.20548), 2607,
App. D) for hybrid Mamba-2: excluding Conv1D weights from Muon gave a consistent ≈0.1 % gain.

**Concrete change to `split_muon_params`:** exclude `Mamba3Mixer.in_proj` (encoder *and* decoder) from
the Muon group, keep `out_proj`, and keep the Relation and MBP matrices — those are single-purpose and
are the case Muon was designed for. If you want the `in_proj` gain too, orthogonalise its eight row
blocks separately instead of as one matrix.

Two honest caveats from the same paper: the gain **shrinks with token budget** (0.116 → 0.063 nats
between 1B and 2.6B tokens; only 0.017 at 19× Chinchilla), so it is a *token-efficiency* effect that
is largest exactly in this repo's compute-constrained, near-single-epoch regime — and at 130M the
0.017-nat gap did **not** transfer to downstream zero-shot scores.

**(2) At this scale Muon wins, and it uses *less* memory than AdamW.**
[**2607.04033**](https://arxiv.org/abs/2607.04033) *(2607, OmniOpt)* tunes 24 optimizers per-scale at
60M / 130M / 350M / 1B. **Muon leads outright at 60M and 130M** (130M perplexity: Muon 21.81, SOAP
22.67, AdamW 23.18). Three numbers that matter on an 8 GB card, at 130M:

| | optimizer state | per-step optimizer time | 130M PPL |
|---|---|---|---|
| Muon | **0.250 GB** | 30.48 ms | **21.81** |
| AdamW | 0.500 GB | 2.31 ms | 23.18 |
| RMNP | **0.250 GB** | **4.63 ms** | 22.54 |
| SOAP | 2.214 GB | 110.4 ms | 22.67 |

Muon carries no second moment, so it **halves** optimizer-state memory — the opposite of what I
assumed. Their mechanism ablation is clean: removing Muon's orthogonalisation collapses quality (PPL
17.78 → 70.74), and replacing the diagonal second moment *with* Newton–Schulz surpasses AdamW
(→ 16.86). Spectral orthogonalisation is the load-bearing component, not the scalar tweaks. Muon is
also marginally *more* LR-robust than AdamW (24.2 % vs 27.7 % worst-case degradation under a 0.2×/5×
LR perturbation).

**This supersedes the cautious-AdamW suggestion I would otherwise have made.** OmniOpt's verdict on
that whole family: *"Most element-wise variants of AdamW do not survive a retuned baseline"* — Adan,
RAdam, NAdam, AdaBelief, Prodigy and MARS-AdamW all land in the weakest tier once AdamW is properly
tuned. **AdEMAMix and PSGD/Kron were not among the 24 benchmarked**, so they remain open questions
rather than refuted ones. **RMNP** is the dark horse for an 8 GB card: near-Muon quality at
near-AdamW cost and half AdamW's state (its ID, 2603.20527, comes from another paper's bibliography
and I did not verify it — treat as a lead).

**(3) A one-line weight-decay change worth +21.7 % at the nearest published scale.**
[**2607.23777**](https://arxiv.org/abs/2607.23777) *(2607, Muon-SW)*: replace the decoupled decay
factor `λ·η_t` with **`λ·η_t² / η_max`**. The Robbins–Monro argument is that an O(η_t) decay term
shifts the stationary point while an O(η_t²) term does not; empirically, under constant decay the RMS
weight norm collapses ~60 % from its post-warm-up peak, while under scaled decay it plateaus.
Measured speedup to equal loss: **21.7 % at 72M params**, 27.5 % at 265M, 29.4 % at 932M. The 72M row
is the closest published scale to this repo's 35M. In `train/muon.py::Muon.step`:

```python
# now:   p.mul_(1.0 - lr * wd)
# ->     p.mul_(1.0 - (lr * lr / lr_max) * wd)      # lr_max = the schedule's peak LR
```

Convergent independent evidence from Defazio's AdamC in
[2605.19095](https://arxiv.org/abs/2605.19095) *(2605)*, which arrives at the same `−γ_t²λz_t` rule.

**(4) Newton–Schulz has a free 1.5–2× in 2026, and a bf16 trap.**
[**2608.11612**](https://arxiv.org/abs/2608.11612) *(2608, Dion3)* introduces **Gram Newton–Schulz**:
because `polar(X) = (XXᵀ)^{−1/2} X`, iterate on the small n×n Gram matrix rather than the large
rectangular one, using only two rectangular matmuls in total. **The output is mathematically
identical** — this is not an approximation — and it gives 1.5× on dense matrices, 2× at aspect ratio
8. That is a drop-in replacement for `newton_schulz` in `train/muon.py`, which currently runs
`A = X @ X.T; B = b*A + c*(A@A); X = a*X + B@X` per iteration — 3 matmuls × 5 iterations = 15 per
matrix, ~1 000 tiny launches per optimizer step across ~70 matrices.

The trap is directly relevant to the bf16-neutrality constraint. The same paper reports their first
implementation underperformed *purely* because a change broke the compiler's fusion of
`W ← (1−η·wd)W − ηO`, creating extra bf16↔fp32 round-trips, and states the normalisation step **must
run in fp32**. Separately, 2607.20548 quantifies the floor: Newton–Schulz in bf16 has resolution
2⁻⁷ ≈ 0.0078 against fp32's ≈1.19e-7, so small singular values are numerical noise rather than signal.
`muon.py::newton_schulz` casts to bf16 at entry (`X = G.to(torch.bfloat16)`), matching the reference
implementations — but the `X / (X.norm() + eps)` normalisation should be fp32. NVIDIA's own settings,
for reference: 16 NS iterations with Polar Express coefficients, ε = 1e-7, β₁ = 0.9, **no Nesterov**
(it did not help), RMS factor √((1−β₁)/(1+β₁)) ≈ 0.2. This repo uses `nesterov=True` and 5 iterations.

**What the implementation already gets right:** decoupled weight decay on the 2-D parameters, and
`rms_scale = 0.2` with `√max(A,B)`, which is the Moonlight RMS-matching convention that makes sharing
one LR with AdamW legitimate. Do **not** mix in the Keller-Jordan convention (`lr = 0.02`,
aspect-ratio-only scaling) — the two are incompatible.

**What *not* to do:** tiled Newton–Schulz ([2606.27216](https://arxiv.org/abs/2606.27216), 2606,
HiMuon) will not pay here. Its end-to-end wins scale with matrix size — 4.7 % per step at Qwen3-0.6B,
8.3 % at 1.7B — and its isolated per-layer speedups need matrices ≥ 1024×2048. The largest matrix in
the `local` preset is 1536×512, at or below the tile size, so tiling has nothing to cut. Its
transferable advice is the opposite one, and it agrees with Dion3: **at small scale kernel-launch cost
dominates the optimizer, so capture the step in a CUDA graph.**

**Still measure it.** Wrap `opt.step()` in `torch.cuda.Event` timing and log it next to
`sec_per_step`. The `T·m/B` cost model still says the Newton–Schulz share is far higher here than in
any published Muon benchmark, because `B` is small. The 2026 evidence says the *quality* is worth it
at 35–130M; it does not say the arithmetic is free. Run rows 1–5 first so the measurement is clean.

### 3.5 Schedule and LR: what 2026 changes, and one schedule built for wall-clock budgets

The current `wsd_lr` — 10 % warm-up, flat, then `1 − √t` decay over the last 20 % to `0.1×` base — was
a good 2025 default. Three 2026 results bear on it, and one of them is a near-exact match for how this
repo actually runs.

**★ The run-length problem has a 2026 answer, and its setup is almost identical to yours.**
[**2607.10959**](https://arxiv.org/abs/2607.10959) *(2607, WSqD — "A Horizon-Free Learning Rate
Schedule")* was evaluated on a **213M LLaMA, bfloat16 mixed precision, single device with gradient
accumulation ×4, AdamW (wd 0.1, β = (0.9, 0.95), clip 1.0), 300-step linear warm-up** — that is this
trainer with a bigger model. It replaces WSD's flat stable phase with **`c₀/√(t + T₀)`**, keeping the
final linear decay (α = 0.2), and proves a minimax-optimal O(1/√T) last-iterate bound with a
horizon-independent scale.

The payoff is exactly the pain point `train.py` has: **WSqD's optimal base LR stayed pinned at 0.0015
across horizons of 15k, 30k, 45k and 60k steps, while WSD's optimum drifted downward as the horizon
grew.** This repo derives `total_steps` from a 5-step throughput probe and then drives the schedule
off wall-clock (`--max-steps 0`), so the effective horizon genuinely varies run to run — the exact
condition under which a horizon-sensitive peak LR costs you. Reported gains 1.01–1.05× when tuned at
10k steps, rising to **1.11× when tuned at 5k** (shorter pilot, bigger win). `T₀` ablation: 500 is too
small, 5000–10000 works, roughly the pilot length. Extending a run means discarding the cooldown
segment and resuming from the pre-decay iterate at step (1−α)T, reusing 80 % of the compute — which
is the `--init-from` workflow this repo already supports.

**★ Whether cooldown helps at all is conditional — and the condition flips between AdamW and Muon.**
[**2607.12360**](https://arxiv.org/abs/2607.12360) *(2607, "Same Loss, Same Noise, Opposite
Schedules")* shows the answer is a joint property of gradient-noise structure and whether the
optimizer normalises its update:

| noise structure | SGD/Adam-like (self-annealing) | signSGD / normalized (**Muon-like**) |
|---|---|---|
| multiplicative | **constant rate optimal, cooldown fraction 0** | cooldown required |
| additive / mixed | cooldown helps | cooldown helps |

A normalized update keeps unit scale and sits on a Θ(η²) floor that only decay removes; an
SGD-like step `ηĝ ∝ ηx` shrinks with the iterate and anneals itself. On their real task **Adam
preferred cooldown fraction 0 and ended 3.5× worse under strong cooldown**, while signSGD was the only
method cooldown helped. Their diagnostic is cheap and worth wiring in next to `B_simple` (§3.2):
regress per-microbatch gradient variance on ‖ĝ‖² locally in time to estimate (σ₀², σ₁²), then check
ρ = σ₀²/(σ₁²‖∇f‖²). **ρ ≪ 1 → multiplicative → cooldown matters little for AdamW but is mandatory for
Muon.** If you adopt Muon (§3.4), keep the cooldown; if you stay on AdamW, the 20 % cooldown is worth
re-testing rather than assumed.

Note the paper's own scope limit, which I am respecting: it explicitly does **not** reproduce the
interior cooldown fractions of 0.2–0.35 used at scale, and argues a stationary landscape-plus-noise
model structurally cannot produce an interior optimum. So it explains the *sign*, not the *shape*, and
it does **not** supersede the empirical 20 % figure — for which the standing reference remains Hägele
et al. [2405.18392](https://arxiv.org/abs/2405.18392) *(background, pre-2026; verified down to 33M
params, found benefits plateau at ~20 % and `1-sqrt` consistently beats linear)*. The current
`wsd_lr` shape is therefore still defensible.

**Schedule-free is better in general but not for short runs.**
[**2605.19095**](https://arxiv.org/abs/2605.19095) *(2605, ScheduleFree+)* beats WSD at every scale and
token budget tested from 120M to 2B; at 1000 tokens-per-parameter it matches a 45 % longer
linear-decay run. But it states plainly that **"no final loss advantage is seen for short duration
(20–100 TPP) runs"** against a tuned linear decay, because the non-steady-state portion of a short run
approaches 50 % of it. This repo's runs are short. It also requires fully-decoupled AdamC with weight
decay in the range **5–50**, not 0.1 — not a drop-in. **Skip it here**; revisit for the flagship.

One finding from that paper worth carrying regardless: **WSD is consistently outperformed by a
properly tuned linear-decay-to-zero at matched horizon**, and *"the suboptimality of the WSD schedule
against properly tuned baseline schedules is not emphasized in the literature."* The repo's
`min_ratio = 0.1` floor (LR stops at 10 % of base rather than reaching ~0) is on the wrong side of
that finding and is a free thing to change.

**Unchanged from my first read, and still correct:**

* **Warm-up 10 % is fine.** At these run lengths that is 270 steps (20 min) to 700 steps (7 h), inside
  the 300–500-step convention and matching H-Net's own 10 % linear warm-up. **No change recommended.**
  (Note §3.4: if you adopt Muon, sign-based and matrix optimizers generally want *longer* warm-up.)
* **LR is in range.** `pilot` at 8e-4 and `local` at 6e-4 bracket H-Net's 6.25e-4 at 760M.
* **Branch your cooldowns.** Checkpoint at the end of the stable phase and run several short cooldowns
  from one trunk instead of re-running whole trainings. `--init-from` already supports this, and it is
  precisely the workflow WSqD is designed around.

**Recommended order:** keep `wsd_lr` for now; adopt the `min_ratio → 0` change immediately (free);
try WSqD as a one-function change once rows 1–5 have stabilised throughput, since its whole value is
that you stop re-tuning the peak LR every time the wall-clock budget changes.

### 3.6 Expectation-setting for the pilot gate

Not an efficiency item, but it determines what "success" should mean and therefore how much
efficiency work is worth. Published byte-level-vs-BPE crossovers are far beyond a 12-hour local run:

* H-Net [2507.07955](https://arxiv.org/abs/2507.07955) *(background, pre-2026 — the only one of the
  five architecture papers that is not 2026)*: the 2-stage model *"overtakes the perplexity
  of a strong tokenized Transformer after just 30B training bytes"* — 15× a 12 h budget here.
* BLT [2412.09871](https://arxiv.org/abs/2412.09871): at fixed inference budget the byte-beats-BPE
  crossover sits at **~150B training bytes** (small budget) to ~1T (large).
* There is **no published BPB for a byte-level model at ~35M params** to anchor against.

A 12 h local run does ~2×10⁹ bytes. Pick a gate that does not require beating a BPE baseline — BPB
*trajectory*, boundary/word alignment (`boundary_on_separator_frac`, already logged), or a
morphology probe — and save the head-to-head for the flagship.

---

## 4. Environment: Triton, WSL2, the allocator

### 4.1 `TRITON_CACHE_AUTOTUNING=1` — the one-line win (rank 0a)

Triton 3.7 has an opt-in **persistent** autotune cache, and it has never been enabled here:
`find ~/.triton/cache -name "*.autotune.json"` returns **0** across ~820 cache entries / 532 MB. So
every fresh process re-runs the full sweep. Mechanism: `triton/runtime/autotuner.py:39` sets
`self.cache_results = (cache_results or knobs.autotuning.cache)`, where `knobs.autotuning.cache` is
the env var `TRITON_CACHE_AUTOTUNING` (`knobs.py:375`); `check_disk_cache`
(`autotuner.py:170-210`) then writes `{fn_name}.autotune.json` keyed on the Triton key, backend hash,
`fn.cache_key`, the tuning key, and every config's repr.

```bash
export TRITON_CACHE_AUTOTUNING=1    # persist autotune choices across processes
export TRITON_PRINT_AUTOTUNING=1    # first run only: prints the key tuple and tuning seconds
```

Measured cross-process on a standalone kernel: **3.05 s → 0.32 s**. Every Mamba-3 SISO autotuner is
eligible, because disk caching is skipped when *any* config has a `pre_hook`
(`autotuner.py:171-174`, *"We can't serialize prehooks, so just give up and run the benchmarks"*) and
**none of the mamba3 SISO kernels has one**. Note the exception: in the **Mamba-2 SSD** path
(`mamba_chunk_scan_combined`, which the dechunk EMA would use), **64 of 195 configs sit in `pre_hook`
kernels** — those can never be disk-cached and will re-tune once per process forever.

Scale of what is being avoided: the Mamba-3 SISO path has **234 configs across 7 autotuners**
(`mamba3_siso_fwd_kernel` 27, `_bwd_kernel_ddt_dtrap_dinput_states` 81, `_bwd_kernel_dzdo` 54, …),
search space `num_stages ∈ {1,2,3} × num_warps ∈ {2,4,8} × maxnreg ∈ {None,128,256}`. The `maxnreg`
axis was added to fix a Blackwell register-spill issue and tripled the sweep for everyone; on Ada
`maxnreg=None` is almost certainly right, so two-thirds of that sweep is pure cost.

If you want zero tuning at all, pin the winners — `autotuner.py:215` gates everything on
`if len(self.configs) > 1`, so a single-element `.configs` list goes straight to `self.configs[0]`
with no key lookup and no benchmark. Mutate the imported `Autotuner` objects at import time, and
**reuse the existing `Config` objects** rather than constructing fresh ones (several mamba3 configs
carry `CHUNK_SIZE` inside their kwargs — it is an autotuned tiling parameter, and the `dzdo` grid is
`lambda META: (cdiv(seqlen, META["CHUNK_SIZE"]), nheads, batch)`). Also clear
`o.early_config_prune`, or the prune function still runs.

### 4.2 The "SSD kernel re-autotunes for every chunk count" premise is wrong — and so was my first correction to it

`morpheme/model/dc.py:144` disables the kernel EMA with the comment *"the Triton SSD kernel
re-autotunes for every new chunk count"*, and the README repeats it. Two layers of correction, the
second of which reverses a recommendation I made earlier in this report.

**Layer 1: the stated mechanism does not exist.** No `@triton.autotune` `key=` list anywhere in
`mamba_ssm` contains `seqlen`, `L`, or `nchunks` — verified by enumerating all 43 `Autotuner`
objects. The autotune key is *only* the values of the named `key=` arguments plus `str(arg.dtype)`
for every argument that has one (`autotuner.py:216-222`).

**Layer 2: there *is* a per-M cost, but it is Triton's separate *compile*-cache specialisation, and
it is bounded and one-time.** Triton specialises integer arguments on divisibility-by-16 and
equal-to-1. Measured on this install: `2048 → ('i32','D')`, `1024 → ('i32','D')` — identical, so
2048→1024 does not even recompile — while `2047 → ('i32','')` does. So M crossing a multiple-of-16
boundary triggers a **single-config recompile**, not a sweep.

The on-disk cache proves the two-mechanism picture. Grouping compiled entries by specialisation class:
`_chunk_scan_fwd_kernel` has two classes of **exactly 11** entries (= its config count → two genuine
sweeps, on two key-sets) and two classes of **1** entry (a lone recompile of the already-chosen winner
in a new class). `mamba3_siso_fwd_kernel` shows the same at scale — **six classes of exactly 27** plus
eleven classes of 1–3. **Sequence-length changes only ever produced the small classes.**

And the counts are tiny: with `bucket = 1` (the old default) M can visit **5** compile classes across
1..2048; with **`chunk_bucket = 64` it visits exactly 2** — `('D','')` and `('D',1)`.

**Where the observed 2.7 s came from.** It is an artifact of the diagnostic script, which timed
`dc._ema` at M ∈ {300, 337, 374, 411, 448, 485} and **divided the total by 6**. Reconstructing:
M=300 paid a full cold-start sweep of every SSD forward and backward kernel (~150 configs, ~15 s),
M=448 paid one specialisation recompile (~3 s, because 448 is divisible by 16 while the others are
not), and the remaining four were free. `(15 + 3) / 6 ≈ 3`. *This split is a reconstruction — the
diagnostic's per-call numbers were never logged, and the GPU was busy with a live run, so I did not
re-measure. §8 has the command to replace it with a measurement.*

**Layer 3, and this reverses rank 6.** I earlier recommended re-enabling `use_kernel` on the strength
of ATDC 2605.30080 §III.C.2 (*"can be efficiently computed via parallel scan kernels by reformulating
the recurrence as a linear SSM"*). That advice was wrong for this particular EMA. `dc.py:192` calls
`mamba_chunk_scan_combined` with **`dstate = 1`** (the EMA is a scalar-decay recurrence), while the
SSD kernels tile the state dimension with `BLOCK_SIZE_K ∈ {32, 64}`. **At `dstate = 1`, ≥97 % of every
dot tile is mask.** The chunked pure-PyTorch scan is the right tool here; ATDC's claim is about the
general case, not about a one-dimensional state.

**So: leave `use_kernel = False`, and fix the comment** — the reason is arithmetic waste at
`dstate=1`, not autotuning. The §2.8 recommendation that survives is the cheap one: raise
`_ema_chunked`'s block size `C` from 64 to 256 to shorten the Python loop. And if the kernel path is
ever revisited, note that `TRITON_CACHE_AUTOTUNING=1` covers only **131 of its 195 configs** — the 64
that live in `pre_hook` kernels (`_chunk_scan_bwd_dc/dx`, `_chunk_cumsum_bwd`,
`_chunk_state_bwd_db/dx/ddAcs`, `_chunk_scan_chunk_state_bwd_dx`) re-sweep every process by design
(`autotuner.py:171-174`).

**What still flips the autotune key in a normal run:** `STORE_SSM_STATES_ADT_OUTV`
(= `needs_backward`), `RETURN_FINAL_STATES`, `HAS_INITIAL_STATES`, `IS_VARLEN`, and dtype. In practice
that is **train ↔ generate**, not train ↔ eval — see the box in §2.8b for why evaluation does *not*
flip it, and for the ~100 MB that costs instead. Run once with `TRITON_PRINT_AUTOTUNING=1` to see the
key tuple and confirm.

### 4.3 Occupancy: the encoder/decoder run 24 CTAs on a 34-SM GPU

`mamba3_siso_fwd.py:630` launches `grid = (nheads, batch)` and each program loops
`for chunk_idx in range(num_chunks)` — the forward kernel is **sequence-serial per (head, batch)**.
Instantiating this repo's exact config (`d_model=384, d_state=64, headdim=64, expand=2, ngroups=1`)
gives `d_inner=768`, **`nheads=12`**, so at batch 2:

> **grid = (12, 2) = 24 CTAs on 34 SMs — 70 % of a single wave, with no cross-CTA latency hiding.**

Upstream's own test runs batch 16 × nheads 32 = 512 CTAs. Two levers:

* **Raise the micro-batch** (rank 3). Batch 3 → 36 CTAs clears one wave; batch 4 → 48; batch 6 → 72 ≈
  two waves. This is a *second, independent* argument for §2.3 and it is nearly free up to 34.
* **Halve `headdim` 64 → 32** → `nheads` 24 → 48 CTAs at batch 2. This changes the model (per-head
  parameters `dt_bias`, `B_bias`, `C_bias`, `D` double), so it is an architecture A/B, not a
  numerics-neutral change. **[unverified — no benchmark was run; treat as a hypothesis]**

Do **not** raise `chunk_size` past 64. The kernel materialises `s_block = tl.dot(q, kᵀ)` as a
`[CHUNK_SIZE, CHUNK_SIZE]` fp32 register tile alongside a persistent `[HEADDIM_V, HEADDIM_QK]` fp32
accumulator (`mamba3_siso_fwd.py:411,417`); at CS=128 `s_block` alone is 64 KB against Ada's
101 376 B per-block shared-memory cap — which is exactly the `OutOfResources` in mamba issue #464 on a
4090. Non-power-of-2 chunk sizes also produce NaN (issue #449). `Mamba3Cfg`'s current
`chunk_size = headdim = d_state = 64` is the paper's own `C = P = N` rule (2603.15569 §3.3) and
should stay.

The upstream author's own diagnosis of this exact situation
([state-spaces/mamba#355](https://github.com/state-spaces/mamba/issues/355)): *"Mamba2 is written
mostly in Triton, so there's a lot of CPU overhead if the layer is so small. Two ways around:
(1) CUDA graph (or torch compile) (2) use a large model."*

### 4.4 Mamba kernel issues worth knowing about

* **`MAMBA_SKIP_CUDA_BUILD=TRUE` is a no-op on this tree.** PR #977 (in the pinned commit `e9594ce`)
  renamed it: `setup.py:41-42` now reads `MAMBA_FORCE_BUILD` / `MAMBA_KEEP_CUDA_BUILD`, both
  defaulting to false, so the CUDA build is already skipped. `README.md` and `cloud/bootstrap.sh`
  should drop the variable (harmless, but misleading).
* **`causal_conv1d` is irrelevant to Mamba-3** and its absence costs nothing. The paper (§3.4) states
  the trapezoidal discretisation plus B/C biases *obviate* the short causal convolution, and Table
  5(a) shows adding it back **hurts** (15.72 → 15.85). `mamba3.py` contains no conv.
* **Issue #1015 (open):** 32-bit pointer overflow past ~61K tokens in the chunked-scan kernels —
  returns finite, plausible, **silently wrong** output. Safe here (batch × seqlen = 4 096), but the
  flagship at batch × 4096 must stay under that bound.
* **Issue #1017 (open):** `mamba3_siso_combined` deviates 3–7.5 % relative from its own reference
  *even in fp32*; CI's `rtol=1e-1` masks it. Relevant if a kernel-vs-reference test in `tests/` ever
  disagrees — that may be upstream, not this repo.
* **PR #997 (merged, = the pinned commit):** `num_stages > 1` silently corrupts
  `mamba3_siso_fwd_kernel` on Blackwell. **Ada is not affected**, but note it before the flagship run
  if that hardware changes.
* **`torch.compile` cannot capture the Mamba call.** `ssd_combined.py` has no `torch.library` /
  `custom_op` registration, so Dynamo graph-breaks by construction; issue #369 (`fullgraph=True`
  fails on the `dt_limit` tuple arg) and #740 (dynamic seqlen → `InductorError: KeyError: 'op3'`)
  confirm it. See §4.7.
* **Upstream's `Mamba3.step()` would raise here** — `mamba3.py:336` hard-asserts on
  `mamba3_step_fn`, and `nvidia-cutlass-dsl` is absent. `morpheme/model/mamba3.py` already
  sidesteps this with its own pure-PyTorch `step()`, which was the right call. But there is a
  **tested Triton alternative** upstream at
  `mamba_ssm/ops/triton/mamba3/mamba3_siso_step.py:233` (`mamba3_siso_step`, 9 autotune configs),
  currently called only from tests. Wiring it into `Mamba3Mixer.step` is the obvious GPU-decode
  speedup for `serve/engine.py` — see §5.

### 4.5 WSL2: what is real, what is folklore, and what it costs

Measured on this machine (driver 610.74, CUDA 13.3, WDDM 3.2) while a run was in flight:

* **Not throttled.** `nvidia-smi -q -d PERFORMANCE`: `SW Power Cap: Not Active`, `HW Slowdown: Not
  Active`, `HW Thermal Slowdown: Not Active`. SM clock **2730 MHz** against a 2535 MHz official boost
  — i.e. boosting *above* spec — at 67–69 °C and 88–90 W of 160 W. Maximum boost plus no throttle
  flags plus 90 W means the SMs are idle between launches. This closes the question: **the 91 W is
  not a power cap, it is an empty pipeline.**
* **`utilization.gpu` is not occupancy.** The
  [nvidia-smi manual](https://docs.nvidia.com/deploy/nvidia-smi/index.html) defines it as *"Percent of
  time over the past sample period during which one or more kernels was executing"* — a duty cycle.
  76–89 % duty at 90 W and 10 % MFU is the signature of many small kernels at low occupancy.
* **How much is WSL2's fault?** NVIDIA's own worst case bounds it. Their
  [WSL2 performance post](https://developer.nvidia.com/blog/leveling-up-cuda-performance-on-wsl2-with-new-enhancements/)
  reports Blender *"within 1 %"* of native Linux, and GenomeWorks — chosen as a deliberate
  short-kernel worst case — at *"equal to or more than 90 % of the native speed"*. The mechanism is
  stated plainly: *"all the GPU operations are serialized through VMBUS and sent to the host kernel
  interface… When the GPU workload submitted by an application is not long enough to overcome that
  latency, a performance gap between native Linux and WSL2 will start to appear."* So the pure
  launch-overhead penalty is bounded at roughly **≤10 %**. **No published per-launch microsecond
  delta for WSL2 vs native exists** — anyone quoting one is guessing.
* **HAGS is already enabled** (dxdiag: `Hardware Scheduling: Enabled:True`, WDDM 3.2). NVIDIA's
  single strongest WSL2 recommendation is already satisfied; there is no win left there.
* **The display cannot be moved off the card.** The CPU is an i7-14700**F** — no iGPU, one adapter,
  `Display Attached: Yes` at 3840-wide, with ~35 Windows processes holding GPU contexts against
  7169/8188 MiB. This is the strongest concrete argument for the native-Linux dual boot: booting to a
  text console reclaims that VRAM and the scheduling contention outright.

**Verdict: fix the workload before fixing the OS.** The ≤10 % WSL2 launch penalty is real but small
next to the 40 syncs (§2.1) and the 24-CTA occupancy (§4.3). The dual boot is worth it for the
*second-order* wins — reclaimed VRAM, a working `expandable_segments`, real OOM instead of silent
sysmem fallback, lockable clocks for reproducible timing — not for the headline number.

### 4.6 `expandable_segments` under WSL2: the actual mechanism, and what to use instead

The README's *"CUDA driver error: device not ready"* is now fully explained.

`expandable_segments` routes PyTorch's allocator through the CUDA VMM APIs. In
`c10/cuda/CUDACachingAllocator.cpp`, `ExpandableSegment` calls `cuMemAddressReserve` → `cuMemCreate`
→ `cuMemMap` → **`cuMemSetAccess`**, each wrapped in `C10_CUDA_DRIVER_CHECK`, which raises
`TORCH_CHECK(false, "CUDA driver error: ", err_str)`. **`CUDA_ERROR_NOT_READY` = 600, and its
`cuGetErrorString` text is exactly `"device not ready"`.** So the message is a VMM driver call
returning 600, printed verbatim.

There is a matching report:
[cuMemSetAccess failed to create GPU mapping](https://forums.developer.nvidia.com/t/cumemsetaccess-failed-to-create-gpu-mapping/377584)
— `cuMemSetAccess()` intermittently returns `CUDA_ERROR_NOT_READY` **on WSL2** while growing a
VMM-backed pool toward the device memory limit, with retry succeeding, and a responder confirming the
same code is clean on native Linux. No NVIDIA staff reply.

**VMM *is* supported on WSL2** — measured `CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED`
(attr 102) = **1** on this box. That is why PyTorch never prints "not supported on this platform": it
compiles as Linux, cannot tell it is on WSL2, enables the feature, and then eats a driver-level
failure. The CUDA-on-WSL guide's limitations section does not list VMM, so this is an **undocumented
driver bug, not a documented restriction** — and the capability check will *not* predict it:

```bash
python3 -c "import ctypes;l=ctypes.CDLL('libcuda.so.1');l.cuInit(0);d=ctypes.c_int();l.cuDeviceGet(ctypes.byref(d),0);v=ctypes.c_int();l.cuDeviceGetAttribute(ctypes.byref(v),102,d);print('VMM supported:',v.value)"
```

It fires specifically at ~87 % VRAM occupancy — the near-limit regime the reports describe. Keep it
off under WSL2; `cloud/launch.py` is right to set it for the H100.

**A second hypothesis for "batch 4 collapsed to 13 kB/s" that deserves testing.** The README
attributes it to allocator fragmentation. There is a competing explanation:
[microsoft/WSL#11050](https://github.com/microsoft/WSL/issues/11050) — *"WSL2 CUDA Does Not Respect
`CUDA - Sysmem Fallback Policy`"*: with the Windows policy set to "Prefer No Sysmem Fallback" and
rebooted, an over-sized allocation on the GPU still *"executes successfully"* instead of OOMing, and
*"just uses the slow fallbacked CPU memory, which causes the inference to be really slow."* Open, no
vendor response. WSL2 runs through WDDM (`nvidia-smi -q` reports `Driver Model: WDDM`), and since
driver 536.40 the Windows driver silently spills CUDA allocations into system RAM rather than failing
([NVIDIA KB 5490](https://nvidia.custhelp.com/app/answers/detail/a_id/5490/)).

**A 6.5× slowdown with no OOM is exactly what a PCIe-backed spill looks like.** Distinguishing test:
re-run batch 4 and watch Windows Task Manager's *"Shared GPU memory"* — `nvidia-smi` inside WSL will
not show it (`memory.total` reads a flat 8188 MiB). If shared memory climbs, it was a spill, not
fragmentation, and no allocator flag will fix it — only using less memory will.

Other documented WSL2 limits worth knowing, all from the
[CUDA on WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html):
Managed Memory *"will not [be supported] for the foreseeable future"* and *"could see reduced
performance and high system memory usage"*; *"Concurrent CPU/GPU access is not supported"*
(measured attr 89 = 0); *"Pinned system memory … availability for applications is limited"*.
**`cudaHostRegister` does not work on WSL2**
([forum](https://forums.developer.nvidia.com/t/cudahostregister-not-supported-on-wsl2/279429), with
an NVIDIA reply) — but `cudaHostAlloc` does, so `pin_memory=True` and `torch.empty(..., pin_memory=True)`
are fine and §2.10 is unaffected. `cudaMallocAsync` / memory pools are supported (measured attr
115 = 1). CUDA IPC is broken, which matters only if this ever goes multi-process.

**`CUDA_MODULE_LOADING=LAZY` is a no-op here** — lazy loading has been the default on all platforms
since CUDA 12.3 and this box is on CUDA 13.3
([docs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/lazy-loading.html)).
Do not bother setting it.

### 4.7 `torch.compile`: what it can and cannot reach in this model

The strategy in §2.7 (compile the leaves) turns out to be what you get anyway, for a hard reason:
**`mamba_ssm` has no `torch.library` / `custom_op` registration**, so Dynamo graph-breaks at every
`mamba3_siso_combined` call by construction (mamba issues #369, #740). `flash_relation` is a
`torch.autograd.Function` with raw Triton launches, which also breaks the graph.

So `torch.compile(model, dynamic=True)` degenerates into compiling exactly the elementwise regions
*between* the kernels — which is the desired outcome, reached automatically. That makes `--compile`
more attractive than before `cf65c84`, because `chunk_bucket = 64` now bounds the number of distinct
M values (~32 possible, realistically ~8–10 in a converged run).

Practical notes for trying it, several of which changed in the 2.11–2.13 line:

* **Do not reach for `dynamic=True`.** The 2.13 dynamic-shapes manual heads that section
  **"torch.compile (dynamic=true) (Not recommended)"** — *"this option is not recommended due to it
  being error prone"* / *"for testing only"*, because it makes *every* dim symbolic including the
  fixed batch and seq. Annotate instead:

  ```python
  torch._dynamo.mark_dynamic(hc, 1, min=128, max=2048, hint_override=<your median M>)
  ```

  `mark_dynamic` must be called **outside** compiled code (it is `@forbid_in_graph`), and it does
  *not* override `force_parameter_static_shapes` (default True) — the flag that beats those is the
  env var `TORCH_COMPILE_DYNAMIC_SOURCES`. Note `strict=` lives on `mark_unbacked`, not on
  `mark_dynamic`; `mark_dynamic` hard-errors if the dim gets specialised, and `maybe_mark_dynamic` is
  the forgiving version.
* ★ **`hint_override` (new in 2.9) is the knob that matters, not `dynamic`.** Inductor emits *one*
  kernel parameterised by a runtime `xnumel`, but `XBLOCK`/`R0_BLOCK` are `tl.constexpr` chosen
  **once** from a single size hint (`codegen/triton.py`: `size_hint = next_power_of_2(int(numel_hint))`).
  With M backed, that hint is whatever the *first* batch produced — and at initialisation the router
  fires on ~50 % of bytes, so the first M is ~1024 while steady state is ~620. Tiles get tuned for the
  wrong problem, and the acknowledged upstream defect is still open
  ([pytorch#148842](https://github.com/pytorch/pytorch/issues/148842), *"We should use max size
  instead of hint size when autotuning"*). Verify with `TORCH_LOGS=output_code` and read the
  `size_hints=` line off the generated kernel against your real M distribution.
* **Raise the recompile limit, and know the failure mode.** `torch._dynamo.config.recompile_limit`
  (renamed from `cache_size_limit` in 2.7; the old name is a live alias) defaults to **8**. On hitting
  it, `torch/_dynamo/cache_size.py` says *"all future compilation attempts will result in the function
  being skipped (run eagerly) … all nested function calls WILL be skipped."* So compile looks fine for
  ~8 steps and then **silently reverts to eager forever**. With ~10 live M buckets you will hit it.
  Set `fail_on_recompile_limit_hit = True` while tuning so it is loud.
* 🚫 **Avoid `mode="reduce-overhead"` and `mode="max-autotune"`.** Both set
  `triton.cudagraphs: True`, and CUDAGraph Trees *"re-record CUDAGraph for every unique shape of an
  input tensor"* — with a bucket ladder that means repeated re-recording (warnings from step 9 via
  `cudagraph_dynamic_shape_warn_limit = 8`, then churn toward
  `cudagraph_unexpected_rerecord_limit = 128`) and a memory-pool blow-up on 8 GB. Use
  **`mode="max-autotune-no-cudagraphs"`**, or set
  `torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True`, or whitelist sizes with
  `triton.cudagraph_capture_sizes`.
* `torch._inductor.config.graph_partition` now defaults to **True** in 2.13, which automatically
  splits graphs around cudagraph-unsafe ops (device copies, control flow, unbacked symints) — the
  automated version of partial capture. Worth knowing it is already on.
* Diagnose with `TORCH_LOGS=recompiles,graph_breaks` (both on by default) and
  `TORCH_LOGS=recompiles_verbose,dynamic,output_code` (off by default, must be named). A
  specialisation looks like `eval Eq(s77, 620) [guard added]`. For a full picture,
  `TORCH_TRACE=/tmp/trace python -m morpheme.train.train … && tlparse /tmp/trace -o out/ --latest`.
  Dozens of graph breaks per step is *expected* here — one pair per Mamba layer and per Relation
  layer.
* One inductor detail that quietly matters: `shape_padding` (default True) **does nothing for M** —
  `pad_mm.py::get_padded_length` returns 0 for any `SymInt` (*"we don't pad x if it is symbolic"*),
  and comprehensive stride padding is skipped for tensors carrying M because `pad_dynamic_shapes`
  defaults False. Another argument for padding M yourself in Python, which `chunk_bucket` already does.
* `train.py` trains through `fwd` (compiled) but evaluates through `model` (eager). That is fine, but
  be deliberate about it — compiling both doubles the number of live graphs against the recompile
  limit.

### 4.8 Filesystem and `.wslconfig`

Measured on this machine, ext4 VHDX vs the 9p `/mnt/d` mount (`msize=65536`, not virtiofs):

| operation | ext4 | 9p (`/mnt/d`) | penalty |
|---|---|---|---|
| sequential write 256 MB (fsync) | 1.6 GB/s | 126 MB/s | **12.7×** |
| sequential read 256 MB (warm cache) | 9.3 GB/s | 222 MB/s | 42× (overstated — warm cache) |
| create 2 000 small files | 0.03 s | 6.49 s | **216×** |

The training **data is already on ext4** (`/home/noahs/data/local_mix.*`) — good. The remaining hot-path
exposure is bigger than the checkpoints: the live process had written **3.8 GB** to
`runs/*/stdout.log` and `log.jsonl` across 9p, because `train.py::log` does `log_f.flush()` **and**
`print(..., flush=True)` on every logging record. Move the whole run directory to `~/runs` and rsync
on completion; that covers §2.9's 424 MB checkpoints and the logs in one change.

There is **no `.wslconfig`**, so the VM defaults to 15 GiB (50 % of 31.8 GB) plus 4 GiB swap. Two
settings are worth adding: `memory=24GB` (stops the VM ballooning against the host) and **`swap=0`**
(gives a clean OOM instead of a silent disk-swap stall). `guiApplications=false` would disable WSLg,
which is currently running — plausibly a small VRAM saving on an 8 GB card, but **[unverified; no
measurement exists]**.

One observed mechanism, flagged but not costed: the training process holds an open fd on
`/usr/lib/wsl/drivers/.../nvcubins.bin`, a **93.6 MB cubin store on a 9p mount**. Under lazy module
loading, kernel binaries are pulled from there on first touch, across 9p. That should be a one-time
warm-up cost and does **not** explain steady-state MFU.

Finally, if you want Nsight: `ncu` on WSL2 needs a Windows-side switch — NVIDIA Control Panel →
Desktop → Enable Developer Settings → Developer → Manage GPU Performance Counters → "Allow access to
all users" — otherwise it fails with *"a driver resource was unavailable or the user does not have
permission"*. `nsys` on WSL2 has had GPU→CPU timestamp problems that silently drop kernels from the
timeline, so prefer `torch.profiler` (which `profile_step.py` already uses).

### 4.9 The allocator, correctly configured — and one 30-second check that could reopen `expandable_segments`

**First: the variable was renamed.** PyTorch 2.10+ uses **`PYTORCH_ALLOC_CONF`**;
`PYTORCH_CUDA_ALLOC_CONF` is a back-compat alias
([2.13 CUDA semantics](https://docs.pytorch.org/docs/2.13/notes/cuda.html#optimizing-memory-usage-with-pytorch-alloc-conf)).
Undocumented gotcha from `c10/core/AllocatorConfig.cpp`: the parser walks
`{PYTORCH_CUDA_ALLOC_CONF, PYTORCH_HIP_ALLOC_CONF, PYTORCH_ALLOC_CONF}` and **parses only the first
one it finds**. Never set both — the legacy name silently wins.

**`roundup_power2_divisions` is the right tool for this workload, and it is off by default.**
Rounding happens in `round_size()` *before* pool selection and the free-block search. At the default
512 B granularity, a `[M, d]` bf16 tensor steps by `2d` bytes per unit of M — with `d = 512` that is
1024 B per step, so **every M in 250–1000 produces a distinct, never-reused block size**.
`roundup_power2_divisions:8` collapses each octave to 8 buckets at ≤12.5 % over-allocation.
Use the **scalar** form, not the `[256:1,512:2,…]` dict: interval index 0 covers everything below
2 MB, so the dict gives no extra resolution for sub-2 MB tensors, and the scalar form also applies to
the small pool.

**Two knobs that are silent no-ops by default** — worth knowing before you waste an evening:

* `garbage_collection_threshold` does nothing unless `per_process_memory_fraction < 1.0`.
  `CUDACachingAllocator.cpp` gates GC on `allowed_memory_maximum.has_value()`, and that is only set
  when the fraction is below 1.0. Default is 1.0, so **GC never runs**. Pair them.
* `max_non_split_rounding_mb` does nothing unless `max_split_size_mb` is set.

**`max_split_size_mb` is probably harmful here** — its minimum accepted value is
`large_segment_size_mb` (20 MB), far above the varying tensors, while `get_free_block` starts refusing
to hand sub-20 MB requests a ≥20 MB cached block. Leave it unset.

Starting point, to sweep against `--bucket`:

```bash
export PYTORCH_ALLOC_CONF=roundup_power2_divisions:8,per_process_memory_fraction:0.9,garbage_collection_threshold:0.8
```

Judge it on `torch.cuda.memory_stats()`: **`num_alloc_retries`** (*"failed cudaMalloc calls that
result in a cache flush and retry"* — non-zero means fragmenting; this is the primary signal),
`requested_bytes` vs `allocated_bytes` (the exact metric for tuning `divisions`), and
`inactive_split_bytes` (the sliver counter). Watch `large_pool` and `small_pool` separately.

That said — **the bigger lever is upstream**. `chunk_bucket = 64` already prevents most of these
slivers from existing, and rank 12 (pinning M) eliminates them entirely. Allocator rounding recovers
slivers; shape stability prevents them.

> ★ **Run this before accepting that `expandable_segments` is unusable.**
> [pytorch#192330](https://github.com/pytorch/pytorch/issues/192330) (opened 2026-08-06, **open**,
> and it explicitly lists `PyTorch 2.13.0+cu126` as affected) isolates the WSL2 failure to a single
> device flag rather than to VMM support in general:
>
> | `gpuDirectRDMACapable` (attr **110**) | `cuMemCreate` | `cuMemMap` | `cuMemSetAccess` |
> |---:|---|---|---|
> | 0 | ✅ | ✅ | ✅ |
> | 1 | ✅ | ✅ | `CUDA_ERROR_UNKNOWN (999)` |
>
> PyTorch sets that capability unconditionally at `CUDACachingAllocator.cpp:583-591`. The reported
> failures are on **Ampere workstation cards (A4000, A5000)**; an **RTX 3090 reported attr 110 = 0 and
> worked fine**. Nobody has reported an Ada consumer card either way. §4.6's measurement here (attr
> **102** = VMM supported = 1) does not settle it — 102 and 110 are different attributes.
>
> ```bash
> python3 -c "import ctypes;l=ctypes.CDLL('libcuda.so.1');l.cuInit(0);d=ctypes.c_int();l.cuDeviceGet(ctypes.byref(d),0);v=ctypes.c_int();l.cuDeviceGetAttribute(ctypes.byref(v),110,d);print('GPUDirect-RDMA-with-VMM:',v.value)"
> ```
>
> If it prints **0**, retry `expandable_segments:True` — it is by far the best fix for continuously
> varying shapes, and most of this subsection becomes moot. If it prints 1, the README's guidance
> stands and now has a mechanism behind it. Either way this is 30 seconds and it resolves a
> constraint the whole memory plan is built on.

Two more allocator notes for completeness. `pinned_use_cuda_host_register:True` should stay **False**
— it routes the host allocator through `cudaHostRegister`, which does not work on WSL2 (§4.6).
And `backend:cudaMallocAsync` probably works under WSL2 (it is WDDM, and memory pools are supported
— measured attr 115 = 1) but is the wrong choice: it cannot be changed at runtime, it *ignores*
`max_split_size_mb`, `roundup_power2_divisions` and `garbage_collection_threshold`, it zeroes most
`memory_stats()` fields, and it **hard-errors** on `_record_memory_history` and `_snapshot` — i.e. it
costs you every diagnostic above.

### 4.10 Combo kernels: PyTorch 2.10 shipped a fix aimed at exactly this problem, and it is off by default

The headline performance item of **PyTorch 2.10 (Jan 2026)** is *"Reduced kernel launch overhead with
combo-kernels horizontal fusion in TorchInductor"* — combining multiple **independent** operations
with no data dependency into a single kernel, as opposed to Inductor's usual vertical
producer–consumer fusion. The release notes say it is *most impactful when models have many small
independent ops*. That is a precise description of §0.2's ~21 600 launches per step.

It is **not enabled by default**. From `torch/_inductor/config.py` (main, Aug 2026):

```python
combo_kernels = False                      # "(Experimental)"
combo_kernels_autotune = 1                 # 0 disable, 1 enable except foreach, 2 all
combo_kernel_foreach_dynamic_shapes = True
benchmark_combo_kernel = False             # benchmark and only keep variants with a real gain
```

So the change is:

```python
import torch._inductor.config as ind
ind.combo_kernels = True
ind.benchmark_combo_kernel = True          # keep only fusions that actually win
```

This composes well with §2.7's "compile the leaves" strategy: the H-Net forward has many *independent*
elementwise chains at the same point in the graph — the eight-way `torch.split` in
`Mamba3Mixer._preprocess`, the two `RMSNorm`s on `B` and `C`, `p1`/`p2`/`info` in `FullRelation`,
the two `SwiGLU` halves — all of which are horizontal-fusion candidates that vertical fusion cannot
touch. Turn it on together with `benchmark_combo_kernel=True` so a fusion that does not pay is
discarded rather than trusted.

Two more 2026 additions worth knowing:

* **`torch.nn.attention.varlen_attn()` (new in 2.10)** — a ragged/packed-sequence attention op with
  forward *and* backward, `torch.compile`-able, A100-or-newer, BF16/FP16. This is the natural 2026
  primitive for the variable-chunk-count problem, and the direction H-Net's own authors took
  (`cu_seqlens` packing, §2.6). It is the thing to reach for if you ever decide bucketing's padding
  waste has become the binding cost — most likely on the H100 flagship rather than here.
* **`torch.cond` became CUDA-Graph-capturable in PyTorch 2.12** (2026-05-19) via CUDA 12.4 conditional
  IF nodes: *"Previously, data-dependent control flow forced fallback to CUDA graph trees because
  branching was evaluated on the CPU."* Relevant to rank 12 — but note the caveat in the same notes:
  *"This currently works with the eager and cudagraphs backends; Inductor support is planned for a
  future release."*

**And a 2026 bound on how much CUDA graphs can buy.** [**2606.05495**](https://arxiv.org/abs/2606.05495)
*(2606, SET)* measures residual scheduling overhead **after** CUDA Graphs on an **RTX 3090**: 45.32 %
of total execution time under static batching, 33.36 % with a queue model, 29.83 % with their
scheduler — and *"at batch size 1, scheduling overhead accounts for 45–51 % of the total execution
time."* Event Tensor makes the structural point behind it: *"CUDA Graphs preserve kernel boundaries
and thus cannot expose inter-kernel parallelism."* Graphs remove launch *submission* cost; they do not
merge kernels. Combo kernels do. That is why rank 0d sits above rank 12 in the table despite being a
one-line change.

> **A framing worth internalising: Ada is a systematic blind spot in 2026 kernel engineering.** Every
> 2026 fused-kernel library and paper I checked gates on Hopper sm_90, Blackwell sm_100/sm_120, or AMD
> gfx1250 — FlashQLA (sm_90/100/120), ThunderKittens 2.0 (*"mainly built and tested for Hopper and
> Blackwell"*, Ampere support dropped), CODA ([2605.19269](https://arxiv.org/abs/2605.19269), Hopper
> ping-pong/TMA), ComFuse ([2608.03537](https://arxiv.org/abs/2608.03537), warp-specialised Hopper),
> [2608.12700](https://arxiv.org/abs/2608.12700) (Blackwell tcgen05 only), and all of torchao's fp8/fp4
> training work (SM100+). PyTorch 2.11's FlashAttention-4 FlexAttention backend is Hopper/Blackwell
> only. **The only 2026 kernel paths that reach an RTX 4060 Ti are TorchInductor/Triton itself and
> Helion** (GA 2026-04-07, PyTorch Foundation-hosted, compiles to autotuned Triton so it inherits
> Triton's Ada support). This is why the hand-written `flash_relation.py` was the right call, and why
> the recommendations here are Inductor/Triton-shaped rather than "adopt library X".

---

## 5. Inference and serving

The studio serves on CPU while the GPU trains (`serve/app.py --device`). Three separable questions:
the acceptance rule, the CPU path, and the GPU decode path.

### 5.1 The acceptance rule is not distribution-preserving, and the fix is well-specified (rank 17)

`serve/engine.py::_generate` accepts a speculative byte when `pm >= params.accept_threshold`
(default τ = 0.9), which is the LCA paper's threshold rule (2608.15454). That rule is **lossy**, and
it is worth being precise about why, because the repo's stated value is that every number the studio
shows is real.

With draft `q` and target `p`, the emitted law under thresholding is

```
P_τ(x) = q(x)·1[p(x) ≥ τ]  +  (1 − Z)·p(x),      Z = Σ_y q(y)·1[p(y) ≥ τ]
```

The exact rule works because `q(x)·min(1, p/q) = min(p,q)` — **the q cancels**. A test that is a
deterministic 0/1 function of `p(x)` alone cannot cancel anything, so conditional on acceptance you
emit the *draft's* relative weights among the survivors. The rejection branch contributes a scalar
multiple of `p`, not the pointwise residual `max(0, p−q)`, so the two branches cannot telescope back
to `p` for any τ. As τ → 0 the output collapses to `q`; as τ grows it collapses to the fallback.
**It interpolates between draft and fallback and never lands on `p`.** A 2026 result puts it
directly: [arXiv 2607.26627](https://arxiv.org/abs/2607.26627), *"Revisiting Lossy Verification in
Speculative Decoding"* — such relaxations *"silently rewrite the decoding distribution, and the
resulting acceleration can come at the cost of unstable, sometimes severely degraded generation
quality"*, and truncation-based rules do not even reproduce the truncated distribution one might
think is being targeted.

Three options, in order of how much they change the studio's story:

1. **Exact.** Accept `x` with probability `min(1, p(x)/q(x))`; on rejection resample from
   `norm(max(0, p − q))`; on full acceptance take a bonus token from the target. Output is
   **identically distributed to sampling from `p` alone**, proved in two lines
   ([Leviathan et al. 2211.17192](https://arxiv.org/abs/2211.17192) Alg. 1;
   [Chen et al. 2302.01318](https://arxiv.org/abs/2302.01318) Thm 1 — same rule, letters swapped).
   This requires a verification forward through the next-byte head, which the LCA paper also
   describes and which is where its headline **1.4–1.7×** comes from (2608.15454 Fig. 8).
   *Background (pre-2026): both speculative-sampling papers are 2022–2023 and nothing in 2026
   supersedes the correctness result — it is a theorem.*
2. **Principled lossy.** Leviathan's *lenience*: accept with `min(1, p(x)/(l·q(x)))`, which carries
   the stated guarantee that *no token is sampled with probability greater than `p(x)/l`*. Or
   Medusa's *typical acceptance* ([2401.10774](https://arxiv.org/abs/2401.10774)):
   `p(x) > min(ε, δ·exp(−H(p)))` with shipped defaults **ε = 0.09, δ = 0.3**, entropy-adaptive so it
   tightens where the model is confident, and the first token is always accepted so it never stalls.
3. **Keep τ, but label it.** If the threshold rule stays, the studio should say the sampled
   distribution is not the model's — that is exactly the kind of honesty strip the repo already has.

**The economics say do this.** With a head-based draft the cost model is a multiplicative overhead
`o` (the head runs inside the same forward), so speedup ≈ `E[#tokens] / o`. At Medusa's measured
`o ≈ 1.22` and γ = 4, **break-even is α ≈ 0.18** — the multi-byte head only has to be right ~18 % of
the time. The runs already log `mbp_top1_acc` at **0.47–0.55**, comfortably above that. And
byte-level units push α higher than BPE tokens by construction:
[2404.19737](https://arxiv.org/abs/2404.19737) §3.3 reports *"On an 8-byte prediction model, the
inference speedup is 6.4×"* (background, pre-2026).

> ⚠️ **Do not tune the head for acceptance rate — 2608.15454 shows that is the wrong objective.**
> Its MLP-MBP baseline (Medusa/Gloeckle-style independent heads on a shared hidden state) reaches
> **58–65 % byte acceptance against LCA's 46–52 %, and consistently loses on downstream metrics at
> comparable throughput.** The independent heads are *confidently wrong*: each future byte is
> predicted from a shared state without conditioning on the other bytes being emitted in parallel,
> whereas LCA's transformer layers condition each byte on the previous chunk and on its offset. This
> repo already has the LCA architecture (`mbp.py::LCAHead` + `lca_mask`), so it is on the right side
> of that result — the point is that `mbp_top1_acc` in the logs is a *diagnostic*, not a target, and
> optimising it would push the head toward the worse design.
>
> Two more numbers from the same paper worth knowing before tuning `n_candidates`: under
> **threshold** acceptance throughput *decreases* monotonically with n (−22 % from n=3 to n=7), while
> under **speculative verification** it *increases* monotonically (**+29–37 %** from n=3 to n=7–8) at
> constant quality. And at n=6 the mean accepted count is 3.05 with 15 % of steps accepting the full
> window, but **no candidate is ever accepted at the first decoding step** — `engine.py` should not
> spend a verification pass there.

**One free throughput trick from the same 2026 line of work.** 2608.15454's `Eff-FxT` baseline —
which it attributes to H-Net — invokes the main (chunk-level) network **only at predicted boundary
positions** and reuses the cached output elsewhere. `HNetForCausalLM.step` already does this
structurally (`if is_boundary: … self.main_network.step(…)`), so this repo has it. Worth recording as
confirmation that the decode path is designed correctly.

### 5.2 SSM state rollback: design for it now, not later

The one thing here that is genuinely painful to retrofit. Rejecting a draft token in a Transformer
means truncating the KV cache. **The Mamba state update is irreversible** — once history is
compressed into the recurrent state you cannot subtract a token back out. Current implementations
keep a separate state snapshot per draft position, which is far heavier than per-token KV
([vLLM RFC #47572](https://github.com/vllm-project/vllm/issues/47572),
[SGLang #28730](https://github.com/sgl-project/sglang/issues/28730)).

The clean fix is [**ReplaySSM** (Tri Dao, 2026)](https://tridao.me/blog/2026/replayssm/): cache recent
*inputs* rather than state, reconstruct state on demand, write the full state back once every L
steps — *"for speculative decode, caching recent inputs makes rollback a ring-buffer pointer move,
cutting the per-step rollback cost from O(T) to O(1)."*

This matters for `morpheme/model/hnet.py`'s `InferenceState`, which currently holds `encoder`,
`dechunk`, and `decoder` states that would all need rolling back on a rejected byte. Restructure
`Mamba3State` handling around input replay **before** implementing exact acceptance (§5.1), because
exact acceptance is precisely what introduces rejections.

### 5.3 CPU serving: the hardware forecloses two obvious ideas

The i7-14700F is Raptor Lake, and **AVX-512 is fused off on client Alder/Raptor Lake**
([Intel support article 000089918](https://www.intel.com/content/www/us/en/support/articles/000089918/processors.html)).
No AVX512-BF16, no AMX. The ISA level is **AVX2 + FMA3 + AVX-VNNI**, which in oneDNN's hierarchy is
`avx2_vnni` — one rung below `avx2_vnni_2`, the level that first adds bf16/fp16 support
([oneDNN CPU dispatcher control](https://uxlfoundation.github.io/oneDNN/dev_guide_cpu_dispatcher_control.html)).

Consequences:

* **bf16 and fp16 CPU inference are dead ends.** There is no native bf16 arithmetic; autocast on CPU
  converts bf16 → two fp32 vectors, computes, converts back. You pay conversion instructions for
  **zero** compute speedup.
* **Skip `intel-extension-for-pytorch`.** Intel's own docs say IPEX is *"optimized for CPUs with
  AVX-512 or above, and functionally works for CPUs with only AVX2"*, and their release notes warn
  that BF16 AMP *"runs abnormally with the extension on the AVX2-only machine"*.
* **int8 dynamic quantization is the only real precision win** — `fbgemm` targets AVX-VNNI's
  `VPDPBUSD`, which this CPU has. It changes numerics, so validate with held-out **BPB**, not by
  eyeballing samples.

**But at this model size, quantization is not the first lever.** Batch-1 byte-by-byte decode is
GEMV, so runtime ≈ time to stream the weights once. At ~40 GB/s effective:

| params | dtype | weight bytes | ms/byte | fits in 33 MB L3? |
|---|---|---|---|---|
| 12.7M (`pilot`) | fp32 | 51 MB | 1.3 | ✗ |
| 12.7M | int8 | 13 MB | 0.3 | **✓** |
| 35.4M (`local`) | fp32 | 141 MB | 3.5 | ✗ |
| 35.4M | int8 | 35 MB | 0.9 | ✗ (marginal) |
| 104.6M (`flagship`) | int8 | 105 MB | 2.6 | ✗ |

Pure memory traffic at the `pilot` size is ~1.3 ms/byte, but an eager PyTorch H-Net decode step —
`Mamba3Mixer.step` alone is dozens of small ops, ×4 layers, plus the Relation cache concat and the
LCA head — easily costs 5–20 ms of Python and dispatch. **At 12–35M you are overhead-bound, not
bandwidth-bound.** So the order is:

1. `torch.inference_mode()` instead of `@torch.no_grad()` on the decode path (`engine.py` and
   `hnet.py::step`) — disables view tracking and version-counter bumps, free at hundreds of tiny ops.
2. `torch.compile` with the Inductor C++/OpenMP backend. Note `mode="reduce-overhead"` is a
   **no-op on CPU** (it is the CUDA-graphs mode); `mode="max-autotune"` on CPU needs frozen weights
   (`torch._inductor.config.freezing = True` under `no_grad`).
3. Pin to P-cores and **sweep thread count 1/2/4/6/8** — Intel's own oneMKL guidance recommends
   P-cores only on hybrid parts, because static OpenMP scheduling finishes when the *slowest* core
   finishes and Gracemont E-cores are roughly half a P-core. `OMP_NUM_THREADS=8`,
   `OMP_PROC_BIND=close`, `OMP_PLACES=cores`, plus `taskset` to the P-core list. At 12M I would
   expect 2–4 threads to win.
4. int8 dynamic quantization on `nn.Linear` only, validated on BPB.

One concrete bug-shaped item: `hnet.py::step` builds
`torch.tensor([[state.cur_offset]], device=h.device)` **every decoded byte**. On CUDA that is a
hidden host-to-device sync (see §2.1); on CPU it is a small allocation in the hot loop. Hoist a
preallocated 1×1 buffer and `fill_` it.

**And budget for contention:** if the GPU trainer ever runs CPU-side optimizer work, or simply while
the data loader runs, the two workloads compete for the same ~90 GB/s of DRAM. CPU decode is pure
bandwidth. They are not independent.

### 5.4 GPU decode: CUDA graphs, and the piece upstream already wrote

If the studio ever serves on the GPU, batch-1 decode is the ideal CUDA-graph target, and the
structure is favourable: **the Mamba SSM state is not a growing cache**. Upstream's
`allocate_inference_cache` returns fixed-shape `conv_state`/`ssm_state` with no `max_seqlen`
dimension, mutated in place — exactly what graph capture wants. `morpheme/model/mamba3.py`'s
`Mamba3State` has the same property. The growing-cache problem applies only to the **Relation**
mixer's `{P2, Ĩ}` cache, which `relation.py` currently grows by `torch.cat` every step — that must
become a preallocated `[B, H, max_seq_len, dh]` buffer plus an `int32` device length tensor, never a
Python-int slice.

The reference recipe is upstream mamba's `mamba_ssm/utils/generation.py`: force
`seqlen_offset = max_seqlen − decoding_seqlen` before warm-up so host-side branches resolve to the
decode path at worst-case extent, warm up on a side stream, capture, then per step copy exactly three
tensors (`lengths_per_sample`, `input_ids`, `position_ids`) and `graph.replay()`. Capture **several
`decoding_seqlen` values** — 1 and γ+1 — so speculative verification is graphed too. Keep the sampler
outside the graph and make it sync-free (`argmax(probs / exponential_())` rather than
`torch.multinomial`). Clone the logits output; the graph rewrites the same buffer each replay.

Expected: **2–4× on wall-clock decode**, top of the range only if the eager timeline actually has
gaps — which for a 12–35M model at batch 1 is near-certain. Per-kernel launch overhead drops from
~3.8 µs (overlapped stream launches) to ~0.5 µs (graph replay)
([NVIDIA CUDA Graphs blog](https://developer.nvidia.com/blog/cuda-graphs/)).

Separately: `morpheme/model/mamba3.py` reimplements `step()` in pure PyTorch because upstream's needs
the CuTe `mamba3_step_fn` (and `nvidia-cutlass-dsl` is absent here — upstream's `Mamba3.step()` would
hard-assert). That was the right call, but upstream **also** ships a tested Triton step kernel at
`mamba_ssm/ops/triton/mamba3/mamba3_siso_step.py:233` (`mamba3_siso_step`, 9 autotune configs),
currently called only from its own tests. Wiring that in is the obvious GPU-decode speedup, and it
keeps the pure-PyTorch path as the Windows/CPU fallback.

---

## 6. 1B parameters on 8 GB: what the arithmetic actually says

Short version: **offload solves memory, not compute.** You can *fit* ~1B params on this box. You
cannot *train* one. The binding constraint at 1B is 44 TFLOPS, not 8 GB.

### 6.1 Throughput estimate, shown fully

ZeRO-Offload stage 2 with the optimizer on CPU
([arXiv 2101.06840](https://arxiv.org/abs/2101.06840), background pre-2026) moves fp32 master weights
and Adam moments to the host and proves the minimum PCIe volume is **4 bytes/param per step**
(2 B gradients down, 2 B params up). At 1B on PCIe 4.0 ×8 (~13 GB/s pinned):

```
D2H gradients 2.0 GB → 154 ms   (streams inside the backward; effectively free)
H2D params    2.0 GB → 154 ms   (on the critical path)
```

CPU Adam is **memory-bound, not compute-bound**. Per parameter per step it touches
2 B (bf16 grad) + 8 B (m) + 8 B (v) + 8 B (fp32 master) + 2 B (bf16 param out) = **28 B**, so 28 GB
at 1B. DDR5-5600 dual-channel peaks at **89.6 GB/s**
([Intel 14th-gen desktop brief](https://www.intel.com/content/www/us/en/products/docs/processors/core/core-14th-gen-desktop-brief.html));
the ZeRO-Offload paper's own 1B measurement (0.22 s on a 256 GB/s machine) implies **49.7 % of peak
achieved**, so scaling gives **~0.63 s per optimizer step here**. Adam's arithmetic is ~11 GFLOP
against ~2.2 TFLOP/s of AVX2 — 5 ms, i.e. **125× headroom**. The missing AVX-512 therefore costs
almost nothing; CPU-Adam's win over eager PyTorch comes from fusing the update into one pass over
DRAM, not from 512-bit registers.

GPU side, with full activation checkpointing: 8N FLOP/token = 8×10⁹. At a realistic 25 % MFU
(11 TFLOPS) that is **1 375 tok/s of raw compute**, 1.49 s for a 2048-token micro-step.

| | with async/delayed CPU update | naive |
|---|---|---|
| GPU fwd+bwd+recompute | 1.49 s | 1.49 s |
| CPU Adam | hidden | 0.63 s exposed |
| H2D params | 0.154 s | 0.154 s |
| **step** | **1.64 s** | **2.27 s** |
| **tok/s** | **~1 250** | **~900** |
| offload share of step | 9.4 % | 34 % |

Knock off 20–25 % for WSL2, Python and imperfect overlap. **Honest answer: ~1 000 tokens/s, range
700–1 500.** For a byte-level model that is **~1 kB/s ≈ 86 MB of text per day.** Note that offload is
*not* the dominant term — with a delayed update it is ~10 % of the step. ZeRO-Offload is doing its job
well and it still does not save you.

### 6.2 Where this machine stops being useful

Chinchilla-optimal (20 tokens/param) at 25 % MFU with checkpointing gives
`t = 160·N² / 1.1×10¹³` seconds:

| budget | max N |
|---|---|
| 3 days | **~133M** |
| 7 days | ~204M |
| 30 days | ~420M |
| **1B** | **~231 days** |

**This is a 100–200M-parameter research machine** — which is one clean doubling past the current
104.6M `flagship` preset, and a perfectly good place to be. Renting is both faster and, at US
residential electricity prices, marginally *cheaper per token*: an H100 SXM at 40 % MFU does ~66 000
tok/s (66×), and 20B tokens costs ~3.5 days / ~$226 rented versus ~231 days / ~$264 of electricity
locally. The local box's advantage is zero friction and zero billing anxiety, not economics. Use it
for what it is genuinely good at: fast 12–105M ablations where a run finishes in hours.

### 6.3 If you want the 1B run anyway, as an engineering exercise

* **RAM binds before VRAM.** The CPU side needs 4 B (master) + 4 B (m) + 4 B (v) + 2 B (pinned
  gradient landing buffer) = **14 B/param** = 14 GB at 1B. WSL2 defaults to **50 % of host RAM**
  = ~16 GB ([Microsoft WSL config docs](https://learn.microsoft.com/en-us/windows/wsl/wsl-config)),
  which will not fit alongside PyTorch and the loader. Set `memory=26GB` in `%UserProfile%\.wslconfig`
  and `wsl --shutdown`. Ceiling ≈ 26 GB / 14 B ≈ **1.85B params**.
* **The NVMe tier is unavailable.** DeepSpeed's `async_io` needs `libaio-dev` (sudo) and is
  unsupported on Windows, so ZeRO-Infinity's disk tier is off the table. RAM is the hard wall.
* **DeepSpeed without `nvcc` does build** — `CPUAdamBuilder` inherits `TorchCPUOpBuilder`, whose
  `builder()` catches `MissingCUDAException` and takes a CPU-only path that drops `-lcudart`/`-lcublas`.
  The failure mode is the *middle* case: `CUDA_HOME`/`CUDA_PATH` resolves but `nvcc` is missing, which
  raises an uncaught `FileNotFoundError`. `unset CUDA_HOME CUDA_PATH` first. **The real blocker is the
  C++ toolchain**, not CUDA — install `gxx_linux-64` and `ninja` from conda-forge into the venv (no
  root needed).
* **Activation memory is not the problem.** At `h=2048, L=24, s=2048, b=1`, Korthikanti's formula
  ([arXiv 2205.05198](https://arxiv.org/abs/2205.05198), background pre-2026) gives 11.5 GB naive,
  3.4 GB with a fused attention kernel, and **~0.35–0.7 GB with full checkpointing** — a 16–32×
  reduction, comfortably inside 8 GB. The cost is the 6N → 8N FLOPs already charged above.
* **Do not use paged optimizers for this.** At 1B you have ~12 GB of optimizer state against 8 GB of
  VRAM, so essentially all of it faults in and out every step: ~24 GB at roughly half of PCIe
  bandwidth ≈ **3.7 s/step**, versus DeepSpeed's explicit 4 GB of pinned traffic ≈ 0.31 s — about
  **12× worse**. bitsandbytes' own docs say paged optimizers *"only become active if you run out of
  GPU memory"* and reach *"about half or worse than the full PCIe memory bandwidth"*. They are an
  OOM-spike absorber, not a pretraining strategy.
* **FSDP alternatives work at `world_size=1`** but pay an eager-optimizer tax. FSDP1's
  `CPUOffload(offload_params=True)` additionally **does not support gradient accumulation outside
  `no_sync()`**, which kills the strategy that amortises offload. FSDP2's `fully_shard(...,
  offload_policy=CPUOffloadPolicy(pin_memory=True))` is the cleanest pure-PyTorch analogue; recover
  most of the CPU-Adam win with `torch.compile(opt.step)` so Inductor fuses the update into one DRAM
  pass.
* **A ~150-line manual loop is a legitimate choice.** CPU master weights, CPU optimizer, explicit
  `.to()` with pinned buffers and a side stream. Identical cost model, no toolchain risk, and you can
  implement the delayed update yourself.

---

## 7. Things that sound good but will not help here

**Raising the ratio-loss weight α to compress harder and go faster.** Tempting: throughput tracks
`bpic` almost linearly (§0.1). But the repo's own sweep already ran the experiment at equal
wall-clock (30 min each, `pilot` preset, batch 2 × accum 8):

| run | α | target N | achieved `val_bpic` | `val_bpb` |
|---|---|---|---|---|
| `sweep_a0.1_n4` | 0.1 | 4.0 | 3.06 | 2.074 |
| `sweep_a0.1_n6` | 0.1 | 6.0 | **3.86** | **1.917** ← best |
| `sweep_a0.3_n4` | 0.3 | 4.0 | 3.46 | 2.143 |
| `sweep_a0.3_n6` | 0.3 | 6.0 | **5.12** | 2.128 |

α = 0.3 bought 33 % more compression and **lost** 0.21 bits/byte at equal wall-clock. The extra bytes
seen did not pay for the per-byte quality loss. The published results agree that target and achieved
ratio are only loosely coupled: ATDC 2605.30080 Table III drives the target to 9.0 and gets BPIC
5.68; H-Net 2507.07955 Table 1 targets N₀ = 6 and achieves BPIC 4.8. Raising N (at α = 0.1) helped
both compression *and* quality; raising α did not. **Keep α = 0.1; get compression from N, not from
α.**

**`torch.backends.cudnn.benchmark = True`.** There are no convolutions anywhere in the model —
`causal_conv1d` is not used by the Mamba-3 SISO path in `mamba3.py`, and there is no `nn.Conv*` in
the repo. Zero effect.

**TF32 (`allow_tf32` / `set_float32_matmul_precision("high")`).** The only genuinely-fp32 matmuls are
`residual_proj` (0.148M params → ~0.7 % of the FLOP budget) and the `[B,nb,C,C] @ [B,nb,C,D]` product
inside `_ema_chunked` (~90 MFLOP per micro-batch). Even a 4× speedup on both is under 1 % of step
time. Not worth the audit — **and there is a trap in it worth stating explicitly**, because this repo
deliberately computes several things in fp32:

> TF32 is a property of **cuBLAS fp32 GEMMs**, selected by dtype, not by lexical position relative to
> an autocast block. `Context.cpp::allowTF32CuBLAS()` consults only `float32Precision(CUDA, MATMUL)`
> — nothing reads autocast state. So `residual_proj` (an explicit `dtype=torch.float32` Linear),
> `_masked_ce`'s `.float()` logits, and `_ema_chunked`'s fp32 matmul would **all silently become
> 10-bit-mantissa** if TF32 were ever enabled globally. The deliberate fp32 in `hnet.py:66` and
> `dc.py`'s EMA would stop being fp32. If you ever do enable TF32 for a benchmark, set
> `torch.backends.cuda.matmul.fp32_precision = "ieee"` (the 2.9+ API; `allow_tf32` is deprecated in
> favour of it) around those regions, or use `NVIDIA_TF32_OVERRIDE=0` as a one-line global A/B.
> Defaults today are on your side: `matmul.allow_tf32` is **False**, `cudnn.allow_tf32` is True but
> there are no convolutions.

A related knob that *is* worth one ablation, and is cheap:
`torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False` (default **True**). With
M ∈ [250, 1000] the model issues many small GEMMs, which is exactly where cuBLAS reaches for split-K
— and split-K reduction in bf16 rather than fp32 is a real accuracy loss. Costs little; measure BPB.

**A hand-written Triton FlashRelation as a *FLOP* win.** It is already written, and it is worth
having — but be clear about why. Relation's pairwise term is only **7.3 %** of the analytic FLOP
budget at `bpic` 3.3 (§0.1); the paper's headline **3.6–4.4× over the materialised reference**
(2608.20172 Table 4a) is measured at T = 1024–4096 where T² dominates, and even there FlashRelation
reaches only **76–85 % of FlashAttention's throughput** (Table 4b/Table 23). Here T ≈ 620. The real
returns are memory (§2.3, which unblocks batch 4) and op count (128 → ~30 per layer), not FLOPs.
Score the kernel on those, not on a hoped-for 4×.

**`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.** Known to crash the WSL2 driver on this
machine (README, reproduced 3×). See §4.6 for the mechanism and §4.9 for the 30-second check that
might reopen it. It stays correct for the Lightning H100 run, where `cloud/launch.py` already sets it.

**fp8 on the Ada tensor cores.** Excluded by the numerics constraint for now, and independently
unattractive: the GEMMs here are small (d_model 512, M ≈ 620, d_ff 1536) and fp8 needs large,
compute-bound GEMMs plus scaling machinery to beat bf16. Revisit at the flagship scale on H100, not
here.

**bf16 optimizer moments with stochastic rounding (fp32 master weights kept).** The brief asked me to
evaluate this honestly, so: **it does not pay at this scale.** At 35.35M params the whole optimizer
footprint is fp32 params 141 MB + grads 141 MB + `exp_avg` 141 MB + `exp_avg_sq` 141 MB ≈ **565 MB of
~7.3 GB usable** — about 8 %. Halving the two moments saves **141 MB**, which does not unlock a
larger micro-batch (§2.3 shows the Relation activations are what moved, in gigabytes) and buys no
speed, because the optimizer is ~11 `_foreach_*` launches out of ~21 600 per step (§3.4). Against
that you would be giving up bit-neutrality, adding a custom optimizer to maintain, and — the real
cost — losing the ability to say a change was bit-neutral when debugging everything else on this
list. At the 104.6M `flagship` the same arithmetic gives 1.67 GB total and a 418 MB saving, which is
still not the binding constraint. **Revisit only if you ever run ≥500M locally**, where it becomes a
genuine enabler rather than a rounding error.

**A fused or "paged" cross-entropy loss.** These exist to stop a `[B·L, V]` logit tensor from
dominating memory, which is a real problem at V = 128 000. Here **V = 264**: the fp32 logits in
`_masked_ce` are `2 × 2047 × 264 × 4` ≈ **4.3 MB**. PyTorch 2.13's new `nn.LinearCrossEntropyLoss`
explicitly targets 100K+ vocabularies. A byte-level model is the one case where the loss is already
free — this is a rare structural *advantage* of the architecture, not a gap.

**A context-length curriculum (train at 1024, extend to 2048).** The compute saving from short
contexts comes from a *quadratic* attention term. The outer network here is Mamba-3 — **linear in L**
— and the Relation main network runs at chunk rate where the quadratic term is 7.3 % of the budget.
At fixed bytes per batch, halving the context saves approximately nothing; you would just process
twice as many sequences. The one exception is the MBP head, which *is* O(L²) — but that is an
argument for fixing §2.2, not for shortening the context. (The published wins for sequence-length
warmup are stability wins at large batch and large LR; at this batch size the authors themselves note
*"efficiency gains were more modest"*.) *Background (pre-2026); no 2026 work found that changes this
for linear-time outer models.*

**Cross-document masking / state reset at document boundaries.** A real effect at long context, but
at **2048 bytes ≈ 455 token-equivalents** most windows sit entirely inside a single FineWeb-Edu
document, so the boundary-spanning rate is low. And a selective SSM can learn to reset its own state
at a BOS/EOS byte through a large Δ. **Measure the boundary-spanning rate on your own shard before
spending engineering here** — if it is under ~10 %, this is not where the compute goes. (Whether a
*forced* Mamba state reset beats the learned one is, as far as I can find, unmeasured — it would be a
cheap and genuinely novel ablation.)

**A multi-process `DataLoader` with `pin_memory` and prefetching.** The micro-batch payload is
`2 × 2049` uint16 ≈ **8 KB** (32 KB after the `int64` cast). Multi-process workers, IPC, pinning and
prefetch all carry fixed per-batch overheads that plausibly exceed the cost of generating this data.
Rank 9 is deliberately scoped to a *single background thread with a reused pinned buffer*, not a
`DataLoader`. Also worth evaluating: the `pilot` shard is 300 MB — just resident it on the GPU and
index it there, and the host is out of the loop entirely. (Under WSL2, `pin_memory=True` is safe
because PyTorch's pinned allocator uses `cudaHostAlloc`; only `cudaHostRegister` is broken.)

**Chasing Mamba-3 kernel hyperparameters.** `Mamba3Cfg` already has `chunk_size = 64`,
`headdim = 64`, `d_state = 64`. Mamba-3 (2603.15569 §3.3) states that for SISO *"setting C = P = N
yields an overall linear-time algorithm"* — i.e. chunk = headdim = state. The config is already at
the paper's recommended point. (Their own experiments use `d_state = 128`, `headdim = 64`, which is a
capacity choice, not a speed one.) The encoder/decoder are also only 10.8 % of parameters and see
fixed-length 2048 inputs, so their shapes never vary and their kernels autotune once.

---

## 8. Measurements to run first

Run these **before** changing anything, so every later number has a baseline. All from the WSL2 venv,
`local` preset, on the real data shard.

**M-zero — three things that cost under five minutes and change what the rest of the list is worth.**

```bash
# (a) Does expandable_segments actually have to stay off? (§4.9)  -- 30 seconds
python3 -c "import ctypes;l=ctypes.CDLL('libcuda.so.1');l.cuInit(0);d=ctypes.c_int();l.cuDeviceGet(ctypes.byref(d),0);v=ctypes.c_int();l.cuDeviceGetAttribute(ctypes.byref(v),110,d);print('GPUDirect-RDMA-with-VMM (110):',v.value)"
# 0  -> retry PYTORCH_ALLOC_CONF=expandable_segments:True; most of §4.9 becomes moot
# 1  -> the README's guidance stands, and now has a mechanism behind it

# (b) Turn on the Triton autotune disk cache and find the real re-tune trigger (§4.1, §4.2)
export TRITON_CACHE_AUTOTUNING=1
export TRITON_PRINT_AUTOTUNING=1     # first run only; prints the key tuple + tuning seconds
find ~/.triton/cache -name '*.autotune.json' | wc -l    # was 0 before this change

# (c) Confirm nothing else is holding the GPU. The desktop alone costs ~1 GB on this box.
nvidia-smi --query-gpu=memory.used,power.draw,clocks.sm --format=csv
nvidia-smi -q -d PERFORMANCE | grep -A6 'Clocks Event Reasons'
```

**M0 — fix the profiler** (§2.5): make it build the model with fp32 parameters under `autocast`, and
default `--grad-accum 8`, so it measures what `train.py` actually runs. Then:

```bash
# M1. Baseline, matching the real trainer exactly.
python -m morpheme.train.profile_step --preset local --data ~/data/local_mix \
  --batch-size 2 --seq-len 2048 --grad-accum 8 --warmup 4 --timed 6 \
  --trace /tmp/base.json
```

Record: `sec_per_step`, `bytes_per_sec`, `bytes_per_chunk`, `tflops`, `mfu`, `peak_mem_GB`, and from
the table the **total CPU time vs total CUDA time**. If CPU ≫ CUDA, §2.1 is confirmed as the top item.

```bash
# M2. Count the synchronisations. Expect ~40 warnings per step today.
PYTHONWARNINGS=always python - <<'EOF'
import torch; torch.cuda.set_sync_debug_mode("warn")
from morpheme.train.profile_step import main
main(["--preset","local","--data","/home/noahs/data/local_mix",
      "--batch-size","2","--grad-accum","8","--warmup","1","--timed","1"])
EOF
```

```bash
# M3. Is batch 4 alive again post-FlashRelation? (rank 3 — cheapest test on the list)
for B in 2 4 8; do
  python -m morpheme.train.profile_step --preset local --data ~/data/local_mix \
    --batch-size $B --seq-len 2048 --grad-accum $((16/B)) --timed 6
done
# and with activation checkpointing on the Relation blocks:
python -m morpheme.train.profile_step --preset local --data ~/data/local_mix \
  --batch-size 4 --grad-accum 4 --ckpt-main --timed 6
```

```bash
# M4. What did FlashRelation and bucketing actually buy?
python -m morpheme.train.profile_step --preset local --data ~/data/local_mix --timed 6            # flash + bucket 64
python -m morpheme.train.profile_step --preset local --data ~/data/local_mix --timed 6 --no-flash
python -m morpheme.train.profile_step --preset local --data ~/data/local_mix --timed 6 --bucket 1
```

```bash
# M5. MBP head cost in isolation (ranks 2 and 14). profile_step has no --config flag yet,
# so add one (three lines, mirroring train.py) or run the trainer for the no-MBP arm.
python - <<'EOF'
from morpheme.config import MorphemeConfig
cfg = MorphemeConfig.local(); cfg.mbp.enabled = False; cfg.save("/tmp/local_nombp.json")
EOF
python -m morpheme.train.profile_step --preset local --data ~/data/local_mix --timed 6   # with head
python -m morpheme.train.train --config /tmp/local_nombp.json --data ~/data/local_mix   --out runs/nombp --batch-size 2 --grad-accum 8 --max-minutes 30                        # without
```

The delta is the true cost of the MBP head; the analytic model says 33 % of FLOPs (18.0 % linear +
14.9 % dense attention). If the measured delta is much larger than 33 %, the masked-SDPA backend is
the culprit and §2.2 is worth more than the table claims.

```bash
# M6. Where the top kernels are. Read the `top CUDA kernels` table from M1's output and check
# how much of the total is elementwise/`vectorized_elementwise_kernel` and how much is GEMM.
# A launch-bound model shows hundreds of sub-10-µs elementwise rows.
```

```bash
# M7. Environment sanity (cheap, run once).
python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability())"
python -c "import triton;print(triton.__version__)"
nvidia-smi --query-gpu=power.draw,power.limit,clocks.sm,clocks.max.sm,memory.used --format=csv
echo "TRITON_CACHE_DIR=$TRITON_CACHE_DIR TRITON_CACHE_AUTOTUNING=$TRITON_CACHE_AUTOTUNING"
# NOTE: 2.10+ reads PYTORCH_ALLOC_CONF; the old name is an alias and WINS if both are set (§4.9)
echo "PYTORCH_ALLOC_CONF=$PYTORCH_ALLOC_CONF PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
df -h /dev/shm; ls ~/runs 2>/dev/null || echo "runs/ still on /mnt/d -- see rank 8"
```

```bash
# M7b. Replace the reconstructed 2.7 s with a measurement (§4.2). Run when the GPU is free.
# Expect: one large first line, one medium line at the FIRST M=448 (it is divisible by 16, so it
# opens a new Triton specialisation class), everything else single-digit ms. The repeats at the end
# must be ~free. Run the whole thing a second time -- with the cache on, line 1 should collapse too.
TRITON_CACHE_AUTOTUNING=1 TRITON_PRINT_AUTOTUNING=1 ~/hnet-venv/bin/python - <<'EOF'
import torch, time
from morpheme.model.dc import DeChunkLayer
dc = DeChunkLayer(256).cuda(); dc.use_kernel = True
x = torch.randn(4, 1100, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
p = torch.rand(4, 1100, device="cuda").clamp(1e-4, 1-1e-4)
for M in (300, 337, 374, 411, 448, 485, 448, 300):
    torch.cuda.synchronize(); t = time.perf_counter()
    dc._ema(x[:, :M], p[:, :M]).float().sum().backward()
    torch.cuda.synchronize(); print(f"M={M:4d}  {(time.perf_counter()-t)*1000:8.1f} ms")
EOF
```

```bash
# M8. Recipe instrumentation, once the syncs are gone (ranks 10, 11). Add to train_step:
#   - accumulate |g|^2 of the FIRST micro-batch and of the full accumulated gradient,
#     then log B_simple = S / |G|^2 as an EWMA (§3.2). Two extra .norm() calls, device-side.
#   - log the loss with the same denominator the gradients use (§3.1).
# Then read B_simple over the first 500 steps: if it starts near 1e3 token-equivalents and
# climbs, ramp --grad-accum 2 -> 4 -> 8 instead of pinning it at 8.
```

---

## 9. 2026 sources read

**33 papers with 2026 arXiv IDs were retrieved and read** for this report, plus the 2026 PyTorch /
Triton / NVIDIA release material listed below. By month: **2608 — 12**, **2607 — 9**, **2606 — 2**,
**2605 — 5**, **2604 — 4**, **2603 — 1**, **2602 — 1**. Anything not in this section and not a 2026
doc, issue or release note is **background (pre-2026)** and is labelled as such at the point of use.

### 2026 papers — architecture and modelling

| ID | Mo | Title | Load-bearing for |
|---|---|---|---|
| 2608.20172 | 2608 | *Ask Self, Ask Others: Relation Is All You Need* | The main network. FlashRelation's structure (App. A.3) that `flash_relation.py` implements; the measured 3.6–4.4× over a materialised reference and the 76–85 %-of-FlashAttention ceiling (Tab. 4); the `{P2, Ĩ}` cache size; the only published hyperparameters at 10M/30M/100M. Confirms both awkward terms (σ diagonal, −λ log i) are hoistable out of the tile loop. |
| 2608.15454 | 2608 | *Dynamic Multi-Byte Prediction With Hierarchical Language Models* (LCA-MBP) | The MBP head, and the closest published system to this repo. The LCA mask; both acceptance rules; **the acceptance-rate trap** (§5.1); n-vs-throughput under each rule; "no candidate accepted at the first decode step"; the `Eff-FxT` boundary-only main-network trick. |
| 2608.05806 | 2608 | *Hierarchical Latent Prediction for Language Models* (HiLP) | Prices MTP at 1B: **training throughput 49 % of next-token** for +0.44 HumanEval. §3.3. |
| 2608.03599 | 2608 | *Disentangling Language Modeling and Boundaries* | **Position paper, flagged as such** — the two decisive experiments are proposed, not run. Carries one useful fact: byteifying an existing subword model costs <1 % of pretraining. |
| 2608.02032 | 2608 | *DART: Decoded Attention over Recurrent States* | Design precedent for the Relation mixer: FlashAttention-style attention over Mamba-2 chunk states, **37.5 % faster than FA-2** at matched shapes, 74 % smaller cache, trained at 130M/370M/780M — credible small-scale evidence. |
| 2607.26627 | 2607 | *Revisiting Lossy Verification in Speculative Decoding* | §5.1: truncation-based acceptance *"silently rewrites the decoding distribution"*. The current-year case against `accept_threshold`. |
| 2605.30080 | 2605 | *ATDC: adaptive targeted dynamic chunking* | The ratio schedule. **The only source stating the dechunk EMA is a linear SSM suited to a parallel scan** (§III.C.2) — H-Net does not say this. Tab. I/III: target ratio and achieved BPIC are loosely coupled (target 9.0 → BPIC 5.68), corroborating §7. |
| 2605.08044 | 2605 | *Fast Byte Latent Transformer* (FastBLT) | BLT successor. **BLT-S self-speculation: up to 77 % memory-bandwidth reduction with no task-performance loss**, acceptance 96.8 % at k=4. The safest variant is the one an H-Net hierarchy can copy. |
| 2603.15569 | 2603 | *Mamba-3* (ICLR 2026) | The encoder/decoder. The `C = P = N` chunk rule `Mamba3Cfg` already satisfies; the 2.5-ops-per-byte decode intensity behind §5.4; and the fusion structure — Mamba-2 decode is `IP, Conv, SSM, Gate, OP`, Mamba-3 is `IP, Rotary, SSM+Gate, OP`, i.e. **two fewer kernels per layer**, with the causal conv subsumed into the trapezoidal discretisation. |
| 2602.06019 | 2602 | *Multi-Token Prediction via Self-Distillation* | §3.3: offline MTP cross-entropy **cannot represent the joint distribution**; ablations favour no auxiliary next-token loss on prefix tokens, which is what `compute_losses` currently does. |

### 2026 papers — optimizers, schedules, batch size

| ID | Mo | Title | Load-bearing for |
|---|---|---|---|
| 2608.03941 | 2608 | *Muon Meets Mamba: Spectral Optimization for State Space Models* | **The single most repo-specific optimizer result of 2026.** Mamba-2 at 130M: Muon's benefit is localised to `out_proj`; on the row-stacked `in_proj` it is worse than AdamW. Basis for the `split_muon_params` fix (rank 16). |
| 2608.11612 | 2608 | *Dion3: Full-Stack Orthogonal Updates* | **Gram Newton–Schulz** — mathematically identical output, 1.5–2× cheaper. Plus an explicit bf16 fusion hazard in the weight update and "normalisation must be fp32". |
| 2608.11859 | 2608 | *Small-Scale Experiments: Are We There Yet?* | How far a <268M-param result can be trusted; extrapolation magnifies sampling error until the law saturates. Tempers §3.2's downward extrapolation. |
| 2608.06398 | 2608 | *EntropyMoE: Entropy-Aware Sparse Expert Routing for Tokenizer-Free LLMs* | **The only 2026 paper that measures byte-patch routing overhead in wall-clock** — and it is a warning: sparse execution is **2.00× slower** than dense BLT despite a 400-parameter router. Also: balance load in **bytes, not chunk counts**. |
| 2607.20548 | 2607 | *SOAP, Muon, and Beyond: Pushing LLM Pretraining Scales* (NVIDIA) | The bf16 Newton–Schulz noise floor (2⁻⁷ vs 2⁻²³); Mamba conv1d must stay on AdamW; NVIDIA's own NS settings (16 iters, Polar Express, no Nesterov). |
| 2607.23777 | 2607 | *Scale Weight Decay and Train Better* (Muon-SW) | `λη_t → λη_t²/η_max`. **+21.7 % at 72M params** — the nearest published scale to this repo. |
| 2607.27731 | 2607 | *Towards joint scaling laws with optimal batch size schedules* | **Rejects the η/B ratio rule by direct test.** Optimal batch *schedule* from LR-schedule shape; and the caveat that the gain **vanishes under pure bf16** but is ~−2.5 % in the mixed regime this repo uses. |
| 2607.12360 | 2607 | *Same Loss, Same Noise, Opposite Schedules* | Cooldown is conditional on optimizer normalisation × noise structure. **AdamW preferred cooldown fraction 0** on their real task; Muon-like optimizers require it. Supplies a measurable diagnostic. |
| 2607.10959 | 2607 | *WSqD: A Horizon-Free Learning Rate Schedule* | **Nearly this exact setup** — 213M, bf16, single device, grad-accum ×4, unknown-length runs. Optimal base LR pinned across 15k–60k horizons where WSD's drifted. §3.5. |
| 2607.04033 | 2607 | *OmniOpt: Taxonomy, Geometry, and Benchmarking of Modern Optimizers* | 24 optimizers properly tuned at 60M/130M/350M/1B with memory and runtime. **Muon leads at 60M and 130M and halves optimizer state.** Supersedes the element-wise-AdamW-variant family. |
| 2607.01487 | 2607 | *How to Allocate Your Tokens? Scaling Laws with Training Steps and Batch Size* | 2026 re-derivation of critical batch size; shows the McCandlish form implies optimal batch = 1; M\* ∝ D^0.566; the ~4×-wide ε-suboptimal window. §3.2. |
| 2606.27216 | 2606 | *Hierarchical Muon: Tiled Newton–Schulz Updates* | Why tiling will **not** pay at 35M (matrices at or below tile size), and the transferable advice that the optimizer step should be CUDA-graph-captured at small scale. |
| 2605.19095 | 2605 | *ScheduleFree+* | Beats WSD 120M–2B — but explicitly **no advantage on short runs**, and needs weight decay 5–50. Also reports WSD loses to a tuned linear-decay-to-zero. |

### 2026 papers — systems, kernels, memory

| ID | Mo | Title | Load-bearing for |
|---|---|---|---|
| 2608.08740 | 2608 | *Memory-Efficient Activation Checkpointing with Sliding Window and Hirschberg's Algorithm* (COLM 2026) | The selective-activation-checkpointing knapsack solver **shipped in PyTorch 2.10**: `dp_knapsack` allocates an O(nW) table and OOMs around n=100; `dp_knapsack_sliding_hirschberg` is O(W), 20× larger problems, 25–28 % faster, same exact optimum. Matters if you ever compile with a memory budget. |
| 2608.08961 | 2608 | *Gradient Under Microscope* | **8–15 % GPU utilisation for memory-bound 1B LLM training vs 96–99 % for vision.** Low utilisation on small-LM training is documented, not a misconfiguration. Also: checkpointing costs 20–60 % runtime; gradient accumulation is Pareto-optimal. §0.2. |
| 2608.03537 | 2608 | *ComFuse* | Joint MatMul + norm/softmax fusion vs TorchInductor (1.02–1.24×). **Hopper-only.** Useful negative: it *loses* on self-attention because Inductor dispatches FlashAttention — fusing compute- and memory-intensive ops is not automatically a win. |
| 2607.22432 | 2607 | *TileSight: A Tile-Centric Analytical GPU Performance Model* | 12.35 % MAPE vs Roofline's 33.85 %. Attractive, but explicitly excludes data-dependent control flow and latency-bound small-batch decode — i.e. **excludes this workload**, and covers no consumer Ada part. |
| 2607.04454 | 2607 | *Correct but Slow: The GPU Kernel Evaluation Gap in Modern DSLs* | Two cheap screening heuristics for "is my Triton kernel actually good" — library-relative efficiency and roofline utilisation — and a failure mode structurally identical to this one (grouped conv degenerating into 512 launches, ≤5.5 % of cuDNN). |
| 2606.05495 | 2606 | *SET: Stream-Event-Triggered Scheduling for CUDA Graph Pipelines* | **Bounds what CUDA Graphs buy.** On an RTX 3090, residual scheduling overhead *after* graphs is 45.32 % (static batching) of total execution, and 45–51 % at batch size 1. §4.10. |
| 2606.00601 | 2606 | *ScanWeaver: Compiler-Driven Parallelization of Affine Recurrences* | Honest negative result: generic MLIR scan lowering does **not** beat the fused Mamba kernel, and its own latency is *"dominated primarily by GPU synchronization, shared-memory staging, and launch overheads rather than arithmetic throughput."* |
| 2605.19269 | 2605 | *CODA: Rewriting Transformer Blocks as GEMM-Epilogue Programs* | The 2026 answer to "fuse norm/gate/projection", including the backward. **Hopper-only** (CuTeDSL ping-pong/TMA), and speedups are modest (1.0–1.6× kernel-level). |
| 2604.13327 | 2604 | *Event Tensor: A Unified Abstraction for Compiling Dynamic Megakernels* (MLSys 2026) | The **5–10 µs per launch** figure used in §0.2 and §2.1, and the structural point that *"CUDA Graphs preserve kernel boundaries and thus cannot expose inter-kernel parallelism."* Inference only — no backward pass. |
| 2604.10597 | 2604 | *COREY: Entropy-Guided Runtime Chunk Scheduling for Selective Scan Kernels* | **The closest published measurement to this machine**: `mamba_ssm` selective scan on an RTX 3070 under WSL2, Triton 3.6, PyTorch 2.11. Latency falls monotonically with chunk size purely from launch count; a per-timestep loop is 48.6× slower than 9 chunked calls. *Self-described as "Concept & Feasibility"; its scheduler is not end-to-end validated.* |
| 2604.05091 | 2604 | *MegaTrain: Full Precision Training of 100B+ LLMs on a Single GPU* | The only 2026 offload system with real consumer numbers (RTX 3090 24 GB: 3B at 33.18 TFLOPS / 1792 tok/s). And the warning that parameter **streaming and CUDA Graphs are mutually exclusive**. |
| 2608.11919 | 2608 | *LazyTrain* | MILP scheduler over MegaTrain; measured PCIe (12 GB/s Gen5 ×16) and NVMe (2.83/3.35 GB/s) rates; its 8-bit-optimizer ablation shows that choice is **memory-motivated, not speed-motivated** (0.3 % throughput). |
| 2608.06838 | 2608 | *StateFlow: Sequence Pipeline Parallelism with Linear Recurrence* | Metadata only. Multi-GPU, ≤32B/256K — not a single-GPU path. Listed so it is not mistaken for one. |
| 2608.12700 | 2608 | *Contract-Grade Verifier + Native Blackwell Backward for Gated-Linear-Recurrence* | Metadata only. **Blackwell tcgen05 only** — evidence for the Ada blind spot in §4.10, not a usable path. |

### 2026 non-paper sources

| source | date | relevance |
|---|---|---|
| [PyTorch 2.10 release blog](https://pytorch.org/blog/pytorch-2-10-release-blog/) | 2026-01 | **Combo-kernels horizontal fusion, headlined as "reduced kernel launch overhead"** (rank 0d); `varlen_attn()`; the shipped SAC knapsack solver. |
| [PyTorch 2.11 release blog](https://pytorch.org/blog/pytorch-2-11-release-blog/) | 2026-03-30 | FlexAttention + FlashAttention-4 backend (Hopper/Blackwell only); CUDA 13 default; 2-month release cadence. |
| [PyTorch 2.12 release blog](https://pytorch.org/blog/pytorch-2-12-release-blog/) | 2026-05-19 | `torch.accelerator.Graph`; **`torch.cond` capturable in CUDA Graphs** (eager/cudagraphs backends; Inductor pending); fused Adagrad; cu126 explicitly retained for older architectures. |
| [PyTorch 2.13 release blog](https://pytorch.org/blog/pytorch-2-13-release-blog/) + `v2.13.0` notes | 2026-07-08 | Triton pin → **3.7.1** (matching this venv); CuTeDSL Inductor backend; `CUDAGraph.get_graph_data()`; off-GIL CUPTI profiler; kernel compilation moved to a subprocess pool. |
| PyTorch 2.13 docs + `torch/_inductor/config.py`, `torch/_dynamo/config.py` (main) | 2026-08 | Exact defaults in §4.7/§4.9/§4.10: `graph_partition = True` (OSS), `triton.cudagraphs = False`, `combo_kernels = False`, `recompile_limit = 8`, `dynamic=True` demoted to "Not recommended", `PYTORCH_ALLOC_CONF` rename. |
| [PyTorch devlog — *Host-to-device syncs are bad too*](https://docs.pytorch.org/devlogs/eager/2026-08-11-hidden-h2d-sync/) | 2026-08-11 | "The bubble afterwards" — the cost model behind rank 1, and the hidden-H2D patterns, one of which is in `hnet.py::step`. |
| [PyTorch devlog — *Pinned memory: why nobody gives it back*](https://docs.pytorch.org/devlogs/eager/2026-08-09-pinned-memory-allocator/) | 2026-08-09 | Corrects the "pinned temp can be freed mid-copy" folklore; the pinned high-water mark is permanent, which compounds WSL2's limited pinned pool. |
| [pytorch#192330](https://github.com/pytorch/pytorch/issues/192330) | 2026-08-06, open | Isolates the WSL2 `expandable_segments` failure to device attribute **110**, and lists 2.13.0+cu126 as affected. Basis for rank 0b. |
| Triton 3.6.0 / 3.7.0 / 3.7.1 releases (GitHub API `published_at`) | 2026-01-21 / 05-07 / 06-18 | `TRITON_CACHE_AUTOTUNING` (rank 0a); `make_block_ptr` deprecated in 3.7 in favour of tensor descriptors; the 3.7 focus is AMD gfx1250 and Blackwell — **nothing Ada-specific**. |
| [state-spaces/mamba](https://github.com/state-spaces/mamba) issues #904, #997, #1014, #1015, #1017 | 2026 | Mamba-3 kernel defects relevant to the flagship run (§4.4), including the silent 32-bit pointer overflow past ~61K tokens. |
| [ReplaySSM (Tri Dao)](https://tridao.me/blog/2026/replayssm/) | 2026 | The O(1) SSM-state rollback design §5.2 says to adopt *before* implementing exact speculative acceptance. |
| Helion 1.0 GA — [Linux Foundation press release](https://www.linuxfoundation.org/) / HPCwire | 2026-04-07 | The one 2026 kernel DSL that is not architecturally gated to Hopper/Blackwell, because it lowers to autotuned Triton. |
| [NVIDIA CUDA on WSL user guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) v13.3 | 2026-06-25 | The limitations in §4.5–4.6: Managed Memory, concurrent CPU/GPU access, pinned-memory ceiling, and the (now stale) claim that NVML utilisation queries are unsupported. |
| [CUDA Toolkit release notes 13.3](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/) | 2026-06-27 | Windows display driver unbundled since CUDA 13.1. |
| [Leveling up CUDA Performance on WSL2](https://developer.nvidia.com/blog/leveling-up-cuda-performance-on-wsl2-with-new-enhancements/) | current | Bounds the WSL2 launch penalty at ≤10 % (GenomeWorks worst case). |
| [microsoft/WSL#11050](https://github.com/microsoft/WSL/issues/11050) | open | WSL2 ignores the Windows sysmem-fallback policy — the competing hypothesis for "batch 4 collapsed to 13 kB/s" (§4.6). |
| torchao v0.16.0 / v0.17.0 / v0.18.0 releases | 2026-02-10 / 03-30 / 08-03 | All fp8/NVFP4 training work is SM100+/Hopper — evidence for §7's fp8 entry and §4.10's Ada blind spot. |

### Where 2026 work does not exist

Stated explicitly, because each of these is a finding rather than a gap in searching.

1. **No 2026 fused-kernel work targets Ada / sm_89 for SSM or linear-attention training.** FlashQLA
   (sm_90/100/120), ThunderKittens 2.0 (Hopper/Blackwell; Ampere support dropped), CODA, ComFuse,
   2608.12700, torchao fp8/fp4 — all gate above Ada. The only 2026 paths that reach this card are
   TorchInductor/Triton and Helion. See §4.10.
2. **No 2026 paper applies megakernels or persistent kernels to *training*.** Event Tensor and all
   2026 megakernel work is inference/serving; none has a backward pass. This is the largest open gap
   relative to this problem.
3. **No 2026 work addresses H-Net-style dynamic chunking at the kernel or systems level.** The three
   2026 H-Net-adjacent papers (2605.30080, 2608.15454, 2605.08044) are modelling or
   inference-throughput papers. **Nothing published addresses the variable-M training tensor-shape
   problem, recompilation, bucketing, or launch counts** — so `chunk_bucket` has no published
   precedent to compare against, and §2.6 is genuinely uncharted.
4. **No 2026 measurement of WSL2 vs native-Linux kernel launch overhead.** The CUDA-on-WSL guide
   v13.3 says nothing about launch latency, submission batching, or WDDM scheduling. COREY ran under
   WSL2 but never compared against native. If launch overhead is the bottleneck, **nobody has
   published how much of it WSL2 contributes** — worth measuring across the dual boot.
5. **No 2026 work targets an 8 GB training budget.** The smallest consumer target in the 2026 offload
   literature is RTX 3090 24 GB.
6. **No 2026 fp8 training results on Ada sm_89**, and **no 2026 work on stochastic rounding** at all.
7. **No 2026 systematic MFU study for sub-100M models.** 2608.08961 is nearest (1B, A100,
   `nvidia-smi` utilisation rather than MFU, no SSMs).
8. **No 2026 paper reports MFU for a byte-level H-Net, or trains one on a consumer GPU.** The five
   architecture papers report RTX 5090, H100, 4× B200, 8× H200; H-Net reports no hardware at all.
   There is no external number to benchmark §0.1 against — the ~9 % here is the datapoint.
9. **No published BPB for a byte-level LM at ~35M params**, so §3.6's gate cannot be set by
   comparison. **And no 2026 training-compute comparison of byte-level vs BPE** — the standing
   references (BLT, H-Net) are both pre-2026 and nothing from 2026 re-measures them.
10. **No 2026 work on sequence packing as data efficiency, document boundaries interacting with SSM
    recurrent state, or context-length curricula.** Two targeted discovery passes returned only
    adjacent systems papers. Any packing or state-reset decision here is an unguided empirical
    choice — which is the honest basis for §7's advice to measure the boundary-spanning rate first.
11. **Relation's FlashRelation has no published backward pass and no released code.** The backward in
    `flash_relation.py` is derived in-repo, which is why `tests/test_flash_relation.py` carries more
    weight than a typical kernel test.
12. **No small-batch or short-sequence analysis in Mamba-3.** Every benchmark is batch 128, and Table
    2's caption states the batch and head dimensions cancel — so the 24-CTA occupancy finding in §4.3
    is not something that paper would have surfaced.

### Leads not verified — do not cite as read

From bibliographies and discovery-tool listings only; I did not retrieve or read these, and did not
confirm the IDs resolve. Recorded so the trail is reproducible, not as citations:
**RMNP 2603.20527** (the most interesting — Tier I in OmniOpt at near-AdamW cost and half AdamW's
state), MuonEq 2603.28254, TrasMuon 2602.13498, Newton-Muon 2604.01472, Aurora 2606.27715,
AdaMTP 2608.00434, Windowed-MTP 2607.21535, LoopMTP 2608.03624, Muse 2607.14536, MALT 2608.05088,
Muon non-convergence 2608.04607, SAM+Muon 2607.26001, Libra 2607.23250, RoutePack 2608.12146,
Right Reset 2608.04330, Stable FP4 2607.24953, Full-Stack FP4 2607.04422, M+Adam 2607.10611,
bf16 collapse 2608.02091, GNMR 2606.00539, MXFP4 pretraining 2605.09825, FBLayout 2607.21624.

One caution carried forward from the search: a discovery tool returned an entry with the identifier
`2607.nvfp4-rl`. That is an alphaXiv-internal id, **not a valid arXiv ID** — do not cite it.
