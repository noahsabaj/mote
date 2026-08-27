# Mote-138M on 8 GB, and what the fused spine kernel bought (2026-08-26 night)

Step 0/3 of `scripts/spine_gate.sh`, run in the window between the T3 E4 arm finishing and E8
starting. `nvidia-smi` free, 7.65 GiB visible to torch, batch 1, `--chunk-bytes 6`.

## The shape does not fit as the gate assumed

The gate estimated ~5.04 GB for Mote-138M at 16384 from Mote-96M's measured 4.31 GB. Measured,
**without activation checkpointing it does not fit at all — not even with the spine off**:

| seq   | `--ckpt-main` | spine  | peak GB | verdict |
|-------|---------------|--------|---------|---------|
| 16384 | no            | off    | > 7.65  | OOM     |
| 16384 | no            | frac   | > 7.65  | OOM     |
| 16384 | no            | expand | > 7.65  | OOM     |

`--ckpt-main` is therefore not a tuning knob at this shape, it is a requirement. The gate's arms
do not pass it, so all three would have OOM'd at step 1/3 after the profile was skipped.

## With `--ckpt-main`, before and after the fused kernel

| seq   | spine  | peak GB (before → after) | B/s (before → after) | vs `off` |
|-------|--------|--------------------------|----------------------|----------|
| 8192  | off    | 3.85                     | 59 905               | 1.00×    |
| 8192  | frac   | 4.25 → **4.12**          | 49 185 → **51 158**  | 0.85×    |
| 8192  | expand | 5.25 → **4.74**          | 35 761 → **41 336**  | 0.69×    |
| 16384 | off    | 5.18                     | 59 986               | 1.00×    |
| 16384 | frac   | 5.98 → **5.74**          | 48 796 → **51 120**  | 0.85×    |
| 16384 | expand | OOM                      | OOM                  | —        |

flops/byte is 231.1 (8192) and ~268 (16384) for all three modes: the spine adds no arithmetic
worth counting, so every millisecond of the gap was bandwidth and launch overhead.

## Two of the gate's premises were wrong

- **"frac … +0 GB"** — measured **+0.56 GB at 16384** (+0.27 at 8192) after the kernel. The
  residual really is the same width; the cost is the spine's own byte-resolution intermediates at
  seven sites, which `--ckpt-main` does not cover because it checkpoints the Relation blocks.
- **"expand … +0.66 GB"** — measured **+0.89 GB at 8192**, and at 16384 it does not fit on this
  card at all. n=4 expanded is not locally measurable at the flagship context.

## Where the cost was, and what the kernel removed

Per-op CUDA diff, `off` → `expand`, seq 8192, before the kernel:

```
aten::bmm         0 -> 214 ms (800 calls)   the two einsums, inner dimension 4
aten::copy_     122 -> 271 ms               reshape/permute materialisations feeding that bmm
elementwise       0 -> 253 ms               the sigmoid/scale/add chain
aten::add_        0 -> 115 ms
aten::mul        71 -> 169 ms
```

Attribution was clean: encoder 61.4 → 153.5 ms, decoder 58.3 → 153.4 ms, and `main_network`
unchanged at 157.3 ms — the spine does not touch it, and the profile agrees.

`mote/model/spine_kernel.py` fuses all four spine contractions into one kernel (they are one
function). It is correct to one fp32 ulp — 1.27e-7 relative for expand, exactly 0 for frac — on
the tensors the model actually passes it, forward and in all four gradients, and is deterministic
on rerun because dH and dp close inside the program with no atomics.

The remaining 0.85× for frac is now `coefficients` — `gen_norm` plus the `phi` projection at seven
sites — which is not fused. That is the next kernel if the gate clears.

## A precision bug the einsum was hiding

`torch.einsum` is on autocast's lower-precision list. The unfused spine therefore computed its
stream mix in **bf16 at every one of the seven sites**, then added an fp32 update on top — so the
accumulator looked fp32 while the mixing was not. Against a forced-fp32 reference that is a
**2.6e-1 to 4.9e-1 relative** difference in the logits of an untrained model, not rounding.

The kernel is fp32 throughout and is the more precise path. Two consequences, both now tested:

- An end-to-end fused-vs-reference comparison measures *that* difference, not the kernel, unless
  the reference is forced out of autocast. This is what made the first end-to-end check read FAIL
  at 3e-1 when the op itself was exact.
- `read` had been relying on the same implicit demotion to hand its sublayer a bf16 `u`. An fp32
  `u` reaches the Relation kernel as an fp32 `p1` against a bf16 `info` and fails its `tl.dot` at
  compile time. The cast is now explicit; `write` keeps X in fp32, where the design puts it.
