# 16384 gate sweep — 2026-08-28

Frozen recipe (`--preset flagship --data data/flagship_mix --seq-len 16384 --batch-size 1 --grad-accum 4 --bucket 64 --ckpt-main`), one `profile_step` process per chunk rate, engine parked on the CPU.
Pass bar: peak ≤ 6.2 GB (the daemon's share beside the locked desktop, docs/results/2026-08-24-evening-gates.md). The untrained router segments real text at 2.1–2.4 bytes/chunk, so the 2.1 row is step 1 of the trunk; 3.3 is where trained routers sit; 6.5 is only ATDC's target.

| bytes/chunk | chunks / 16384 | peak GB | fits ≤ 6.2 | B/s | s/step | MFU | FLOPs/byte (M) |
|---|---|---|---|---|---|---|---|
| 2.1 | 7801 | 4.71 | yes | 29025 | 2.2579 | 0.474 | 719.4 |
| 2.5 | 6553 | 4.63 | yes | 35124 | 1.8658 | 0.446 | 559.7 |
| 3.3 | 4964 | 4.53 | yes | 46973 | 1.3952 | 0.412 | 386.7 |
| 4.0 | 4096 | 4.47 | yes | 56552 | 1.1589 | 0.393 | 306.5 |

## Reading

- **Memory is flat in the rate**: 4.47 → 4.71 GB from 4.0 down to 2.1 bytes/chunk. With `--ckpt-main` the
  Relation main network is checkpointed, so the peak is the byte-level Mamba-3 tape, which does not depend
  on the chunk count. Step 1 of the trunk (untrained router, 2.1–2.4) fits with 1.5 GB to spare against
  the 6.2 GB desktop-locked ceiling; the daemon's own CUDA context (~0.3 GB) still leaves >1 GB.
  **No init guardrail is needed for memory**, so bounded routing ships as it is: unbound by default, a
  path that is bit-identical to today's router when it does not bind (tests/test_bounded_routing.py). The
  35M A/B that was to test "what the trunk runs" has nothing to measure and is not queued.
- **Throughput is not flat**: 29.0 → 56.6 KB/s over the same span, 47.0 KB/s at the trained rate 3.3.
  The freeze's 68.1 KB/s (docs/results/2026-08-23-fedora-day1.md) was measured at a forced 6.0 — ATDC's
  target, not a rate any Mote router has produced. **The trunk's constant is 47 KB/s**: 7 days ≈ 28 GB
  ≈ 2.8 passes over the 10 GB mix A (the plan had 4), and the first hours run nearer 30 KB/s while the
  router's rate climbs from ~2.2 to ~3.3. `docs/shape.md`'s "68.1 KB/s @ 4.31 GB" should be read as
  "47 KB/s @ 4.5 GB".
- MFU falls as the rate rises (0.474 → 0.393): the main network is the well-utilised part, and fewer chunks
  mean less of it per byte.
