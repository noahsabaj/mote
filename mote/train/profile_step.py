"""Profile one training step: where the time goes, per module and per CUDA kernel.

    python -m mote.train.profile_step --preset local --data ~/data/local_mix --batch-size 2 --seq-len 2048

Prints achieved TFLOPS / MFU, the forward time per top-level module (record_function ranges via
hooks), backward and optimizer time, the 25 most expensive CUDA kernels, and peak memory. A Chrome
trace is written next to the report (--trace).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from ..config import PRESETS, MoteConfig, normalize_preset, resolve_preset
from ..data.loader import ByteShard
from ..model.hnet import HNetForCausalLM
from .flops import flops_per_byte, peak_tflops_for
from .train import compute_losses

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="mote-35m", type=normalize_preset, help=", ".join(PRESETS))
    ap.add_argument("--data", required=True)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--dense-mbp", action="store_true", help="dense masked attention in the multi-byte head (reference path)")
    ap.add_argument("--no-mbp", action="store_true", help="build the model without the multi-byte head")
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--timed", type=int, default=6, help="unprofiled steps used for the throughput number")
    ap.add_argument("--ckpt-main", action="store_true", help="activation checkpointing on the Relation blocks")
    ap.add_argument("--no-flash", action="store_true", help="materialized Relation instead of the Triton kernel")
    ap.add_argument("--bucket", type=int, default=None, help="override chunk-count bucket (1 = off)")
    ap.add_argument("--trace", default=None, help="write a Chrome trace json here")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--chunk-bytes", type=float, default=None, help="profiling only: force a boundary every N bytes (trained routers land near 5-6), so a preset with no checkpoint is profiled at a realistic chunk count")
    ap.add_argument("--spine", default=None, choices=["off", "expand", "frac"], help="hyper-connection spine mode (mote/model/spine.py)")
    ap.add_argument("--spine-n", type=int, default=None, help="streams (expand) or slices (frac)")
    ap.add_argument("--out", default=None, help="also write the result json here")
    ap.add_argument("--init-from", default=None, help="profile a trained checkpoint: a random router makes ~9 chunks per 2048 bytes, a trained one ~370, and the main network and dechunk costs scale with that")
    args = ap.parse_args(argv)

    device = torch.device(args.device)
    cfg: MoteConfig = resolve_preset(args.preset)
    if args.spine is not None:
        cfg.spine.mode = args.spine
    if args.spine_n is not None:
        cfg.spine.n = args.spine_n
    if args.bucket is not None:
        cfg.dc.chunk_bucket = args.bucket
    if args.no_flash:
        import mote.model.relation as R

        R.USE_FLASH = False
    if args.dense_mbp:
        import mote.model.mbp as MBP

        MBP.USE_BLOCK_LOCAL = False
    if args.no_mbp:
        cfg.mbp.enabled = False
    torch.manual_seed(0)
    model = HNetForCausalLM(cfg, device=device)  # fp32 parameters under autocast, exactly like the trainer
    if args.init_from:
        ck = torch.load(args.init_from, map_location="cpu", weights_only=False)
        cfg = MoteConfig.from_dict(ck["config"])
        if args.bucket is not None:
            cfg.dc.chunk_bucket = args.bucket
        if args.no_mbp:
            cfg.mbp.enabled = False
        model = HNetForCausalLM(cfg, device=device)
        model.load_state_dict(ck["model"], strict=False)
    if args.chunk_bytes:
        # The router's own p still flows (its parameters get gradients as in training); only the
        # selection is forced to a fixed period, which is what sets the main-network and dechunk cost.
        from mote.model.dc import RoutingOutput

        router = model.routing_module
        real_forward = router.forward
        period = float(args.chunk_bytes)

        def forced_forward(hidden, mask, state=None):
            out = real_forward(hidden, mask, state)
            pos = torch.arange(hidden.shape[1], device=hidden.device, dtype=torch.float32)
            forced = ((pos / period).floor() != ((pos - 1) / period).floor())[None, :] & mask
            sel = forced.long()
            return RoutingOutput(out.boundary_prob, forced, out.boundary_prob.gather(-1, sel.unsqueeze(-1)))

        router.forward = forced_forward
    if args.ckpt_main:
        model.main_network.grad_checkpoint = True
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), fused=device.type == "cuda")
    shard = ByteShard(args.data, "train")
    gen = torch.Generator().manual_seed(0)

    # forward-time attribution: one record_function range per top-level module
    ranges = {}

    def pre(name):
        def hook(mod, inp):
            rf = record_function(name)
            rf.__enter__()
            ranges[name] = rf
        return hook

    def post(name):
        def hook(mod, inp, out):
            ranges.pop(name).__exit__(None, None, None)
        return hook

    for name in ["embeddings", "encoder", "routing_module", "chunk_layer", "main_network", "dechunk_layer", "decoder", "lm_head", "mbp_head"]:
        mod = getattr(model, name, None)
        if mod is not None:
            mod.register_forward_pre_hook(pre("fwd:" + name))
            mod.register_forward_hook(post("fwd:" + name))

    def step():
        opt.zero_grad(set_to_none=True)
        bpic = 0.0
        for _ in range(args.grad_accum):
            batch, _ = shard.sample_batch(args.batch_size, args.seq_len, gen)
            batch = batch.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                with record_function("forward+loss"):
                    loss, n, stats, _ = compute_losses(model, batch, cfg.dc.target_ratio_init, cfg.mbp.loss_weight, cfg.dc.ratio_loss_weight, None)
            with record_function("backward"):
                (loss / (n * args.grad_accum)).backward()
            bpic = bpic + stats["bpic"] / args.grad_accum
        with record_function("optimizer"):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        return bpic

    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    for _ in range(args.warmup):
        step()
    sync()
    t0 = time.time()
    bpic = 0.0
    for _ in range(args.timed):
        bpic += step() / args.timed
    sync()
    sec = (time.time() - t0) / args.timed
    bpic = float(bpic)
    bytes_per_step = args.batch_size * args.seq_len * args.grad_accum
    fl = flops_per_byte(model, args.seq_len, bpic)
    tflops = fl * bytes_per_step / sec / 1e12
    peak = peak_tflops_for(device) if device.type == "cuda" else None
    result = {
        "preset": args.preset, "spine": cfg.spine.mode, "spine_n": cfg.spine.n, "params": model.num_params(), "batch": args.batch_size, "seq_len": args.seq_len,
        "grad_accum": args.grad_accum, "bucket": cfg.dc.chunk_bucket, "flash": not args.no_flash, "ckpt_main": args.ckpt_main,
        "block_local_mbp": not args.dense_mbp, "mbp": not args.no_mbp,
        "sec_per_step": round(sec, 4), "bytes_per_sec": round(bytes_per_step / sec), "bytes_per_chunk": round(bpic, 2),
        "flops_per_byte_M": round(fl / 1e6, 1), "tflops": round(tflops, 2), "mfu": round(tflops / peak, 3) if peak else None,
        "peak_mem_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device.type == "cuda" else None,
    }
    print(json.dumps(result, indent=1), flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=1))

    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device.type == "cuda" else [])
    with profile(activities=activities, record_shapes=False, profile_memory=False, with_stack=False) as prof:
        step()
        sync()
    key = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
    rows = prof.key_averages()
    print("\n== forward per module / phases (device time, ms) ==")
    for r in rows:
        if r.key.startswith("fwd:") or r.key in ("forward+loss", "backward", "optimizer"):
            t = (r.device_time_total if device.type == "cuda" else r.cpu_time_total) / 1000
            print(f"  {r.key:<24} {t:8.2f}")
    print("\n== top CUDA kernels ==")
    print(rows.table(sort_by=key, row_limit=25, max_name_column_width=70))
    if args.trace:
        prof.export_chrome_trace(args.trace)
        print("trace written:", args.trace)


if __name__ == "__main__":
    main()
