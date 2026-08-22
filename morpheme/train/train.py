"""Pretraining loop for the byte-level H-Net.

    L = CE_nbp + λ1 · CE_mbp + α · L_ratio(N(step))

Features: bf16 autocast, AdamW(0.9, 0.95) with per-stage LR multipliers and a no-decay set,
warmup-stable-decay schedule, gradient clipping, ATDC target-ratio schedule, periodic evaluation
(val bits/byte, bytes-per-chunk, boundary/word alignment, multi-byte-head accuracy, a chunked text
sample), JSONL logging, atomic checkpoints every N minutes with auto-resume, and a wall-clock budget.

    python -m morpheme.train.train --preset pilot --data data/fineweb_edu_pilot --out runs/pilot \
        --batch-size 16 --seq-len 2048 --max-minutes 60
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from ..config import MorphemeConfig
from ..data.loader import ByteShard
from ..model.dc import atdc_target_ratio, bytes_per_chunk, ratio_loss
from ..model.hnet import HNetForCausalLM
from ..tokenizer import ByteTokenizer

LN2 = math.log(2.0)


# --------------------------------------------------------------------------------------
def build_optimizer(model: HNetForCausalLM, lr: float, weight_decay: float, stage_lr_mult, betas=(0.9, 0.95)):
    groups = model.stage_param_groups()
    param_groups = []
    for stage, params in groups.items():
        decay, no_decay = [], []
        for p in params:
            if not p.requires_grad:
                continue
            if p.ndim < 2 or getattr(p, "_no_weight_decay", False):
                no_decay.append(p)
            else:
                decay.append(p)
        mult = stage_lr_mult[stage]
        if decay:
            param_groups.append({"params": decay, "weight_decay": weight_decay, "lr_mult": mult, "stage": stage})
        if no_decay:
            param_groups.append({"params": no_decay, "weight_decay": 0.0, "lr_mult": mult, "stage": stage})
    opt = torch.optim.AdamW(param_groups, lr=lr, betas=betas, eps=1e-8, fused=torch.cuda.is_available())
    return opt


def wsd_lr(step: int, total: int, base: float, warmup_frac: float = 0.1, decay_frac: float = 0.2, min_ratio: float = 0.1) -> float:
    warm = max(int(total * warmup_frac), 1)
    decay_start = int(total * (1 - decay_frac))
    if step < warm:
        return base * (step + 1) / warm
    if step < decay_start:
        return base
    # inverse-square-root style decay to min_ratio over the last decay_frac
    t = (step - decay_start) / max(total - decay_start, 1)
    return base * max(min_ratio, 1.0 - (1.0 - min_ratio) * math.sqrt(t))


def set_lr(opt, lr: float):
    for g in opt.param_groups:
        g["lr"] = lr * g["lr_mult"]


# --------------------------------------------------------------------------------------
def compute_losses(model: HNetForCausalLM, batch: torch.Tensor, target_ratio: float, mbp_weight: float, ratio_weight: float):
    inputs, targets = batch[:, :-1], batch[:, 1:]
    out = model(inputs)
    V = out.logits.shape[-1]
    ce = F.cross_entropy(out.logits.reshape(-1, V).float(), targets.reshape(-1))
    mask = torch.ones_like(inputs, dtype=torch.bool)
    lr_ = ratio_loss(out.routing.boundary_prob, out.routing.boundary_mask, mask, target_ratio)
    loss = ce + ratio_weight * lr_
    stats = {"ce": ce.item(), "ratio": lr_.item(), "bpic": bytes_per_chunk(out.routing.boundary_mask, mask)}
    if out.mbp_logits is not None and mbp_weight > 0:
        ce_m = F.cross_entropy(out.mbp_logits.reshape(-1, V).float(), targets.reshape(-1))
        loss = loss + mbp_weight * ce_m
        stats["ce_mbp"] = ce_m.item()
    return loss, stats, out


@torch.no_grad()
def evaluate(model: HNetForCausalLM, shard: ByteShard, batch_size: int, seq_len: int, max_batches: int, device, target_ratio: float):
    model.eval()
    tot_nll, tot_tok, tot_bytes, tot_chunks, mbp_correct, mbp_tot = 0.0, 0, 0, 0, 0, 0
    word_hits, boundary_count = 0, 0
    for batch in shard.sequential_batches(batch_size, seq_len, max_batches):
        batch = batch.to(device, non_blocking=True)
        inputs, targets = batch[:, :-1], batch[:, 1:]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(inputs)
        V = out.logits.shape[-1]
        nll = F.cross_entropy(out.logits.reshape(-1, V).float(), targets.reshape(-1), reduction="sum")
        tot_nll += nll.item()
        tot_tok += targets.numel()
        bm = out.routing.boundary_mask
        tot_bytes += bm.numel()
        tot_chunks += bm.sum().item()
        # boundary alignment: fraction of boundaries (excluding position 0) whose previous byte is space/newline/punct
        prev = inputs[:, :-1]
        is_sep = (prev == 32) | (prev == 10) | ((prev >= 33) & (prev <= 47)) | ((prev >= 58) & (prev <= 64))
        b_inner = bm[:, 1:]
        word_hits += (b_inner & is_sep).sum().item()
        boundary_count += b_inner.sum().item()
        if out.mbp_logits is not None:
            pred = out.mbp_logits.argmax(-1)
            mbp_correct += (pred == targets).sum().item()
            mbp_tot += targets.numel()
    model.train()
    res = {
        "val_bpb": tot_nll / max(tot_tok, 1) / LN2,
        "val_bpic": tot_bytes / max(tot_chunks, 1),
        "target_ratio": target_ratio,
        "boundary_on_separator_frac": word_hits / max(boundary_count, 1),
    }
    if mbp_tot:
        res["mbp_top1_acc"] = mbp_correct / mbp_tot
    return res


@torch.no_grad()
def chunk_sample(model: HNetForCausalLM, text: str, device) -> str:
    """Render learned chunk boundaries of a text with '|' (for eyeballing word-likeness)."""
    tok = ByteTokenizer()
    ids = torch.tensor([tok.encode(text, add_bos=True)], device=device)
    model.eval()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        out = model(ids)
    model.train()
    bm = out.routing.boundary_mask[0].tolist()
    raw = ids[0].tolist()
    pieces = []
    for i, (b, t) in enumerate(zip(bm, raw)):
        if b and i > 0:
            pieces.append("|")
        if t < 256:
            pieces.append(chr(t) if 32 <= t < 127 else "·")
    return "".join(pieces)


# --------------------------------------------------------------------------------------
def save_checkpoint(path: Path, model, opt, step: int, cfg: MorphemeConfig, extra: Dict):
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step, "config": cfg.to_dict(), "extra": extra}, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path, model, opt=None):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    if opt is not None and "optimizer" in ck:
        opt.load_state_dict(ck["optimizer"])
    return ck["step"], ck.get("extra", {})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="pilot", choices=["pilot", "flagship"])
    ap.add_argument("--config", default=None, help="JSON config overriding the preset")
    ap.add_argument("--data", required=True, help="shard prefix, e.g. data/fineweb_edu_pilot")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--stage-lr-mult", type=float, nargs=2, default=[2.0, 1.0], help="outer, main")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=0, help="0 = derive from --max-minutes after a throughput probe")
    ap.add_argument("--max-minutes", type=float, default=60.0)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--ckpt-minutes", type=float, default=10.0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = MorphemeConfig.load(args.config) if args.config else getattr(MorphemeConfig, args.preset)()
    cfg.max_seq_len = max(cfg.max_seq_len, args.seq_len)
    cfg.save(out_dir / "config.json")
    (out_dir / "run.json").write_text(json.dumps({**vars(args), "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2))

    model = HNetForCausalLM(cfg, device=device)
    n_params = model.num_params()
    print(f"params: {n_params/1e6:.2f}M | device: {device} | kernels: mamba3={__import__('morpheme.model.mamba3', fromlist=['x']).HAS_MAMBA3_KERNEL} ssd={__import__('morpheme.model.dc', fromlist=['x']).HAS_SSD_KERNEL}", flush=True)
    opt = build_optimizer(model, args.lr, args.weight_decay, args.stage_lr_mult)

    train_shard = ByteShard(args.data, "train")
    val_shard = ByteShard(args.data, "val")
    gen = torch.Generator().manual_seed(args.seed)

    step, t_start = 0, time.time()
    ckpt_path = out_dir / "last.pt"
    extra = {}
    if args.resume and ckpt_path.exists():
        step, extra = load_checkpoint(ckpt_path, model, opt)
        if "generator_state" in extra:
            gen.set_state(torch.tensor(extra["generator_state"], dtype=torch.uint8))
        print(f"resumed from step {step}", flush=True)

    fwd = torch.compile(model) if args.compile else model
    log_f = open(out_dir / "log.jsonl", "a", encoding="utf-8")

    def log(rec: Dict):
        rec["step"] = step
        rec["elapsed_min"] = (time.time() - t_start) / 60
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()
        print(json.dumps(rec), flush=True)

    # ---- total steps: fixed, or derived from a short throughput probe ----------------------------
    total_steps = args.max_steps
    tokens_per_step = args.batch_size * args.seq_len * args.grad_accum
    model.train()

    def train_step(target_ratio: float):
        opt.zero_grad(set_to_none=True)
        agg = {}
        for _ in range(args.grad_accum):
            batch = train_shard.sample_batch(args.batch_size, args.seq_len, gen).to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss, stats, _ = compute_losses(fwd, batch, target_ratio, cfg.mbp.loss_weight, cfg.dc.ratio_loss_weight)
            (loss / args.grad_accum).backward()
            for k, v in stats.items():
                agg[k] = agg.get(k, 0.0) + v / args.grad_accum
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip).item()
        opt.step()
        agg["grad_norm"] = gnorm
        return agg

    if total_steps == 0:
        probe_steps = 5
        set_lr(opt, args.lr * 0.01)
        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.time()
        for _ in range(probe_steps):
            train_step(cfg.dc.target_ratio_init)
        torch.cuda.synchronize() if device.type == "cuda" else None
        sec_per_step = (time.time() - t0) / probe_steps
        budget_sec = args.max_minutes * 60 - (time.time() - t_start)
        total_steps = max(int(budget_sec / sec_per_step * 0.9), 50)
        log({"probe_sec_per_step": sec_per_step, "bytes_per_sec": tokens_per_step / sec_per_step, "total_steps": total_steps})

    last_ckpt = time.time()
    t_log = time.time()
    while step < total_steps:
        target_ratio = atdc_target_ratio(step, total_steps, cfg.dc.target_ratio_init, cfg.dc.target_ratio_final, cfg.dc.schedule_warmup_frac)
        lr = wsd_lr(step, total_steps, args.lr)
        set_lr(opt, lr)
        stats = train_step(target_ratio)
        step += 1
        if step % args.log_every == 0:
            dt = time.time() - t_log
            t_log = time.time()
            rec = {"lr": lr, "target_ratio": target_ratio, "bytes_per_sec": tokens_per_step * args.log_every / dt, "train_bpb": stats["ce"] / LN2}
            rec.update(stats)
            log(rec)
        if step % args.eval_every == 0 or step == total_steps:
            ev = evaluate(model, val_shard, args.batch_size, args.seq_len, args.eval_batches, device, target_ratio)
            ev["sample"] = chunk_sample(model, "The router compares each byte with the one before it. Where they stop looking alike, it draws a boundary.", device)
            log({"eval": ev})
        if (time.time() - last_ckpt) / 60 >= args.ckpt_minutes or step == total_steps:
            save_checkpoint(ckpt_path, model, opt, step, cfg, {"generator_state": gen.get_state().tolist(), "total_steps": total_steps, "n_params": n_params, "bytes_seen": step * tokens_per_step})
            last_ckpt = time.time()
            log({"checkpoint": str(ckpt_path)})
        if (time.time() - t_start) / 60 > args.max_minutes:
            log({"stopped": "time budget"})
            break

    save_checkpoint(ckpt_path, model, opt, step, cfg, {"generator_state": gen.get_state().tolist(), "total_steps": total_steps, "n_params": n_params, "bytes_seen": step * tokens_per_step})
    log({"done": True, "final_step": step})
    log_f.close()


if __name__ == "__main__":
    main()
