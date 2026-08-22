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
def _masked_ce(logits: torch.Tensor, targets: torch.Tensor, loss_mask: Optional[torch.Tensor]) -> torch.Tensor:
    V = logits.shape[-1]
    if loss_mask is None:
        return F.cross_entropy(logits.reshape(-1, V).float(), targets.reshape(-1))
    per = F.cross_entropy(logits.reshape(-1, V).float(), targets.reshape(-1), reduction="none")
    w = loss_mask.reshape(-1).float()
    return (per * w).sum() / w.sum().clamp(min=1.0)


def compute_losses(model: HNetForCausalLM, batch: torch.Tensor, target_ratio: float, mbp_weight: float, ratio_weight: float, loss_mask: Optional[torch.Tensor] = None):
    inputs, targets = batch[:, :-1], batch[:, 1:]
    tmask = loss_mask[:, 1:] if loss_mask is not None else None  # mask is per target position
    out = model(inputs)
    ce = _masked_ce(out.logits, targets, tmask)
    mask = torch.ones_like(inputs, dtype=torch.bool)
    lr_ = ratio_loss(out.routing.boundary_prob, out.routing.boundary_mask, mask, target_ratio)
    loss = ce + ratio_weight * lr_
    stats = {"ce": ce.item(), "ratio": lr_.item(), "bpic": bytes_per_chunk(out.routing.boundary_mask, mask)}
    if out.mbp_logits is not None and mbp_weight > 0:
        ce_m = _masked_ce(out.mbp_logits, targets, tmask)
        loss = loss + mbp_weight * ce_m
        stats["ce_mbp"] = ce_m.item()
    return loss, stats, out


@torch.no_grad()
def evaluate(model: HNetForCausalLM, shard: ByteShard, batch_size: int, seq_len: int, max_batches: int, device, target_ratio: float):
    model.eval()
    tot_nll, tot_tok, tot_bytes, tot_chunks, mbp_correct, mbp_tot = 0.0, 0, 0, 0, 0, 0
    word_hits, boundary_count = 0, 0
    for batch, lmask in shard.sequential_batches(batch_size, seq_len, max_batches):
        batch = batch.to(device, non_blocking=True)
        inputs, targets = batch[:, :-1], batch[:, 1:]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(inputs)
        V = out.logits.shape[-1]
        nll = F.cross_entropy(out.logits.reshape(-1, V).float(), targets.reshape(-1), reduction="none")
        if lmask is not None:
            w = lmask[:, 1:].to(device).reshape(-1).float()
            tot_nll += (nll * w).sum().item()
            tot_tok += int(w.sum().item())
        else:
            tot_nll += nll.sum().item()
            tot_tok += targets.numel()
        bm = out.routing.boundary_mask
        tot_bytes += bm.numel()
        tot_chunks += bm.sum().item()
        # boundary/word alignment (excluding position 0): a boundary is "word-aligned" if its previous byte is a
        # separator (space/newline/punct) OR the boundary byte itself is one (chunks that start with the space).
        def sep(t):
            return (t == 32) | (t == 10) | ((t >= 33) & (t <= 47)) | ((t >= 58) & (t <= 64))

        prev, cur = inputs[:, :-1], inputs[:, 1:]
        b_inner = bm[:, 1:]
        word_hits += (b_inner & (sep(prev) | sep(cur))).sum().item()
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


def ckpt_path_exists(out_dir: Path) -> bool:
    return (out_dir / "last.pt").exists()


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
    ap.add_argument("--sft", action="store_true", help="train on an SFT shard (assistant-byte loss mask)")
    ap.add_argument("--init-from", default=None, help="checkpoint to initialize weights from (e.g. pretrain -> SFT)")
    ap.add_argument("--ratio-weight", type=float, default=None, help="override dc.ratio_loss_weight (α)")
    ap.add_argument("--target-ratio", type=float, nargs=2, default=None, metavar=("INIT", "FINAL"), help="override the ATDC target-ratio schedule endpoints")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = MorphemeConfig.load(args.config) if args.config else getattr(MorphemeConfig, args.preset)()
    cfg.max_seq_len = max(cfg.max_seq_len, args.seq_len)
    if args.ratio_weight is not None:
        cfg.dc.ratio_loss_weight = args.ratio_weight
    if args.target_ratio is not None:
        cfg.dc.target_ratio_init, cfg.dc.target_ratio_final = args.target_ratio
    cfg.save(out_dir / "config.json")
    (out_dir / "run.json").write_text(json.dumps({**vars(args), "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2))

    model = HNetForCausalLM(cfg, device=device)
    n_params = model.num_params()
    print(f"params: {n_params/1e6:.2f}M | device: {device} | kernels: mamba3={__import__('morpheme.model.mamba3', fromlist=['x']).HAS_MAMBA3_KERNEL} ssd={__import__('morpheme.model.dc', fromlist=['x']).HAS_SSD_KERNEL}", flush=True)
    opt = build_optimizer(model, args.lr, args.weight_decay, args.stage_lr_mult)

    train_shard = ByteShard(args.data, "train", sft=args.sft)
    val_shard = ByteShard(args.data, "val", sft=args.sft)
    if args.init_from and not (args.resume and ckpt_path_exists(out_dir)):
        _step, _ = load_checkpoint(Path(args.init_from), model, None)
        print(f"initialized weights from {args.init_from} (step {_step})", flush=True)
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
            batch, lmask = train_shard.sample_batch(args.batch_size, args.seq_len, gen)
            batch = batch.to(device, non_blocking=True)
            lmask = lmask.to(device, non_blocking=True) if lmask is not None else None
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss, stats, _ = compute_losses(fwd, batch, target_ratio, cfg.mbp.loss_weight, cfg.dc.ratio_loss_weight, lmask)
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
    time_driven = args.max_steps == 0  # schedules follow wall-clock progress; step count is only an estimate
    budget_sec = args.max_minutes * 60

    def progress() -> float:
        if time_driven:
            return min((time.time() - t_start) / budget_sec, 1.0)
        return min(step / max(total_steps, 1), 1.0)

    while (progress() < 1.0) if time_driven else (step < total_steps):
        pr = progress()
        horizon = 1000
        sched_step = int(pr * horizon)
        target_ratio = atdc_target_ratio(sched_step, horizon, cfg.dc.target_ratio_init, cfg.dc.target_ratio_final, cfg.dc.schedule_warmup_frac)
        lr = wsd_lr(sched_step, horizon, args.lr)
        set_lr(opt, lr)
        stats = train_step(target_ratio)
        step += 1
        if step % args.log_every == 0:
            dt = time.time() - t_log
            t_log = time.time()
            rec = {"lr": lr, "target_ratio": target_ratio, "bytes_per_sec": tokens_per_step * args.log_every / dt, "train_bpb": stats["ce"] / LN2}
            rec.update(stats)
            log(rec)
        if step % args.eval_every == 0:
            ev = evaluate(model, val_shard, args.batch_size, args.seq_len, args.eval_batches, device, target_ratio)
            ev["sample"] = chunk_sample(model, "The router compares each byte with the one before it. Where they stop looking alike, it draws a boundary.", device)
            log({"eval": ev})
        if (time.time() - last_ckpt) / 60 >= args.ckpt_minutes:
            save_checkpoint(ckpt_path, model, opt, step, cfg, {"generator_state": gen.get_state().tolist(), "total_steps": total_steps, "n_params": n_params, "bytes_seen": step * tokens_per_step})
            last_ckpt = time.time()
            log({"checkpoint": str(ckpt_path)})
        if (time.time() - t_start) / 60 > args.max_minutes:
            log({"stopped": "time budget"})
            break
    ev = evaluate(model, val_shard, args.batch_size, args.seq_len, args.eval_batches, device, cfg.dc.target_ratio_final)
    ev["sample"] = chunk_sample(model, "The router compares each byte with the one before it. Where they stop looking alike, it draws a boundary.", device)
    log({"eval": ev, "final": True})

    save_checkpoint(ckpt_path, model, opt, step, cfg, {"generator_state": gen.get_state().tolist(), "total_steps": total_steps, "n_params": n_params, "bytes_seen": step * tokens_per_step})
    log({"done": True, "final_step": step})
    log_f.close()


if __name__ == "__main__":
    main()
