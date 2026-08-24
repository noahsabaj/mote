"""Pretraining loop for the byte-level H-Net.

    L = CE_nbp + λ1 · CE_mbp + α · L_ratio(N(step))

Features: bf16 autocast, AdamW(0.9, 0.95) with per-stage LR multipliers and a no-decay set,
warmup-stable-decay schedule, gradient clipping, ATDC target-ratio schedule, periodic evaluation
(val bits/byte, bytes-per-chunk, boundary/word alignment, multi-byte-head accuracy, a chunked text
sample), JSONL logging, atomic checkpoints every N minutes with auto-resume, and a wall-clock budget.

    python -m mote.train.train --preset pilot --data data/fineweb_edu_pilot --out runs/pilot \
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

from ..config import MoteConfig
from ..data.loader import ByteShard, MixedShard
from ..model.dc import atdc_target_ratio, bytes_per_chunk, ratio_loss
from ..model.hnet import HNetForCausalLM
from ..tokenizer import ByteTokenizer
from .flops import flops_per_byte, peak_tflops_for
from .muon import Muon, split_muon_params

LN2 = math.log(2.0)


# --------------------------------------------------------------------------------------
class MultiOpt:
    """Several optimizers driven as one (Muon for hidden matrices + AdamW for the rest)."""

    def __init__(self, opts):
        self.opts = opts

    @property
    def param_groups(self):
        return [g for o in self.opts for g in o.param_groups]

    def zero_grad(self, set_to_none=True):
        for o in self.opts:
            o.zero_grad(set_to_none=set_to_none)

    def step(self):
        for o in self.opts:
            o.step()

    def state_dict(self):
        return {"multi": [o.state_dict() for o in self.opts]}

    def load_state_dict(self, sd):
        for o, s in zip(self.opts, sd["multi"]):
            o.load_state_dict(s)


def build_optimizer(model: HNetForCausalLM, lr: float, weight_decay: float, stage_lr_mult, betas=(0.9, 0.95), optimizer: str = "adamw"):
    groups = model.stage_param_groups()
    muon_ids = {id(p) for p in split_muon_params(model)[0]} if optimizer in ("muon", "muonsw") else set()
    adam_groups, muon_groups = [], []
    for stage, params in groups.items():
        decay, no_decay, muon = [], [], []
        for p in params:
            if not p.requires_grad:
                continue
            if id(p) in muon_ids:
                muon.append(p)
            elif p.ndim < 2 or getattr(p, "_no_weight_decay", False):
                no_decay.append(p)
            else:
                decay.append(p)
        mult = stage_lr_mult[stage]
        if decay:
            adam_groups.append({"params": decay, "weight_decay": weight_decay, "lr_mult": mult, "stage": stage})
        if no_decay:
            adam_groups.append({"params": no_decay, "weight_decay": 0.0, "lr_mult": mult, "stage": stage})
        if muon:
            muon_groups.append({"params": muon, "weight_decay": weight_decay, "lr_mult": mult, "stage": stage})
    adam = torch.optim.AdamW(adam_groups, lr=lr, betas=betas, eps=1e-8, fused=torch.cuda.is_available())
    if not muon_groups:
        return adam
    # lr_max for Muon-SW is the schedule's peak LR times the group's multiplier; set_lr scales `lr` by the
    # same multiplier, so η_t/η_max is the schedule's own fraction.
    for g in muon_groups:
        g["lr_max"] = lr * g["lr_mult"]
    return MultiOpt([adam, Muon(muon_groups, lr=lr, momentum=0.95, nesterov=True, weight_decay=weight_decay, sw_decay=(optimizer == "muonsw"))])


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


def schedule_lr(kind: str, step: int, total: int, base: float, min_ratio: float = 0.1) -> float:
    """The three schedules of the pipeline (docs/shape.md, signed 2026-08-24).

    `wsd`      warmup-stable-decay over the budget — the lab arms.
    `trunk`    warmup (10 %) then constant, never decays — the flagship trunk; cooldowns branch off it.
    `cooldown` decay only: base -> min_ratio over the whole run, no warmup — a branch started with
               `--init-from` a trunk snapshot.
    """
    if kind == "trunk":
        warm = max(int(total * 0.1), 1)
        return base * (step + 1) / warm if step < warm else base
    if kind == "cooldown":
        t = min(step / max(total, 1), 1.0)
        return base * max(min_ratio, 1.0 - (1.0 - min_ratio) * math.sqrt(t))
    return wsd_lr(step, total, base, min_ratio=min_ratio)


def parse_mix_spec(spec: str):
    """`PREFIX:SHARE[:plain]` -> (prefix, share, plain). `plain` reads an SFT shard's bytes without its
    loss mask (chat/identity bytes as ordinary LM data in a cooldown mix)."""
    parts = spec.split(":")
    plain = parts[-1] == "plain"
    if plain:
        parts = parts[:-1]
    return ":".join(parts[:-1]), float(parts[-1]), plain


def set_lr(opt, lr: float):
    for g in opt.param_groups:
        g["lr"] = lr * g["lr_mult"]


# --------------------------------------------------------------------------------------
def _masked_ce_sum(logits: torch.Tensor, targets: torch.Tensor, loss_mask: Optional[torch.Tensor], weights: Optional[torch.Tensor] = None):
    """(sum of per-token CE over counted tokens, number of counted tokens) — both 0-dim tensors.
    `weights` (same shape as targets) scales each token's loss; the count is unweighted."""
    V = logits.shape[-1]
    per = F.cross_entropy(logits.reshape(-1, V).float(), targets.reshape(-1), reduction="none")
    if weights is not None:
        per = per * weights.reshape(-1).float()
    if loss_mask is None:
        return per.sum(), torch.tensor(float(per.numel()), device=per.device)
    w = loss_mask.reshape(-1).float()
    return (per * w).sum(), w.sum()


def compute_losses(model: HNetForCausalLM, batch: torch.Tensor, target_ratio: float, mbp_weight: float, ratio_weight: float, loss_mask: Optional[torch.Tensor] = None):
    """Returns (loss_unnormalised, n_tokens, stats, out). The loss is a SUM over counted tokens (plus the
    ratio loss scaled by the token count), so that gradient accumulation normalises ONCE by the total
    token count across micro-batches — a mean of per-micro-batch means would up-weight windows with few
    assistant bytes (SFT). Every stat is a 0-dim device tensor: nothing here synchronises."""
    inputs, targets = batch[:, :-1], batch[:, 1:]
    tmask = loss_mask[:, 1:] if loss_mask is not None else None  # mask is per target position
    out = model(inputs)
    ce_sum, n = _masked_ce_sum(out.logits, targets, tmask)
    mask = torch.ones_like(inputs, dtype=torch.bool)
    lr_ = ratio_loss(out.routing.boundary_prob, out.routing.boundary_mask, mask, target_ratio)
    n_safe = n.clamp(min=1.0)
    loss = ce_sum + ratio_weight * lr_ * n_safe
    stats = {"ce_sum": ce_sum.detach(), "n": n.detach(), "ratio": lr_.detach(), "bpic": bytes_per_chunk(out.routing.boundary_mask, mask)}
    if out.mbp_logits is not None and mbp_weight > 0:
        gamma = getattr(model.cfg.mbp, "position_gamma", 0.0) if hasattr(model, "cfg") else 0.0
        pw = torch.exp(-out.offset.float() / gamma) if gamma and gamma > 0 else None  # earlier draft slots matter more
        ce_m_sum, _ = _masked_ce_sum(out.mbp_logits, targets, tmask, pw)
        loss = loss + mbp_weight * ce_m_sum
        stats["ce_mbp_sum"] = ce_m_sum.detach()
    return loss, n_safe, stats, out


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
def save_checkpoint(path: Path, model, opt, step: int, cfg: MoteConfig, extra: Dict):
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


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="pilot", choices=["smoke", "pilot", "local", "flagship"])
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
    ap.add_argument("--optimizer", default="adamw", choices=["adamw", "muon", "muonsw"], help="muon: Newton-Schulz updates for hidden 2-D matrices, AdamW for the rest; muonsw: Muon with \u03b7\u00b2-scaled weight decay (2607.23777)")
    ap.add_argument("--ckpt-main", action="store_true", help="activation checkpointing on the Relation blocks (bit-neutral, ~30%% more compute, much less memory)")
    ap.add_argument("--bucket", type=int, default=None, help="chunk-count bucket (default from the preset, 64); 1 = exact shapes")
    ap.add_argument("--no-mbp", action="store_true", help="A/B: train without the multi-byte head")
    ap.add_argument("--mix", action="append", default=[], metavar="PREFIX:SHARE[:plain]", help="extra shard mixed into training by share, e.g. data/sft_identity:0.05 (repeatable); ':plain' reads an SFT shard without its loss mask")
    ap.add_argument("--beta2", type=float, default=0.95, help="AdamW \u03b2\u2082 (2608.16760: the convergence threshold rises as the batch shrinks; A/B 0.99/0.997 at our 32 kB steps)")
    ap.add_argument("--mbp-weight", type=float, default=None, help="\u03bb1 for the multi-byte head loss (preset default 1.0)")
    ap.add_argument("--mbp-gamma", type=float, default=None, help="position weighting exp(-offset/\u03b3) on the head loss (0 = off)")
    ap.add_argument("--mbp-transition", action="store_true", help="add the V\u00d7V byte-transition bias to the head (DSpark Markov head)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--sft", action="store_true", help="train on an SFT shard (assistant-byte loss mask)")
    ap.add_argument("--init-from", default=None, help="checkpoint to initialize weights from (e.g. pretrain -> SFT)")
    ap.add_argument("--ratio-weight", type=float, default=None, help="override dc.ratio_loss_weight (\u03b1)")
    ap.add_argument("--bf16-residual", action="store_true", help="A/B: keep the residual stream in bf16 instead of fp32")
    ap.add_argument("--relation-window", type=int, default=None, help="A/B: each chunk sees at most the last N chunks (materialized path)")
    ap.add_argument("--attention-main", action="store_true", help="ablation: parameter-matched causal attention instead of Relation in the main network")
    ap.add_argument("--jepa", choices=["minimal", "ema", "sigreg"], default=None, help="JEPA aux loss on the byte encoder (lab arms, docs/shape.md 2026-08-24)")
    ap.add_argument("--jepa-weight", type=float, default=0.05, help="weight of the JEPA aux loss")
    ap.add_argument("--no-flash", action="store_true", help="A/B: materialized Relation instead of the Triton kernel")
    ap.add_argument("--target-ratio", type=float, nargs=2, default=None, metavar=("INIT", "FINAL"), help="override the ATDC target-ratio schedule endpoints")
    ap.add_argument("--schedule", default="wsd", choices=["wsd", "trunk", "cooldown"], help="wsd: warmup-stable-decay over the budget (lab arms); trunk: warmup then constant, no decay (the flagship trunk); cooldown: decay only, lr -> 0.1x over the run (a branch started with --init-from a trunk snapshot). docs/shape.md pipeline")
    ap.add_argument("--snapshot-steps", type=int, default=0, help="also keep a weights-only snap_<step>.pt every N steps (the branch points for cooldowns)")
    return ap


class Trainer:
    """The training loop as a drivable object (decided 2026-08-23, docs/shape.md).

    `run()` is a generator that yields ("slice", None) after every accumulation micro-batch and
    ("step", None) after every optimizer step. `main()` simply drains it, which reproduces the old
    monolithic loop exactly; the daemon (mote.serve.jobs) drives it one yield at a time and slips
    generation between slices. `request_stop()` ends the run gracefully at the next step boundary
    (final eval + checkpoint still happen), which is what the studio's Stop button calls.
    """

    def __init__(self, argv_or_args=None):
        args = build_argparser().parse_args(argv_or_args) if (argv_or_args is None or isinstance(argv_or_args, list)) else argv_or_args
        self.args = args
        torch.manual_seed(args.seed)
        self.device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.out_dir = out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)

        cfg = MoteConfig.load(args.config) if args.config else getattr(MoteConfig, args.preset)()
        cfg.max_seq_len = max(cfg.max_seq_len, args.seq_len)
        if args.ratio_weight is not None:
            cfg.dc.ratio_loss_weight = args.ratio_weight
        if args.target_ratio is not None:
            cfg.dc.target_ratio_init, cfg.dc.target_ratio_final = args.target_ratio
        cfg.save(out_dir / "config.json")
        (out_dir / "run.json").write_text(json.dumps({**vars(args), "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2))

        if args.bucket is not None:
            cfg.dc.chunk_bucket = args.bucket
        if args.no_mbp:
            cfg.mbp.enabled = False
        if args.mbp_weight is not None:
            cfg.mbp.loss_weight = args.mbp_weight
        if args.mbp_gamma is not None:
            cfg.mbp.position_gamma = args.mbp_gamma
        if args.mbp_transition:
            cfg.mbp.transition = True
        if args.bf16_residual:
            cfg.residual_in_fp32 = False
        if args.relation_window is not None:
            cfg.main.window_chunks = args.relation_window
        if args.attention_main:
            cfg.main.mixer = "attention"
        if args.no_flash:
            from ..model import relation as _relation
            _relation.USE_FLASH = False
        self.cfg = cfg
        self.model = model = HNetForCausalLM(cfg, device=device)
        if args.ckpt_main:
            model.main_network.grad_checkpoint = True
        self.n_params = model.num_params()
        self.peak_tflops = peak_tflops_for(device) if device.type == "cuda" else None
        print(f"params: {self.n_params/1e6:.2f}M | device: {device} | kernels: mamba3={__import__('mote.model.mamba3', fromlist=['x']).HAS_MAMBA3_KERNEL} ssd={__import__('mote.model.dc', fromlist=['x']).HAS_SSD_KERNEL}", flush=True)
        self.opt = build_optimizer(model, args.lr, args.weight_decay, args.stage_lr_mult, betas=(0.9, args.beta2), optimizer=args.optimizer)

        self.jepa, self.jepa_opt, self._enc_h = None, None, None
        if args.jepa:
            from .jepa import JepaAux

            self.jepa = JepaAux(model, args.jepa, cfg.d_model_outer).to(device)
            model.encoder.register_forward_hook(
                lambda m, i, o: setattr(self, "_enc_h", o if torch.is_tensor(o) else o[0]))
            self.jepa_opt = torch.optim.AdamW(self.jepa.parameters(), lr=args.lr, betas=(0.9, args.beta2), weight_decay=0.0)
            print(f"jepa aux: {args.jepa}, weight {args.jepa_weight}, {sum(q.numel() for q in self.jepa.parameters())/1e6:.2f}M aux params", flush=True)

        train_shard = ByteShard(args.data, "train", sft=args.sft)
        self.val_shard = ByteShard(args.data, "val", sft=args.sft)
        if args.mix:
            extras = []
            for spec in args.mix:
                prefix, share, plain = parse_mix_spec(spec)
                extras.append((ByteShard(prefix, "train", sft=args.sft or plain, plain=plain), share))
            main_w = max(1.0 - sum(w for _, w in extras), 0.0)
            train_shard = MixedShard([train_shard] + [s for s, _ in extras], [main_w] + [w for _, w in extras])
            print("training mix:", {args.data: main_w, **{spec: parse_mix_spec(spec)[1] for spec in args.mix}}, flush=True)
        self.train_shard = train_shard
        if args.init_from and not (args.resume and ckpt_path_exists(out_dir)):
            _step, _ = load_checkpoint(Path(args.init_from), model, None)
            print(f"initialized weights from {args.init_from} (step {_step})", flush=True)
        self.gen = torch.Generator().manual_seed(args.seed)

        self.step, self.t_start = 0, time.time()
        self.sched_total = None  # step horizon of the trunk/cooldown schedules: fixed at the first probe, survives resume
        self.ckpt_path = out_dir / "last.pt"
        if args.resume and self.ckpt_path.exists():
            self.step, extra = load_checkpoint(self.ckpt_path, model, self.opt)
            if args.schedule != "wsd":
                self.sched_total = extra.get("sched_total") or extra.get("total_steps")
            if "generator_state" in extra:
                self.gen.set_state(torch.tensor(extra["generator_state"], dtype=torch.uint8))
            if self.jepa is not None and "jepa" in extra:
                self.jepa.load_state_dict(extra["jepa"])
                self.jepa_opt.load_state_dict(extra["jepa_opt"])
            print(f"resumed from step {self.step}", flush=True)

        self.fwd = torch.compile(model) if args.compile else model
        self.log_f = open(out_dir / "log.jsonl", "a", encoding="utf-8")
        self.tokens_per_step = args.batch_size * args.seq_len * args.grad_accum
        self.total_steps = args.max_steps
        self.time_driven = args.max_steps == 0  # schedules follow wall-clock progress; step count is only an estimate
        self.budget_sec = args.max_minutes * 60
        self._stop = False
        self.stopped_reason = None

    # ------------------------------------------------------------------------------
    def log(self, rec: Dict):
        rec["step"] = self.step
        rec["elapsed_min"] = (time.time() - self.t_start) / 60
        self.log_f.write(json.dumps(rec) + "\n")
        self.log_f.flush()
        print(json.dumps(rec), flush=True)

    def request_stop(self, reason: str = "requested"):
        """Graceful: the run ends at the next step boundary, with the final eval and checkpoint."""
        self._stop = True
        self.stopped_reason = reason

    def save(self):
        extra = {"generator_state": self.gen.get_state().tolist(), "total_steps": self.total_steps,
                 "sched_total": self.sched_total, "schedule": self.args.schedule,
                 "n_params": self.n_params, "bytes_seen": self.step * self.tokens_per_step}
        if self.jepa is not None:
            extra["jepa"] = self.jepa.state_dict()
            extra["jepa_opt"] = self.jepa_opt.state_dict()
        save_checkpoint(self.ckpt_path, self.model, self.opt, self.step, self.cfg, extra)

    def _train_step(self, target_ratio: float):
        """One optimizer step as a generator: yields ("slice", None) after each accumulation micro-batch
        (the daemon's preemption points), returns the stats dict of 0-dim device tensors. Calling
        float() on them is the one sync per logging interval, as before."""
        args, device = self.args, self.device
        self.opt.zero_grad(set_to_none=True)
        if self.jepa_opt is not None:
            self.jepa_opt.zero_grad(set_to_none=True)
        agg = {}
        total_n = None
        for _ in range(args.grad_accum):
            batch, lmask = self.train_shard.sample_batch(args.batch_size, args.seq_len, self.gen)
            batch = batch.to(device, non_blocking=True)
            lmask = lmask.to(device, non_blocking=True) if lmask is not None else None
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss, n, stats, out_fwd = compute_losses(self.fwd, batch, target_ratio, self.cfg.mbp.loss_weight, self.cfg.dc.ratio_loss_weight, lmask)
                if self.jepa is not None:
                    aux, jstats = self.jepa(batch[:, :-1], self._enc_h)
                    loss = loss + args.jepa_weight * aux * n  # same sum-normalisation as CE
                    bp = out_fwd.routing.boundary_prob
                    pb = (bp[..., 1] if bp.dim() == 3 else bp).float().clamp(1e-6, 1 - 1e-6)
                    jstats["jepa_bent"] = (-(pb * pb.log() + (1 - pb) * (1 - pb).log())).mean().detach()
                    stats = {**stats, **jstats}
            loss.backward()  # unnormalised sum; normalised once below
            total_n = n if total_n is None else total_n + n
            for k, v in stats.items():
                agg[k] = v if k not in agg else agg[k] + v
            yield ("slice", None)
        grads = [p.grad for p in self.model.parameters() if p.grad is not None]
        if self.jepa is not None:
            grads += [p.grad for p in self.jepa.parameters() if p.grad is not None]
        torch._foreach_div_(grads, total_n)
        gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
        self.opt.step()
        if self.jepa is not None:
            torch.nn.utils.clip_grad_norm_(self.jepa.parameters(), args.clip)
            self.jepa_opt.step()
            self.jepa.ema_update(self.model)
        out = {"ce": agg["ce_sum"] / total_n, "ratio": agg["ratio"] / args.grad_accum, "bpic": agg["bpic"] / args.grad_accum, "grad_norm": gnorm}
        for k in agg:
            if k.startswith("jepa_"):
                out[k] = agg[k] / args.grad_accum
        if "ce_mbp_sum" in agg:
            out["ce_mbp"] = agg["ce_mbp_sum"] / total_n
        return out

    def _progress(self) -> float:
        if self.time_driven:
            return min((time.time() - self.t_start) / self.budget_sec, 1.0)
        return min(self.step / max(self.total_steps, 1), 1.0)

    def _running(self) -> bool:
        if self.args.schedule == "cooldown":  # a branch ends where its decay ends, by step, resume-proof
            return self.step < self.sched_total
        return (self._progress() < 1.0) if self.time_driven else (self.step < self.total_steps)

    def snapshot(self) -> Path:
        """Weights-only `snap_<step>.pt`: a branch point (`--init-from` loads it; no optimizer state)."""
        p = self.out_dir / f"snap_{self.step:08d}.pt"
        tmp = p.with_suffix(".tmp")
        torch.save({"model": self.model.state_dict(), "step": self.step, "config": self.cfg.to_dict(),
                    "extra": {"sched_total": self.sched_total, "schedule": self.args.schedule}}, tmp)
        os.replace(tmp, p)
        self.log({"snapshot": str(p)})
        return p

    def run(self):
        """The whole run. Drain it (`for _ in t.run(): pass`) for the old behaviour."""
        args, cfg, device = self.args, self.cfg, self.device
        self.model.train()

        if self.total_steps == 0:
            probe_steps = 5
            set_lr(self.opt, args.lr * 0.01)
            torch.cuda.synchronize() if device.type == "cuda" else None
            t0 = time.time()
            for _ in range(probe_steps):
                yield from self._train_step(cfg.dc.target_ratio_init)
                yield ("step", None)
            torch.cuda.synchronize() if device.type == "cuda" else None
            sec_per_step = (time.time() - t0) / probe_steps
            budget_sec = args.max_minutes * 60 - (time.time() - self.t_start)
            self.total_steps = max(int(budget_sec / sec_per_step * 0.9), 50)
            self.log({"probe_sec_per_step": sec_per_step, "bytes_per_sec": self.tokens_per_step / sec_per_step, "total_steps": self.total_steps})

        if self.sched_total is None:
            self.sched_total = self.total_steps
        last_ckpt = time.time()
        t_log = time.time()
        snap_idx = self.step // args.snapshot_steps if args.snapshot_steps else 0

        while self._running() and not self._stop:
            # wsd follows wall-clock progress (arms compare at equal wall-clock); trunk/cooldown follow
            # the step horizon fixed at the first probe, so a resume continues the schedule instead of
            # restarting it
            pr = self._progress() if args.schedule == "wsd" else min(self.step / max(self.sched_total, 1), 1.0)
            horizon = 1000
            sched_step = int(pr * horizon)
            # a cooldown branch starts from a trunk that finished its ATDC ramp: hold the final target
            if args.schedule == "cooldown":
                target_ratio = cfg.dc.target_ratio_final
            else:
                target_ratio = atdc_target_ratio(sched_step, horizon, cfg.dc.target_ratio_init, cfg.dc.target_ratio_final, cfg.dc.schedule_warmup_frac)
            lr = schedule_lr(args.schedule, sched_step, horizon, args.lr)
            set_lr(self.opt, lr)
            stats = yield from self._train_step(target_ratio)
            self.step += 1
            if args.snapshot_steps and self.step // args.snapshot_steps > snap_idx:
                snap_idx = self.step // args.snapshot_steps
                self.snapshot()
            if self.step % args.log_every == 0:
                dt = time.time() - t_log
                t_log = time.time()
                stats = {k: float(v) for k, v in stats.items()}  # the one sync per logging interval
                rec = {"lr": lr, "target_ratio": target_ratio, "bytes_per_sec": self.tokens_per_step * args.log_every / dt, "train_bpb": stats["ce"] / LN2}
                rec.update(stats)
                rec["tflops"] = flops_per_byte(self.model, args.seq_len, stats.get("bpic", 1.0)) * rec["bytes_per_sec"] / 1e12
                if self.peak_tflops:
                    rec["mfu"] = rec["tflops"] / self.peak_tflops
                self.log(rec)
            if self.step % args.eval_every == 0:
                ev = evaluate(self.model, self.val_shard, args.batch_size, args.seq_len, args.eval_batches, device, target_ratio)
                ev["sample"] = chunk_sample(self.model, "The router compares each byte with the one before it. Where they stop looking alike, it draws a boundary.", device)
                self.log({"eval": ev})
            if (time.time() - last_ckpt) / 60 >= args.ckpt_minutes:
                self.save()
                last_ckpt = time.time()
                self.log({"checkpoint": str(self.ckpt_path)})
            if (time.time() - self.t_start) / 60 > args.max_minutes:
                self.log({"stopped": "time budget"})
                break
            yield ("step", None)
        if self._stop:
            self.log({"stopped": self.stopped_reason or "requested"})

        ev = evaluate(self.model, self.val_shard, args.batch_size, args.seq_len, args.eval_batches, device, cfg.dc.target_ratio_final)
        ev["sample"] = chunk_sample(self.model, "The router compares each byte with the one before it. Where they stop looking alike, it draws a boundary.", device)
        self.log({"eval": ev, "final": True})

        self.save()
        self.log({"done": True, "final_step": self.step})

    def close(self):
        self.log_f.close()


def main(argv=None):
    t = Trainer(argv)
    try:
        for _ in t.run():
            pass
    finally:
        t.close()


if __name__ == "__main__":
    main()
