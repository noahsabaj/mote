"""Pretraining loop for the byte-level H-Net.

    L = CE_nbp + α · L_ratio(N(step))

Features: bf16 autocast, AdamW(0.9, 0.95) with per-stage LR multipliers and a no-decay set,
warmup-stable-decay schedule, gradient clipping, ATDC target-ratio schedule, periodic evaluation
(val bits/byte, bytes-per-chunk, boundary/word alignment, a chunked text sample), JSONL logging, atomic checkpoints every N minutes with auto-resume, and a wall-clock budget.

    python -m mote.train.train --preset pilot --data data/fineweb_edu_pilot --out runs/pilot \
        --batch-size 16 --seq-len 2048 --max-minutes 60
"""

from __future__ import annotations

import argparse
from collections import deque
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..config import PRESETS, MoteConfig, normalize_preset, resolve_preset
from ..model import triton_lock

triton_lock.install()  # the trainer shares autotuned kernels with serving replies in the daemon
from ..data.loader import ByteShard, MixedShard
from ..model.dc import atdc_target_ratio, bytes_per_chunk, ratio_loss
from ..model.moe import collect_moe, moe_modules
import mote

from .. import determinism, runinfo
from .flops import _n as _n_active
from ..model.feedback import feedback_from
from ..model.hnet import HNetForCausalLM, strip_retired
from ..tokenizer import OFFSET_ID, ByteTokenizer
from .flops import flops_per_byte, peak_tflops_for
from . import elr
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


BRANCH_SCHEDULES = ("branch", "constant")
BRANCH_DECAY_FRAC = 0.2  # the 2x2's fork point: constant to here, then decay (docs/shape.md § mid)
# 0.2 sits between the two windows 2608.24814 App. F.1 tested at a fixed peak lr and budget: at 0.1 the
# weight-decayed run never overtook the unregularised one, at 0.3 it did and finished lower — same runs,
# opposite verdict, because the gain is acquired early and only revealed once the low-ELR phase runs long
# enough for the accumulated optimisation noise to be forgotten. So the window is a variable, not a
# constant, and the mid stage measures it (`--branch-decay-frac`) instead of assuming 0.2.


def schedule_lr(kind: str, step: int, total: int, base: float, min_ratio: float = 0.1,
                decay_frac: float = BRANCH_DECAY_FRAC) -> float:
    """The schedules of the pipeline (docs/shape.md § mid, re-signed 2026-08-26).

    `wsd`      warmup-stable-decay over the budget — the lab arms.
    `trunk`    warmup (10 %) then constant, never decays — the flagship trunk; branches fork off it.
               This is Warmup-Stable-Only, which 2603.16127 measured as the best pre-training schedule
               *for post-SFT quality* at 1B and 8B, beating every decay variant it lost to on val loss.
    `branch`   constant for the first 80 %, then decay to `min_ratio` over the last 20 % — the mid-training
               branches. `constant` is the same run without the decay, which is the other arm of the 2x2.

    The old `cooldown` (base -> 0.1x over the whole branch, as 1-sqrt(t)) is retired. Two 2026 results
    killed it. Index-1.9B (2607.09885 §6.4-6.5) found the schedule alone is worth almost nothing at 0.1B,
    and that what pays is decay *combined with* a data-quality raise — but that combination only worked
    under WSD: cosine plus curated data scored *below* plain cosine, because "the cosine tail leaves too
    little learning rate" to adapt to the distribution shift. 2605.25698 names the same conflict formally
    (the model meets its best data exactly when its learning intensity is weakest) and measures +3.27 on a
    600M dense model for fixing it. 1-sqrt(t) is the worst shape for this: it is concave, so it was down
    to 55 % of peak by the first quarter of the branch — precisely when mix C's shift arrives.
    """
    if kind == "trunk":
        warm = max(int(total * 0.1), 1)
        return base * (step + 1) / warm if step < warm else base
    if kind == "constant":
        return base
    if kind == "branch":
        t = min(step / max(total, 1), 1.0)
        if t < 1.0 - decay_frac:
            return base
        u = (t - (1.0 - decay_frac)) / decay_frac
        return base * max(min_ratio, 1.0 - (1.0 - min_ratio) * u)  # linear, straight to min_ratio
    return wsd_lr(step, total, base, min_ratio=min_ratio)


def parse_mix_spec(spec: str):
    """`PREFIX:SHARE[:plain][:fim]` -> (prefix, share, plain, fim).

    `plain` reads an SFT shard's bytes without its loss mask (chat/identity bytes as ordinary LM data in a
    mid-training mix). `fim` additionally permutes each training window around one of its tool calls
    (mote.data.loader.fim_window, 2607.12463) — the tool-trace shard is the only one it is used on, since
    a window with no `<|call|>` in it passes through unchanged."""
    parts = spec.split(":")
    fim = "fim" in parts[2:]
    plain = "plain" in parts[2:] or fim  # a FIM shard is read without its mask by construction
    parts = [q for q in parts if q not in ("plain", "fim")]
    return ":".join(parts[:-1]), float(parts[-1]), plain, fim


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


def _targets(batch: torch.Tensor, loss_mask: Optional[torch.Tensor]):
    """(inputs, targets, target mask) for a window, honouring the 2606.16246 offset marker."""
    inputs, targets = batch[:, :-1], batch[:, 1:]
    tmask = loss_mask[:, 1:] if loss_mask is not None else None  # mask is per target position
    # Target offset prediction (2606.16246 §2.3), read back off the window rather than passed in: a window
    # that opens `<|offset|> <digit>` declares that its label at position t is x_{t+i}, not x_{t+1}. The
    # tail where t+i runs past the window is dropped from the loss. Nothing happens when the flag is off,
    # which is every run to date and the mid-training 2x2.
    if batch.shape[1] > 2 and bool((batch[:, 0] == OFFSET_ID).any()):
        i = int(batch[0, 1].item() - ord("0"))
        if i > 1:
            L = inputs.shape[1]
            idx = torch.arange(L, device=batch.device) + i
            valid = idx < batch.shape[1]
            targets = batch[:, idx.clamp(max=batch.shape[1] - 1)]
            keep = valid[None, :].expand_as(targets)
            tmask = keep.to(targets.dtype) if tmask is None else tmask * keep.to(tmask.dtype)
    return inputs, targets, tmask


def _pass_loss(model, out, inputs, targets, tmask, target_ratio: float, ratio_weight: float):
    """The loss of one forward pass: CE sum + ratio (+ MoE). Every stat a 0-dim tensor."""
    ce_sum, n = _masked_ce_sum(out.logits, targets, tmask)
    mask = torch.ones_like(inputs, dtype=torch.bool)
    lr_ = ratio_loss(out.routing.boundary_prob, out.routing.boundary_mask, mask, target_ratio)
    n_safe = n.clamp(min=1.0)
    loss = ce_sum + ratio_weight * lr_ * n_safe
    stats = {"ce_sum": ce_sum.detach(), "n": n.detach(), "ratio": lr_.detach(), "bpic": bytes_per_chunk(out.routing.boundary_mask, mask)}
    moe_aux, moe_stats = collect_moe(model)
    if moe_aux is not None:  # balance / z losses of the MoE FFNs, per-token like the ratio loss
        loss = loss + moe_aux * n_safe
        stats.update(moe_stats)
    return loss, n_safe, stats


def iter_pass_losses(model: HNetForCausalLM, batch: torch.Tensor, target_ratio: float, ratio_weight: float,
                     loss_mask: Optional[torch.Tensor] = None, passes: int = 1, gen: Optional[torch.Generator] = None, detach: bool = False):
    """Latent feedback's multi-pass teacher forcing (2608.08888 §3.3), one pass at a time: pass 1 is the
    ordinary forward; pass k fuses pass k−1's top states, shifted right, into the plain inputs (prefix
    mixin sampled on `gen`) and re-runs the stack in parallel. Yields (loss, n, stats, out) per pass so a
    caller may back-propagate each pass as it goes (`detach=True`: the memory fallback — one pass of
    activations alive at a time)."""
    inputs, targets, tmask = _targets(batch, loss_mask)
    out = model(inputs)
    yield (*_pass_loss(model, out, inputs, targets, tmask, target_ratio, ratio_weight), out)
    prev = out
    for _ in range(passes - 1):
        fb = feedback_from(prev, gen, detach=detach, mixin=True)
        prev = model(inputs, feedback=fb)
        yield (*_pass_loss(model, prev, inputs, targets, tmask, target_ratio, ratio_weight), prev)


def compute_losses(model: HNetForCausalLM, batch: torch.Tensor, target_ratio: float, ratio_weight: float,
                   loss_mask: Optional[torch.Tensor] = None, passes: int = 1, gen: Optional[torch.Generator] = None):
    """Returns (loss_unnormalised, n_tokens, stats, out). The loss is a SUM over counted tokens (plus the
    ratio loss scaled by the token count), so that gradient accumulation normalises ONCE by the total
    token count across micro-batches — a mean of per-micro-batch means would up-weight windows with few
    assistant bytes (SFT). Every stat is a 0-dim device tensor: nothing here synchronises.

    `passes` > 1 adds the latent-feedback passes (λ = 1, averaged over the extra passes as in the paper's
    eq. 12); `n` stays pass 1's token count, so a feedback micro-batch weighs about twice a plain one."""
    it = iter_pass_losses(model, batch, target_ratio, ratio_weight, loss_mask, passes, gen, detach=False)
    loss, n_safe, stats, out = next(it)
    if passes > 1:
        extra, ce_fb = 0.0, 0.0
        for lk, _, sk, _ in it:
            extra = extra + lk
            ce_fb = ce_fb + sk["ce_sum"]
        loss = loss + extra / (passes - 1)
        stats["ce_fb_sum"] = ce_fb / (passes - 1)
    return loss, n_safe, stats, out


def evaluate(model: HNetForCausalLM, shard: ByteShard, batch_size: int, seq_len: int, max_batches: int, device, target_ratio: float,
             spread: bool = False, feedback_passes: int = 0):
    """`spread=False` reads the head of the val shard = its first source only (fineweb_edu on the mixes);
    `spread=True` spaces the windows over the whole shard. Off by default so every arm in a queue is
    measured like its control; the trunk and the branches run with --eval-spread."""
    gen = evaluate_batches(model, shard, batch_size, seq_len, max_batches, device, target_ratio, spread, feedback_passes)
    while True:
        try:
            next(gen)
        except StopIteration as done:
            return done.value


@torch.no_grad()
def evaluate_batches(model: HNetForCausalLM, shard: ByteShard, batch_size: int, seq_len: int, max_batches: int, device,
                     target_ratio: float, spread: bool = False, feedback_passes: int = 0):
    """`evaluate` as a generator that yields after every window: the daemon runs each window as its own
    slice under the GPU gate, so a chat waits for one window instead of the whole evaluation (a 16-window
    EMA eval held the gate for minutes and a reply showed no first byte for as long, QA 2026-08-24)."""
    model.eval()
    tot_nll, tot_tok, tot_bytes, tot_chunks = 0.0, 0, 0, 0
    # latent feedback: k fused prefill passes over the same windows (2608.08888 eqs. 10-11) — pass k fuses
    # pass k−1's top states, shifted right, into the plain inputs; val_bpb_fb1 is what the gate reads
    n_fb = feedback_passes if getattr(model, "feedback_level", "off") != "off" else 0
    fb_nll = [0.0] * n_fb
    word_hits, boundary_count = 0, 0
    bp_all = []
    moe_loads, moe_batches = {}, 0
    for batch, lmask in shard.sequential_batches(batch_size, seq_len, max_batches, spread=spread):
        batch = batch.to(device, non_blocking=True)
        inputs, targets = batch[:, :-1], batch[:, 1:]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(inputs)
            fb_outs, prev = [], out
            for _ in range(n_fb):
                prev = model(inputs, feedback=feedback_from(prev, None, detach=True, mixin=False))
                fb_outs.append(prev)
        V = out.logits.shape[-1]
        w = lmask[:, 1:].to(device).reshape(-1).float() if lmask is not None else None

        def nll_sum(logits):
            nll = F.cross_entropy(logits.reshape(-1, V).float(), targets.reshape(-1), reduction="none")
            return (nll * w).sum().item() if w is not None else nll.sum().item()

        tot_nll += nll_sum(out.logits)
        tot_tok += int(w.sum().item()) if w is not None else targets.numel()
        for k, o in enumerate(fb_outs):
            fb_nll[k] += nll_sum(o.logits)
        bm = out.routing.boundary_mask
        tot_bytes += bm.numel()
        tot_chunks += bm.sum().item()
        # The p at which a plain threshold would reproduce this run's own boundary rate. Decode sees
        # one byte and has no sequence to project over (signed 2026-08-27), so it thresholds; under
        # bounded routing the projection has already moved the rate away from the p > 0.5 the router
        # would have picked, and this is the number that puts decode back on the same rate. It equals
        # 0.5 exactly when nothing bound.
        bp_all.append(out.routing.boundary_prob[..., 1].detach().float().flatten().cpu())
        # boundary/word alignment (excluding position 0): a boundary is "word-aligned" if its previous byte is a
        # separator (space/newline/punct) OR the boundary byte itself is one (chunks that start with the space).
        def sep(t):
            return (t == 32) | (t == 10) | ((t >= 33) & (t <= 47)) | ((t >= 58) & (t <= 64))

        prev, cur = inputs[:, :-1], inputs[:, 1:]
        b_inner = bm[:, 1:]
        word_hits += (b_inner & (sep(prev) | sep(cur))).sum().item()
        boundary_count += b_inner.sum().item()
        for i, m in enumerate(moe_modules(model)):  # expert usage per layer over the eval windows
            moe_loads[i] = moe_loads.get(i, 0.0) + m.stats["load"].float().cpu()
            moe_batches += 1
        yield None  # a slice boundary for the daemon
    model.train()
    res = {
        "val_bpb": tot_nll / max(tot_tok, 1) / LN2,
        **{f"val_bpb_fb{k + 1}": v / max(tot_tok, 1) / LN2 for k, v in enumerate(fb_nll)},
        "val_bpic": tot_bytes / max(tot_chunks, 1),
        "target_ratio": target_ratio,
        "boundary_on_separator_frac": word_hits / max(boundary_count, 1),
    }
    if bp_all:
        p_all = torch.cat(bp_all)
        rate = tot_chunks / max(tot_bytes, 1)  # the fraction of positions that ended up boundaries
        res["val_decode_threshold"] = float(torch.quantile(p_all, max(1.0 - rate, 0.0)))
    if moe_loads:
        n_layers = len(moe_loads)
        loads = [moe_loads[i] / max(moe_batches // n_layers, 1) for i in range(n_layers)]
        vio = [float((l.max() - l.mean()) / l.mean().clamp_min(1e-9)) for l in loads]
        res["moe_maxvio"] = sum(vio) / n_layers
        res["moe_maxvio_max"] = max(vio)
        res["moe_load"] = [[round(float(v), 4) for v in l] for l in loads]
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
    """The one writer of a Mote checkpoint. `opt=None` (a weights-only snapshot, a post-training stage) leaves the
    optimizer out rather than writing `None` — a resume that found `"optimizer": None` used to crash in
    `MultiOpt.load_state_dict`."""
    tmp = path.with_suffix(".tmp")
    # Everything here is tensors and plain Python, which is what lets every reader use
    # torch.load(weights_only=True) — the safe loader, and the default since 2.6. We had been
    # opting out of it at all twelve load sites for no reason: a .pt is a pickle, and this repo is
    # public. If a future field ever needs a real class, register it with
    # torch.serialization.add_safe_globals rather than turning the flag back off.
    ck = {"model": model.state_dict(), "step": step, "config": cfg.to_dict(), "extra": extra}
    if opt is not None:
        ck["optimizer"] = opt.state_dict()
    torch.save(ck, tmp)
    os.replace(tmp, path)


def ckpt_path_exists(out_dir: Path) -> bool:
    return (out_dir / "last.pt").exists()


VOCAB_SIZED = ("embeddings.weight", "lm_head.weight")


def pad_vocab_rows(sd: Dict, model) -> Dict:
    """A checkpoint with fewer embedding rows than the model (264 before 2026-08-24, 272 after) loads with its rows
    copied and the model's fresh init kept for the spare ones — `--init-from` an older run. Rows are never dropped."""
    cur = model.state_dict()
    out = dict(sd)
    for key in VOCAB_SIZED:
        if key not in sd or key not in cur or tuple(sd[key].shape) == tuple(cur[key].shape):
            continue
        old, new = sd[key], cur[key]
        if old.ndim != new.ndim or any(o > n for o, n in zip(old.shape, new.shape)):
            continue  # a real mismatch: let load_state_dict report it
        padded = new.detach().clone()
        padded[tuple(slice(0, o) for o in old.shape)] = old.to(padded.dtype)
        out[key] = padded
    return out


def load_checkpoint(path: Path, model, opt=None, ck: Optional[Dict] = None, allow_new: tuple = ()):
    """`ck`: the checkpoint if the caller already read it (a resume reads it first for its config)."""
    if ck is None:
        ck = torch.load(path, map_location="cpu", weights_only=True)
    sd = pad_vocab_rows(strip_retired(ck["model"]), model)
    if allow_new:  # modules the checkpoint predates (an --init-from into a latent-feedback config) start fresh
        res = model.load_state_dict(sd, strict=False)
        stray = [k for k in res.missing_keys if not k.startswith(tuple(allow_new))]
        assert not stray and not res.unexpected_keys, f"checkpoint/model mismatch: missing {stray}, unexpected {res.unexpected_keys}"
        if res.missing_keys:
            print(f"fresh modules not in {path}: {sorted({k.split('.')[0] for k in res.missing_keys})}", flush=True)
    else:
        model.load_state_dict(sd)
    if opt is not None and ck.get("optimizer") is not None:
        opt.load_state_dict(ck["optimizer"])
    return ck["step"], ck.get("extra", {})


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="mote-11m", type=normalize_preset,
                    help="model size: " + ", ".join(PRESETS) + " (the retired role names still resolve)")
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
    ap.add_argument("--tf32", action="store_true", help="TF32 tensor-core inputs for the fp32 matmuls (the fp32 residual projection); a numerics change to the frozen fp32-residual path, screened as an arm (2026-08-24)")
    ap.add_argument("--optimizer", default="adamw", choices=["adamw", "muon", "muonsw"], help="muon: Newton-Schulz updates for hidden 2-D matrices, AdamW for the rest; muonsw: Muon with \u03b7\u00b2-scaled weight decay (2607.23777)")
    ap.add_argument("--branch-decay-frac", type=float, default=BRANCH_DECAY_FRAC, help="fraction of a `branch` run spent decaying (default 0.2). 2608.24814 App. F.1: 0.1 was too short to reveal the gain weight decay had already accumulated and 0.3 was long enough, at the same peak lr and budget \u2014 so the mid 2x2 runs this as a third arm rather than assuming it")
    ap.add_argument("--qk-norm", action="store_true", help="RMSNorm the Relation evidence projections p1/p2 per head before RoPE (QK-Norm). Not loss-neutral here — silu(u) and sigmoid(u/\u03c4_s) are not scale-covariant — so \u03c4_s and \u03bb are re-gated with it (mote/model/relation.py HeadRMSNorm)")
    ap.add_argument("--tau-s", type=float, default=None, help="Relation Self temperature (preset default 2.0); swept with --qk-norm because QK-Norm rescales the evidence u")
    ap.add_argument("--lambda-init", type=float, default=None, help="Relation count-calibration \u03bb\u2080 (preset default 0.5); swept with --qk-norm for the same reason")
    ap.add_argument("--elr-trace-out", default=None, metavar="PATH", help="record this run's per-matrix ‖W‖_F on the logging cadence to PATH (default runs/<out>/elr_trace.json when --elr-match is used elsewhere); the reference half of an ELR-matched pair (2608.24814 App. B.2)")
    ap.add_argument("--elr-match", default=None, metavar="PATH", help="track the ELR schedule in an --elr-trace-out file: this run keeps its own optimizer and norm control, and only its per-matrix learning rates are adapted, η_i = η^eff,ref · ‖W_i‖. If the losses then collapse, the norm-control difference acted through ELR. Muon matrices only — the AdamW groups keep the schedule")
    ap.add_argument("--norm-guard", default="stop", choices=["off", "warn", "stop"], help="watch for ‖W‖_F falling faster than weight decay alone could take it — an update systematically anti-aligned with its own weights, which is what the lr_sweep_12e-4 collapse was. `stop` ends the run gracefully and HOLDS the queue; `warn` only logs")
    ap.add_argument("--norm-guard-slack", type=float, default=1.25,
                    help="how far past the weight-decay budget ‖W‖ may fall before --norm-guard trips; 1.0 is the "
                         "bound an orthogonal update cannot cross, the slack absorbs the aligned component and noise")
    ap.add_argument("--ckpt-main", action="store_true", help="activation checkpointing on the Relation blocks (bit-neutral, ~30%% more compute, much less memory)")
    ap.add_argument("--bucket", type=int, default=None, help="chunk-count bucket (default from the preset, 64); 1 = exact shapes")
    ap.add_argument("--no-mbp", action="store_true", help=argparse.SUPPRESS)  # the head is gone (2026-08-29); a no-op so queued argv still parses
    ap.add_argument("--mix", action="append", default=[], metavar="PREFIX:SHARE[:plain]", help="extra shard mixed into training by share, e.g. data/sft_identity:0.05 (repeatable); ':plain' reads an SFT shard without its loss mask")
    ap.add_argument("--beta2", type=float, default=0.95, help="AdamW \u03b2\u2082 (2608.16760: the convergence threshold rises as the batch shrinks; A/B 0.99/0.997 at our 32 kB steps)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fast", action="store_true", help="give up bitwise reproducibility for ~20 %% throughput. Runs are reproducible by default: cuBLAS pinned to a fixed workspace plus the two-pass Relation backward, which together are the whole story (mote/determinism.py). A `--fast` number and a default number are not comparable, so the mode is recorded in run.json")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                    help="cuda (default) or cpu. There is no silent fallback: a job that asked for the GPU and cannot "
                         "have it fails at once. 2026-08-28: the daemon started before the driver, the trainer fell "
                         "back on its own, and a flagship at 16384 on the CPU reference path took 23.5 GB — the kernel "
                         "OOM-killer took the whole daemon, three times")
    ap.add_argument("--sft", action="store_true", help="train on an SFT shard (assistant-byte loss mask)")
    ap.add_argument("--keep", default=None, metavar="NPY",
                    help="window starts from mote.data.select_sft: the main shard is sampled from these only (SFT-1's signed one-shot difficulty selection)")
    ap.add_argument("--reselect-every", type=float, default=0.0, metavar="FRAC",
                    help="re-select the SFT windows every FRAC of the budget with the live model (the trajectory rule, "
                         "docs/research/curriculum-2026-08-28.md): pass 0 at the start fixes tau_m = the shard's 20th-percentile "
                         "loss, later passes drop what is mastered (a floor share stays) and what has not moved. 0 = off. "
                         "Signed 2026-08-28 as A/B-gated: it becomes SFT-1's default only if the static-vs-dynamic pair passes")
    ap.add_argument("--reselect-windows", type=int, default=0, help="score an evenly spread sample of this many windows per pass (0 = all; ~4 GPU-min per 127 MB at 2048)")
    ap.add_argument("--reselect-floor", type=float, default=0.1, help="share of mastered windows kept in the mix (forgetting)")
    ap.add_argument("--reselect-eps", type=float, default=0.02, help="a window whose loss moved less than this since the last pass is stuck")
    ap.add_argument("--init-from", default=None, help="checkpoint to initialize weights from (e.g. pretrain -> SFT)")
    ap.add_argument("--ratio-weight", type=float, default=None, help="override dc.ratio_loss_weight (\u03b1)")
    ap.add_argument("--moe", type=int, default=None, metavar="E", help="mixture of experts in the main-network FFNs: E experts (signed 2026-08-24, docs/shape.md \"MoE\")")
    ap.add_argument("--moe-topk", type=int, default=2, help="active experts per chunk")
    ap.add_argument("--moe-ff", type=int, default=None, help="expert hidden width (default d_ff // topk: active FLOPs match the dense FFN)")
    ap.add_argument("--moe-router", default="lossfree", choices=["lossfree", "aux"], help="lossfree: DeepSeek-V3 bias balancing + seq-level balance loss; aux: Switch softmax + load-balance loss + z-loss")
    ap.add_argument("--moe-dense-first", action="store_true", help="layer 0 keeps the dense FFN")
    ap.add_argument("--moe-aux-weight", type=float, default=None, help="balance-loss weight (default 1e-4 lossfree / 1e-2 aux)")
    ap.add_argument("--moe-gamma", type=float, default=None, help="lossfree: expert-bias step per optimizer step (default 1e-3)")
    ap.add_argument("--moe-gate-scale", type=float, default=None, help="routed-output scale (default: Moonlight's computed factor for lossfree, 1.0 for aux)")
    ap.add_argument("--no-flash", action="store_true", help="A/B: materialized Relation instead of the Triton kernel")
    ap.add_argument("--target-ratio", type=float, nargs=2, default=None, metavar=("INIT", "FINAL"), help="override the ATDC target-ratio schedule endpoints")
    ap.add_argument("--adapt-threshold", type=float, default=None, help="ATDC's proficiency trigger tau (2605.30080 Alg. 1): raise the target ratio by --adapt-rate while the windowed mean LM loss is below this. Mote shipped the paper's schedule without this trigger until 2026-08-27; None keeps it off")
    ap.add_argument("--adapt-rate", type=float, default=None, help="ATDC's gamma, the boost applied when the trigger fires (paper: 1.05)")
    ap.add_argument("--bound-ceiling", type=float, default=None, help="bounded routing (GeneZip 2602.17739): cap boundaries per window at this multiple of the target rate L/N. A GUARDRAIL at loose values (the arena can then never overflow its capacity); tightening it toward 1.0 makes it a compression-rate controller, which is past the paper's evidence and wants its own A/B")
    ap.add_argument("--bound-floor", type=int, default=None, help="bounded routing: minimum boundaries per window (the paper's K_min=8 keeps the router from collapsing)")
    ap.add_argument("--schedule", default="wsd", choices=["wsd", "trunk", "branch", "constant"], help="wsd: warmup-stable-decay over the budget (lab arms); trunk: warmup then constant, never decays (the flagship trunk); branch: constant for 80%% then decay to --min-lr-ratio (a mid-training branch off a trunk snapshot); constant: the same branch without the decay, the other arm of the 2x2. docs/shape.md § mid")
    ap.add_argument("--min-lr-ratio", type=float, default=None, help="where a decay lands, as a fraction of --lr. Default depends on the schedule: 0 for a branch (straight to zero — Bergsma 2502.15938, and 2602.06797's optimal schedules also terminate at 0), 0.1 for wsd, which every lab arm to date was run with and must stay comparable to.")
    ap.add_argument("--snapshot-steps", type=int, default=0, help="also keep a weights-only snap_<step>.pt every N steps (the branch points for the mid-training branches)")
    ap.add_argument("--snapshot-at", type=float, default=None, metavar="FRAC", help="also snapshot the first time the run crosses this fraction of its horizon. The 2x2 forks here: the decayed arm keeps going under --schedule branch, and the no-decay arm resumes from this snapshot under --schedule constant for the remaining bytes, so the two are token-matched (docs/shape.md § mid)")
    # 2606.16246's three augmentations, all OFF by default. Built 2026-08-26 and deliberately held out of
    # the mid-training 2x2 so its verdict stays attributable to the data changes; they are their own
    # comparison afterwards, which costs nothing because the shards are already built.
    ap.add_argument("--aug-noise", type=float, default=0.0, help="replace this fraction of content bytes with random bytes (their best single augmentation: post-decay val loss 4.000 -> 3.826 at 15 %%; 5 %% in the winning three-way combination)")
    ap.add_argument("--aug-r2l", type=float, default=0.0, help="reverse this share of training windows over codepoints, marked with <|r2l|> (-> 3.910 at 0.5; reversing raw bytes would be corruption, not reversal)")
    ap.add_argument("--aug-offset", type=int, default=1, help="largest target offset i for x_{t+i} prediction, sampled exponentially per micro-batch (-> 3.870 at 5, and the highest downstream mean of any single method). 1 = ordinary next-byte prediction")
    ap.add_argument("--eval-spread", action="store_true", help="evaluate on windows spread over the whole val shard instead of its head (= first source only); use for the trunk and branches, never mid-queue")
    ap.add_argument("--feedback", choices=["off", "byte", "chunk"], default=None, help="latent feedback (2608.08888; docs/results/2026-08-28-latent-feedback-prereg.md): fuse the previous position's top state into the next input — `byte` (decoder top -> next byte's encoder input) or `chunk` (main top -> next chunk's main input). Sets the config's feedback level on a fresh run or --init-from; a resume keeps the checkpoint's")
    ap.add_argument("--feedback-mix", default="0.75,0.22,0.03", help="probabilities of 1-, 2-, 3-pass micro-batches (the paper's whole-run mixture; the 3 %% three-pass batches are what makes the feedback map a contraction)")
    ap.add_argument("--feedback-window", type=int, default=0, help="bytes per multi-pass micro-batch (0 = the full --seq-len): the first memory fallback — pass-1 activations stay alive through the extra passes")
    ap.add_argument("--feedback-detach", action="store_true", help="detach the carried state and back-propagate each pass on its own (one pass of activations at a time): the second memory fallback; loses the auxiliary gradient into the earlier pass")
    ap.add_argument("--feedback-jitter", type=float, default=None, help="override FeedbackCfg.jitter (uniform ±σ on the carried state in training passes; paper 0.02)")
    ap.add_argument("--eval-feedback-passes", type=int, default=0, help="also evaluate with k fused prefill passes (2608.08888 eqs. 10-11; logs val_bpb_fb1..k) — the gate reads k=1")
    ap.add_argument("--eval-ema", type=float, default=0.0, help="also evaluate an EMA of the weights (per-step decay; 0 = off): the decayed-quality stand-in for constant-LR runs that the LR-vs-horizon fit reads (2608.20061 §2.2.1; mote/train/lr_horizon.py); logs val_bpb_ema")
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
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available and this job asked for it (--device cuda is the default). "
                               "Pass --device cpu to train on the CPU on purpose; nothing falls back to it silently "
                               "(2026-08-28: a flagship at 16384 on the CPU took 23.5 GB and the kernel OOM-killer took the daemon)")
        # Before any CUDA work. In the resident daemon this is process state shared with serving,
        # so close() puts it back — a --fast job must not leave the next one non-reproducible.
        determinism.apply(not args.fast)
        torch.manual_seed(args.seed)
        self.device = device = torch.device(args.device)
        self.out_dir = out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)

        self._resume_ck = None
        if args.resume and ckpt_path_exists(out_dir):
            # A resume continues the run as it was built: the checkpoint's own config beats today's preset
            # defaults AND the run's config.json — a resume that failed under a new default had already
            # rewritten config.json with that default (2026-08-24: 264 -> 272 vocab rows, twice).
            self._resume_ck = torch.load(out_dir / "last.pt", map_location="cpu", weights_only=True)
            cfg = MoteConfig.from_dict(self._resume_ck["config"])
        else:
            cfg = MoteConfig.load(args.config) if args.config else resolve_preset(args.preset)
        cfg.max_seq_len = max(cfg.max_seq_len, args.seq_len)
        if args.ratio_weight is not None:
            cfg.dc.ratio_loss_weight = args.ratio_weight
        if args.target_ratio is not None:
            cfg.dc.target_ratio_init, cfg.dc.target_ratio_final = args.target_ratio
        if args.adapt_threshold is not None:
            cfg.dc.adapt_threshold = args.adapt_threshold
        if args.adapt_rate is not None:
            cfg.dc.adapt_rate = args.adapt_rate
        if args.bound_ceiling is not None:
            cfg.dc.bound_ceiling = args.bound_ceiling
        if args.bound_floor is not None:
            cfg.dc.bound_floor = args.bound_floor
        if args.feedback is not None:
            cfg.feedback.level = args.feedback
        if args.feedback_jitter is not None:
            cfg.feedback.jitter = args.feedback_jitter
        (out_dir / "run.json").write_text(json.dumps(
            {**vars(args), "determinism": determinism.state(), "code": mote.CODE_VERSION,
             "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2))

        if args.bucket is not None:
            cfg.dc.chunk_bucket = args.bucket
        if args.moe:
            cfg.main.moe_experts = args.moe
            cfg.main.moe_topk = args.moe_topk
            cfg.main.moe_router = args.moe_router
            cfg.main.moe_dense_first = args.moe_dense_first
            if args.moe_ff is not None:
                cfg.main.moe_d_ff = args.moe_ff
            if args.moe_aux_weight is not None:
                cfg.main.moe_aux_weight = args.moe_aux_weight
            if args.moe_gamma is not None:
                cfg.main.moe_bias_gamma = args.moe_gamma
            if args.moe_gate_scale is not None:
                cfg.main.moe_gate_scale = args.moe_gate_scale
        if args.qk_norm:
            cfg.main.qk_norm = True
        if args.tau_s is not None:
            cfg.main.tau_s = args.tau_s
        if args.lambda_init is not None:
            cfg.main.lambda_init = args.lambda_init
        if args.no_flash:
            from ..model import relation as _relation
            _relation.USE_FLASH = False
        # after every architecture flag, not before them: config.json is the run's readable record and used
        # to be written while the config was still half-built, so the architecture flags (--moe, --qk-norm,
        # ...) never appeared in it. The checkpoint always carried the finished config, so resumes were
        # right and only the file on disk was wrong.
        cfg.save(out_dir / "config.json")
        self.cfg = cfg
        # ATDC's proficiency trigger (2605.30080 Alg. 1 line 11) averages this window of LM loss
        self._loss_window: deque = deque(maxlen=max(int(cfg.dc.adapt_window), 1))
        self.model = model = HNetForCausalLM(cfg, device=device)
        if args.ckpt_main:
            model.main_network.grad_checkpoint = True
        self.n_params = model.num_params()
        self.peak_tflops = peak_tflops_for(device) if device.type == "cuda" else None
        print(f"params: {self.n_params/1e6:.2f}M | device: {device} | kernels: mamba3={__import__('mote.model.mamba3', fromlist=['x']).HAS_MAMBA3_KERNEL} ssd={__import__('mote.model.dc', fromlist=['x']).HAS_SSD_KERNEL}", flush=True)
        self._moe = moe_modules(model)
        if self._moe:
            m0 = self._moe[0]
            print(f"moe: {len(self._moe)} layers x {m0.n_experts} experts, top-{m0.top_k}, d_ff {m0.d_ff}, router {m0.router_kind}, gate scale {m0.scale:.3f}, "
                  f"active {_n_active(model)/1e6:.2f}M of {self.n_params/1e6:.2f}M params", flush=True)
        self.opt = build_optimizer(model, args.lr, args.weight_decay, args.stage_lr_mult, betas=(0.9, args.beta2), optimizer=args.optimizer)

        # ELR (2608.24814): η/‖W‖_F is the coordinate loss dynamics actually follow, so every run logs it and
        # any comparison across optimizers, weight decays or norm-control methods is read on it rather than on
        # the nominal lr. Empty for an AdamW run, which switches all three features off.
        self._elr_named = elr.muon_named_matrices(model, self.opt)
        self.norms = elr.NormTracker(self._elr_named)
        self.norm_guard = elr.NormGuard(slack=args.norm_guard_slack) if (self.norms and args.norm_guard != "off") else None
        self.elr_trace = elr.ELRTrace(meta={"lr": args.lr, "weight_decay": args.weight_decay,
                                            "optimizer": args.optimizer, "schedule": args.schedule}) if args.elr_trace_out else None
        self._elr_trace_path = None
        if args.elr_trace_out:
            q = Path(args.elr_trace_out)
            self._elr_trace_path = q if q.is_absolute() or q.parent != Path(".") else Path(args.out) / q
        self.elr_match = None
        if args.elr_match:
            if not self.norms:
                raise SystemExit("--elr-match needs Muon matrices; this run is on AdamW")
            self.elr_match = elr.ELRMatcher(elr.ELRTrace.load(Path(args.elr_match)), self._elr_named)
            print(f"elr: tracking {args.elr_match} — {len(self.elr_match.trace.samples)} samples over "
                  f"steps {self.elr_match.trace.samples[0].step}..{self.elr_match.trace.samples[-1].step}", flush=True)

        if args.reselect_every > 0 and not args.sft:
            raise ValueError("--reselect-every selects assistant-byte windows: it needs --sft")
        aug = {"noise": args.aug_noise, "r2l": args.aug_r2l, "offset_max": args.aug_offset}
        train_shard = ByteShard(args.data, "train", sft=args.sft, seed=args.seed, keep=args.keep, **aug)
        self._main_shard = train_shard  # the shard --keep / --reselect-every select from (the mix wraps it)
        self.val_shard = ByteShard(args.data, "val", sft=args.sft)
        self._reselect_next = 0.0 if args.reselect_every > 0 else None  # pass 0 at the start, then every FRAC
        self._reselect_prev = None  # the previous pass's per-window losses
        self._reselect_tau = None  # tau_m, fixed by pass 0
        if args.mix:
            extras = []
            for spec in args.mix:
                prefix, share, plain, fim = parse_mix_spec(spec)
                extras.append((ByteShard(prefix, "train", sft=args.sft or plain, plain=plain, fim=fim,
                                         seed=args.seed + len(extras) + 1, **aug), share))
            main_w = max(1.0 - sum(w for _, w in extras), 0.0)
            train_shard = MixedShard([train_shard] + [s for s, _ in extras], [main_w] + [w for _, w in extras])
            print("training mix:", {args.data: main_w, **{spec: parse_mix_spec(spec)[1] for spec in args.mix}}, flush=True)
        self.train_shard = train_shard
        if args.init_from and not (args.resume and ckpt_path_exists(out_dir)):
            _step, _ = load_checkpoint(Path(args.init_from), model, None, allow_new=("fusion.",))
            print(f"initialized weights from {args.init_from} (step {_step})", flush=True)
        # A per-schedule default, not a global one: `wsd` has always decayed to 0.1x and every lab arm on
        # record was run that way, so changing its floor would silently make new arms incomparable to their
        # own controls. Only the branches go to zero.
        self.min_lr_ratio = args.min_lr_ratio if args.min_lr_ratio is not None else (
            0.0 if args.schedule in BRANCH_SCHEDULES else 0.1)
        self.gen = torch.Generator().manual_seed(args.seed)
        # latent feedback: the 1-/2-/3-pass micro-batch mixture, drawn on the same generator (reproducible)
        self.feedback_mix = None
        if cfg.feedback.level != "off":
            mix = [float(v) for v in args.feedback_mix.split(",")]
            assert len(mix) == 3 and all(v >= 0 for v in mix) and sum(mix) > 0, "--feedback-mix wants three non-negative weights"
            self.feedback_mix = [v / sum(mix) for v in mix]
        self.feedback_window = int(args.feedback_window or 0)

        self.step, self.t_start = 0, time.time()
        self.sched_total = None  # step horizon of the trunk/branch schedules: fixed at the first probe, survives resume
        self.ckpt_path = out_dir / "last.pt"
        if args.resume and self.ckpt_path.exists():
            self.step, extra = load_checkpoint(self.ckpt_path, model, self.opt, ck=self._resume_ck)
            self._resume_ck = None
            if args.schedule != "wsd":
                self.sched_total = extra.get("sched_total") or extra.get("total_steps")
            if "generator_state" in extra:
                self.gen.set_state(torch.tensor(extra["generator_state"], dtype=torch.uint8))
            self._ema_state = extra.get("ema")
            # the wall clock survives a resume too (max-minutes, the wsd progress and elapsed_min continue
            # instead of restarting); checkpoints from before 2026-08-24 carry no clock — read the log's last
            elapsed = extra.get("elapsed_sec")
            if elapsed is None:
                elapsed = runinfo.last_elapsed_sec(out_dir)
            self.t_start = time.time() - float(elapsed or 0.0)
            print(f"resumed from step {self.step} at {(elapsed or 0.0) / 60:.1f} min", flush=True)

        self.ema_decay = float(getattr(args, "eval_ema", 0.0) or 0.0)
        self.ema = [p.detach().clone() for p in model.parameters()] if self.ema_decay > 0 else None
        if self.ema is not None and getattr(self, "_ema_state", None):
            for e, s in zip(self.ema, self._ema_state):
                e.copy_(s.to(e.device))
        self.fwd = torch.compile(model) if args.compile else model
        if getattr(args, "tf32", False):
            torch.set_float32_matmul_precision("high")  # TF32 for fp32 GEMMs; off (="highest") is the frozen default
        log_path = out_dir / "log.jsonl"
        if not (args.resume and self.step > 0) and log_path.exists() and log_path.stat().st_size > 0:
            # a fresh start into a used directory keeps the old run's log aside instead of appending to it
            n = 1
            while (out_dir / f"log.{n}.jsonl").exists():
                n += 1
            log_path.rename(out_dir / f"log.{n}.jsonl")
        self.log_f = open(log_path, "a", encoding="utf-8")
        self.tokens_per_step = args.batch_size * args.seq_len * args.grad_accum
        # What the run is doing right now, read by the daemon for the studio ("train", "eval 3/16",
        # "eval ema 3/16", "checkpoint") — a reply that waits on the GPU gate can say why (2026-08-25).
        self.phase = "train"
        self.total_steps = args.max_steps
        self.time_driven = args.max_steps == 0  # schedules follow wall-clock progress; step count is only an estimate
        self.budget_sec = args.max_minutes * 60
        self._stop = False
        self.stopped_reason = None

    # ------------------------------------------------------------------------------
    def log(self, rec: Dict):
        rec["step"] = self.step
        rec["elapsed_min"] = (time.time() - self.t_start) / 60
        if self.device.type == "cuda":
            rec["peak_gb"] = torch.cuda.max_memory_allocated() / 2**30  # the process peak (serving residue included in the daemon)
        self.log_f.write(json.dumps(rec) + "\n")
        self.log_f.flush()
        print(json.dumps(rec), flush=True)

    def request_stop(self, reason: str = "requested"):
        """Graceful: the run ends at the next step boundary, with the final eval and checkpoint."""
        self._stop = True
        self.stopped_reason = reason

    def save(self):
        self.phase = "checkpoint"
        try:
            self._save_checkpoint()
        finally:
            self.phase = "train"

    def _save_checkpoint(self):
        extra = {"generator_state": self.gen.get_state().tolist(), "total_steps": self.total_steps,
                 "sched_total": self.sched_total, "schedule": self.args.schedule,
                 "n_params": self.n_params, "bytes_seen": self.step * self.tokens_per_step,
                 "elapsed_sec": time.time() - self.t_start}
        if self.ema is not None:
            extra["ema"] = self.ema
        save_checkpoint(self.ckpt_path, self.model, self.opt, self.step, self.cfg, extra)

    @torch.no_grad()
    def _swap_in_ema(self):
        """Put the EMA weights into the model and return the raw ones (swap back with `_restore`)."""
        params = list(self.model.parameters())
        raw = [p.detach().clone() for p in params]
        for p, e in zip(params, self.ema):
            p.copy_(e)
        return raw

    @torch.no_grad()
    def _restore(self, raw):
        for p, r in zip(self.model.parameters(), raw):
            p.copy_(r)

    def _evaluate(self, target_ratio: float):
        """A generator (`ev = yield from self._evaluate(...)`): every eval window yields a ("slice", None)
        so the daemon can slot a reply between windows; the EMA pass swaps weights in and out around its
        own windows and never leaves them swapped across a yield's caller boundary."""
        args, device = self.args, self.device

        def windows():
            return evaluate_batches(self.model, self.val_shard, args.batch_size, args.seq_len, args.eval_batches, device,
                                    target_ratio, spread=args.eval_spread, feedback_passes=args.eval_feedback_passes)

        def drain(gen, label):
            i = 0
            while True:
                self.phase = f"{label} {i + 1}/{args.eval_batches}"
                try:
                    next(gen)
                except StopIteration as done:
                    return done.value
                i += 1
                yield ("slice", None)

        try:
            ev = yield from drain(windows(), "eval")
            if self.ema is not None:
                raw = self._swap_in_ema()
                try:
                    ev["val_bpb_ema"] = (yield from drain(windows(), "eval ema"))["val_bpb"]
                finally:
                    self._restore(raw)
        finally:
            self.phase = "train"
        return ev

    def _train_step(self, target_ratio: float):
        """One optimizer step as a generator: yields ("slice", None) after each accumulation micro-batch
        (the daemon's preemption points), returns the stats dict of 0-dim device tensors. Calling
        float() on them is the one sync per logging interval, as before."""
        args, device = self.args, self.device
        self.opt.zero_grad(set_to_none=True)
        agg = {}
        total_n = None
        for _ in range(args.grad_accum):
            batch, lmask = self.train_shard.sample_batch(args.batch_size, args.seq_len, self.gen)
            batch = batch.to(device, non_blocking=True)
            lmask = lmask.to(device, non_blocking=True) if lmask is not None else None
            n_pass = self._feedback_passes()
            if n_pass > 1 and self.feedback_window and batch.shape[1] > self.feedback_window + 1:
                batch = batch[:, : self.feedback_window + 1]  # the memory fallback: shorter multi-pass windows
                lmask = lmask[:, : self.feedback_window + 1] if lmask is not None else None
            if n_pass > 1 and args.feedback_detach:
                # one pass of activations alive at a time: back-propagate each pass as it is produced
                stats = {}
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    it = iter_pass_losses(self.fwd, batch, target_ratio, self.cfg.dc.ratio_loss_weight, lmask,
                                          passes=n_pass, gen=self.gen, detach=True)
                    for j, (lk, nk, sk, _o) in enumerate(it):
                        (lk if j == 0 else lk / (n_pass - 1)).backward()
                        if j == 0:
                            n, stats = nk, sk
                        else:
                            stats["ce_fb_sum"] = stats.get("ce_fb_sum", 0.0) + sk["ce_sum"] / (n_pass - 1)
            else:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    loss, n, stats, _out = compute_losses(self.fwd, batch, target_ratio, self.cfg.dc.ratio_loss_weight, lmask,
                                                          passes=n_pass, gen=self.gen)
                loss.backward()  # unnormalised sum; normalised once below
            if self.feedback_mix is not None:
                stats["passes"] = torch.tensor(float(n_pass), device=device)
            total_n = n if total_n is None else total_n + n
            for k, v in stats.items():
                agg[k] = v if k not in agg else agg[k] + v
            yield ("slice", None)
        grads = [p.grad for p in self.model.parameters() if p.grad is not None]
        torch._foreach_div_(grads, total_n)
        gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
        self.opt.step()
        for m in self._moe:
            m.update_bias()  # lossfree routing: the expert bias follows the step's load, never the gradient
        if self.ema is not None:
            with torch.no_grad():
                torch._foreach_lerp_(self.ema, [p.detach() for p in self.model.parameters()], 1.0 - self.ema_decay)
        out = {"ce": agg["ce_sum"] / total_n, "ratio": agg["ratio"] / args.grad_accum, "bpic": agg["bpic"] / args.grad_accum, "grad_norm": gnorm}
        for k in agg:
            if k.startswith("moe_"):
                out[k] = agg[k] / args.grad_accum
        if "ce_fb_sum" in agg:
            out["ce_fb"] = agg["ce_fb_sum"] / total_n  # the feedback passes' CE per token (mean over the extra passes)
        if "passes" in agg:
            out["passes"] = agg["passes"] / args.grad_accum
        return out

    def _feedback_passes(self) -> int:
        """1, 2 or 3 forward passes for the next micro-batch, drawn from --feedback-mix on the run's
        generator; always 1 when latent feedback is off (and nothing is drawn, so plain runs are unchanged)."""
        if self.feedback_mix is None:
            return 1
        u = float(torch.rand((), generator=self.gen))
        p1, p2 = self.feedback_mix[0], self.feedback_mix[0] + self.feedback_mix[1]
        return 1 + int(u >= p1) + int(u >= p2)

    def _reselect(self):
        """One re-selection pass with the live model (docs/research/curriculum-2026-08-28.md). A generator: the
        GPU gate is released between scoring batches, so replies slot in as they do between training slices."""
        from ..data.select_sft import trajectory_keep, window_losses, window_starts

        args = self.args
        shard = self._main_shard
        starts = window_starts(shard, args.seq_len, args.reselect_windows)
        losses = np.full(len(starts), np.nan, dtype=np.float64)
        self.model.eval()
        try:
            for i in range(0, len(starts), args.batch_size):
                chunk = starts[i : i + args.batch_size]
                losses[i : i + len(chunk)] = window_losses(self.model, shard, chunk, args.seq_len, args.batch_size, self.device)
                yield ("slice", None)
        finally:
            self.model.train()
        keep, rep = trajectory_keep(starts, losses, self._reselect_prev, self._reselect_tau, args.reselect_eps,
                                    args.reselect_floor, seed=args.seed + self.step)
        if self._reselect_tau is None:
            self._reselect_tau = rep["tau_m"]
        self._reselect_prev = losses
        shard.keep = keep
        self._reselect_next += args.reselect_every
        self.log({"reselect": rep, "step": self.step, "progress": round(self._progress(), 3)})

    def _progress(self) -> float:
        if self.time_driven:
            return min((time.time() - self.t_start) / self.budget_sec, 1.0)
        return min(self.step / max(self.total_steps, 1), 1.0)

    def _running(self) -> bool:
        if self.args.schedule in BRANCH_SCHEDULES:  # a branch ends on its step horizon, resume-proof
            return self.step < self.sched_total
        return (self._progress() < 1.0) if self.time_driven else (self.step < self.total_steps)

    def snapshot(self) -> Path:
        """Weights-only `snap_<step>.pt`: a branch point (`--init-from` loads it; no optimizer state)."""
        p = self.out_dir / f"snap_{self.step:08d}.pt"
        save_checkpoint(p, self.model, None, self.step, self.cfg, {"sched_total": self.sched_total, "schedule": self.args.schedule})
        self.log({"snapshot": str(p)})
        return p

    def run(self):
        """The whole run. Drain it (`for _ in t.run(): pass`) for the old behaviour."""
        args, cfg, device = self.args, self.cfg, self.device
        self.model.train()
        if self._reselect_next is not None and self.step == 0:
            yield from self._reselect()  # pass 0: tau_m and the first keep, before any training

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
        snapped_at = args.snapshot_at is not None and (self.sched_total or 0) and self.step >= args.snapshot_at * self.sched_total

        while self._running() and not self._stop:
            # wsd follows wall-clock progress (arms compare at equal wall-clock); trunk/branch follow
            # the step horizon fixed at the first probe, so a resume continues the schedule instead of
            # restarting it
            pr = self._progress() if args.schedule == "wsd" else min(self.step / max(self.sched_total, 1), 1.0)
            horizon = 1000
            sched_step = int(pr * horizon)
            # a branch starts from a trunk that finished its ATDC ramp: hold the final target
            if args.schedule in BRANCH_SCHEDULES:
                target_ratio = cfg.dc.target_ratio_final
            else:
                target_ratio = atdc_target_ratio(
                    sched_step, horizon, cfg.dc.target_ratio_init, cfg.dc.target_ratio_final,
                    cfg.dc.schedule_warmup_frac,
                    loss_window=(sum(self._loss_window) / len(self._loss_window)
                                 if len(self._loss_window) >= cfg.dc.adapt_window else None),
                    threshold=cfg.dc.adapt_threshold, rate=cfg.dc.adapt_rate)
            # the router prices its own budget off the live target (mote/model/dc.py::_bound)
            self.model.routing_module.target_ratio = target_ratio
            lr = schedule_lr(args.schedule, sched_step, horizon, args.lr, min_ratio=self.min_lr_ratio,
                             decay_frac=args.branch_decay_frac)
            set_lr(self.opt, lr)
            if self.elr_match is not None:
                # after set_lr, which would otherwise overwrite it: param_lr wins for Muon's matrices, the
                # schedule still drives the AdamW groups (they are identical between the arms being compared)
                lr = self.elr_match.apply(self.opt, self.step) or lr
            stats = yield from self._train_step(target_ratio)
            self._loss_window.append(float(stats["ce"]))  # ATDC's proficiency trigger reads this window
            self.step += 1
            if args.snapshot_steps and self.step // args.snapshot_steps > snap_idx:
                snap_idx = self.step // args.snapshot_steps
                self.snapshot()
            if args.snapshot_at is not None and not snapped_at and pr >= args.snapshot_at:
                snapped_at = True
                print(f"fork snapshot at {pr:.3f} of the horizon (step {self.step})", flush=True)
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
                if self.norms:
                    sample = self.norms.sample(self.step, lr, self.args.weight_decay)  # 61 norms in one device->host transfer
                    rec.update(self.norms.record(sample))
                    if self.elr_match is not None:
                        self.elr_match.refresh(sample)
                    if self.elr_trace is not None:
                        # rewritten whole, so not on every sample: a 2-h run logs ~3,000 of them and
                        # saving each one would push several GB through the disk for a 2 MB file
                        self.elr_trace.add(sample)
                        if len(self.elr_trace.samples) % elr.SAVE_EVERY == 0:
                            self.elr_trace.save(self._elr_trace_path)
                    if self.norm_guard is not None:
                        tripped = self.norm_guard.update(sample)
                        if tripped:
                            rec["norm_guard"] = tripped
                            if args.norm_guard == "stop":
                                self.request_stop(tripped)  # already prefixed "norm collapse"; jobs.py holds the queue on it
                self.log(rec)
            if self.step % args.eval_every == 0:
                ev = yield from self._evaluate(target_ratio)
                ev["sample"] = chunk_sample(self.model, "The router compares each byte with the one before it. Where they stop looking alike, it draws a boundary.", device)
                self.log({"eval": ev})
            if self._reselect_next is not None and self._reselect_next < 1.0 and self._progress() >= self._reselect_next:
                yield from self._reselect()
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
        if self.elr_trace is not None:  # the throttled save above can be up to SAVE_EVERY samples behind
            self.elr_trace.save(self._elr_trace_path)

        if self.stopped_reason != "interrupted":  # an interrupted run resumes: checkpoint only, no final eval
            ev = yield from self._evaluate(cfg.dc.target_ratio_final)
            ev["sample"] = chunk_sample(self.model, "The router compares each byte with the one before it. Where they stop looking alike, it draws a boundary.", device)
            self.log({"eval": ev, "final": True})

        self.save()
        self.log({"done": True, "final_step": self.step, "interrupted": self.stopped_reason == "interrupted"})

    def close(self):
        self.log_f.close()
        determinism.apply(True)  # restore the default for whatever runs next in this process


def main(argv=None):
    t = Trainer(argv)
    try:
        for _ in t.run():
            pass
    finally:
        t.close()


if __name__ == "__main__":
    main()
