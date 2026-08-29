"""Latent feedback — the full-bandwidth transformer's inter-step channel (Wang et al. 2608.08888).

A standard decoder feeds only the sampled token back to the bottom of the stack; the previous position's
top-layer state, the most processed thing the model has, is discarded. Here it is fused into the next
input through a gated linear unit,

    u_t = RMSNorm( W_U · h_{t-1} ⊙ σ(W_G · x_t) )

with the state on the value path and the plain input only as the gate: an additive fusion would let the
model zero the state path and recover its ordinary input, so reading the state is mandatory. The first
position of a sequence stays plain.

Training keeps parallel teacher forcing by paying the sequential dependency across a few PASSES rather
than across positions: pass 1 is the ordinary forward; pass k shifts pass k−1's top states one position
to the right, fuses them with the plain inputs, and re-runs the stack in parallel over all positions.
`prefix mixin` reverts a random prefix of every sequence to plain inputs in each pass, so training covers
the prompt-then-generate shape of inference (prompt plain, generated positions fused). Where the fusion
sits in the H-Net (`FeedbackCfg.level`) is decided by the config; this module is the arithmetic.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .norm import RMSNorm


class LatentFusion(nn.Module):
    """u = RMSNorm(W_U h_prev ⊙ σ(W_G x)). `d_state` is the width of the carried state, `d_in` the width of
    the input it replaces (the two coincide at both of Mote's fusion points, but need not)."""

    def __init__(self, d_state: int, d_in: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        fk = {"device": device, "dtype": dtype}
        self.w_u = nn.Linear(d_state, d_in, bias=False, **fk)
        self.w_g = nn.Linear(d_in, d_in, bias=False, **fk)
        self.norm = RMSNorm(d_in, eps=eps, **fk)

    def forward(self, h_prev: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.w_u(h_prev.to(x.dtype)) * torch.sigmoid(self.w_g(x)))


@dataclass
class FeedbackInput:
    """What a feedback pass needs from the pass before it.

    `top`: the previous pass's top states, [B, T, d_state] — unshifted; the model shifts them right by one
    position. `plain`: [B, T] bool, True where the plain input is kept (position 0 always, plus the
    prefix-mixin prefix). `detach`: cut the gradient into the previous pass (the memory fallback).
    `front`: chunk-level only — the previous pass's (h, residual, routing, hc_plain, next_mask), reused
    because the encoder and the router see plain bytes in every pass."""

    top: torch.Tensor
    plain: torch.Tensor
    detach: bool = False
    front: Optional[tuple] = None
    gen: Optional[torch.Generator] = None  # the run's CPU generator: jitter drawn on it keeps runs and resumes bitwise reproducible


def shift_right(h: torch.Tensor) -> torch.Tensor:
    """h_{t-1} at position t; zeros at t = 0 (position 0 is always plain, so the zero row is never read)."""
    return torch.cat([torch.zeros_like(h[:, :1]), h[:, :-1]], dim=1)


def plain_mask(batch: int, length: int, prefix: Optional[torch.Tensor], device) -> torch.Tensor:
    """[B, T] bool: True where the input stays plain. `prefix` [B] long = the number of leading positions
    kept plain (the prefix mixin's p, sampled by the caller); None = position 0 only."""
    pos = torch.arange(length, device=device)[None, :].expand(batch, -1)
    if prefix is None:
        return pos == 0
    return pos <= prefix.to(device)[:, None].clamp(min=0)


def sample_prefix(batch: int, length: int, gen: Optional[torch.Generator]) -> torch.Tensor:
    """The prefix mixin: p ~ U{0, …, T−1} per sequence, drawn on the CPU generator so runs stay
    bitwise reproducible. p = 0 keeps only position 0 plain (a fully fused sequence)."""
    return torch.randint(0, max(length, 1), (batch,), generator=gen)


def fuse(fusion: LatentFusion, x_plain: torch.Tensor, fb: FeedbackInput, jitter: float, training: bool) -> torch.Tensor:
    """The pass's inputs: `x_plain` where `fb.plain`, the fusion of the shifted previous top state and
    `x_plain` elsewhere. Jitter (uniform ±σ on the carried state) only in training passes."""
    prev = fb.top.detach() if fb.detach else fb.top
    prev = shift_right(prev)
    if training and jitter > 0:
        if fb.gen is not None:  # drawn on the CPU generator (checkpointed with the run), then moved
            noise = torch.rand(prev.shape, generator=fb.gen, dtype=torch.float32).to(prev.device, prev.dtype)
        else:
            noise = torch.rand_like(prev)
        prev = prev + (noise * 2.0 - 1.0) * jitter
    fused = fusion(prev, x_plain)
    return torch.where(fb.plain[..., None], x_plain, fused.to(x_plain.dtype))


def detach_tree(o):
    """`detach()` over tensors nested in tuples, lists, NamedTuples and dataclasses (a pass's front half)."""
    if isinstance(o, torch.Tensor):
        return o.detach()
    if isinstance(o, tuple):
        return type(o)(*[detach_tree(x) for x in o]) if hasattr(o, "_fields") else tuple(detach_tree(x) for x in o)
    if isinstance(o, list):
        return [detach_tree(x) for x in o]
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return dataclasses.replace(o, **{f.name: detach_tree(getattr(o, f.name)) for f in dataclasses.fields(o)})
    return o


def feedback_from(prev, gen: Optional[torch.Generator], detach: bool = False, mixin: bool = True) -> FeedbackInput:
    """The next pass's input from the previous pass's output (`prev.top`, `prev.front`). `mixin` samples
    the prefix-mixin prefix on `gen` (training); evaluation and serving keep only position 0 plain."""
    top = prev.top
    assert top is not None, "feedback_from: the model's feedback level is off"
    B, T = top.shape[:2]
    prefix = sample_prefix(B, T, gen) if mixin else None
    # detach mode back-propagates each pass on its own, so the reused front half must not reach into the
    # previous pass's (already freed) graph either
    front = detach_tree(prev.front) if detach else prev.front
    return FeedbackInput(top=top, plain=plain_mask(B, T, prefix, top.device), detach=detach, front=front, gen=gen)
