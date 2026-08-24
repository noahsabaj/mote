"""Muon — momentum + Newton-Schulz orthogonalization for 2-D weight matrices (Jordan et al. 2024),
with the update scaled to match AdamW's per-element RMS (Liu et al. 2025, "Muon is scalable"), so
the same learning rate and weight decay as AdamW can be used. Everything that is not a hidden
2-D matrix (embeddings / tied head, norms, biases, SSM scalars, λ, Givens angles) stays on AdamW.

Muon-SW (2607.23777): identical update, but the decoupled weight decay is scaled by η_t/η_max —
the decay term becomes O(η²) and no longer shifts the stationary point as the schedule cools
(`sw_decay=True`, `lr_max` = the schedule's peak learning rate).
"""

from __future__ import annotations

import torch


@torch.no_grad()
def newton_schulz(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Quintic Newton-Schulz iteration approximating the orthogonal factor of G (bf16, as in the reference).
    G may carry leading batch dims ([..., m, n]): every matrix is normalised by its own Frobenius norm and
    the iteration runs as batched matmuls — one launch per step for a whole shape group (2026-08-24)."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(torch.float32)
    X = (X / (X.norm(dim=(-2, -1), keepdim=True) + eps)).to(torch.bfloat16)  # normalise in fp32, iterate in bf16
    transposed = X.shape[-2] > X.shape[-1]
    if transposed:
        X = X.transpose(-2, -1)
    for _ in range(steps):
        A = X @ X.transpose(-2, -1)
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.transpose(-2, -1)
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3, momentum: float = 0.95, nesterov: bool = True, weight_decay: float = 0.0, ns_steps: int = 5, rms_scale: float = 0.2, sw_decay: bool = False, lr_max: float = 1.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay, ns_steps=ns_steps, rms_scale=rms_scale, sw_decay=sw_decay, lr_max=lr_max)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, mom, wd = group["lr"], group["momentum"], group["weight_decay"]
            # momentum per parameter, then one batched Newton-Schulz per matrix shape (the 12 Relation layers'
            # 768² projections are one launch per iteration instead of 48). A 3-D parameter is a stack of
            # matrices (MoE expert weights [E, m, n]): every slice is orthogonalised on its own, in the same launch.
            by_shape: dict = {}
            for p in group["params"]:
                if p.grad is None:
                    continue
                assert p.ndim in (2, 3), "Muon is for 2-D matrices (or 3-D stacks of them)"
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(mom).add_(g)
                upd = g.add(buf, alpha=mom) if group["nesterov"] else buf
                m, n = p.shape[-2], p.shape[-1]
                by_shape.setdefault((m, n), []).append((p, upd.reshape(-1, m, n)))
            for (m, n), items in by_shape.items():
                stacked = torch.cat([u for _, u in items], dim=0) if len(items) > 1 else items[0][1]
                orth = newton_schulz(stacked, steps=group["ns_steps"])
                scale = group["rms_scale"] * max(m, n) ** 0.5
                decay = (lr * lr / group["lr_max"] if group["sw_decay"] else lr) * wd if wd else 0.0
                i = 0
                for p, u in items:
                    o = orth[i:i + u.shape[0]].reshape(p.shape)
                    i += u.shape[0]
                    if decay:
                        p.mul_(1.0 - decay)
                    p.add_(o.to(p.dtype) * scale, alpha=-lr)
        return loss


def split_muon_params(model) -> tuple[list, list]:
    """(muon_params, other_params): hidden 2-D matrices and MoE expert stacks (`_muon_stack`, [E, m, n]) go
    to Muon; embeddings / tied head, everything else with ndim != 2, Mamba-3's `in_proj` and the MoE
    router (`_no_muon`) go to AdamW. `in_proj` row-stacks eight unrelated sub-projections (z, x, B, C,
    dt, A, trap, angles); orthogonalising it as one matrix is measured to be worse than AdamW on Mamba
    (2608.03941), while `out_proj` carries Muon's gain."""
    from ..model.mamba3 import Mamba3Mixer

    skip = {id(model.embeddings.weight), id(model.lm_head.weight)}
    for m in model.modules():
        if isinstance(m, Mamba3Mixer):
            skip.add(id(m.in_proj.weight))
    muon, other = [], []
    seen = set()
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        stack = p.ndim == 3 and getattr(p, "_muon_stack", False)
        if (p.ndim == 2 or stack) and id(p) not in skip and not getattr(p, "_no_muon", False):
            muon.append(p)
        else:
            other.append(p)
    return muon, other
