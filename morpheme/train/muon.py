"""Muon — momentum + Newton-Schulz orthogonalization for 2-D weight matrices (Jordan et al. 2024),
with the update scaled to match AdamW's per-element RMS (Liu et al. 2025, "Muon is scalable"), so
the same learning rate and weight decay as AdamW can be used. Everything that is not a hidden
2-D matrix (embeddings / tied head, norms, biases, SSM scalars, λ, Givens angles) stays on AdamW.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def newton_schulz(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Quintic Newton-Schulz iteration approximating the orthogonal factor of G (bf16, as in the reference)."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(torch.bfloat16)
    X = X / (X.norm() + eps)
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3, momentum: float = 0.95, nesterov: bool = True, weight_decay: float = 0.0, ns_steps: int = 5, rms_scale: float = 0.2):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay, ns_steps=ns_steps, rms_scale=rms_scale)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, mom, wd = group["lr"], group["momentum"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                assert p.ndim == 2, "Muon is for 2-D matrices only"
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(mom).add_(g)
                upd = g.add(buf, alpha=mom) if group["nesterov"] else buf
                upd = newton_schulz(upd, steps=group["ns_steps"]).to(p.dtype)
                upd = upd * (group["rms_scale"] * max(p.shape[0], p.shape[1]) ** 0.5)
                if wd:
                    p.mul_(1.0 - lr * wd)
                p.add_(upd, alpha=-lr)
        return loss


def split_muon_params(model) -> tuple[list, list]:
    """(muon_params, other_params): hidden 2-D matrices go to Muon; embeddings / tied head and everything
    with ndim != 2 go to AdamW."""
    emb_ids = {id(model.embeddings.weight), id(model.lm_head.weight)}
    muon, other = [], []
    seen = set()
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        if p.ndim == 2 and id(p) not in emb_ids:
            muon.append(p)
        else:
            other.append(p)
    return muon, other
