"""Residual blocks and isotropic stacks (the encoder, main network and decoder of the H-Net).

Pre-norm blocks with the residual stream kept in fp32, exactly as in the official H-Net code.
Every mixer exposes ``forward(x, ...)``, ``step(x, state)`` and ``allocate_inference_cache``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .mamba3 import Mamba3Mixer
from .norm import RMSNorm
from .relation import FullRelation, SwiGLU


class Block(nn.Module):
    def __init__(self, d_model: int, mixer: nn.Module, mlp: Optional[nn.Module], eps: float = 1e-5, residual_in_fp32: bool = True, device=None, dtype=None):
        super().__init__()
        fk = {"device": device, "dtype": dtype}
        self.residual_in_fp32 = residual_in_fp32
        self.norm1 = RMSNorm(d_model, eps=eps, **fk)
        self.mixer = mixer
        self.mlp = mlp
        self.norm2 = RMSNorm(d_model, eps=eps, **fk) if mlp is not None else None

    @property
    def height(self) -> int:
        return 2 if self.mlp is not None else 1

    def forward(self, hidden: torch.Tensor, residual: Optional[torch.Tensor], cache: Any = None, return_cache: bool = False):
        hidden, residual = self.norm1(hidden, residual=residual, prenorm=True, residual_in_fp32=self.residual_in_fp32)
        new_cache = None
        if isinstance(self.mixer, FullRelation):
            if return_cache:
                hidden, new_cache = self.mixer(hidden, cache=cache, return_cache=True)
            else:
                hidden = self.mixer(hidden, cache=cache)
        else:  # Mamba-3
            if return_cache:
                hidden, new_cache = self.mixer(hidden, return_final_states=True)
            else:
                hidden = self.mixer(hidden)
        if self.mlp is not None:
            hidden, residual = self.norm2(hidden, residual=residual, prenorm=True, residual_in_fp32=self.residual_in_fp32)
            hidden = self.mlp(hidden)
        return hidden, residual, new_cache

    def step(self, hidden: torch.Tensor, residual: Optional[torch.Tensor], cache: Any):
        hidden, residual = self.norm1(hidden, residual=residual, prenorm=True, residual_in_fp32=self.residual_in_fp32)
        if isinstance(self.mixer, FullRelation):
            hidden, new_cache = self.mixer(hidden, cache=cache, return_cache=True)
        else:
            hidden, new_cache = self.mixer.step(hidden, cache)
        if self.mlp is not None:
            hidden, residual = self.norm2(hidden, residual=residual, prenorm=True, residual_in_fp32=self.residual_in_fp32)
            hidden = self.mlp(hidden)
        return hidden, residual, new_cache

    def allocate_inference_cache(self, batch_size: int, device, dtype=None):
        if isinstance(self.mixer, FullRelation):
            return None  # Relation cache is built lazily from the first prefill
        return self.mixer.allocate_inference_cache(batch_size, device, dtype)


class Isotropic(nn.Module):
    """A stack of blocks followed by a final RMSNorm that folds in the residual."""

    def __init__(self, blocks: List[Block], d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.layers = nn.ModuleList(blocks)
        self.rmsnorm = RMSNorm(d_model, eps=eps, device=device, dtype=dtype)

    @property
    def height(self) -> int:
        return sum(b.height for b in self.layers)

    def forward(self, hidden: torch.Tensor, caches: Optional[List[Any]] = None, return_caches: bool = False):
        residual = None
        new_caches: List[Any] = []
        for i, layer in enumerate(self.layers):
            cache = caches[i] if caches is not None else None
            hidden, residual, c = layer(hidden, residual, cache=cache, return_cache=return_caches)
            new_caches.append(c)
        hidden = self.rmsnorm(hidden, residual=residual, prenorm=False, residual_in_fp32=True)
        return (hidden, new_caches) if return_caches else hidden

    def step(self, hidden: torch.Tensor, caches: List[Any]) -> Tuple[torch.Tensor, List[Any]]:
        residual = None
        new_caches: List[Any] = []
        for layer, cache in zip(self.layers, caches):
            hidden, residual, c = layer.step(hidden, residual, cache)
            new_caches.append(c)
        hidden = self.rmsnorm(hidden, residual=residual, prenorm=False, residual_in_fp32=True)
        return hidden, new_caches

    def allocate_inference_cache(self, batch_size: int, device, dtype=None) -> List[Any]:
        return [layer.allocate_inference_cache(batch_size, device, dtype) for layer in self.layers]


# --------------------------------------------------------------------------------------
def make_mamba3_stack(n_layers: int, d_model: int, cfg, eps: float, layer_offset: int = 0, device=None, dtype=None) -> Isotropic:
    """Encoder / decoder: Mamba-3 mixers without FFN ('m' blocks)."""
    blocks = []
    for i in range(n_layers):
        mixer = Mamba3Mixer(
            d_model,
            d_state=cfg.d_state,
            expand=cfg.expand,
            headdim=cfg.headdim,
            ngroups=cfg.ngroups,
            rope_fraction=cfg.rope_fraction,
            A_floor=cfg.A_floor,
            chunk_size=cfg.chunk_size,
            layer_idx=layer_offset + i,
            device=device,
            dtype=dtype,
        )
        blocks.append(Block(d_model, mixer, None, eps=eps, device=device, dtype=dtype))
    return Isotropic(blocks, d_model, eps=eps, device=device, dtype=dtype)


def make_relation_stack(cfg, eps: float, device=None, dtype=None) -> Isotropic:
    """Main network: Full Relation mixers with SwiGLU FFNs ('R' blocks)."""
    blocks = []
    for i in range(cfg.n_layers):
        mixer = FullRelation(
            cfg.d_model,
            cfg.n_heads,
            layer_idx=i,
            tau_s=cfg.tau_s,
            lambda_init=cfg.lambda_init,
            rope_theta=cfg.rope_theta,
            givens=cfg.givens,
            device=device,
            dtype=dtype,
        )
        mlp = SwiGLU(cfg.d_model, cfg.d_ff, device=device, dtype=dtype)
        blocks.append(Block(cfg.d_model, mixer, mlp, eps=eps, device=device, dtype=dtype))
    return Isotropic(blocks, cfg.d_model, eps=eps, device=device, dtype=dtype)
