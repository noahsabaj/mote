"""RMSNorm with the fused-residual interface the H-Net blocks expect.

Mirrors the semantics of flash_attn's Triton RMSNorm used by the official H-Net code:
``norm(x, residual, prenorm=True)`` adds the residual (in fp32), returns the normalized
tensor in the input dtype and the new fp32 residual stream. Pure PyTorch so it runs anywhere;
torch.compile fuses it well enough for our sizes.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))

    def _norm(self, x: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * self.weight.float()).to(out_dtype)

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        prenorm: bool = False,
        residual_in_fp32: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        out_dtype = x.dtype
        if residual is not None:
            x = (x.float() + residual.float()) if residual_in_fp32 else x + residual
        elif residual_in_fp32 and prenorm:
            x = x.float()
        y = self._norm(x, out_dtype)
        if prenorm:
            return y, x
        return y
