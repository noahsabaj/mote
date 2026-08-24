"""RMSNorm with the fused-residual interface the H-Net blocks expect.

Mirrors the semantics of flash_attn's Triton RMSNorm used by the official H-Net code:
``norm(x, residual, prenorm=True)`` adds the residual (in fp32), returns the normalized
tensor in the input dtype and the new fp32 residual stream. Pure PyTorch so it runs anywhere;
torch.compile fuses it well enough for our sizes.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

try:  # the fused Triton add+RMSNorm from the Mamba checkout (same flash_attn semantics as the reference below)
    from mamba_ssm.ops.triton.layer_norm import rms_norm_fn as _fused_rms_norm  # type: ignore

    HAS_FUSED_NORM = True
except Exception:  # CPU-only installs, or no Triton
    _fused_rms_norm = None
    HAS_FUSED_NORM = False


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
        # Fused Triton path (phase K, 2026-08-23): one kernel for add+norm instead of the mul/copy chain
        # that profiling showed at ~39% of GPU time. Exactness vs the reference is tested in
        # tests/test_fused_norm.py; MOTE_NO_FUSED_NORM=1 is the kill switch.
        if HAS_FUSED_NORM and x.is_cuda and not os.environ.get("MOTE_NO_FUSED_NORM"):
            return _fused_rms_norm(x, self.weight, None, residual=residual, eps=self.eps,
                                   prenorm=prenorm, residual_in_fp32=residual_in_fp32)
        out_dtype = x.dtype
        if residual is not None:
            x = (x.float() + residual.float()) if residual_in_fp32 else x + residual
        elif residual_in_fp32 and prenorm:
            x = x.float()
        y = self._norm(x, out_dtype)
        if prenorm:
            return y, x
        return y
