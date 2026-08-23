"""Analytic training FLOPs per byte for the H-Net, so every run can report TFLOPS and MFU.

The usual 6·N rule (forward + backward ≈ 6 FLOPs per parameter per token) applied per stage:
byte-level modules see every byte, the main network sees one position per chunk, and the two
attention-like mixers add their pairwise matmuls (4·T·d per position forward, ×3 with backward).
The SSM scan inside Mamba-3 is omitted (well under 1 % at these widths).
"""

from __future__ import annotations

import torch

# dense bf16 tensor-core peak with fp32 accumulation, TFLOPS (vendor data sheets)
PEAK_TFLOPS = {
    "NVIDIA GeForce RTX 4060 Ti": 44.1,
    "NVIDIA GeForce RTX 4090": 165.2,
    "NVIDIA H100 80GB HBM3": 989.0,
    "NVIDIA H100 PCIe": 756.0,
    "NVIDIA RTX PRO 6000 Blackwell Workstation Edition": 503.8,
    "NVIDIA A100-SXM4-80GB": 312.0,
}


def _n(module) -> int:
    seen, n = set(), 0
    for p in module.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            n += p.numel()
    return n


def flops_per_byte(model, seq_len: int, bytes_per_chunk: float) -> float:
    cfg = model.cfg
    D0 = cfg.d_model_outer
    bpc = max(float(bytes_per_chunk), 1.0)
    T_main = seq_len / bpc  # chunk-level positions per sequence

    byte_level = _n(model.encoder) + _n(model.decoder) + _n(model.routing_module) + _n(model.residual_proj)
    byte_level += _n(model.lm_head)  # next-byte head
    main = _n(model.main_network) + (model.pad_dimension.numel() if model.pad_dimension is not None else 0)
    fl = 6.0 * byte_level + 6.0 * main / bpc
    # Relation pairwise matmuls (U = P1·P2ᵀ and Y = F·Ĩ): 4·T·d per position forward, ×3 with backward
    fl += 12.0 * T_main * cfg.main.d_model * cfg.main.n_layers / bpc
    if model.mbp_head is not None:
        fl += 6.0 * (_n(model.mbp_head) + _n(model.lm_head))  # head runs over every byte + shared lm_head again
        fl += 12.0 * seq_len * D0 * cfg.mbp.n_layers  # LCA attention (dense mask; block-sparse would be less)
    return fl


def peak_tflops_for(device) -> float | None:
    if not torch.cuda.is_available():
        return None
    name = torch.cuda.get_device_name(device)
    for key, val in PEAK_TFLOPS.items():
        if name.startswith(key):
            return val
    return None
