import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

from .rotary import apply_rotary_emb, apply_block_rotations, compute_rope_frequencies


@dataclass
class SISAConfig:
    d_model: int = 768
    n_heads: int = 12
    d_head: int = 64
    d_s: int = 32  # SSM channel dimension (must be even)
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    init_b_alpha: float = -5.0
    init_lambda: float = 0.31
    clamp_minimax: float = 11.0
    dropout: float = 0.0
    bias: bool = False


class SISAAttention(nn.Module):
    """
    SISA: SSM-Informed Softmax Attention
    Implements score-level fusion via augmented Q/K and a single SDPA call:
    Score_ij = (q_i^T k_j) / sqrt(d_h) + lambda * C_bar_i^T B_bar_j
    """

    def __init__(self, config: SISAConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.d_s = config.d_s
        self.scale = 1.0 / math.sqrt(self.d_head)
        assert self.d_s % 2 == 0, "d_s (SSM channel dimension) must be even."

        # Content attention projections
        self.q_proj = nn.Linear(self.d_model, self.n_heads * self.d_head, bias=config.bias)
        self.k_proj = nn.Linear(self.d_model, self.n_heads * self.d_head, bias=config.bias)
        self.v_proj = nn.Linear(self.d_model, self.n_heads * self.d_head, bias=config.bias)
        self.out_proj = nn.Linear(self.n_heads * self.d_head, self.d_model, bias=config.bias)

        # SSM channel projections
        self.b_proj = nn.Linear(self.d_model, self.n_heads * self.d_s, bias=config.bias)
        self.c_proj = nn.Linear(self.d_model, self.n_heads * self.d_s, bias=config.bias)

        # SSM decay: alpha_t = exp(-softplus(w_alpha^T x_t + b_alpha))
        self.w_alpha = nn.Linear(self.d_model, self.n_heads, bias=True)
        # Initialize b_alpha to config.init_b_alpha (half-life ~100 tokens)
        nn.init.zeros_(self.w_alpha.weight)
        nn.init.constant_(self.w_alpha.bias, config.init_b_alpha)

        # SSM phase (rotation frequencies): theta_t = W_theta x_t
        self.w_theta = nn.Linear(self.d_model, self.n_heads * (self.d_s // 2), bias=config.bias)
        nn.init.normal_(self.w_theta.weight, std=0.02)

        # Per-head learnable lambda in fp32
        # Initialize lambda_raw such that softplus(lambda_raw) approx init_lambda
        init_raw = math.log(math.exp(config.init_lambda) - 1.0) if config.init_lambda > 0 else -1.0
        self.lambda_raw = nn.Parameter(torch.full((self.n_heads,), init_raw, dtype=torch.float32))

        # Precompute RoPE table
        cos, sin = compute_rope_frequencies(self.d_head, config.max_seq_len, theta=config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def get_lambda(self) -> torch.Tensor:
        """Returns positive per-head lambda."""
        return F.softplus(self.lambda_raw)

    def forward(
        self,
        x: torch.Tensor,
        past_key_values: Optional[Dict[str, Any]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Forward pass for SISA.
        x: [batch, seq_len, d_model]
        """
        B, L, _ = x.shape
        device = x.device
        dtype = x.dtype

        # 1. Content projections [B, L, H, d_head] -> [B, H, L, d_head]
        q = self.q_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        # 2. SSM channel projections [B, L, H, d_s] -> [B, H, L, d_s]
        b_vec = self.b_proj(x).view(B, L, self.n_heads, self.d_s).transpose(1, 2)
        c_vec = self.c_proj(x).view(B, L, self.n_heads, self.d_s).transpose(1, 2)

        # Decay: log_alpha = -softplus(w_alpha(x)) in fp32
        log_alpha = -F.softplus(self.w_alpha(x).to(torch.float32)).view(B, L, self.n_heads).transpose(1, 2)  # [B, H, L]

        # Phase: theta = w_theta(x) in fp32 [B, H, L, d_s/2]
        theta = self.w_theta(x).view(B, L, self.n_heads, self.d_s // 2).transpose(1, 2).to(torch.float32)

        # Handle incremental decoding with cache
        if past_key_values is not None:
            prev_k = past_key_values.get("key")
            prev_v = past_key_values.get("value")
            prev_g = past_key_values.get("cum_g")  # [B, H, 1]
            prev_phi = past_key_values.get("cum_phi")  # [B, H, 1, d_s/2]
            past_len = prev_k.shape[2] if prev_k is not None else 0

            # Positional RoPE for incremental step
            cos = self.rope_cos[past_len : past_len + L].to(dtype=dtype, device=device)
            sin = self.rope_sin[past_len : past_len + L].to(dtype=dtype, device=device)
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)

            # Cumulative decay & phase
            cum_log_alpha = log_alpha.cumsum(dim=-1)
            cum_theta = theta.cumsum(dim=-2)

            if prev_g is not None:
                g = prev_g + cum_log_alpha
                phi = prev_phi + cum_theta
            else:
                g = cum_log_alpha
                phi = cum_theta

            # Clamp g for numerical stability in float16/bf16
            g_clamped = torch.clamp(g, -self.config.clamp_minimax, 0.0)

            exp_c = torch.exp(g_clamped).unsqueeze(-1).to(dtype)
            exp_b = torch.exp(-g_clamped).unsqueeze(-1).to(dtype)

            # Apply data-dependent rotations
            c_bar = exp_c * apply_block_rotations(c_vec, phi.to(dtype), transpose=False)
            b_bar = exp_b * apply_block_rotations(b_vec, phi.to(dtype), transpose=False)

            # Compute scale s = d_h^(1/4) * sqrt(lambda)
            lam = self.get_lambda().view(1, self.n_heads, 1, 1).to(dtype)
            s = (self.d_head ** 0.25) * torch.sqrt(lam)

            # Augmented Q and K
            q_aug = torch.cat([q, s * c_bar], dim=-1)
            k_aug = torch.cat([k, s * b_bar], dim=-1)

            if prev_k is not None:
                k_full = torch.cat([prev_k, k_aug], dim=2)
                v_full = torch.cat([prev_v, v], dim=2)
            else:
                k_full = k_aug
                v_full = v

            new_cache = {
                "key": k_full,
                "value": v_full,
                "cum_g": g[..., -1:],
                "cum_phi": phi[..., -1:, :],
            }

            out = F.scaled_dot_product_attention(
                q_aug, k_full, v_full,
                scale=self.scale,
                is_causal=(L > 1 and prev_k is None),
                dropout_p=self.config.dropout if self.training else 0.0,
            )

        else:
            # Parallel training / prefill mode
            cos = self.rope_cos[:L].to(dtype=dtype, device=device)
            sin = self.rope_sin[:L].to(dtype=dtype, device=device)
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)

            # Cumulative decay & phase in fp32
            g = log_alpha.cumsum(dim=-1)  # [B, H, L] (monotonically <= 0)
            phi = theta.cumsum(dim=-2)    # [B, H, L, d_s/2]

            g_clamped = torch.clamp(g, -self.config.clamp_minimax, 0.0)

            exp_c = torch.exp(g_clamped).unsqueeze(-1).to(dtype)
            exp_b = torch.exp(-g_clamped).unsqueeze(-1).to(dtype)

            # Apply data-dependent 2x2 block rotations R(Phi)
            c_bar = exp_c * apply_block_rotations(c_vec, phi.to(dtype), transpose=False)
            b_bar = exp_b * apply_block_rotations(b_vec, phi.to(dtype), transpose=False)

            # s = d_head^(1/4) * sqrt(lambda)
            lam = self.get_lambda().view(1, self.n_heads, 1, 1).to(dtype)
            s = (self.d_head ** 0.25) * torch.sqrt(lam)

            # Augmented Q, K: shape [B, H, L, d_head + d_s]
            q_aug = torch.cat([q, s * c_bar], dim=-1)
            k_aug = torch.cat([k, s * b_bar], dim=-1)

            # Single SDPA call with scale = 1 / sqrt(d_head)
            out = F.scaled_dot_product_attention(
                q_aug, k_aug, v,
                scale=self.scale,
                is_causal=True,
                dropout_p=self.config.dropout if self.training else 0.0,
            )

            new_cache = None
            if use_cache:
                new_cache = {
                    "key": k_aug,
                    "value": v,
                    "cum_g": g[..., -1:],
                    "cum_phi": phi[..., -1:, :],
                }

        # Transpose back: [B, H, L, d_head] -> [B, L, H * d_head] -> Output projection
        out = out.transpose(1, 2).contiguous().view(B, L, self.n_heads * self.d_head)
        return self.out_proj(out), new_cache
