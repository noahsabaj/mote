import math
from typing import Dict, Any


def compute_sisa_parameter_budget(
    d_model: int,
    n_heads: int,
    d_s: int,
    d_ff_standard: int,
) -> Dict[str, Any]:
    """
    Computes exact parameter overhead for SISA's SSM projections
    and calculates the reduced SwiGLU FFN dimension d_ff_red to parameter-match
    a standard Transformer baseline.
    
    P_SSM = 2 * d * h * d_s (W_B, W_C)
          + d * h + h (w_alpha, b_alpha)
          + d * h * (d_s / 2) (W_theta)
          + h (lambda)
    """
    p_ssm = (
        2 * d_model * n_heads * d_s
        + d_model * n_heads + n_heads
        + d_model * n_heads * (d_s // 2)
        + n_heads
    )

    # SwiGLU has 3 matrices: W_gate, W_up, W_down of size [d_model, d_ff]
    # Total SwiGLU params per layer = 3 * d_model * d_ff
    # We want 3 * d_model * (d_ff - d_ff_red) ≈ P_SSM
    # => d_ff_diff = ceil(P_SSM / (3 * d_model))
    d_ff_diff = math.ceil(p_ssm / (3 * d_model))
    d_ff_red = max(d_ff_standard - d_ff_diff, 1)

    ffn_reduction_pct = ((d_ff_standard - d_ff_red) / d_ff_standard) * 100.0

    return {
        "d_model": d_model,
        "n_heads": n_heads,
        "d_s": d_s,
        "p_ssm_per_layer": p_ssm,
        "d_ff_standard": d_ff_standard,
        "d_ff_red": d_ff_red,
        "ffn_reduction_pct": ffn_reduction_pct,
    }


def match_sisa_ffn_dim(d_model: int, n_heads: int, d_s: int, d_ff_standard: int) -> int:
    """Convenience function returning the reduced SwiGLU FFN dimension."""
    budget = compute_sisa_parameter_budget(d_model, n_heads, d_s, d_ff_standard)
    return budget["d_ff_red"]
