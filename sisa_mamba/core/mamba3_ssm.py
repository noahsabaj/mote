import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

from .rotary import apply_block_rotations


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = torch.mean(x ** 2, dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight


@dataclass
class Mamba3Config:
    d_model: int = 768
    d_state: int = 64        # SSM state dimension N (must be even for complex RoPE)
    d_head: int = 64         # Head dimension P
    n_heads: int = 12        # Number of heads (d_inner = n_heads * d_head)
    mimo_rank: int = 1       # MIMO rank R (1 for SISO, >=2 for MIMO)
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4
    bias: bool = False
    use_bc_norm: bool = True
    use_bc_bias: bool = True
    trapezoidal: bool = True # Enable exponential-trapezoidal 3-term recurrence
    complex_rope: bool = True# Enable complex-valued state via RoPE trick


class Mamba3SSM(nn.Module):
    """
    Mamba-3 State Space Model layer:
    - Exponential-Trapezoidal Discretization (O(Delta^3) error, implicit width-2 input convolution)
    - Complex-Valued State Transitions with Data-Dependent RoPE Trick
    - BC Normalization and Learnable Biases
    - Multi-Input Multi-Output (MIMO) SSM with Rank-R matrix projections
    """

    def __init__(self, config: Mamba3Config):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.d_state = config.d_state
        self.d_head = config.d_head
        self.n_heads = config.n_heads
        self.d_inner = config.n_heads * config.d_head
        self.mimo_rank = config.mimo_rank
        self.trapezoidal = config.trapezoidal
        self.complex_rope = config.complex_rope

        assert self.d_state % 2 == 0, "d_state must be even for complex RoPE."

        # Input projections: x, z (gate), B, C, dt, theta (phase), lambda (trapezoidal gate)
        # x input projection
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=config.bias) # produces x and z

        # B and C projections: [D -> n_heads * d_state * R] or shared
        self.b_proj = nn.Linear(self.d_model, self.n_heads * self.d_state * self.mimo_rank, bias=config.bias)
        self.c_proj = nn.Linear(self.d_model, self.n_heads * self.d_state * self.mimo_rank, bias=config.bias)

        # BC Normalization
        if config.use_bc_norm:
            self.b_norm = RMSNorm(self.d_state * self.mimo_rank)
            self.c_norm = RMSNorm(self.d_state * self.mimo_rank)
        else:
            self.b_norm = nn.Identity()
            self.c_norm = nn.Identity()

        # Learnable channel biases for B and C (initialized to 1.0 as discovered in Mamba-3 paper)
        if config.use_bc_bias:
            self.b_bias = nn.Parameter(torch.ones(self.n_heads, self.d_state * self.mimo_rank))
            self.c_bias = nn.Parameter(torch.ones(self.n_heads, self.d_state * self.mimo_rank))
        else:
            self.b_bias = None
            self.c_bias = None

        # dt projection (step size Delta_t)
        self.dt_proj = nn.Linear(self.d_model, self.n_heads, bias=True)
        # Initialize dt bias
        dt_init = torch.exp(
            torch.rand(self.n_heads) * (math.log(config.dt_max) - math.log(config.dt_min))
            + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        inv_dt = dt_init + torch.log(-torch.expm1(-dt_init))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # Real decay parameter A (negative scalar per head)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, self.n_heads + 1, dtype=torch.float32)))

        # Complex rotation phase projection theta (frequencies for N/2 2x2 rotation blocks)
        if self.complex_rope:
            self.theta_proj = nn.Linear(self.d_model, self.n_heads * (self.d_state // 2), bias=config.bias)
            nn.init.normal_(self.theta_proj.weight, std=0.02)
        else:
            self.theta_proj = None

        # Trapezoidal gate projection u_t: lambda_t = sigmoid(u_t)
        if self.trapezoidal:
            self.trap_proj = nn.Linear(self.d_model, self.n_heads, bias=config.bias)
        else:
            self.trap_proj = None

        # MIMO rank expansion / reduction weights
        if self.mimo_rank > 1:
            # Per-head learnable scaling vector for MIMO input expansion [d_head, R]
            self.x_mimo_scale = nn.Parameter(torch.randn(self.n_heads, self.d_head, self.mimo_rank) * 0.02)
            # MIMO output down-projection [P * R -> P]
            self.mimo_out_proj = nn.Linear(self.d_head * self.mimo_rank, self.d_head, bias=config.bias)
        else:
            self.x_mimo_scale = None
            self.mimo_out_proj = None

        # Final output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=config.bias)

    def _discretize(self, u: torch.Tensor):
        """
        Computes discrete parameters alpha_t, beta_t, gamma_t, and rotation angles Phi_t.
        u: input [B, L, D]
        """
        B, L, _ = u.shape

        # Delta_t = softplus(dt_proj(u)) [B, L, H] -> [B, H, L]
        dt = F.softplus(self.dt_proj(u)).view(B, L, self.n_heads).transpose(1, 2)
        A = -torch.exp(self.A_log).view(1, self.n_heads, 1)  # [1, H, 1] negative

        # alpha_t = exp(dt * A)
        alpha = torch.exp(dt * A)  # [B, H, L]

        if self.trapezoidal and self.trap_proj is not None:
            # lambda_t = sigmoid(trap_proj(u))
            lam = torch.sigmoid(self.trap_proj(u)).view(B, L, self.n_heads).transpose(1, 2)  # [B, H, L]
            beta = (1.0 - lam) * dt * alpha  # [B, H, L]
            gamma = lam * dt                 # [B, H, L]
        else:
            beta = torch.zeros_like(alpha)
            gamma = dt

        # Complex phase angles
        if self.complex_rope and self.theta_proj is not None:
            # theta_t [B, L, H, N/2] -> [B, H, L, N/2]
            theta = self.theta_proj(u).view(B, L, self.n_heads, self.d_state // 2).transpose(1, 2)
            # Rotation angle step: Delta_t * theta_t
            d_phi = dt.unsqueeze(-1) * theta
            phi = d_phi.cumsum(dim=2)  # Cumulative angle [B, H, L, N/2]
        else:
            phi = None

        return alpha, beta, gamma, phi, dt

    def forward(
        self,
        u: torch.Tensor,
        recurrent_state: Optional[Dict[str, Any]] = None,
        use_state: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Forward pass for Mamba-3.
        u: [batch, seq_len, d_model]
        """
        B, L, D = u.shape
        device = u.device
        dtype = u.dtype

        # 1. Project input to x and z (gate)
        xz = self.in_proj(u)  # [B, L, 2 * d_inner]
        x, z = xz.chunk(2, dim=-1)  # each [B, L, d_inner]
        x = x.view(B, L, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, L, P]
        z = z.view(B, L, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, L, P]

        # 2. Project B and C
        # b_raw, c_raw: [B, L, H, N * R] -> [B, H, L, N * R]
        b_raw = self.b_proj(u).view(B, L, self.n_heads, self.d_state * self.mimo_rank).transpose(1, 2)
        c_raw = self.c_proj(u).view(B, L, self.n_heads, self.d_state * self.mimo_rank).transpose(1, 2)

        # Apply BC norm and biases
        b_normed = self.b_norm(b_raw)
        c_normed = self.c_norm(c_raw)
        if self.b_bias is not None:
            b_normed = b_normed + self.b_bias.view(1, self.n_heads, 1, self.d_state * self.mimo_rank)
            c_normed = c_normed + self.c_bias.view(1, self.n_heads, 1, self.d_state * self.mimo_rank)

        # 3. Discretize
        dt = F.softplus(self.dt_proj(u)).view(B, L, self.n_heads).transpose(1, 2)
        A = -torch.exp(self.A_log).view(1, self.n_heads, 1)
        alpha = torch.exp(dt * A)

        if self.trapezoidal and self.trap_proj is not None:
            lam = torch.sigmoid(self.trap_proj(u)).view(B, L, self.n_heads).transpose(1, 2)
            beta = (1.0 - lam) * dt * alpha
            gamma = lam * dt
        else:
            beta = torch.zeros_like(alpha)
            gamma = dt

        # Complex phase angles
        if self.complex_rope and self.theta_proj is not None:
            theta = self.theta_proj(u).view(B, L, self.n_heads, self.d_state // 2).transpose(1, 2)
            d_phi = dt.unsqueeze(-1) * theta
            if recurrent_state is not None and "phi" in recurrent_state:
                phi = recurrent_state["phi"] + d_phi.cumsum(dim=2)
            else:
                phi = d_phi.cumsum(dim=2)
        else:
            phi = None

        # 4. Complex RoPE trick: Rotate B and C vectors
        if self.complex_rope and phi is not None:
            if self.mimo_rank == 1:
                b_rot = apply_block_rotations(b_normed, phi, transpose=True)
                c_rot = apply_block_rotations(c_normed, phi, transpose=True)
            else:
                b_reshaped = b_normed.view(B, self.n_heads, L, self.mimo_rank, self.d_state)
                c_reshaped = c_normed.view(B, self.n_heads, L, self.mimo_rank, self.d_state)
                phi_expanded = phi.unsqueeze(3).expand(-1, -1, -1, self.mimo_rank, -1)
                b_rot = apply_block_rotations(b_reshaped, phi_expanded, transpose=True).view_as(b_normed)
                c_rot = apply_block_rotations(c_reshaped, phi_expanded, transpose=True).view_as(c_normed)
        else:
            b_rot = b_normed
            c_rot = c_normed

        # 5. Execute Recurrence / State Update
        if recurrent_state is not None:
            prev_h = recurrent_state.get("h", torch.zeros(B, self.n_heads, self.d_state, self.d_head, device=device, dtype=torch.float32))
            prev_bx = recurrent_state.get("bx", torch.zeros(B, self.n_heads, self.d_state, self.d_head, device=device, dtype=torch.float32))

            if self.mimo_rank == 1:
                bx_curr = torch.einsum("bhl n, bhl p -> bhl np", b_rot, x).view(B, self.n_heads, L, self.d_state, self.d_head)
            else:
                x_mimo = x.unsqueeze(-1) * self.x_mimo_scale.unsqueeze(0).unsqueeze(2)
                b_mimo = b_rot.view(B, self.n_heads, L, self.d_state, self.mimo_rank)
                bx_curr = torch.einsum("bhlnr, bhlpr -> bhlnp", b_mimo, x_mimo)

            outputs = []
            curr_h = prev_h
            for t in range(L):
                a_t = alpha[:, :, t].unsqueeze(-1).unsqueeze(-1).to(torch.float32)
                b_t = beta[:, :, t].unsqueeze(-1).unsqueeze(-1).to(torch.float32)
                g_t = gamma[:, :, t].unsqueeze(-1).unsqueeze(-1).to(torch.float32)

                bx_t = bx_curr[:, :, t].to(torch.float32)
                curr_h = a_t * curr_h + b_t * prev_bx + g_t * bx_t
                prev_bx = bx_t

                if self.mimo_rank == 1:
                    c_t = c_rot[:, :, t].to(torch.float32)
                    y_t = torch.einsum("bhn, bhnp -> bhp", c_t, curr_h)
                else:
                    c_mimo_t = c_rot[:, :, t].view(B, self.n_heads, self.d_state, self.mimo_rank).to(torch.float32)
                    y_mimo_t = torch.einsum("bhnr, bhnp -> bhpr", c_mimo_t, curr_h)
                    y_t = self.mimo_out_proj(y_mimo_t.reshape(B, self.n_heads, self.d_head * self.mimo_rank))

                outputs.append(y_t.unsqueeze(2))

            y = torch.cat(outputs, dim=2).to(dtype)
            new_state = {"h": curr_h, "bx": prev_bx, "phi": phi[:, :, -1:, :] if phi is not None else None}

        else:
            if self.mimo_rank == 1:
                bx = torch.einsum("bhl n, bhl p -> bhl np", b_rot, x).view(B, self.n_heads, L, self.d_state, self.d_head)
            else:
                x_mimo = x.unsqueeze(-1) * self.x_mimo_scale.unsqueeze(0).unsqueeze(2)
                b_mimo = b_rot.view(B, self.n_heads, L, self.d_state, self.mimo_rank)
                bx = torch.einsum("bhlnr, bhlpr -> bhlnp", b_mimo, x_mimo)

            h = torch.zeros(B, self.n_heads, self.d_state, self.d_head, device=device, dtype=torch.float32)
            prev_bx = torch.zeros_like(h)
            outputs = []

            for t in range(L):
                a_t = alpha[:, :, t].unsqueeze(-1).unsqueeze(-1).to(torch.float32)
                b_t = beta[:, :, t].unsqueeze(-1).unsqueeze(-1).to(torch.float32)
                g_t = gamma[:, :, t].unsqueeze(-1).unsqueeze(-1).to(torch.float32)

                bx_t = bx[:, :, t].to(torch.float32)
                h = a_t * h + b_t * prev_bx + g_t * bx_t
                prev_bx = bx_t

                if self.mimo_rank == 1:
                    c_t = c_rot[:, :, t].to(torch.float32)
                    y_t = torch.einsum("bhn, bhnp -> bhp", c_t, h)
                else:
                    c_mimo_t = c_rot[:, :, t].view(B, self.n_heads, self.d_state, self.mimo_rank).to(torch.float32)
                    y_mimo_t = torch.einsum("bhnr, bhnp -> bhpr", c_mimo_t, h)
                    y_t = self.mimo_out_proj(y_mimo_t.reshape(B, self.n_heads, self.d_head * self.mimo_rank))

                outputs.append(y_t.unsqueeze(2))

            y = torch.cat(outputs, dim=2).to(dtype)
            new_state = {"h": h, "bx": prev_bx, "phi": phi[:, :, -1:, :] if phi is not None else None} if use_state else None

        # 6. Apply Swish/SiLU Gating with z
        y = y * F.silu(z)  # [B, H, L, P]

        # 7. Merge heads and down-project to d_model
        y = y.transpose(1, 2).contiguous().view(B, L, self.d_inner)
        out = self.out_proj(y)

        return out, new_state
