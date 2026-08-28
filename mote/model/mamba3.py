"""Mamba-3 SISO mixer (Lahoti, Li, Chen, Wang, Bick, Kolter, Dao, Gu 2026).

Parameter names and shapes match ``mamba_ssm.modules.mamba3.Mamba3`` (SISO, no out-proj norm)
exactly, so checkpoints are interchangeable with the upstream module. Three execution paths:

* training / prefill on CUDA with Triton available  -> upstream fused kernel ``mamba3_siso_combined``
* anywhere else (Windows-native, CPU, tests)         -> pure-PyTorch port of upstream's reference, run in
  windows of REF_CHUNK positions with the recurrent state carried between them (2026-08-28: the whole-
  sequence reference is O(L²) in memory — +6.2 GB and 18.6 s for one 4096-byte window of the flagship)
* decode (one byte at a time)                         -> pure-PyTorch recurrent step

Upstream's own ``step`` needs the CuTe ``mamba3_step_fn`` ("only tested on H100"), which is why
the recurrent step is reimplemented here. State tuple = (angle, ssm, k, v), see ``Mamba3State``.
"""

from __future__ import annotations

import math
import os
from typing import NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined  # type: ignore

    HAS_MAMBA3_KERNEL = True
except Exception:  # pragma: no cover
    mamba3_siso_combined = None
    HAS_MAMBA3_KERNEL = False

from .norm import RMSNorm


# The CPU reference path materialises [B,H,L,L] four times over (`_segsum` twice, the masked QK and its
# exp): 16 GiB per tensor at 16384. Windowing it at this many positions makes that L×REF_CHUNK — 256 keeps
# the per-window matmuls large enough to be cheap and the transient under ~50 MB at the flagship's width.
REF_CHUNK = int(os.environ.get("MOTE_MAMBA3_REF_CHUNK", 256))


class Mamba3State(NamedTuple):
    angle: torch.Tensor  # [B, H, num_rope_angles] fp32, cumulative rotation phase mod 2π
    ssm: torch.Tensor  # [B, H, headdim, d_state] fp32
    k: torch.Tensor  # [B, H, d_state]  last rotated B (trapezoid needs the previous input)
    v: torch.Tensor  # [B, H, headdim]  last x


def heavy_tail_activation(x: torch.Tensor) -> torch.Tensor:
    """f(x) = 1 + x (x >= 0), 1 / (1 - x) (x < 0): positive, continuous, differentiable at 0."""
    neg = x.clamp_max(0)
    pos = x.clamp_min(0)
    return pos + torch.reciprocal(1 - neg)


def _rotary_pairs(t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate interleaved (2i, 2i+1) pairs of the last dim; pad identity when fewer angles than pairs."""
    tr = t.view(*t.shape[:-1], -1, 2)
    t0, t1 = tr[..., 0], tr[..., 1]
    if cos.shape[-1] < t0.shape[-1]:
        pad = t0.shape[-1] - cos.shape[-1]
        cos = F.pad(cos, (0, pad), value=1.0)
        sin = F.pad(sin, (0, pad), value=0.0)
    r0 = t0 * cos - t1 * sin
    r1 = t0 * sin + t1 * cos
    return torch.stack([r0, r1], dim=-1).view_as(t)


def _segsum(x: torch.Tensor) -> torch.Tensor:
    """x: [..., T] -> [..., T, T] with entry (t, s) = sum_{s < u <= t} x_u (−inf above the diagonal)."""
    T = x.size(-1)
    x = x.unsqueeze(-1).expand(*x.shape, T)  # [..., T(d), T(e)]
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    xs = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=0)
    return xs.masked_fill(~mask, -torch.inf)


class Mamba3Mixer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        rope_fraction: float = 0.5,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        A_floor: float = 1e-4,
        chunk_size: int = 64,
        layer_idx: Optional[int] = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        fk = {"device": device, "dtype": dtype}
        self.d_model, self.d_state, self.expand, self.headdim = d_model, d_state, expand, headdim
        self.chunk_size, self.layer_idx, self.A_floor = chunk_size, layer_idx, A_floor
        self.mimo_rank = 1
        self.d_inner = int(expand * d_model)
        assert self.d_inner % headdim == 0
        self.nheads = self.d_inner // headdim
        self.num_bc_heads = ngroups
        assert self.nheads % ngroups == 0

        assert rope_fraction in (0.5, 1.0)
        self.rotary_dim_divisor = int(2 / rope_fraction)
        split = int(d_state * rope_fraction)
        if split % 2 != 0:
            split -= 1
        self.num_rope_angles = split // 2
        assert self.num_rope_angles > 0

        # in_proj output order: [z, x, B, C, dd_dt, dd_A, trap, angle]
        d_in_proj = 2 * self.d_inner + 2 * d_state * ngroups + 3 * self.nheads + self.num_rope_angles
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=False, **fk)

        _dt = torch.exp(torch.rand(self.nheads, dtype=torch.float32) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        _dt = torch.clamp(_dt, min=dt_init_floor)
        self.dt_bias = nn.Parameter((_dt + torch.log(-torch.expm1(-_dt))).to(device=device))
        self.dt_bias._no_weight_decay = True

        self.B_bias = nn.Parameter(torch.ones(self.nheads, 1, d_state, dtype=torch.float32, device=device))
        self.C_bias = nn.Parameter(torch.ones(self.nheads, 1, d_state, dtype=torch.float32, device=device))
        self.B_norm = RMSNorm(d_state, eps=1e-5, **fk)
        self.C_norm = RMSNorm(d_state, eps=1e-5, **fk)

        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False, **fk)
        self.telemetry: Optional[dict] = None  # set to a dict by the serving engine to collect live values
        # Decode-graph telemetry: {"retention": [H], "trapezoid": [H]} device buffers the step writes with
        # copy_ instead of .tolist() — nothing may synchronise inside a CUDA-graph capture (serve/graph.py).
        self.telemetry_dev: Optional[dict] = None

    # ------------------------------------------------------------------------------
    def _preprocess(self, u: torch.Tensor):
        """u: [B, L, D] -> z, x [B,L,H,P]; Bn, Cn [B,L,G,N]; ADT, DT, trap [B,H,L]; angles [B,L,H,A] fp32."""
        Bsz, L, _ = u.shape
        proj = self.in_proj(u)
        z, x, Bm, Cm, dd_dt, dd_A, trap, angles = torch.split(
            proj,
            [self.d_inner, self.d_inner, self.d_state * self.num_bc_heads, self.d_state * self.num_bc_heads,
             self.nheads, self.nheads, self.nheads, self.num_rope_angles],
            dim=-1,
        )
        z = z.view(Bsz, L, self.nheads, self.headdim)
        x = x.view(Bsz, L, self.nheads, self.headdim)
        Bm = Bm.view(Bsz, L, self.num_bc_heads, self.d_state)
        Cm = Cm.view(Bsz, L, self.num_bc_heads, self.d_state)
        trap = trap.transpose(1, 2)  # [B, H, L]
        A = -heavy_tail_activation(dd_A.float())
        A = torch.clamp(A, max=-self.A_floor)
        DT = F.softplus(dd_dt.float() + self.dt_bias.float())
        ADT = (A * DT).transpose(1, 2)  # [B, H, L]
        DT = DT.transpose(1, 2)
        angles = angles.unsqueeze(-2).expand(-1, -1, self.nheads, -1).to(torch.float32)
        Bn = self.B_norm(Bm)
        Cn = self.C_norm(Cm)
        return z, x, Bn, Cn, ADT, DT, trap, angles

    def _expand_groups(self, t: torch.Tensor) -> torch.Tensor:
        """[B, L, G, N] -> [B, L, H, N] (repeat each group over its heads)."""
        if t.shape[2] == self.nheads:
            return t
        return t.repeat_interleave(self.nheads // self.num_bc_heads, dim=2)

    # ------------------------------------------------------------------------------
    def forward(self, u: torch.Tensor, return_final_states: bool = False, initial_states: Optional[Mamba3State] = None):
        z, x, Bn, Cn, ADT, DT, trap, angles = self._preprocess(u)
        use_kernel = HAS_MAMBA3_KERNEL and u.is_cuda
        if use_kernel:
            # Input_States order per upstream: (angle [B,H,A], ssm [B,H,P,N], k [B,H,N], v [B,H,P]).
            # The kernel does not cast these, and Triton specialises on the pointer dtypes: feed them
            # in the layout the kernel itself returns (angle/ssm/k fp32, v bf16, contiguous). A state
            # produced by the eager `step` carries v in fp32 — without this cast the first warm
            # continuation after a decode compiled a second variant (measured 3.5 s, 2026-08-24).
            inp = None
            if initial_states is not None:
                inp = (initial_states.angle.float().contiguous(), initial_states.ssm.float().contiguous(),
                       initial_states.k.float().contiguous(), initial_states.v.to(torch.bfloat16).contiguous())
            out = mamba3_siso_combined(
                Q=Cn, K=Bn, V=x, ADT=ADT, DT=DT, Trap=trap,
                Q_bias=self.C_bias.squeeze(1), K_bias=self.B_bias.squeeze(1),
                Angles=angles, D=self.D if torch.is_grad_enabled() else self.D.detach(), Z=z, chunk_size=self.chunk_size,
                Input_States=inp, return_final_states=return_final_states, cu_seqlens=None,
            )
            if return_final_states:
                y, last_angle, last_state, last_k, last_v, *_ = out
                state = Mamba3State(last_angle.float(), last_state.float(), last_k, last_v)
            else:
                y, state = out, None
        else:
            y, state = self._reference_forward_chunked(Cn, Bn, x, ADT, DT, trap, angles, z, initial_states)
        y = y.reshape(u.shape[0], u.shape[1], self.d_inner)
        out = self.out_proj(y.to(u.dtype))
        return (out, state) if return_final_states else out

    # --- pure PyTorch (port of upstream mamba3_siso_fwd_ref, batch mode) ----------------
    def _reference_forward_chunked(self, Q, K, V, ADT, DT, Trap, Angles, Z, initial: Optional[Mamba3State],
                                   chunk: Optional[int] = None):
        """`_reference_forward` in windows of `chunk` positions, the recurrent state carried between them.

        Exact algebra: a window that starts from the previous window's final state is the reference's own
        resume (the trapezoid's previous-input term rides in `state.k/state.v`, the rotation phase in
        `state.angle`), so the L×L work becomes L×chunk with the same result. Floating point is not bitwise —
        each window exponentiates its own partial sums — and the tests pin the difference to fp32 rounding;
        a sequence that fits one window takes exactly the old path. Measured 2026-08-28 before this: the
        flagship read one 4096-byte window on the CPU at +6.2 GB and 18.6 s, the path the trunk serves from.
        """
        L = Q.shape[1]
        C = chunk or REF_CHUNK
        if C <= 0 or L <= C:
            return self._reference_forward(Q, K, V, ADT, DT, Trap, Angles, Z, initial)
        ys, state = [], initial
        for a in range(0, L, C):
            b = min(a + C, L)
            y, state = self._reference_forward(Q[:, a:b], K[:, a:b], V[:, a:b], ADT[:, :, a:b], DT[:, :, a:b],
                                               Trap[:, :, a:b], Angles[:, a:b], Z[:, a:b], state)
            ys.append(y)
        return torch.cat(ys, dim=1), state

    def _reference_forward(self, Q, K, V, ADT, DT, Trap, Angles, Z, initial: Optional[Mamba3State]):
        dtype = torch.float32
        Q = self._expand_groups(Q).to(dtype)
        K = self._expand_groups(K).to(dtype)
        V = V.to(dtype)
        Z = Z.to(dtype)
        ADT = ADT.float()
        DT = DT.float()
        Trap = torch.sigmoid(Trap.float())  # [B,H,L]
        Angles = torch.tanh(Angles) * math.pi  # [B,L,H,A]
        Qb = self.C_bias.squeeze(1).to(dtype)
        Kb = self.B_bias.squeeze(1).to(dtype)
        Bsz, L, H, N = Q.shape
        P = V.shape[-1]
        TWO_PI = 2 * math.pi

        ang = torch.cumsum(Angles * DT.transpose(1, 2).unsqueeze(-1), dim=1)  # [B,L,H,A]
        if initial is not None:
            ang = ang + initial.angle.unsqueeze(1)
        ang = ang - TWO_PI * torch.floor(ang / TWO_PI)
        final_angle = ang[:, -1]

        if initial is not None:
            scalar = DT[:, :, 0] * (1 - Trap[:, :, 0])  # [B,H]
            acc0 = initial.ssm.float() + initial.v.float()[:, :, :, None] * initial.k.float()[:, :, None, :] * scalar[:, :, None, None]
        else:
            acc0 = torch.zeros(Bsz, H, P, N, device=Q.device, dtype=dtype)

        DT_sh = F.pad(DT[:, :, 1:], (0, 1))
        Trap_sh = F.pad(Trap[:, :, 1:], (0, 1))
        shifted_gamma = DT_sh * (1 - Trap_sh)  # [B,H,L]
        scale = DT * Trap + DT_sh * (1 - Trap_sh)  # [B,H,L]

        Q = Q + Qb[None, None]
        K = K + Kb[None, None]
        qk_dot = (K * Q).sum(-1) * shifted_gamma.transpose(1, 2)  # [B,L,H]

        cos, sin = torch.cos(ang), torch.sin(ang)
        Q = _rotary_pairs(Q, cos, sin)
        K = _rotary_pairs(K, cos, sin)
        final_k = K[:, -1]
        final_v = V[:, -1]
        Ks = K * scale.transpose(1, 2).unsqueeze(-1)

        QK = torch.einsum("blhn,bshn->bhls", Q, Ks)
        QK = torch.tril(QK) * torch.exp(_segsum(ADT))  # [B,H,L,L]
        out = torch.einsum("bhls,bshp->blhp", QK, V)
        if initial is not None:
            da_cs = torch.exp(torch.cumsum(ADT, dim=-1))  # [B,H,L]
            out = out + torch.einsum("bhpn,blhn,bhl->blhp", acc0, Q, da_cs)
        out = out + self.D.float()[None, None, :, None] * V
        out = out - V * qk_dot.unsqueeze(-1)
        out = out * Z * torch.sigmoid(Z)

        da_last = torch.exp(ADT.sum(-1))  # [B,H]
        da_rev = torch.exp(ADT.sum(-1, keepdim=True) - torch.cumsum(ADT, dim=-1))  # [B,H,L]
        Vs = V * da_rev.transpose(1, 2).unsqueeze(-1)
        final_ssm = acc0 * da_last[:, :, None, None] + torch.einsum("blhn,blhp->bhpn", Ks, Vs)
        return out, Mamba3State(final_angle, final_ssm, final_k, final_v)

    # --- recurrent decode -------------------------------------------------------------
    def allocate_inference_cache(self, batch_size: int, device, dtype=None) -> Mamba3State:
        return Mamba3State(
            angle=torch.zeros(batch_size, self.nheads, self.num_rope_angles, device=device, dtype=torch.float32),
            ssm=torch.zeros(batch_size, self.nheads, self.headdim, self.d_state, device=device, dtype=torch.float32),
            k=torch.zeros(batch_size, self.nheads, self.d_state, device=device, dtype=torch.float32),
            v=torch.zeros(batch_size, self.nheads, self.headdim, device=device, dtype=torch.float32),
        )

    def step(self, u: torch.Tensor, state: Mamba3State) -> Tuple[torch.Tensor, Mamba3State]:
        """u: [B, 1, D]. Pure-PyTorch port of upstream's mamba3_siso_step_ref for one token."""
        z, x, Bn, Cn, ADT, DT, trap, angles = self._preprocess(u)
        q = (self._expand_groups(Cn)[:, 0].float() + self.C_bias.squeeze(1).float()[None])  # [B,H,N]
        k = (self._expand_groups(Bn)[:, 0].float() + self.B_bias.squeeze(1).float()[None])
        v = x[:, 0].float()  # [B,H,P]
        zz = z[:, 0].float()
        adt, dt, tr = ADT[:, :, 0], DT[:, :, 0], torch.sigmoid(trap[:, :, 0].float())  # [B,H]
        ang = torch.tanh(angles[:, 0]) * math.pi  # [B,H,A]

        TWO_PI = 2 * math.pi
        angle = state.angle + ang * dt.unsqueeze(-1)
        angle = angle - TWO_PI * torch.floor(angle / TWO_PI)
        cos, sin = torch.cos(angle), torch.sin(angle)
        q_rot = _rotary_pairs(q, cos, sin)
        k_rot = _rotary_pairs(k, cos, sin)

        alpha = torch.exp(adt)
        if self.telemetry_dev is not None:
            self.telemetry_dev["retention"].copy_(alpha[0])
            self.telemetry_dev["trapezoid"].copy_(tr[0])
        elif self.telemetry is not None and not (u.is_cuda and torch.cuda.is_current_stream_capturing()):
            # host copies are illegal inside a CUDA-graph capture (the flagship's first capture found this,
            # 2026-08-24: the 35M never captures because of its multi-byte head); the decoder's readings
            # come from telemetry_dev when the graph path owns the mixer
            self.telemetry["retention"] = alpha[0].tolist()  # per-head exp(A·Δt) for this byte
            self.telemetry["trapezoid"] = tr[0].tolist()
        beta = (1 - tr) * dt * alpha
        gamma = tr * dt
        ssm = alpha[:, :, None, None] * state.ssm
        ssm = ssm + beta[:, :, None, None] * (state.v.float()[:, :, :, None] * state.k.float()[:, :, None, :])
        ssm = ssm + gamma[:, :, None, None] * (v[:, :, :, None] * k_rot[:, :, None, :])
        out = torch.einsum("bhpn,bhn->bhp", ssm, q_rot)
        out = out + self.D.float()[None, :, None] * v
        out = out * zz * torch.sigmoid(zz)
        y = out.reshape(u.shape[0], 1, self.d_inner).to(u.dtype)
        return self.out_proj(y), Mamba3State(angle, ssm, k_rot, v)
