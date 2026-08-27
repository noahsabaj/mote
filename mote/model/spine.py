"""The hyper-connection spine: one multi-stream residual at byte resolution (signed 2026-08-26).

An H-Net has no single long residual stream — the chunker resets it three times — so the spine runs
where the bytes are: ``embed → encoder(3) → ⟦the whole chunk stage⟧ → decoder(3) → head``, seven
sites over one stream, with the main network keeping its own plain residual inside (different width,
different resolution). Each site is a hyper-connection (Zhu et al. 2409.19606):

    u  = RMSNorm(H_pre · X)          read  n streams down to one d-dim layer input
    y  = F(u)                        the site's own function (a Mamba-3 block, or the chunk stage)
    X  = H_res · X + H_post^T · y    mix the streams, write the update back

**Why not the Birkhoff polytope.** mHC (2512.24880) constrains H_res to be doubly stochastic. That
is two constraints doing two jobs: an affine part (rows and columns sum to 1) that conserves the
mean across streams, and non-negativity, which bounds ‖H_res‖₂ ≤ 1. Only the first is load-bearing
for the identity mapping; non-negativity is a *sufficient but unnecessary* way to get the norm
bound, and it is the part that costs you. Three groups measured the damage independently: residual
mixing decays to near-identity and one stream comes to dominate (2606.03483), a data-dependent
diagonal beats Sinkhorn outright (2604.21254 Tab. 6), and under bistochastic skips >75 % of the
Jacobian's spectral mass sits at zero — "spectral stalling", three quarters of the directions get no
gradient (2602.18308 §7.1).

So the spine constrains the property it actually needs. Following sHC (2603.20896 §5.1), write

    H_res = J + H_disp,   J = (1/n)·11ᵀ,   H_disp ∈ Z_n = {H : H·1 = 0, 1ᵀ·H = 0}

J annihilates the component orthogonal to 1 and H_disp annihilates the component along it, so they
act on orthogonal subspaces and ‖J + H_disp‖₂ = max(1, ‖H_disp‖₂). Bounding ‖H_disp‖₂ ≤ 1 therefore
gives mean conservation *and* the spectral bound, with negative entries allowed. The Birkhoff
polytope is strictly contained in this set, so nothing is given up.

`project` selects the manifold; every variant in the literature is a control arm:
    spectral_sphere  sHC — H_disp = P·(U Σ Vᵀ)·Pᵀ, U/V Cayley-orthogonal, |Σ_ii| ≤ 1   (default)
    orthogonal       JPmHC — as above with Σ = I; norm-preserving, cannot attenuate a stream
    sinkhorn         mHC — 20 Sinkhorn-Knopp iterations onto the Birkhoff polytope
    perm_convex      mHC-lite — a softmax over the n! permutation matrices; exactly doubly stochastic
    diag             Hyperloop — diagonal sigmoid; no cross-stream mixing, and NOT mean-conserving
    none             unconstrained HC

`mode` selects how the streams are made:
    expand  n copies of the byte state           (memory ×n, the whole literature's form)
    frac    the byte state split into n slices   (Frac-Connections 2503.14125; memory unchanged)
"""

from __future__ import annotations

import itertools
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import RMSNorm

PRE_BIAS = 2.0    # the read's rotating one-hot: site k prefers stream k mod n by e^4 ≈ 7.4 : 1
POST_BIAS = 0.0   # 2·σ(0) = 1 exactly, so at init every stream receives the update with weight 1 (HC's B = 1)
# sHC uses 4.0, but tanh saturates: sech²(4) = 0.0013 damps every gradient into the singular values
# by 750x, so Σ would sit pinned at 1 and H_res = J + (I−J) = I — reproducing by construction the
# identity degeneration sHC exists to prevent. 3.0 costs 0.5% per site off identity (3.5% across the
# seven, which the streams' own norm absorbs) and damps 7.4x less. Depth is why the paper's constant
# does not transfer: it is spent per site, and this spine has seven, not twenty-four.
S_BIAS = 3.0
# `frac` can afford more saturation and needs it: its slices differ, so (I−J)X is non-zero and the
# per-site 0.5% attenuation compounds to 3.0% across the seven sites, where `expand`'s identical
# streams pass a mean-conserving mixer with no loss at all. Measured the other way too — frac's b_res
# gradients run 2.2e-2 against expand's 1e-3, an order of magnitude of headroom to spend. 4.0 puts
# frac's init error at 0.5%.
S_BIAS_FRAC = 4.0
LSS_STD = 0.02    # Learned Stream Scaling (2606.03483): streams start close but never identical
SK_ITERS = 20     # mHC's choice, kept for the `sinkhorn` control arm


def helmert(n: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """[n, n-1] orthonormal basis of 1ᵀx = 0. Closed form, so it is bit-identical on every device.

    P Pᵀ = I − J, which is what makes `H_disp = P M Pᵀ` land in Z_n and makes M = I give H_res = I."""
    p = torch.zeros(n, n - 1, device=device, dtype=dtype)
    for i in range(1, n):
        s = 1.0 / math.sqrt(i * (i + 1))
        p[:i, i - 1] = s
        p[i, i - 1] = -i * s
    return p


def _skew(v: torch.Tensor, m: int) -> torch.Tensor:
    """[..., m(m-1)/2] → [..., m, m] skew-symmetric (strict upper triangle, antisymmetric below)."""
    a = v.new_zeros(*v.shape[:-1], m, m)
    if m > 1:
        iu = torch.triu_indices(m, m, offset=1, device=v.device)
        a[..., iu[0], iu[1]] = v
        a = a - a.transpose(-1, -2)
    return a


def _cayley(a: torch.Tensor) -> torch.Tensor:
    """(I − A)(I + A)⁻¹ for skew-symmetric A — orthogonal by construction, and at A = 0 it is I.

    At n = 4 this inverts a 3×3, which is cheap enough to sit inside a fused kernel later."""
    m = a.shape[-1]
    eye = torch.eye(m, device=a.device, dtype=a.dtype).expand_as(a)
    return torch.linalg.solve((eye + a).transpose(-1, -2), (eye - a).transpose(-1, -2)).transpose(-1, -2)


def _sinkhorn(raw: torch.Tensor, iters: int = SK_ITERS) -> torch.Tensor:
    """mHC's projection: exponentiate, then alternate row and column normalisation."""
    m = torch.exp(raw - raw.amax(dim=(-2, -1), keepdim=True))
    for _ in range(iters):
        m = m / m.sum(dim=-2, keepdim=True).clamp_min(1e-12)
        m = m / m.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return m


def n_cols(project: str, n: int) -> int:
    """How many numbers the generator must emit per token to build one H_res."""
    k = max((n - 1) * (n - 2) // 2, 0)
    return {
        "spectral_sphere": 2 * k + (n - 1),
        "orthogonal": k,
        "sinkhorn": n * n,
        "perm_convex": math.factorial(n),
        "diag": n,
        "none": n * n,
    }[project]


class ResMixer(nn.Module):
    """Builds H_res ∈ [..., n, n] from `n_cols(project, n)` raw numbers per token."""

    def __init__(self, n: int, project: str = "spectral_sphere", device=None):
        super().__init__()
        self.n, self.project = n, project
        self.k = max((n - 1) * (n - 2) // 2, 0)
        if project in ("spectral_sphere", "orthogonal"):
            self.register_buffer("basis", helmert(n, device=device), persistent=False)
        if project == "perm_convex":
            perms = torch.stack([torch.eye(n)[list(p)] for p in itertools.permutations(range(n))])
            self.register_buffer("perms", perms.to(device=device), persistent=False)
            # index of the identity permutation, so the bias can be initialised to concentrate on it
            self.identity_perm = next(i for i, p in enumerate(itertools.permutations(range(n))) if list(p) == list(range(n)))

    def bias_init(self, s_bias: float = S_BIAS) -> torch.Tensor:
        """The static bias that makes H_res ≈ I before anything is learned."""
        n, cols = self.n, n_cols(self.project, self.n)
        b = torch.zeros(cols)
        if self.project == "spectral_sphere":
            b[2 * self.k:] = s_bias                       # U = V = I (zero skew), Σ = tanh(4) ≈ I
        elif self.project == "sinkhorn":
            b = torch.full((cols,), -8.0)
            b.view(n, n).fill_diagonal_(0.0)              # exp() then concentrates on the diagonal
        elif self.project == "perm_convex":
            b = torch.full((cols,), -8.0)
            b[self.identity_perm] = 0.0                   # softmax concentrates on the identity
        elif self.project == "diag":
            b = torch.full((cols,), 6.0)                  # σ(6) = 0.9975; a sigmoid cannot reach 1, and `diag` is a control arm
        elif self.project == "none":
            b = torch.eye(n).flatten().clone()
        return b

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        n = self.n
        if self.project in ("spectral_sphere", "orthogonal"):
            m = n - 1
            u = _cayley(_skew(raw[..., : self.k], m))
            if self.project == "orthogonal":
                core = u
            else:
                v = _cayley(_skew(raw[..., self.k : 2 * self.k], m))
                sigma = torch.tanh(raw[..., 2 * self.k :])          # |Σ_ii| ≤ 1  ⇒  ‖H_disp‖₂ ≤ 1
                core = u * sigma.unsqueeze(-2) @ v.transpose(-1, -2)
            p = self.basis.to(raw.dtype)
            disp = p @ core @ p.transpose(-1, -2)
            return disp + (1.0 / n)                                 # J = (1/n)·11ᵀ is a constant add
        raw = raw.unflatten(-1, (n, n)) if self.project in ("sinkhorn", "none") else raw
        if self.project == "sinkhorn":
            return _sinkhorn(raw)
        if self.project == "perm_convex":
            return torch.tensordot(raw.softmax(-1), self.perms.to(raw.dtype), dims=([-1], [0]))
        if self.project == "diag":
            return torch.diag_embed(torch.sigmoid(raw))
        return raw                                                   # "none": unconstrained HC


class Spine(nn.Module):
    """One site of the spine: read the streams, hand `u` to the caller's F, write the update back.

    Split into `read` / `write` because F is anything from a Mamba-3 block to the entire chunk stage.
    The generator is a single zero-initialised projection emitting every coefficient at once — the
    three mappings are separate output columns, so one matmul is exactly equivalent to mHC's three
    and costs one kernel launch instead of three.
    """

    def __init__(self, d_model: int, n: int, site_idx: int, mode: str = "expand",
                 project: str = "spectral_sphere", dynamic: bool = True, post_scale: float = 2.0,
                 eps: float = 1e-5, device=None):
        super().__init__()
        if mode not in ("expand", "frac"):
            raise ValueError(f"mode must be expand|frac, got {mode!r}")
        if mode == "frac" and d_model % n:
            raise ValueError(f"frac needs n | d_model; {d_model} % {n} = {d_model % n}")
        self.n, self.mode, self.dynamic, self.post_scale = n, mode, dynamic, post_scale
        self.site_idx = site_idx  # global across the spine: the rotating read is the symmetry break
        self.d_stream = d_model if mode == "expand" else d_model // n
        flat = n * self.d_stream  # n·d for expand, d for frac — frac's generator is n× cheaper to read

        # `frac`'s read is an m x m mix of slices. Unconstrained, and initialised to exactly I:
        # it is not carried across sites, so it needs no norm bound, and a manifold that only
        # approximates the identity would attenuate the residual before training starts.
        self.pre_mixer = ResMixer(n, "none", device=device) if mode == "frac" else None
        self.res_mixer = ResMixer(n, project, device=device)
        c_pre = n * n if mode == "frac" else n
        c_post, c_res = n, n_cols(project, n)
        self.splits = (c_pre, c_post, c_res)

        self.gen_norm = RMSNorm(flat, eps=eps, device=device, dtype=torch.float32)
        if dynamic:
            self.phi = nn.Linear(flat, c_pre + c_post + c_res, bias=False, device=device, dtype=torch.float32)
            nn.init.zeros_(self.phi.weight)          # a static hyper-connection until training moves it
            self.phi.weight._no_reinit = True        # _init_weights must not overwrite the zero
            self.phi.weight._no_muon = True          # a coefficient generator, not a hidden matrix
        else:
            self.phi = None
        self.alpha = nn.Parameter(torch.full((3,), 0.01, device=device))  # mHC's small gating factors

        # Static biases. `_no_weight_decay` because HC keeps the static component out of decay, and
        # `_no_muon` because Newton-Schulz orthogonalising a doubly-stochastic bias fights the manifold.
        b_pre = self.pre_mixer.bias_init() if mode == "frac" else torch.full((n,), -PRE_BIAS)
        if mode == "expand":
            b_pre[site_idx % n] = PRE_BIAS           # rotating read: HC's e_{k mod n}, the one thing
        self.b_pre = nn.Parameter(b_pre.to(device))  # in HC's init that breaks stream symmetry
        self.b_post = nn.Parameter(torch.full((n,), POST_BIAS, device=device))
        self.b_res = nn.Parameter(self.res_mixer.bias_init(S_BIAS if mode == "expand" else S_BIAS_FRAC).to(device))
        for p in (self.b_pre, self.b_post, self.b_res, self.alpha):
            p._no_weight_decay = True
            p._no_muon = True

    def coefficients(self, x: torch.Tensor):
        """x: [..., n, d_stream] → (H_pre, H_post, H_res), all in fp32."""
        xf = x.float()
        flat = self.gen_norm(xf.flatten(-2))
        if self.phi is not None:
            g_pre, g_post, g_res = self.phi(flat).split(self.splits, dim=-1)
        else:
            z = flat.new_zeros(*flat.shape[:-1], sum(self.splits))
            g_pre, g_post, g_res = z.split(self.splits, dim=-1)
        a = self.alpha.float()
        h_post = self.post_scale * torch.sigmoid(a[1] * g_post + self.b_post)
        h_res = self.res_mixer(a[2] * g_res + self.b_res)
        if self.mode == "frac":
            h_pre = self.pre_mixer(a[0] * g_pre + self.b_pre)
        else:
            h_pre = torch.sigmoid(a[0] * g_pre + self.b_pre)
        return h_pre, h_post, h_res

    def read(self, x: torch.Tensor):
        """[..., n, d_stream] → (u [..., d_model], carried coefficients for `write`)."""
        h_pre, h_post, h_res = self.coefficients(x)
        if self.mode == "expand":
            u = torch.einsum("...nd,...n->...d", x.float(), h_pre)
        else:
            u = torch.einsum("...nm,...md->...nd", h_pre, x.float()).flatten(-2)
        return u, (h_post, h_res)

    def write(self, x: torch.Tensor, y: torch.Tensor, carried) -> torch.Tensor:
        """X ← H_res·X + H_postᵀ·y. `y` is the site's output at d_model; X stays fp32."""
        h_post, h_res = carried
        yf = y.float()
        if self.mode == "frac":
            yf = yf.unflatten(-1, (self.n, self.d_stream))
            update = h_post.unsqueeze(-1) * yf
        else:
            update = h_post.unsqueeze(-1) * yf.unsqueeze(-2)
        return torch.einsum("...nm,...md->...nd", h_res, x.float()) + update


class StreamExpand(nn.Module):
    """Make the streams, and fold them back. Owns the Learned Stream Scaling that breaks symmetry.

    HC replicates the state exactly, which leaves every stream in the fixed subspace of the stream-
    permutation symmetry — training then has to invent the asymmetry, and what it invents is one
    dominant stream (2606.03483). LSS gives each stream a near-identity diagonal scale instead: n·d
    parameters, no matmul, and the streams start distinguishable.

    Streams are read out by a learned convex combination rather than summed. Summing needs HC's √n
    correction on exactly the modules `_init_weights` already scales by 1/√n_res, and stacking two
    corrections on one line is how that silently goes wrong; softmax weights sum to 1, so the scale
    is preserved for free.

    The read-out must be NON-uniform, and that is not a nicety. A plain mean cannot see a
    mean-conserving mixer: dL/dH_res comes out constant down the destination axis, H_res's columns
    sum to 1 by construction, and the two contract to exactly zero. Measured on a seven-site stack,
    a mean read-out leaves the final site's H_res at a gradient of 2e-8 while the six upstream sites
    sit at 1e-3..6e-3 — the last mixer is dead, because no non-uniform reader follows it. A learned
    convex read is the cheapest thing that gives it one. (`frac` needs no such fix: its read-out is
    a concatenation, which reads every slice distinctly.)
    """

    def __init__(self, d_model: int, n: int, mode: str = "expand", lss: bool = True, device=None):
        super().__init__()
        self.n, self.mode = n, mode
        self.d_stream = d_model if mode == "expand" else d_model // n
        s = torch.ones(n, self.d_stream, device=device)
        if lss:
            s = s + LSS_STD * torch.randn_like(s)
        self.scale = nn.Parameter(s)
        self.read_out = nn.Parameter(LSS_STD * torch.randn(n, device=device)) if mode == "expand" else None
        for p in (self.scale, self.read_out):
            if p is not None:
                p._no_weight_decay = True
                p._no_muon = True

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        x = h.float().unsqueeze(-2) if self.mode == "expand" else h.float().unflatten(-1, (self.n, self.d_stream))
        return x * self.scale

    def collapse(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "frac":
            return x.flatten(-2)
        return torch.einsum("...nd,n->...d", x.float(), self.read_out.float().softmax(-1))
