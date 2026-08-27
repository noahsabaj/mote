"""The fused spine contraction against the materialised function it replaces.

Every claim the kernel makes is checked here rather than asserted: that it is the same function as
`spine_mix_reference` in forward and in all four gradients, that it holds for both spine modes and
for shapes that are not round, and — the one that actually bit — that the reference is only that
function with autocast OFF, because `torch.einsum` is on autocast's lower-precision list.
"""

from __future__ import annotations

import pytest
import torch

from mote.model.spine_kernel import HAS_TRITON, spine_mix, spine_mix_reference

cuda = pytest.mark.skipif(not torch.cuda.is_available() or not HAS_TRITON, reason="needs CUDA + Triton")

# (n_in, n_out, D, has_p, y_per_i) — the four shapes the spine actually launches, plus two edges
CASES = [
    pytest.param(4, 1, 512, False, False, id="expand-read"),
    pytest.param(4, 4, 512, True, False, id="expand-write"),
    pytest.param(4, 4, 128, False, False, id="frac-read"),
    pytest.param(4, 4, 128, True, True, id="frac-write"),
    pytest.param(2, 2, 512, True, False, id="n2"),
    pytest.param(4, 4, 96, True, False, id="D-not-a-power-of-two"),
]


def _inputs(n_in, n_out, D, has_p, y_per_i, device, seed=0, y_dtype=torch.float32):
    torch.manual_seed(seed)
    B, L = 2, 129  # a token count that is not a multiple of anything
    x = torch.randn(B, L, n_in, D, device=device, requires_grad=True)
    h = torch.randn(B, L, n_out, n_in, device=device, requires_grad=True)
    p = torch.randn(B, L, n_out, device=device, requires_grad=True) if has_p else None
    y = None
    if has_p:
        shape = (B, L, n_out, D) if y_per_i else (B, L, D)
        y = torch.randn(*shape, device=device, dtype=y_dtype, requires_grad=True)
    return x, h, p, y


def _close(got, want, name):
    """Tolerance from the tensor's own dtype. A bf16 result is a single rounding of an fp32 value,
    and the kernel's accumulation order differs from the reference's by ~1e-7 — enough to land on
    the other side of a bf16 boundary, so bf16 tensors are allowed two ulp and fp32 tensors are
    held to fp32 rounding."""
    g, w = got.float(), want.float()
    mag = max(w.abs().max().item(), 1e-6)
    ulp = 2.0 ** -8 if want.dtype is torch.bfloat16 else 2.0 ** -23
    tol = 2 * ulp * mag
    err = (g - w).abs().max().item()
    assert err <= tol, f"{name}: {err:.3e} > {tol:.3e} ({want.dtype}, |max| {mag:.3f})"


@cuda
@pytest.mark.parametrize("n_in,n_out,D,has_p,y_per_i", CASES)
def test_fused_matches_the_materialised_function(n_in, n_out, D, has_p, y_per_i):
    dev = "cuda"
    a = _inputs(n_in, n_out, D, has_p, y_per_i, dev)
    b = tuple(None if t is None else t.detach().clone().requires_grad_() for t in a)
    out_k = spine_mix(a[0], a[1], a[2], a[3], n_out=n_out, y_per_i=y_per_i)
    out_r = spine_mix_reference(b[0], b[1], b[2], b[3], y_per_i=y_per_i)
    assert out_k.shape == out_r.shape
    _close(out_k, out_r, "forward")

    g = torch.randn_like(out_r)
    out_k.backward(g)
    out_r.backward(g)
    for name, ta, tb in zip(("x", "h", "p", "y"), a, b):
        if ta is None:
            continue
        assert ta.grad is not None, f"no gradient reached {name}"
        _close(ta.grad, tb.grad, f"{name} gradient")


@cuda
def test_the_reference_is_only_the_fp32_function_with_autocast_off():
    """torch.einsum is autocast-listed. Left under autocast the reference silently computes the
    stream mix in bf16 — which is what the unfused spine did at every one of its seven sites, and
    is a different function from the kernel by three orders of magnitude, not by rounding."""
    x, h, p, y = _inputs(4, 4, 512, True, False, "cuda")
    with torch.no_grad():
        exact = spine_mix_reference(x, h, p, y)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            demoted = spine_mix_reference(x, h, p, y)
            fused = spine_mix(x, h, p, y, n_out=4)
    assert demoted.dtype == torch.float32  # the trailing `+ update` promotes it back
    scale = exact.abs().max().item()
    assert (demoted - exact).abs().max().item() / scale > 1e-3   # the einsum really did demote
    assert (fused - exact).abs().max().item() / scale < 1e-6     # the kernel did not


@cuda
def test_read_hands_the_sublayer_the_autocast_dtype():
    """`u` feeds a sublayer that runs in bf16, and eventually the Relation kernel, whose tl.dot
    rejects an fp32 operand against a bf16 one at compile time. The einsum this replaced cast
    implicitly; the kernel has to do it on purpose."""
    from mote.config import SpineCfg  # noqa: F401
    from mote.model.spine import Spine

    s = Spine(512, 4, site_idx=0, mode="expand").cuda()
    x = torch.randn(2, 16, 4, 512, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        u, _ = s.read(x)
    assert u.dtype == torch.bfloat16
    u_off, _ = s.read(x)
    assert u_off.dtype == torch.float32  # outside autocast it stays in the spine's own precision


@cuda
def test_the_state_it_carries_stays_fp32():
    """X is the residual. It is fp32 by construction and must not follow `y` down to bf16."""
    from mote.model.spine import Spine

    s = Spine(512, 4, site_idx=1, mode="expand").cuda()
    x = torch.randn(2, 16, 4, 512, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        u, carried = s.read(x)
        out = s.write(x, torch.randn(2, 16, 512, device="cuda", dtype=torch.bfloat16), carried)
    assert out.dtype == torch.float32
    assert out.shape == x.shape


@cuda
def test_it_is_deterministic():
    """No atomics: dH and dp close inside the program, so a rerun is bit-identical. A norm guard
    that trips on a rate needs the rate to be a property of the run and not of the scheduler."""
    x, h, p, y = _inputs(4, 4, 512, True, False, "cuda")
    outs = []
    for _ in range(3):
        for t in (x, h, p, y):
            if t.grad is not None:
                t.grad = None
        o = spine_mix(x, h, p, y, n_out=4)
        o.backward(torch.ones_like(o))
        outs.append((o.detach().clone(), x.grad.clone(), h.grad.clone()))
    for o, dx, dh in outs[1:]:
        assert torch.equal(o, outs[0][0])
        assert torch.equal(dx, outs[0][1])
        assert torch.equal(dh, outs[0][2])


def test_the_cpu_path_is_the_reference():
    """No CUDA, no Triton: `spine_mix` must still be the same function, so tests and the CPU
    serving path do not silently exercise different maths."""
    x, h, p, y = _inputs(4, 4, 64, True, False, "cpu")
    got = spine_mix(x, h, p, y, n_out=4)
    want = spine_mix_reference(x, h, p, y)
    assert torch.equal(got, want)


@cuda
@pytest.mark.parametrize("out_dtype", [torch.float32, torch.bfloat16], ids=["out-fp32", "out-bf16"])
@pytest.mark.parametrize("y_per_i", [False, True], ids=["y-shared", "y-per-stream"])
def test_it_reads_and_writes_native_dtypes(out_dtype, y_per_i):
    """The point of the dtype plumbing: `y` arrives bf16 under autocast and `u` leaves bf16, and
    neither is materialised as an fp32 copy on the way. X, H and P stay fp32 — X is the residual.
    Gradients come back in each input's own dtype, which is what autograd requires."""
    D = 128 if y_per_i else 512
    a = _inputs(4, 4, D, True, y_per_i, "cuda", y_dtype=torch.bfloat16)
    b = tuple(None if t is None else t.detach().clone().requires_grad_() for t in a)

    out_k = spine_mix(a[0], a[1], a[2], a[3], n_out=4, y_per_i=y_per_i, out_dtype=out_dtype)
    out_r = spine_mix_reference(b[0], b[1], b[2], b[3], y_per_i=y_per_i, out_dtype=out_dtype)
    assert out_k.dtype is out_dtype
    assert a[3].dtype is torch.bfloat16, "y must not have been promoted"
    _close(out_k, out_r, "forward")

    g = torch.randn_like(out_r)
    out_k.backward(g)
    out_r.backward(g)
    assert a[0].grad.dtype is torch.float32, "dx must stay fp32 — X is the residual"
    assert a[1].grad.dtype is torch.float32 and a[2].grad.dtype is torch.float32
    assert a[3].grad.dtype is torch.bfloat16, "dy must match y's dtype"
    for name, ta, tb in zip(("x", "h", "p", "y"), a, b):
        _close(ta.grad, tb.grad, f"{name} gradient")


@cuda
def test_read_asks_the_kernel_for_its_output_dtype():
    """`read` used to cast `u` down after the fact, which is a full copy of [B,L,d] at every site.
    It now hands the dtype to the kernel, so the conversion happens in registers."""
    from mote.model.spine import Spine

    s = Spine(512, 4, site_idx=0, mode="expand").cuda()
    x = torch.randn(2, 16, 4, 512, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        u, _ = s.read(x)
    assert u.dtype is torch.bfloat16
    assert s.read(x)[0].dtype is torch.float32
