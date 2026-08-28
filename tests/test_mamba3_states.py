"""Input_States wiring (docs/shape.md, serving prefix cache): resuming the Triton kernel from a
cached state must equal running the whole sequence in one call. The old 0.03-0.15 logit drift came
from falling back to the fp32 reference for resumed prefills; in-kernel resume agrees to ~6e-3 at
the mixer output, with state components inside bf16 accumulation noise."""

import pytest
import torch

from mote.model.mamba3 import HAS_MAMBA3_KERNEL, Mamba3Mixer

needs_gpu = pytest.mark.skipif(
    not (torch.cuda.is_available() and HAS_MAMBA3_KERNEL), reason="needs CUDA + the Mamba-3 kernel"
)


def _mixer(seed=0):
    torch.manual_seed(seed)
    return Mamba3Mixer(d_model=64, d_state=16, headdim=16, expand=2, layer_idx=0, device="cuda").eval()


@needs_gpu
@torch.no_grad()
def test_kernel_resume_equals_one_call():
    m = _mixer()
    torch.manual_seed(1)
    u = torch.randn(2, 128, 64, device="cuda")
    y_all, state_all = m(u, return_final_states=True)
    for cut in (48, 64, 96):  # off-chunk and on-chunk splits (chunk_size 64)
        y1, s1 = m(u[:, :cut], return_final_states=True)
        y2, s2 = m(u[:, cut:], return_final_states=True, initial_states=s1)
        y_cat = torch.cat([y1, y2], dim=1)
        assert (y_cat.float() - y_all.float()).abs().max() < 2e-2, cut
        assert (s2.ssm - state_all.ssm).abs().max() < 5e-2, cut
        assert (torch.cos(s2.angle) - torch.cos(state_all.angle)).abs().max() < 5e-2, cut
        assert torch.equal(s2.v, state_all.v), cut  # last-position v is copied, not accumulated


@needs_gpu
@torch.no_grad()
def test_kernel_resume_matches_reference_resume():
    m = _mixer()
    torch.manual_seed(2)
    u = torch.randn(1, 96, 64, device="cuda")
    _, s1 = m(u[:, :40], return_final_states=True)
    y_kernel = m(u[:, 40:], initial_states=s1)

    z, x, Bn, Cn, ADT, DT, trap, angles = m._preprocess(u[:, 40:])
    y_ref, _ = m._reference_forward(Cn, Bn, x, ADT, DT, trap, angles, z, s1)
    y_ref = m.out_proj(y_ref.reshape(1, 56, -1).to(u.dtype))
    assert (y_kernel.float() - y_ref.float()).abs().max() < 5e-2


# ---- the windowed CPU reference (finding 8 of the serving audit, built 2026-08-28) ----------------
def _cpu_mixer(seed=0):
    torch.manual_seed(seed)
    return Mamba3Mixer(d_model=64, d_state=16, headdim=16, expand=2, layer_idx=0, device="cpu").eval()


def _parts(m, u):
    z, x, Bn, Cn, ADT, DT, trap, angles = m._preprocess(u)
    return Cn, Bn, x, ADT, DT, trap, angles, z


def _slice(parts, a, b):
    Cn, Bn, x, ADT, DT, trap, angles, z = parts
    return Cn[:, a:b], Bn[:, a:b], x[:, a:b], ADT[:, :, a:b], DT[:, :, a:b], trap[:, :, a:b], angles[:, a:b], z[:, a:b]


@torch.no_grad()
def test_windowed_reference_matches_the_whole_sequence_reference():
    """Windows on and off the sequence's own boundaries, with and without an initial state: the same numbers
    to fp32 rounding. What used to be four [B,H,L,L] tensors is now four [B,H,chunk,chunk]."""
    m = _cpu_mixer()
    torch.manual_seed(3)
    parts = _parts(m, torch.randn(2, 300, 64))
    y_ref, s_ref = m._reference_forward(*parts, None)
    for C in (64, 100, 256):
        y_c, s_c = m._reference_forward_chunked(*parts, None, chunk=C)
        assert (y_c - y_ref).abs().max() < 1e-4, C
        assert (s_c.ssm - s_ref.ssm).abs().max() < 1e-4, C
        assert (torch.cos(s_c.angle) - torch.cos(s_ref.angle)).abs().max() < 1e-5, C
        assert torch.equal(s_c.v, s_ref.v), C  # the last x is copied, not accumulated
        assert (s_c.k - s_ref.k).abs().max() < 1e-5, C  # the last B, rotated by the carried phase: rounding only
    _, s1 = m._reference_forward(*_slice(parts, 0, 40), None)
    y_ref2, _ = m._reference_forward(*_slice(parts, 40, 300), s1)
    y_c2, _ = m._reference_forward_chunked(*_slice(parts, 40, 300), s1, chunk=64)
    assert (y_c2 - y_ref2).abs().max() < 1e-4


@torch.no_grad()
def test_a_sequence_that_fits_one_window_takes_the_old_path_bit_for_bit(monkeypatch):
    import mote.model.mamba3 as M

    m = _cpu_mixer()
    torch.manual_seed(4)
    u = torch.randn(1, 200, 64)
    y_ref, _ = m._reference_forward(*_parts(m, u), None)
    y_ref = m.out_proj(y_ref.reshape(1, 200, -1).to(u.dtype))
    monkeypatch.setattr(M, "REF_CHUNK", 10_000)
    assert torch.equal(m(u), y_ref)  # no split: exactly what the model computed before today
    monkeypatch.setattr(M, "REF_CHUNK", 64)
    assert (m(u) - y_ref).abs().max() < 1e-4  # split: fp32 rounding, nothing else


def test_the_windowed_path_trains_to_the_same_gradients(monkeypatch):
    """Backward runs through the window loop (CPU trainers in tests, boxes without the kernel)."""
    import mote.model.mamba3 as M

    m = _cpu_mixer().train()
    torch.manual_seed(5)
    u = torch.randn(1, 130, 64)
    grads = {}
    for C in (10_000, 32):
        monkeypatch.setattr(M, "REF_CHUNK", C)
        m.zero_grad()
        m(u).square().mean().backward()
        grads[C] = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    assert grads[32].keys() == grads[10_000].keys() and grads[32]
    for n in grads[32]:
        assert torch.isfinite(grads[32][n]).all(), n
        assert (grads[32][n] - grads[10_000][n]).abs().max() < 1e-4 * (1 + grads[10_000][n].abs().max()), n
