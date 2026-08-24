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
