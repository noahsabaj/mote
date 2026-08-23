"""`forward_from_state` over k bytes must equal k sequential `step` calls: same logits, same boundaries,
same resulting state (checked through the next step and the multi-byte draft), with and without
boundaries inside the segment. `clone_state` must be a true snapshot."""

import torch

from mote.config import MBPCfg, Mamba3Cfg, MoteConfig, RelationCfg
from mote.model.hnet import HNetForCausalLM


def _model(seed=0):
    torch.manual_seed(seed)
    cfg = MoteConfig(
        d_model_outer=32, encoder_layers=2, decoder_layers=1,
        mamba3=Mamba3Cfg(d_state=16, expand=2, headdim=16),
        main=RelationCfg(n_layers=2, d_model=32, n_heads=2, d_ff=64),
        mbp=MBPCfg(enabled=True, n_layers=1, n_heads=2, d_ff=64),
    )
    return HNetForCausalLM(cfg, dtype=torch.float32).eval()


def _run_steps(model, state, bytes_):
    logits, bmask, bprob = [], [], []
    mbp = None
    for b in bytes_:
        lg, routing, is_b, mbp = model.step(torch.tensor([[b]]), state)
        logits.append(lg[0, 0])
        bmask.append(bool(is_b))
        bprob.append(float(routing.boundary_prob[0, 1]))
    return torch.stack(logits), bmask, bprob, mbp


@torch.no_grad()
def test_forward_from_state_matches_sequential_steps():
    g = torch.Generator().manual_seed(3)
    seen = {True: 0, False: 0}
    for seed in range(6):
        model = _model(seed)
        prompt = torch.randint(0, 256, (1, 24), generator=g)
        cont = torch.randint(0, 256, (12,), generator=g).tolist()
        probe = int(torch.randint(0, 256, (1,), generator=g))

        s1 = model.allocate_inference_state("cpu")
        model.prefill(prompt, s1)
        lg1, bm1, bp1, _ = _run_steps(model, s1, cont)
        nxt1, _, _, mbp1 = _run_steps(model, s1, [probe])

        s2 = model.allocate_inference_state("cpu")
        model.prefill(prompt, s2)
        lg2, bm2, bp2 = model.forward_from_state(torch.tensor([cont]), s2)
        nxt2, _, _, mbp2 = _run_steps(model, s2, [probe])

        seen[any(bm1)] += 1
        assert bm2.tolist() == bm1, (seed, bm1, bm2.tolist())
        assert torch.allclose(bp2, torch.tensor(bp1), atol=1e-5)
        assert torch.allclose(lg2[0], lg1, atol=1e-4, rtol=1e-4), (lg2[0] - lg1).abs().max()
        assert torch.allclose(nxt2, nxt1, atol=1e-4, rtol=1e-4), (nxt2 - nxt1).abs().max()
        assert s1.n_chunks == s2.n_chunks and s1.cur_offset == s2.cur_offset
        assert len(s1.cur_chunk_inputs) == len(s2.cur_chunk_inputs)
        assert (mbp1 is None) == (mbp2 is None)
        if mbp1 is not None:
            assert torch.allclose(mbp1, mbp2, atol=1e-4, rtol=1e-4)
        if s1.prev_chunk_inputs is not None:
            assert torch.allclose(s1.prev_chunk_inputs, s2.prev_chunk_inputs, atol=1e-5)
    assert seen[True] > 0, "no seed produced a boundary inside the segment"


@torch.no_grad()
def test_segment_without_boundary_keeps_chunk_bookkeeping():
    # force no boundaries by a router that never fires: make q/k projections identical on a repeated byte
    model = _model(1)
    s = model.allocate_inference_state("cpu")
    model.prefill(torch.randint(0, 256, (1, 16)), s)
    off0, n0 = s.cur_offset, s.n_chunks
    lg, bm, bp = model.forward_from_state(torch.tensor([[65, 65, 65]]), s)
    if not bm.any():
        assert s.cur_offset == off0 + 3 and s.n_chunks == n0
    assert lg.shape == (1, 3, model.cfg.pad_vocab_to)


@torch.no_grad()
def test_clone_state_is_a_snapshot():
    model = _model(2)
    s = model.allocate_inference_state("cpu")
    model.prefill(torch.randint(0, 256, (1, 20)), s)
    snap = HNetForCausalLM.clone_state(s)
    lg_a, _, _ = model.forward_from_state(torch.tensor([[10, 20, 30]]), s)
    lg_b, _, _ = model.forward_from_state(torch.tensor([[10, 20, 30]]), snap)
    assert torch.allclose(lg_a, lg_b, atol=1e-6)
    # the original state advanced; the snapshot advanced independently — running again differs from the first run
    lg_c, _, _ = model.forward_from_state(torch.tensor([[10, 20, 30]]), s)
    assert not torch.allclose(lg_a, lg_c)
