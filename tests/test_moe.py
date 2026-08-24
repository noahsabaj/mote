"""MoE FFN (mote/model/moe.py, signed 2026-08-24): the dense and loop paths agree in outputs and gradients
under both routers, decode (one chunk) equals the batched path, the loss-free bias moves towards balance,
Muon treats an expert stack as its slices, the trainer runs it end to end with telemetry, checkpoints and
configs carry the MoE fields, the engine serves it, and the dense path compiles fullgraph (the decode graph).
The grouped bf16 CUDA path is checked against the loop path when a GPU is present (T1 on the 4060 Ti)."""

import json
import shutil
import threading

import numpy as np
import pytest
import torch

from mote.config import MBPCfg, Mamba3Cfg, MoteConfig, RelationCfg
from mote.model.hnet import HNetForCausalLM
from mote.model.moe import MoESwiGLU, collect_moe, gate_scale
from mote.model.relation import SwiGLU
from mote.train.muon import Muon, split_muon_params

D, F_, E, K = 32, 16, 4, 2


def _mod(router: str, E_: int = E, k: int = K, seed: int = 0) -> MoESwiGLU:
    torch.manual_seed(seed)
    return MoESwiGLU(D, F_, E_, top_k=k, router=router)


def _run(m: MoESwiGLU, x: torch.Tensor, mask, path: str):
    m.zero_grad(set_to_none=True)
    m.dense_threshold = 10 ** 9 if path == "dense" else -1
    xin = x.clone().requires_grad_(True)
    y = m(xin, mask)
    aux, stats = collect_moe(m)
    (y.square().mean() + aux).backward()
    return y.detach(), xin.grad.clone(), {n: p.grad.clone() for n, p in m.named_parameters()}, stats


@pytest.mark.parametrize("router", ["lossfree", "aux"])
def test_dense_and_loop_paths_agree(router):
    m = _mod(router)
    with torch.no_grad():
        m.expert_bias.uniform_(-0.2, 0.2)  # the bias takes part in selection (ignored by the aux router)
    x = torch.randn(3, 40, D)
    mask = torch.ones(3, 40, dtype=torch.bool)
    mask[1, 30:] = False  # padded chunk rows
    a = _run(m, x, mask, "dense")
    b = _run(m, x, mask, "loop")
    assert torch.allclose(a[0], b[0], atol=1e-5, rtol=1e-5), (a[0] - b[0]).abs().max()
    assert torch.allclose(a[1], b[1], atol=1e-5, rtol=1e-5)
    for n in a[2]:
        assert torch.allclose(a[2][n], b[2][n], atol=1e-5, rtol=1e-4), n
    assert m.router.weight.grad.abs().sum() > 0  # the router learns through the gate weights
    load = m.stats["load"]
    assert abs(float(load.sum()) - 1.0) < 1e-5 and float(m.stats["maxvio"]) >= 0
    assert 0 < float(a[3]["moe_topk_mass"]) <= 1.0
    assert torch.isfinite(a[3]["moe_aux"])


def test_padded_rows_leave_the_statistics():
    m = _mod("lossfree", k=1)
    x = torch.randn(1, 20, D)
    x[0, 10:] = 0.0  # pads route on the bias alone; they must not count
    mask = torch.zeros(1, 20, dtype=torch.bool)
    mask[0, :10] = True
    m(x, mask)
    with_mask = m.stats["load"].clone()
    m(x[:, :10])
    assert torch.allclose(with_mask, m.stats["load"], atol=1e-6)


def test_decode_row_equals_batched():
    m = _mod("lossfree").eval()
    x = torch.randn(2, 20, D)
    m.dense_threshold = -1
    full = m(x)  # loop path over 40 rows
    m.dense_threshold = 16
    for b, t in ((0, 3), (1, 19)):
        one = m(x[b:b + 1, t:t + 1])  # [1,1,D]: the shape Block.step and the decode graph feed
        assert torch.allclose(one[0, 0], full[b, t], atol=1e-5), (b, t)


def test_bias_update_moves_towards_balance():
    m = _mod("lossfree", k=1).train()
    m._load_acc.copy_(torch.tensor([10.0, 2.0, 2.0, 2.0]))
    m.update_bias()
    b = m.expert_bias
    assert b[0] < 0 and bool((b[1:] > 0).all()) and abs(float(b.sum())) < 1e-6  # centred, overloaded goes down
    assert float(m._load_acc.sum()) == 0.0
    a = _mod("aux")
    a._load_acc.fill_(3.0)
    a.update_bias()
    assert bool((a.expert_bias == 0).all())


def test_gate_scale_matches_moonlight():
    assert abs(gate_scale(64, 6) - 2.446) < 0.05  # 2502.16982 App. C: 2.446 at 64 experts / top-6
    assert gate_scale(4, 1) == pytest.approx(1.0)
    for e in (4, 8):
        assert 1.2 < gate_scale(e, 2) < 1.45


def test_muon_stack_equals_its_slices():
    torch.manual_seed(0)
    stack = torch.nn.Parameter(torch.randn(3, 24, 40) * 0.1)
    g = torch.randn(3, 24, 40) * 0.01
    slices = [torch.nn.Parameter(stack[i].detach().clone()) for i in range(3)]
    o1 = Muon([stack], lr=1e-2, weight_decay=0.1)
    o2 = Muon(slices, lr=1e-2, weight_decay=0.1)
    for it in range(3):
        stack.grad = g * (1 + 0.1 * it)
        for i, s in enumerate(slices):
            s.grad = g[i] * (1 + 0.1 * it)
        o1.step()
        o2.step()
    for i in range(3):
        assert torch.allclose(stack[i], slices[i], atol=2e-3, rtol=1e-2), i


def _tiny_cfg(**moe):
    return MoteConfig(
        d_model_outer=32, encoder_layers=1, decoder_layers=1,
        main=RelationCfg(n_layers=2, d_model=32, n_heads=2, d_ff=64, moe_experts=4, moe_topk=2, **moe),
        mbp=MBPCfg(n_layers=1, n_heads=2, d_ff=64, n_candidates=3, enabled=False),
        mamba3=Mamba3Cfg(d_state=16, headdim=16, expand=2), max_seq_len=256,
    )


def test_model_builds_trains_and_roundtrips(tmp_path):
    from mote.train.flops import _n
    from mote.train.train import compute_losses

    cfg = _tiny_cfg(moe_dense_first=True)
    torch.manual_seed(0)
    model = HNetForCausalLM(cfg)
    layers = model.main_network.layers
    assert isinstance(layers[0].mlp, SwiGLU) and isinstance(layers[1].mlp, MoESwiGLU)
    assert layers[1].mlp.d_ff == 32  # d_ff // topk: active FLOPs of the dense FFN
    ids = torch.randint(0, 256, (2, 96))
    loss, n, stats, _ = compute_losses(model, ids, 5.0, 0.0, 0.03)
    assert "moe_aux" in stats and "moe_maxvio" in stats
    loss.backward()
    assert layers[1].mlp.w1.grad is not None and layers[1].mlp.router.weight.grad is not None
    muon, other = split_muon_params(model)
    assert any(p is layers[1].mlp.w1 for p in muon) and any(p is layers[1].mlp.router.weight for p in other)
    assert _n(model.main_network) < sum(p.numel() for p in model.main_network.parameters())
    # config and state-dict round trips carry the MoE fields and the expert bias, not the load accumulator
    cfg2 = MoteConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))
    assert cfg2.main.moe_experts == 4 and cfg2.main.moe_dense_first and cfg2.main.moe_router == "lossfree"
    with torch.no_grad():
        layers[1].mlp.expert_bias.fill_(0.1)
    sd = model.state_dict()
    assert "main_network.layers.1.mlp.expert_bias" in sd and not any(k.endswith("_load_acc") for k in sd)
    m2 = HNetForCausalLM(cfg2)
    m2.load_state_dict(sd)
    assert float(m2.main_network.layers[1].mlp.expert_bias[0]) == pytest.approx(0.1)
    # step (decode) path runs through the MoE with one chunk
    state = model.allocate_state(1) if hasattr(model, "allocate_state") else None
    if state is not None:
        model.eval()
        model.prefill(ids[:1, :40], state)


def _fixture(tmp_path):
    cfg = _tiny_cfg()
    cfg.mbp.enabled = True
    cfg_path = tmp_path / "tiny.json"
    cfg.save(cfg_path)
    rng = np.random.default_rng(0)
    for split, n in (("train", 20000), ("val", 4000)):
        rng.integers(0, 256, size=n, dtype=np.uint16).tofile(tmp_path / f"tiny.{split}.bin")
    (tmp_path / "tiny.meta.json").write_text(json.dumps({"train": {"file": "tiny.train.bin"}, "val": {"file": "tiny.val.bin"}}))
    return cfg_path, tmp_path / "tiny"


@pytest.mark.parametrize("router,opt", [("lossfree", "muon"), ("aux", "adamw")])
def test_trainer_runs_moe(tmp_path, router, opt):
    from mote.train.train import Trainer

    cfg_path, prefix = _fixture(tmp_path)
    out = tmp_path / f"run_{router}"
    argv = ["--config", str(cfg_path), "--data", str(prefix), "--out", str(out), "--batch-size", "2", "--seq-len", "64",
            "--grad-accum", "2", "--lr", "1e-3", "--eval-every", "3", "--eval-batches", "1", "--log-every", "1",
            "--ckpt-minutes", "99999", "--max-minutes", "99999", "--max-steps", "3", "--no-mbp",
            "--moe", "4", "--moe-topk", "2", "--moe-router", router, "--optimizer", opt]
    t = Trainer(argv)
    for _ in t.run():
        pass
    t.close()
    recs = [json.loads(l) for l in (out / "log.jsonl").read_text().splitlines()]
    steps = [r for r in recs if "moe_maxvio" in r]
    assert len(steps) == 3 and all(r["moe_maxvio"] >= 0 for r in steps)
    evs = [r["eval"] for r in recs if "eval" in r]
    assert evs and "moe_load" in evs[-1] and len(evs[-1]["moe_load"]) == 2
    ck = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
    assert ck["config"]["main"]["moe_experts"] == 4
    bias = ck["model"]["main_network.layers.0.mlp.expert_bias"]
    if router == "lossfree":
        assert float(bias.abs().sum()) > 0  # the bias moved
    else:
        assert float(bias.abs().sum()) == 0


def test_engine_serves_moe(tmp_path):
    from mote.serve.engine import Engine, GenParams

    cfg = _tiny_cfg()
    torch.manual_seed(0)
    model = HNetForCausalLM(cfg)
    run = tmp_path / "runs" / "moe_tiny"
    run.mkdir(parents=True)
    torch.save({"model": model.state_dict(), "step": 3, "config": cfg.to_dict(), "extra": {"bytes_seen": 3000}}, run / "last.pt")
    eng = Engine(run / "last.pt", device="cpu")
    eng.gpu_gate = threading.Lock()
    out = []
    eng.generate([{"role": "user", "content": "hello there"}], GenParams(max_bytes=16, temperature=0.0), out.append, threading.Event())
    assert [e for e in out if e.get("type") == "done"], out


@pytest.mark.skipif(shutil.which("g++") is None, reason="Inductor on CPU needs a C++ compiler")
def test_dense_path_compiles_fullgraph():
    torch._dynamo.reset()
    m = _mod("lossfree").eval()
    m.dense_threshold = 10 ** 9
    fn = torch.compile(m, fullgraph=True)
    x = torch.randn(1, 1, D)
    y = fn(x)
    assert torch.allclose(y, m(x), atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="grouped_mm is a CUDA bf16 path")
@pytest.mark.parametrize("router", ["lossfree", "aux"])
def test_grouped_path_matches_loop_on_cuda(router):
    m = _mod(router).cuda()
    x = torch.randn(2, 300, D, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        m.dense_threshold = -1
        ref = m._loop(*_prep(m, x))
        got = m._grouped(*_prep(m, x))
    assert torch.allclose(ref.float(), got.float(), atol=3e-2, rtol=3e-2), (ref.float() - got.float()).abs().max()


def _prep(m: MoESwiGLU, x: torch.Tensor):
    xf = x.reshape(-1, x.shape[-1])
    _, _, topi, w = m._route(xf)
    return xf, topi, w
