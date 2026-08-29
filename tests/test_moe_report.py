"""mote.eval.moe_report: NMI/JSD of per-domain routing distributions, and the report over a tiny MoE
checkpoint with two synthetic domain slices."""

import json
import math

import numpy as np
import torch

from mote.config import RelationCfg
from mote.eval.moe_report import mi_and_jsd, run
from mote.model.hnet import HNetForCausalLM
from conftest import tiny_cfg


def test_nmi_and_jsd_extremes():
    same = torch.tensor([[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]])
    r = mi_and_jsd(same)
    assert abs(r["nmi"]) < 1e-6 and abs(r["jsd"]) < 1e-6
    disjoint = torch.tensor([[0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5]])
    r = mi_and_jsd(disjoint)
    assert abs(r["nmi"] - 1.0) < 1e-6 and abs(r["jsd"] - math.log(2)) < 1e-6  # fully specialised
    assert mi_and_jsd(torch.tensor([[1.0, 0.0]]))["nmi"] == 0.0  # one domain: nothing to separate


def test_report_over_tiny_checkpoint(tmp_path):
    cfg = tiny_cfg(main=RelationCfg(n_layers=2, d_model=32, n_heads=2, d_ff=64, moe_experts=4, moe_topk=2))
    torch.manual_seed(0)
    model = HNetForCausalLM(cfg)
    ck = tmp_path / "last.pt"
    torch.save({"model": model.state_dict(), "step": 1, "config": cfg.to_dict(), "extra": {}}, ck)
    dom = tmp_path / "val"
    dom.mkdir()
    rng = np.random.default_rng(0)
    rng.integers(0, 128, size=6000, dtype=np.uint16).tofile(dom / "ascii.val.bin")
    rng.integers(128, 256, size=6000, dtype=np.uint16).tofile(dom / "high.val.bin")
    res = run(ck, dom, batches=2, seq_len=128, device=torch.device("cpu"))
    assert res["layers"] == 2 and res["experts"] == 4 and set(res["domains"]) == {"ascii", "high"}
    for layer in res["per_layer"]:
        assert 0.0 <= layer["nmi"] <= 1.0 + 1e-6 and layer["jsd"] >= 0 and layer["maxvio"] >= 0
    assert set(res["summary"]) == {"maxvio_mean", "nmi_mean", "nmi_last", "jsd_mean"}
    json.dumps(res)  # serialisable
