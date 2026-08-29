"""mote/eval/rl_taxonomy.py: the Table-14 categories on hand-made policies, and the end-to-end walk over held-out
sim tasks with two tiny checkpoints (every expert step scored over the state's legal actions)."""

import torch

from mote.config import Mamba3Cfg, MoteConfig, RelationCfg
from mote.eval.rl_taxonomy import CATEGORIES, categorize, run, state_walk
from mote.model.hnet import HNetForCausalLM
from mote.sim.tasks import heldout_tasks


def test_categories_follow_table_14():
    k, eps = 3, 0.05
    p0 = [0.5, 0.3, 0.1, 0.06, 0.04]
    assert categorize(p0, [0.6, 0.25, 0.1, 0.03, 0.02], 0, k, eps) == "gt_amplification"
    assert categorize(p0, [0.4, 0.3, 0.1, 0.1, 0.1], 0, k, eps) == "other"  # in top-k both times but lost mass
    assert categorize(p0, [0.3, 0.3, 0.05, 0.05, 0.3], 4, k, eps) == "tail_discovery"  # from 0.04 (< eps) into the top-3
    assert categorize(p0, [0.3, 0.3, 0.05, 0.3, 0.05], 3, k, eps) == "topk_correction"  # from 0.06 (>= eps)
    assert categorize(p0, [0.1, 0.4, 0.3, 0.15, 0.05], 0, k, eps) == "gt_regression"
    assert categorize(p0, [0.7, 0.2, 0.05, 0.03, 0.02], 4, k, eps) == "wrong_mode_amp"  # a* still out, wrong top-1 grew
    assert categorize(p0, [0.4, 0.3, 0.2, 0.06, 0.04], 4, k, eps) == "other"


def _ckpt(tmp_path, name, seed):
    cfg = MoteConfig(
        d_model_outer=32, encoder_layers=1, decoder_layers=1,
        main=RelationCfg(n_layers=1, d_model=32, n_heads=2, d_ff=64),
        mamba3=Mamba3Cfg(d_state=16, headdim=16, expand=2), max_seq_len=4096,
    )
    torch.manual_seed(seed)
    model = HNetForCausalLM(cfg)
    run_dir = tmp_path / "runs" / name
    run_dir.mkdir(parents=True)
    torch.save({"model": model.state_dict(), "step": 1, "config": cfg.to_dict(), "extra": {}}, run_dir / "last.pt")
    return run_dir / "last.pt"


def test_state_walk_and_end_to_end(tmp_path):
    tasks = heldout_tasks(2)
    for t in tasks:
        states = state_walk(t)
        assert len(states) == len(t.expert)
        for st in states:
            assert st["legal"][st["gt"]] in t.expert and len(st["legal"]) >= 1
    before, after = _ckpt(tmp_path, "sft", 0), _ckpt(tmp_path, "rl", 1)
    res = run(str(before), str(after), n=2, k=3, eps_tail=0.05, device=torch.device("cpu"), batch=8)
    assert res["n_tasks"] == 2 and res["n_states"] == sum(len(t.expert) for t in tasks)
    for b, cats in res["categories"].items():
        assert abs(sum(cats.values()) - 1.0) < 1e-6 and set(cats) == set(CATEGORIES)
    for b, row in res["before"].items():
        assert 1.0 <= row["gt_mean_rank"] and 0.0 <= row["gt_mean_p"] <= 1.0
    solo = run(str(before), None, n=1, k=3, eps_tail=0.05, device=torch.device("cpu"), batch=8)
    assert "categories" not in solo and solo["after_ckpt"] is None
