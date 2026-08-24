"""`--eval-ema` logs an EMA validation beside the raw one and survives a checkpoint; the LR-vs-horizon fit
(mote/train/lr_horizon.py) recovers a planted slope from synthetic runs."""

import json
import math

import numpy as np
import torch

from mote.config import MBPCfg, Mamba3Cfg, MoteConfig, RelationCfg
from mote.train.lr_horizon import fit, parabola_vertex, predict, read_run
from mote.train.train import Trainer


def _fixture(tmp_path):
    cfg = MoteConfig(
        d_model_outer=32, encoder_layers=1, decoder_layers=1,
        main=RelationCfg(n_layers=1, d_model=32, n_heads=2, d_ff=64),
        mbp=MBPCfg(n_layers=1, n_heads=2, d_ff=64, n_candidates=3, enabled=False),
        mamba3=Mamba3Cfg(d_state=16, headdim=16, expand=2), max_seq_len=256,
    )
    cfg_path = tmp_path / "tiny.json"
    cfg.save(cfg_path)
    rng = np.random.default_rng(0)
    for split, n in (("train", 20000), ("val", 4000)):
        rng.integers(0, 256, size=n, dtype=np.uint16).tofile(tmp_path / f"tiny.{split}.bin")
    (tmp_path / "tiny.meta.json").write_text(json.dumps({"train": {"file": "tiny.train.bin"}, "val": {"file": "tiny.val.bin"}}))
    return cfg_path, tmp_path / "tiny"


def test_eval_ema_logged_and_checkpointed(tmp_path):
    cfg_path, prefix = _fixture(tmp_path)
    out = tmp_path / "run"
    argv = ["--config", str(cfg_path), "--data", str(prefix), "--out", str(out), "--batch-size", "2", "--seq-len", "64",
            "--grad-accum", "1", "--lr", "1e-3", "--eval-every", "2", "--eval-batches", "1", "--log-every", "1",
            "--ckpt-minutes", "0", "--max-minutes", "99999", "--max-steps", "2", "--no-mbp", "--eval-ema", "0.5"]
    t = Trainer(argv)
    for _ in t.run():
        pass
    t.close()
    evs = [json.loads(l)["eval"] for l in (out / "log.jsonl").read_text().splitlines() if "\"eval\"" in l]
    assert evs and all("val_bpb_ema" in e for e in evs)
    ck = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
    names = [n for n, _ in t.model.named_parameters()]
    assert "ema" in ck["extra"] and len(ck["extra"]["ema"]) == len(names)
    # the model's raw weights were restored after the EMA eval: they differ from the EMA copy
    raw = torch.cat([ck["model"][n].flatten().float() for n in names])
    ema = torch.cat([v.flatten().float() for v in ck["extra"]["ema"]])
    assert raw.numel() == ema.numel() and not torch.allclose(raw, ema)
    # resume restores the EMA and keeps training
    t2 = Trainer(argv[:-2] + ["--eval-ema", "0.5", "--resume", "--max-steps", "3"])
    assert torch.allclose(t2.ema[0].cpu(), ck["extra"]["ema"][0].cpu())
    t2.close()


def _synthetic(tmp_path, lrs, beta=-0.25, eta0=1e-3, D0=1e6):
    """val bpb = base(D) + 0.8·(ln lr − ln η*(D))² with ln η*(D) = ln eta0 + beta·ln(D/D0)."""
    runs = []
    for lr in lrs:
        run = tmp_path / f"lr_{lr:g}"
        run.mkdir()
        (run / "run.json").write_text(json.dumps({"lr": lr, "batch_size": 1, "seq_len": 1000, "grad_accum": 1}))
        lines = []
        for step in range(1000, 20001, 1000):
            D = step * 1000
            eta = eta0 * (D / D0) ** beta
            bpb = 1.0 + 0.5 * (D / D0) ** -0.2 + 0.8 * (math.log(lr) - math.log(eta)) ** 2
            lines.append(json.dumps({"eval": {"val_bpb": bpb}, "step": step, "elapsed_min": step / 100}))
        (run / "log.jsonl").write_text("\n".join(lines))
        runs.append(run)
    return runs


def test_fit_recovers_planted_slope(tmp_path):
    runs = _synthetic(tmp_path, [4e-4, 8e-4, 1.6e-3])
    res = fit([read_run(r) for r in runs])
    assert res["n_usable"] >= 10 and abs(res["beta"] + 0.25) < 0.02 and res["r2"] > 0.99
    eta = predict(res, 1e9)
    assert abs(math.log(eta) - math.log(1e-3 * (1e9 / 1e6) ** -0.25)) < 0.05


def test_parabola_vertex_exact():
    xs = [-1.0, 0.0, 1.0]
    ys = [2.0 * (x - 0.3) ** 2 + 1.0 for x in xs]
    a, b, c, xstar = parabola_vertex(xs, ys)
    assert abs(a - 2.0) < 1e-9 and abs(xstar - 0.3) < 1e-9
