"""Val bits-per-byte of a checkpoint on the shared val shard and the per-domain slices — the guard metric
of the mid-training gate (docs/shape.md § pipeline: anneal val bpb ≤ control + 0.005, per-domain visible).

    python -m mote.eval.val_bpb --checkpoint runs/branch_anneal/last.pt [--data data/flagship_mix] \
        [--domains data/flagship_val] [--batches 64] [--seq-len 16384]

Loads the model directly (no serving engine), runs the trainer's `evaluate` on sequential windows, and
writes val_bpb.json next to the checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from ..config import MoteConfig
from ..data.loader import ByteShard
from ..model.hnet import HNetForCausalLM
from ..train.train import evaluate


def raw_shard(path: Path) -> ByteShard:
    """A ByteShard over a bare uint16 .bin (the per-domain val slices carry no meta of their own)."""
    sh = ByteShard.__new__(ByteShard)
    sh.meta, sh.sft, sh.mask = {"file": str(path)}, False, None
    sh.data = np.memmap(path, dtype=np.uint16, mode="r")
    sh.n = len(sh.data)
    return sh


def load_model(checkpoint: str | Path, device: torch.device):
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = MoteConfig.from_dict(ck["config"])
    model = HNetForCausalLM(cfg, device=device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg, ck.get("step")


def run(checkpoint: str | Path, data: Optional[str], domains: Optional[str], batches: int, seq_len: Optional[int],
        batch_size: int, device: torch.device) -> Dict:
    model, cfg, step = load_model(checkpoint, device)
    seq_len = seq_len or cfg.max_seq_len
    ratio = cfg.dc.target_ratio_final
    out: Dict = {"checkpoint": str(checkpoint), "step": step, "seq_len": seq_len, "batches": batches, "domains": {}}
    with torch.no_grad():
        # spread=True: windows over the whole shard; the head of a source-blocked shard is one source only
        if data:
            ev = evaluate(model, ByteShard(data, "val"), batch_size, seq_len, batches, device, ratio, spread=True)
            out["val_bpb"] = ev["val_bpb"]
            out["val"] = {k: v for k, v in ev.items() if isinstance(v, (int, float))}
        if domains:
            d = Path(domains)
            for p in sorted(d.glob("*.val.bin")):
                ev = evaluate(model, raw_shard(p), batch_size, seq_len, batches, device, ratio, spread=True)
                out["domains"][p.name.split(".")[0]] = ev["val_bpb"]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", default="data/flagship_mix", help="shard prefix whose val split is the shared val (mix A's); '' to skip")
    ap.add_argument("--domains", default="data/flagship_val", help="directory of <domain>.val.bin slices; '' to skip")
    ap.add_argument("--batches", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=None, help="default: the checkpoint's max_seq_len")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None, help="default: val_bpb.json next to the checkpoint")
    args = ap.parse_args(argv)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    res = run(args.checkpoint, args.data or None, args.domains or None, args.batches, args.seq_len, args.batch_size, device)
    out = Path(args.out) if args.out else Path(args.checkpoint).parent / "val_bpb.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
