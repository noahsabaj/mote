"""Build the flagship pretraining mix as packed uint16 shards.

    python -m mote.data.build_mix --out data/pretrain_mix --target-gb 8 --val-mb 32
    python -m mote.data.build_mix --out data/mix_probe --target-gb 0.02 --val-mb 1   # quick probe of every source

Streams every source in `sources.PRETRAIN`, keeps documents within the byte window, interleaves
them according to the mix shares, and writes train/val shards with BOS/EOS separators.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, Iterator, List

import numpy as np

from ..tokenizer import BOS_ID, EOS_ID, VOCAB_SIZE
from .sources import PRETRAIN, PretrainSource


def stream_source(src: PretrainSource, min_bytes: int, max_bytes: int) -> Iterator[bytes]:
    from datasets import load_dataset

    kwargs = dict(split=src.split, streaming=True)
    if src.name:
        kwargs["name"] = src.name
    ds = load_dataset(src.path, **kwargs)
    for row in ds:
        try:
            if not src.keep(row):
                continue
            t = src.text(row)
        except Exception:
            continue
        if not t:
            continue
        b = t.encode("utf-8")
        if min_bytes <= len(b) <= max_bytes:
            yield b


def build(out: Path, target_bytes: int, val_bytes: int, min_bytes: int, max_bytes: int, seed: int, sources: List[PretrainSource]):
    rng = random.Random(seed)
    iters: Dict[str, Iterator[bytes]] = {s.key: stream_source(s, min_bytes, max_bytes) for s in sources}
    budget = {s.key: int(s.share * target_bytes) for s in sources}
    val_budget = {s.key: int(s.share * val_bytes) for s in sources}
    written = {s.key: 0 for s in sources}
    exhausted = set()
    t0 = time.time()

    def take(key: str) -> bytes | None:
        try:
            return next(iters[key])
        except StopIteration:
            exhausted.add(key)
            return None
        except Exception as e:  # network hiccup: drop the source rather than the run
            print(f"  source {key} failed: {type(e).__name__}: {str(e)[:120]}", flush=True)
            exhausted.add(key)
            return None

    def fill(path: Path, budgets: Dict[str, int]):
        total = sum(budgets.values())
        # Write through a memory-mapped file: the OS pages it to disk as it fills, so a 10 GB corpus
        # (20 GB of uint16 ids) never has to sit in RAM. Truncated to the written length at the end.
        buf = np.memmap(path, dtype=np.uint16, mode="w+", shape=(total + max_bytes + 2,))
        n, n_docs, got = 0, 0, {k: 0 for k in budgets}
        keys = list(budgets)
        while keys:
            # pick the source furthest behind its share
            key = min(keys, key=lambda k: got[k] / max(budgets[k], 1))
            doc = take(key)
            if doc is None or got[key] >= budgets[key]:
                keys.remove(key)
                continue
            need = len(doc) + 2
            if n + need > len(buf):
                break
            buf[n] = BOS_ID
            buf[n + 1 : n + 1 + len(doc)] = np.frombuffer(doc, dtype=np.uint8)
            buf[n + 1 + len(doc)] = EOS_ID
            n += need
            got[key] += need
            n_docs += 1
            if n_docs % 5000 == 0:
                print(f"  {path.name}: {n_docs} docs {n/1e6:.0f} MB ({n/1e6/(time.time()-t0):.2f} MB/s) " + " ".join(f"{k}={got[k]/1e6:.0f}" for k in got), flush=True)
        buf.flush()
        del buf
        with open(path, "r+b") as f:
            f.truncate(n * 2)
        return n, n_docs, got

    print("val shard ...", flush=True)
    nv, dv, gv = fill(out.with_suffix(".val.bin"), val_budget)
    print("train shard ...", flush=True)
    nt, dt, gt = fill(out.with_suffix(".train.bin"), budget)
    meta = {
        "dtype": "uint16", "vocab_size": VOCAB_SIZE,
        "train": {"ids": nt, "docs": dt, "file": out.with_suffix(".train.bin").name, "per_source_bytes": gt},
        "val": {"ids": nv, "docs": dv, "file": out.with_suffix(".val.bin").name, "per_source_bytes": gv},
        "sources": [{"key": s.key, "path": s.path, "name": s.name, "share": s.share, "note": s.note} for s in sources],
        "filters": {"min_bytes": min_bytes, "max_bytes": max_bytes}, "exhausted": sorted(exhausted),
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k != "sources"}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-gb", type=float, default=8.0)
    ap.add_argument("--val-mb", type=float, default=32.0)
    ap.add_argument("--min-bytes", type=int, default=128)
    ap.add_argument("--max-bytes", type=int, default=4096)
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these source keys (for probing)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    sources = [s for s in PRETRAIN if not args.only or s.key in args.only]
    if args.only:  # renormalize shares
        tot = sum(s.share for s in sources)
        for s in sources:
            s.share = s.share / tot
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    build(out, int(args.target_gb * 1e9), int(args.val_mb * 1e6), args.min_bytes, args.max_bytes, args.seed, sources)


if __name__ == "__main__":
    main()
