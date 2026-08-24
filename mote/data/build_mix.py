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

import time
from pathlib import Path
from typing import Iterator, List

import numpy as np

from ..tokenizer import BOS_ID, EOS_ID, VOCAB_SIZE
from .sources import FLAGSHIP, PRETRAIN, PretrainSource


def _chunks(b: bytes, max_bytes: int) -> Iterator[bytes]:
    """Split an over-long document at paragraph breaks (books): each piece is its own document."""
    while len(b) > max_bytes:
        cut = b.rfind(b"\n\n", max_bytes // 2, max_bytes)
        if cut < 0:
            cut = max_bytes
        yield b[:cut]
        b = b[cut:].lstrip(b"\n")
    if b:
        yield b


def stream_source(src: PretrainSource, min_bytes: int, max_bytes: int) -> Iterator[bytes]:
    from datasets import load_dataset

    lo = src.min_bytes if src.min_bytes is not None else min_bytes
    hi = src.max_bytes if src.max_bytes is not None else max_bytes
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
        if src.chunk and len(b) > hi:
            for piece in _chunks(b, hi):
                if lo <= len(piece) <= hi:
                    yield piece
            continue
        if lo <= len(b) <= hi:
            yield b


def build(out: Path, target_bytes: int, val_bytes: int, min_bytes: int, max_bytes: int, seed: int, sources: List[PretrainSource]):
    import gc

    budget = {s.key: int(s.share * target_bytes) for s in sources}
    val_budget = {s.key: int(s.share * val_bytes) for s in sources}
    exhausted = set()
    t0 = time.time()

    # One source at a time: 20 concurrent streaming iterators held ~20 GB of parquet read buffers
    # and the kernel OOM-killed the first build (2026-08-23). Sequential filling bounds memory at the
    # heaviest single source; the loader samples random windows, so source-blocked order is harmless.
    # Writes go through memory-mapped files the OS pages to disk; truncated to the written length.
    train_path, val_path = out.with_suffix(".train.bin"), out.with_suffix(".val.bin")
    train_buf = np.memmap(train_path, dtype=np.uint16, mode="w+", shape=(sum(budget.values()) + max_bytes + 2,))
    val_buf = np.memmap(val_path, dtype=np.uint16, mode="w+", shape=(sum(val_budget.values()) + max_bytes + 2,))
    nt = nv = dt = dv = 0
    gt = {s.key: 0 for s in sources}
    gv = {s.key: 0 for s in sources}

    for src in sources:
        it = stream_source(src, min_bytes, max_bytes)

        def take() -> bytes | None:
            try:
                return next(it)
            except StopIteration:
                exhausted.add(src.key)
                return None
            except Exception as e:  # network hiccup: drop the source rather than the run
                print(f"  source {src.key} failed: {type(e).__name__}: {str(e)[:120]}", flush=True)
                exhausted.add(src.key)
                return None

        # val quota first, then train, from the same iterator: the shards never share a document
        for buf, got, quota, is_val in ((val_buf, gv, val_budget[src.key], True), (train_buf, gt, budget[src.key], False)):
            n = nv if is_val else nt
            while got[src.key] < quota:
                doc = take()
                if doc is None:
                    break
                need = len(doc) + 2
                if n + need > len(buf):
                    break
                buf[n] = BOS_ID
                buf[n + 1 : n + 1 + len(doc)] = np.frombuffer(doc, dtype=np.uint8)
                buf[n + 1 + len(doc)] = EOS_ID
                n += need
                got[src.key] += need
                if is_val:
                    nv, dv = n, dv + 1
                else:
                    nt, dt = n, dt + 1
                if (dt + dv) % 5000 == 0:
                    print(f"  {src.key}: {dt + dv} docs, train {nt/1e6:.0f} MB val {nv/1e6:.0f} MB ({(nt + nv)/1e6/(time.time()-t0):.2f} MB/s)", flush=True)
        print(f"  {src.key} done: train {gt[src.key]/1e6:.0f}/{budget[src.key]/1e6:.0f} MB val {gv[src.key]/1e6:.0f} MB" + (" (exhausted)" if src.key in exhausted else ""), flush=True)
        del it
        gc.collect()

    for buf in (train_buf, val_buf):
        buf.flush()
    del train_buf, val_buf
    with open(train_path, "r+b") as f:
        f.truncate(nt * 2)
    with open(val_path, "r+b") as f:
        f.truncate(nv * 2)
    meta = {
        "dtype": "uint16", "vocab_size": VOCAB_SIZE,
        "train": {"ids": nt, "docs": dt, "file": out.with_suffix(".train.bin").name, "per_source_bytes": gt},
        "val": {"ids": nv, "docs": dv, "file": out.with_suffix(".val.bin").name, "per_source_bytes": gv},
        "sources": [{"key": s.key, "path": s.path, "name": s.name, "share": s.share, "note": s.note} for s in sources],
        "filters": {"min_bytes": min_bytes, "max_bytes": max_bytes,
                    "overrides": {s.key: [s.min_bytes, s.max_bytes] for s in sources if s.min_bytes or s.max_bytes}},
        "exhausted": sorted(exhausted),
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k != "sources"}, indent=2), flush=True)
    # pyarrow's background IO threads crash the interpreter during normal teardown (PyGILState_Release
    # fatal error); everything is flushed and truncated by now, so skip Python finalization entirely.
    os._exit(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-gb", type=float, default=8.0)
    ap.add_argument("--val-mb", type=float, default=32.0)
    ap.add_argument("--min-bytes", type=int, default=128)
    ap.add_argument("--max-bytes", type=int, default=4096)
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these source keys (for probing)")
    ap.add_argument("--list", default="pretrain", choices=["pretrain", "flagship"], help="which source registry to build from")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    registry = FLAGSHIP if args.list == "flagship" else PRETRAIN
    sources = [s for s in registry if not args.only or s.key in args.only]
    if args.only:  # renormalize shares
        tot = sum(s.share for s in sources)
        for s in sources:
            s.share = s.share / tot
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    build(out, int(args.target_gb * 1e9), int(args.val_mb * 1e6), args.min_bytes, args.max_bytes, args.seed, sources)


if __name__ == "__main__":
    main()
