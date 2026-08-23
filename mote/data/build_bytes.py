"""Build packed byte shards for pretraining.

Streams documents from Hugging Face, keeps those within a byte-length window, and writes
``{name}.bin`` as uint16 token ids (256 byte values + specials) with an EOS id between documents,
plus ``{name}.meta.json``. uint16 costs 2x the raw bytes but keeps the special ids in-band.

Example (pilot, ~300 MB of FineWeb-Edu):
    python -m mote.data.build_bytes --out data/fineweb_edu_pilot --target-mb 300 --val-mb 8
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..tokenizer import BOS_ID, EOS_ID, VOCAB_SIZE

SOURCES = {
    "fineweb-edu": dict(path="HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", text_key="text"),
}


def stream_docs(source: str, min_bytes: int, max_bytes: int, min_score: float):
    from datasets import load_dataset

    spec = SOURCES[source]
    ds = load_dataset(spec["path"], name=spec["name"], split=spec["split"], streaming=True)
    for row in ds:
        if min_score > 0 and row.get("int_score", 99) < min_score:
            continue
        b = row[spec["text_key"]].encode("utf-8")
        if min_bytes <= len(b) <= max_bytes:
            yield b


def write_shard(path: Path, docs, target_bytes: int, log_every: int = 2000):
    buf = np.empty(target_bytes + 2 * 65536, dtype=np.uint16)
    n, n_docs, t0 = 0, 0, time.time()
    for b in docs:
        need = len(b) + 2
        if n + need > target_bytes:
            break
        buf[n] = BOS_ID
        buf[n + 1 : n + 1 + len(b)] = np.frombuffer(b, dtype=np.uint8)
        buf[n + 1 + len(b)] = EOS_ID
        n += need
        n_docs += 1
        if n_docs % log_every == 0:
            print(f"  {path.name}: {n_docs} docs, {n/1e6:.1f} MB, {n/1e6/(time.time()-t0):.2f} MB/s", flush=True)
    buf[:n].tofile(path)
    return n, n_docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="fineweb-edu", choices=list(SOURCES))
    ap.add_argument("--out", required=True, help="output prefix, e.g. data/fineweb_edu_pilot")
    ap.add_argument("--target-mb", type=float, default=300.0)
    ap.add_argument("--val-mb", type=float, default=8.0)
    ap.add_argument("--min-bytes", type=int, default=256)
    ap.add_argument("--max-bytes", type=int, default=4096)
    ap.add_argument("--min-score", type=float, default=3.0, help="FineWeb-Edu int_score threshold (0 = off)")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    docs = stream_docs(args.source, args.min_bytes, args.max_bytes, args.min_score)
    print(f"writing val shard ({args.val_mb} MB) ...")
    nv, dv = write_shard(out.with_suffix(".val.bin"), docs, int(args.val_mb * 1e6))
    print(f"writing train shard ({args.target_mb} MB) ...")
    nt, dt = write_shard(out.with_suffix(".train.bin"), docs, int(args.target_mb * 1e6))
    meta = {
        "source": args.source,
        "dtype": "uint16",
        "vocab_size": VOCAB_SIZE,
        "train": {"ids": nt, "docs": dt, "file": out.with_suffix(".train.bin").name},
        "val": {"ids": nv, "docs": dv, "file": out.with_suffix(".val.bin").name},
        "filters": {"min_bytes": args.min_bytes, "max_bytes": args.max_bytes, "min_score": args.min_score},
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
