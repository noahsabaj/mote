"""Build the SFT shard: chat conversations rendered with the byte chat template plus a loss mask.

Output: ``{out}.sft.train.bin`` (uint16 ids), ``{out}.sft.train.mask.bin`` (uint8, 1 on assistant bytes + EOS),
same for val, and ``{out}.sft.meta.json`` with per-conversation offsets. Conversations are packed back to back;
the loader cuts fixed windows and the mask decides which positions train.

    python -m morpheme.data.build_sft --out data/sft_mix --target-mb 300 --val-mb 8
    python -m morpheme.data.build_sft --out data/sft_probe --target-mb 2 --val-mb 0.5   # probe every source
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np

from ..tokenizer import ByteTokenizer, ChatMessage
from .sources import SFT, SFTSource


def _oasst_threads(rows: Iterator[dict], max_msgs: int = 12) -> Iterator[List[dict]]:
    """Assemble English prompter/assistant threads from OASST2 message rows (tree-structured)."""
    by_id: Dict[str, dict] = {}
    children: Dict[Optional[str], List[str]] = defaultdict(list)
    for r in rows:
        if r.get("lang") != "en":
            continue
        by_id[r["message_id"]] = r
        children[r.get("parent_id")].append(r["message_id"])
    # walk from roots down the first child each time (greedy longest single path)
    for root in children[None]:
        path, cur = [], root
        while cur and len(path) < max_msgs:
            m = by_id[cur]
            role = "user" if m["role"] == "prompter" else "assistant"
            path.append({"role": role, "content": m["text"]})
            kids = children.get(cur, [])
            cur = kids[0] if kids else None
        if len(path) >= 2 and path[-1]["role"] == "assistant":
            yield path


def stream_source(src: SFTSource, max_bytes: int) -> Iterator[List[dict]]:
    from datasets import load_dataset

    kwargs = dict(split=src.split, streaming=True)
    if src.name:
        kwargs["name"] = src.name
    ds = load_dataset(src.path, **kwargs)
    if src.key == "oasst2":
        gen = _oasst_threads(r for r in ds if src.keep(r))
    else:
        def gen_fn():
            for r in ds:
                try:
                    if not src.keep(r):
                        continue
                    m = src.messages(r)
                except Exception:
                    continue
                if m:
                    yield m
        gen = gen_fn()
    for msgs in gen:
        msgs = [m for m in msgs if m.get("role") in ("system", "user", "assistant") and isinstance(m.get("content"), str)]
        if len(msgs) < 2 or not any(m["role"] == "assistant" for m in msgs):
            continue
        if any(not m["content"].strip() for m in msgs):  # redacted / empty turns
            continue
        if msgs[-1]["role"] != "assistant":
            msgs = msgs[: max(i for i, m in enumerate(msgs) if m["role"] == "assistant") + 1]
        size = sum(len(m["content"].encode("utf-8")) + 2 for m in msgs) + 1
        if size <= max_bytes:
            yield msgs


def build(out: Path, target_bytes: int, val_bytes: int, max_bytes: int, sources: List[SFTSource]):
    tok = ByteTokenizer()
    iters = {s.key: stream_source(s, max_bytes) for s in sources}
    exhausted = set()
    t0 = time.time()

    def fill(prefix: str, total: int):
        budgets = {s.key: int(s.share * total) for s in sources}
        got = {k: 0 for k in budgets}
        ids_buf = np.empty(total + max_bytes + 8, dtype=np.uint16)
        mask_buf = np.zeros(total + max_bytes + 8, dtype=np.uint8)
        n, n_conv = 0, 0
        keys = list(budgets)
        while keys:
            key = min(keys, key=lambda k: got[k] / max(budgets[k], 1))
            if got[key] >= budgets[key] or key in exhausted:
                keys.remove(key)
                continue
            try:
                msgs = next(iters[key])
            except StopIteration:
                exhausted.add(key)
                continue
            except Exception as e:
                print(f"  source {key} failed: {type(e).__name__}: {str(e)[:120]}", flush=True)
                exhausted.add(key)
                continue
            ids, mask = tok.format_chat_with_loss_mask([ChatMessage(m["role"], m["content"]) for m in msgs])
            if n + len(ids) > len(ids_buf):
                break
            ids_buf[n : n + len(ids)] = np.asarray(ids, dtype=np.uint16)
            mask_buf[n : n + len(ids)] = np.asarray(mask, dtype=np.uint8)
            n += len(ids)
            got[key] += len(ids)
            n_conv += 1
            if n_conv % 2000 == 0:
                print(f"  {prefix}: {n_conv} convs {n/1e6:.0f} MB ({n/1e6/(time.time()-t0):.2f} MB/s) " + " ".join(f"{k}={got[k]/1e6:.1f}" for k in got), flush=True)
        ids_buf[:n].tofile(out.parent / f"{out.name}.sft.{prefix}.bin")
        mask_buf[:n].tofile(out.parent / f"{out.name}.sft.{prefix}.mask.bin")
        return n, n_conv, got

    print("val ...", flush=True)
    nv, cv, gv = fill("val", val_bytes)
    print("train ...", flush=True)
    nt, ct, gt = fill("train", target_bytes)
    meta = {
        "dtype": "uint16", "mask_dtype": "uint8", "vocab_size": tok.vocab_size,
        "train": {"ids": nt, "convs": ct, "file": f"{out.name}.sft.train.bin", "mask_file": f"{out.name}.sft.train.mask.bin", "per_source_bytes": gt},
        "val": {"ids": nv, "convs": cv, "file": f"{out.name}.sft.val.bin", "mask_file": f"{out.name}.sft.val.mask.bin", "per_source_bytes": gv},
        "sources": [{"key": s.key, "path": s.path, "name": s.name, "share": s.share, "note": s.note} for s in sources],
        "max_bytes": max_bytes, "exhausted": sorted(exhausted),
    }
    (out.parent / f"{out.name}.sft.meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k != "sources"}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-mb", type=float, default=300.0)
    ap.add_argument("--val-mb", type=float, default=8.0)
    ap.add_argument("--max-bytes", type=int, default=4096, help="max bytes per conversation (incl. template)")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    sources = [s for s in SFT if not args.only or s.key in args.only]
    if args.only:
        tot = sum(s.share for s in sources)
        for s in sources:
            s.share = s.share / tot
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    build(out, int(args.target_mb * 1e6), int(args.val_mb * 1e6), args.max_bytes, sources)


if __name__ == "__main__":
    main()
