"""Build shards from local JSONL — the sim generator's outputs, exported conversations, expert traces.

    # plain-LM shard for a cooldown mix: narratives as documents, QA conversations as chat-template bytes
    python -m mote.data.build_local --out data/sim_plain --text data/sim_state.narrative.jsonl --chat data/sim_state.qa.jsonl
    # SFT shard (assistant-byte loss mask) for post-training
    python -m mote.data.build_local --out data/sim_sft --chat data/sim_state.qa.jsonl --sft

Text rows `{"text": ...}` become BOS + bytes + EOS; chat rows `{"messages": [{role, content}, ...]}` become
the byte chat template (role id + bytes + EOS per turn), with the assistant-bytes mask when `--sft`.
Rows are shuffled together; the first `--val-frac` of them form the val split (a document never
straddles the split). Output layout matches build_mix (plain) / build_sft (sft), so ByteShard reads both.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, Tuple

import numpy as np

from ..tokenizer import BOS_ID, EOS_ID, VOCAB_SIZE, ByteTokenizer, ChatMessage


def _rows(path: str | Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _docs(text_paths: Sequence[str | Path], chat_paths: Sequence[str | Path], max_bytes: int) -> Tuple[List[Tuple[List[int], List[int]]], dict]:
    tok = ByteTokenizer()
    docs: List[Tuple[List[int], List[int]]] = []
    counts = {}
    for p in text_paths:
        n = 0
        for r in _rows(p):
            b = (r.get("text") or "").encode("utf-8")
            if not b or len(b) + 2 > max_bytes:
                continue
            ids = [BOS_ID] + list(b) + [EOS_ID]
            docs.append((ids, [0] * len(ids)))
            n += 1
        counts[str(p)] = n
    for p in chat_paths:
        n = 0
        for r in _rows(p):
            msgs = r.get("messages")
            if not msgs:
                continue
            ids, mask = tok.format_chat_with_loss_mask([ChatMessage(m["role"], m["content"]) for m in msgs])
            if len(ids) > max_bytes:
                continue
            docs.append((ids, mask))
            n += 1
        counts[str(p)] = n
    return docs, counts


def _write(path: Path, arrays: Iterable[np.ndarray], dtype) -> int:
    n = 0
    with open(path, "wb") as f:
        for a in arrays:
            a.astype(dtype).tofile(f)
            n += len(a)
    return n


def build_local(out: Path, text_paths: Sequence[str | Path] = (), chat_paths: Sequence[str | Path] = (), sft: bool = False,
                val_frac: float = 0.02, max_bytes: int = 16384, seed: int = 0) -> dict:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    docs, counts = _docs(text_paths, chat_paths, max_bytes)
    if not docs:
        raise SystemExit("no documents within --max-bytes")
    random.Random(seed).shuffle(docs)
    n_val = int(round(len(docs) * val_frac))
    splits = {"val": docs[:n_val], "train": docs[n_val:]}
    tag = f"{out.name}.sft" if sft else out.name
    meta = {"dtype": "uint16", "vocab_size": VOCAB_SIZE, "format": "sft" if sft else "plain",
            "sources": [{"path": p, "rows": n} for p, n in counts.items()], "max_bytes": max_bytes}
    if sft:
        meta["mask_dtype"] = "uint8"
    for split, items in splits.items():
        data_file = f"{tag}.{split}.bin"
        n_ids = _write(out.parent / data_file, (np.asarray(ids, dtype=np.uint16) for ids, _ in items), np.uint16)
        rec = {"ids": n_ids, ("convs" if sft else "docs"): len(items), "file": data_file}
        if sft:
            rec["mask_file"] = f"{tag}.{split}.mask.bin"
            _write(out.parent / rec["mask_file"], (np.asarray(m, dtype=np.uint8) for _, m in items), np.uint8)
        meta[split] = rec
    (out.parent / f"{tag}.meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="shard prefix, e.g. data/sim_plain")
    ap.add_argument("--text", nargs="*", default=[], help="JSONL files of {text} rows")
    ap.add_argument("--chat", nargs="*", default=[], help="JSONL files of {messages} rows")
    ap.add_argument("--sft", action="store_true", help="write an SFT shard (assistant-byte loss mask) instead of a plain-LM shard")
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--max-bytes", type=int, default=16384, help="drop documents/conversations longer than this (incl. template)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    meta = build_local(Path(args.out), args.text, args.chat, sft=args.sft, val_frac=args.val_frac, max_bytes=args.max_bytes, seed=args.seed)
    print(json.dumps({k: v for k, v in meta.items() if k in ("format", "train", "val", "sources")}, indent=2))


if __name__ == "__main__":
    main()
