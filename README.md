# Morpheme

A byte-level language model that learns its own tokenizer, and a studio for chatting with it and
looking inside. Working name; the package is `morpheme/`.

**Architecture** — a one-stage H-Net (Hwang, Wang & Gu, 2025) over raw UTF-8 bytes:

```
bytes ─▶ Mamba-3 encoder ─▶ dynamic chunking ─▶ Relation main network ─▶ dechunk (EMA) ─▶ Mamba-3 decoder ─▶ next byte
                                           └──────────────────────────────────────────▶ multi-byte head (LCA) ─▶ several bytes at once
```

* **Encoder / decoder**: official Mamba-3 SISO mixers (`state-spaces/mamba`, Triton kernels on Linux/WSL2;
  a weight-compatible pure-PyTorch path everywhere else, including a recurrent decode step) — `morpheme/model/mamba3.py`.
* **Dynamic chunking**: cosine-similarity router, straight-through boundaries, EMA dechunking, ratio loss with the
  ATDC target schedule — `morpheme/model/dc.py`.
* **Main network**: Full Relation (Ge, Yang & Nie, 2026): Self/Exchange relations, count calibration λ, Givens head mixing,
  `{P2, Ĩ}` decode cache — `morpheme/model/relation.py`.
* **Multi-byte prediction**: Latent-Causal-Attention head (Owodunni et al., 2026) that proposes the rest of a chunk
  in parallel; bytes are accepted while the head's confidence ≥ τ — `morpheme/model/mbp.py`.
* **Tokenizer**: none. 256 byte values + `<|bos|> <|eos|> <|pad|> <|system|> <|user|> <|assistant|>` — `morpheme/tokenizer.py`.

Every number the studio shows is real: checkpoint metadata, live tensors, run logs. Undertrained checkpoints are
labelled as such.

## Status (2026-08-22)

* Model, trainer, serving engine and API are implemented and tested (`tests/`, incl. prefill+step decoding
  reproducing the full forward logits).
* Local pilot (12.7M params, RTX 4060 Ti, WSL2): dynamic chunking finds word-like boundaries within minutes;
  see `runs/pilot_*/log.jsonl`.
* Flagship training on Lightning.ai (`cloud/`) has **not** been run yet.
* Frontend (`web/`, Vite + Svelte 5) in progress.

## Setup

Python ≥ 3.11. Two environments are useful on Windows:

* **Windows venv** (`.venv`): tests, data building, serving with the pure-PyTorch paths.
  `uv pip install -e . --python .venv/Scripts/python.exe`
* **WSL2 Ubuntu** (`~/hnet-venv`): the official Mamba-3 Triton kernels for training. Once:
  ```bash
  uv venv ~/hnet-venv --python 3.12
  uv pip install --python ~/hnet-venv/bin/python torch --index-url https://download.pytorch.org/whl/cu126
  uv pip install --python ~/hnet-venv/bin/python "triton>=3.5" einops numpy huggingface_hub transformers datasets fastapi "uvicorn[standard]" websockets pytest
  MAMBA_SKIP_CUDA_BUILD=TRUE uv pip install --python ~/hnet-venv/bin/python -e /mnt/d/Code/Storage/mamba --no-deps --no-build-isolation
  ```
  `cloud/bootstrap.sh` does the same on a Lightning studio.

## Data

```bash
python -m morpheme.data.build_bytes --out data/fineweb_edu_pilot --target-mb 300 --val-mb 8     # pilot: FineWeb-Edu ≤4 KB docs
python -m morpheme.data.build_mix   --out data/pretrain_mix --target-gb 10 --val-mb 64           # flagship mix (morpheme/data/sources.py)
python -m morpheme.data.build_sft   --out data/sft_mix --target-mb 300 --val-mb 8                # chat SFT mix with loss masks
```

Shards are uint16 ids (bytes + specials) with BOS/EOS separators; SFT shards add a uint8 mask (1 on assistant bytes).

## Train

```bash
# in WSL2 (kernels):
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m morpheme.train.train \
  --preset pilot --data ~/data/fineweb_edu_pilot --out runs/pilot_1h --batch-size 4 --grad-accum 4 --seq-len 2048 --max-minutes 60
# SFT from a pretrained checkpoint:
python -m morpheme.train.train --preset pilot --sft --init-from runs/pilot_1h/last.pt --data data/sft_mix --out runs/pilot_sft --max-minutes 20
```

`log.jsonl` records train loss/BPB, bytes-per-chunk, ratio loss, multi-byte-head loss, and periodic evals
(val BPB, boundary/separator alignment, MBP top-1, a chunked text sample). `last.pt` is written atomically
every `--ckpt-minutes`; `--resume` continues. With `--max-steps 0` (default) the LR and ratio schedules follow
wall-clock progress toward `--max-minutes`.

Notes learned the hard way: batch 4 × accum 4 at 2048 bytes is the 8 GB sweet spot (at initialization the router
fires on ~50% of bytes, so the materialized Relation attention is ~6× larger than after convergence); the Triton
SSD kernel re-autotunes for every new chunk count, so the EMA dechunk uses a chunked pure-PyTorch scan instead.

## Serve

```bash
python -m morpheme.serve.app --checkpoint runs/pilot_1h/last.pt --port 7860
```

`docs/api.md` is the contract: `/api/model`, `/api/checkpoints` (+ hot-swap), `/api/training/runs`,
`/v1/chat/completions` (OpenAI-compatible SSE), and `/ws/generate` streaming per-byte events (probabilities,
entropy, chunk boundaries, multi-byte acceptances, UTF-8 assembly, live Mamba-3 retention and Relation exchange mass).
The built frontend (`web/dist`) is served at `/`.

## Frontend

`web/` is a Vite + Svelte 5 + TypeScript app (see `web/README.md`): `npm install`, `npm run dev` (standalone against a
clearly-labelled dev mock), `npm run build` (→ `web/dist`, served by the backend at `/`), `npm run check`. One page: the
conversation, a one-line honesty strip, and Structure/Bytes toggles under each reply; Model, Diagnostics and Training
open as sheets. Everything it shows comes from the API above.

## Cloud (Lightning.ai)

`cloud/launch.py` drives a studio through the SDK; nothing starts without `--go`. `plan` prints the budget math.
Training runs on an interruptible H100 and auto-resumes from the studio disk.

## Tests

```bash
python -m pytest -q tests/
```

## References

H-Net 2507.07955 · Mamba-3 2603.15569 · Relation 2608.20172 · LCA multi-byte prediction 2608.15454 · ATDC 2605.30080.
