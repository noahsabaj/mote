# Mote

A byte-level language model that learns its own tokenizer, and a studio for chatting with it and looking
inside. One resident process on one consumer GPU trains it and serves it.

**Architecture** — a one-stage H-Net (Hwang, Wang & Gu, 2025) over raw UTF-8 bytes: Mamba-3 at byte
resolution outside, blocks with the Relation mixer (Ge, Yang & Nie, 2026 — in place of attention) at chunk
resolution inside, and a learned dynamic chunker between them.

```
bytes ─▶ Mamba-3 encoder ─▶ dynamic chunking ─▶ Relation main network ─▶ dechunk (EMA) ─▶ Mamba-3 decoder ─▶ next byte
```

* **Encoder / decoder**: Mamba-3 SISO mixers (the official Triton kernels on Linux; a weight-compatible pure-PyTorch
  path everywhere else, including a recurrent decode step) — `mote/model/mamba3.py`.
* **Dynamic chunking**: cosine-similarity router, straight-through boundaries, EMA dechunking, ratio loss with the
  ATDC target schedule; bounded routing as the serving arena's guardrail — `mote/model/dc.py`. The chunk rate is an
  observable of a trained model (about 3.3 bytes a chunk), not a setting.
* **Main network**: Full Relation — Self/Exchange relations, count calibration λ, Givens head mixing, a `{P2, Ĩ}`
  decode cache; the fused Triton kernel `FlashRelation` is exact to the reference — `mote/model/relation.py`,
  `mote/model/flash_relation.py`.
* **Tokenizer**: none. 256 byte values plus 15 protocol ids (chat roles, tool call/result, fill-in-the-middle, thinking,
  reversal and offset markers) — `mote/tokenizer.py`.

**Sizes** (`mote/config.py`, named by parameter count): Mote-1M, Mote-11M, Mote-32M, **Mote-96M** — the flagship, a
16384-byte window, frozen 2026-08-24 — and Mote-138M. Every number the studio shows is real: checkpoint metadata,
live tensors, run logs; undertrained checkpoints are labelled as such.

## What is where

* `mote/model` — the model. `mote/train` — the trainer (bf16 autocast, Muon + AdamW, WSD/trunk/branch schedules,
  ELR logging and a norm guard, atomic checkpoints, bitwise-reproducible by default), the post-training stages
  (`dpo`, `kto`, `rlvr`) and the lr-vs-horizon fit.
* `mote/infer` — inference: the generation engine, one CUDA graph per byte for decoding, the prefix store that
  keeps conversations warm, context folding. `mote/identity.py` — what the model is told about itself.
* `mote/serve` — the resident daemon: a FastAPI app, the training job queue that drives the trainer one accumulation
  slice at a time between replies, the preference store, device pairing. `mote/cli.py` — `mote …`.
* `mote/data` — shard builders (pretraining mixes, SFT, identity and spec documents, on-policy replay).
  `mote/sim` — a small ECS world simulator: the tool environment for RLVR and its expert traces. `mote/eval` —
  the probes and the branch gate.
* `web/` — the studio: Vite + Svelte 5 + TypeScript, built with Bun, served by the daemon at `/`.
* `docs/shape.md` — the design and its current state; `docs/api.md` — the HTTP/WebSocket contract;
  `docs/runbook.md` — running it; `docs/results/` and `docs/research/` — dated records and reading notes.

## Setup (Linux)

Python ≥ 3.11, a CUDA GPU for training (serving runs on the CPU too), [Bun](https://bun.com/install) for the studio.

```bash
uv venv .venv && uv pip install --python .venv/bin/python -e .
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu126   # or the wheel for your driver
# the Mamba-3 Triton kernels (optional; the pure-PyTorch paths are used without them):
git clone https://github.com/state-spaces/mamba.git ~/Development/mamba && git -C ~/Development/mamba checkout e9594ce
MAMBA_SKIP_CUDA_BUILD=TRUE uv pip install --python .venv/bin/python -e ~/Development/mamba --no-deps --no-build-isolation
uv pip install --python .venv/bin/python "triton>=3.5" einops
git config core.hooksPath .githooks        # refuses profiler blobs and credential-shaped strings
```

## Data

```bash
.venv/bin/python -m mote.data.build_mix   --out data/flagship_mix --list flagship --target-gb 10 --val-mb 128   # the trunk's mix A (mote/data/sources.py)
.venv/bin/python -m mote.data.build_sft   --out data/sft_mix --target-mb 300 --val-mb 8                           # chat SFT with loss masks
.venv/bin/python -m mote.data.build_local --out data/local_mix                                                       # the small local mix the arms use
```

Shards are uint16 ids (bytes + protocol ids) with BOS/EOS separators; SFT shards add a uint8 mask (1 on assistant
bytes). `data/` is gitignored.

## Train

```bash
.venv/bin/python -m mote.train.train --preset mote-32m --data data/local_mix --out runs/local --batch-size 4 \
    --grad-accum 4 --seq-len 2048 --optimizer muon --max-minutes 60
# SFT from a checkpoint:
.venv/bin/python -m mote.train.train --preset mote-32m --sft --init-from runs/local/last.pt --data data/sft_mix --out runs/local_sft --max-minutes 20
```

`log.jsonl` records train bits/byte, bytes per chunk, the ratio loss, throughput, TFLOPS/MFU, the parameter norms and
ELR, and periodic evals (val bits/byte, boundary/separator alignment, a chunked text sample). `last.pt` is written
atomically every `--ckpt-minutes`; `--resume` continues, clock included. With `--max-steps 0` (the default) the
schedules follow wall-clock progress toward `--max-minutes`. `python -m mote.train.train --help` lists every flag;
`python -m mote.train.profile_step` prints where a step's time goes.

On the resident daemon the same args go through the queue: `mote train start -- <args>` (see the runbook).

## Serve

```bash
mote service install     # once: the token, .mote/config.json, a systemd user unit; starts the studio
mote build               # after changes: svelte-check + build + tests + restart + the pairing link
mote status | logs | restart | pair
```

The studio is at `http://127.0.0.1:7861`; `docs/remote-access.md` is how a phone reaches it (Tailscale). The API
(`docs/api.md`) is `/api/model`, `/api/checkpoints` (+ hot-swap), `/api/training/*` (the queue), `/api/context`,
`/api/prefs/*`, `/v1/chat/completions` (OpenAI-compatible SSE) and `/ws/generate`, which streams every byte with its
probability, entropy, chunk boundary, live Mamba-3 retention and Relation exchange mass. A manual form:

```bash
.venv/bin/python -m mote.serve.app --checkpoint runs/local/last.pt --port 7861
```

`docs/prefs.md` is the preference loop (votes, marks, a challenger checkpoint for blind A/B, Claude as the second
rater under `docs/rubric.md`); `docs/context.md` is the prefix store and context folding; `docs/checkpoints.md` the
checkpoint picker; `docs/search.md` a designed, not built, web-search tool.

## Tests

```bash
.venv/bin/python -m pytest -q tests/          # the GPU tests skip without a card
```

## References

H-Net 2507.07955 · Mamba-3 2603.15569 · Relation 2608.20172 · ATDC 2605.30080 · Muon-SW 2607.23777 · ELR 2608.24814.

## License

Dual-licensed, at your option, under either

* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE), <http://www.apache.org/licenses/LICENSE-2.0>), or
* MIT license ([LICENSE-MIT](LICENSE-MIT), <http://opensource.org/licenses/MIT>).

Unless you state otherwise, any contribution you intentionally submit for inclusion in this work, as defined in
Apache-2.0, is dual-licensed on the same terms with no additional conditions.

The model architecture follows published work cited above; the implementation, the dynamic-chunking studio, the
Triton kernels and the training pipeline are original. Checkpoints and datasets are not covered by this license
and are not distributed here.
