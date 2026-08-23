# Morpheme Studio — backend API contract (v1)

Served by FastAPI (`morpheme.serve.app`) on `http://127.0.0.1:7860`. The built frontend (`web/dist`)
is served at `/`. All values are real: they come from the loaded checkpoint, live tensors, or run logs.
There is no decorative data anywhere.

## Authentication (optional)

When the server is started with `--token` (or `MORPHEME_TOKEN`), every `/api/*` and `/v1/*`
request needs `Authorization: Bearer <token>` (otherwise `401` with `WWW-Authenticate: Bearer`),
and `/ws/generate` expects a first frame `{"type": "auth", "token": "<token>"}` answered by
`{"type": "auth_ok"}` — a wrong or missing frame closes the socket with code `4401`. An auth frame is
answered with `auth_ok` even when no token is configured, so clients that hold one can always wait for it.
`/api/health`, `/api/pair` and the static frontend are always open. Pairing: `GET /pair` (loopback only)
shows a QR of `<public-url>/#token=<token>` plus a 6-digit code; `POST /api/pair {"code"}` returns
`{"token"}` for a valid, unexpired, unused code (400 otherwise, 429 after 10 attempts/min). The UI reads
`#token=` from the URL fragment on load and scrubs it. See `docs/remote-access.md`.

## HTTP

### `GET /api/health`
`{ "ok": true, "model_loaded": bool }`

### `GET /api/model`
```json
{
  "name": "pilot_1h/last.pt",
  "params": 12660000,
  "status": "pilot",                 // "pilot" | "undertrained" | "flagship" — honesty label, shown verbatim
  "status_note": "Trained 1 hour on 300 MB of FineWeb-Edu on an RTX 4060 Ti. Expect fluent nonsense.",
  "checkpoint": { "path": "runs/pilot_1h/last.pt", "step": 3100, "bytes_seen": 101580800,
                  "val_bpb": 1.63, "trained_minutes": 60.2, "created_at": "2026-08-22T14:05:11Z" },
  "architecture": { "outer_width": 256, "encoder_layers": 2, "decoder_layers": 2,
                    "main": "Relation 6L/384/8 heads", "mbp_layers": 2 },
  "context_limit_bytes": 2048,
  "device": { "name": "NVIDIA GeForce RTX 4060 Ti", "vram_total_mb": 8188, "vram_used_mb": 912 },
  "kernels": { "mamba3": true, "ssd": true },
  "defaults": { "temperature": 0.8, "top_p": 0.9, "max_bytes": 512, "n_candidates": 3 },
  "probe": { "identity_acc": 0.83, "hold_rate": 0.5, "concede_rate": 0.75, "n_identity": 6, "n_facts": 8,
             "identity_acc_seen": 1.0, "hold_rate_seen": 0.875, "concede_rate_seen": 0.75, "n_identity_seen": 6, "n_facts_seen": 8 },
  "identity_card": "You are Mote, a small byte-level language model ..."
}
```

### `POST /api/context`  body `{ "messages": [...], "max_bytes": 512, "fold": "auto", "card": null, "prev": null | { "from", "card" } }`
What the next prompt would look like, without generating — for the studio's meter and fold line:
`{ "used": 1830, "limit": 2048, "reserve": 512, "fold": null | { "from", "turns", "card" }, "truncated": false, "reusable": 1402 }`.
`prev` is the client's last fold, kept while the prompt still fits; `reusable` is how many bytes of that prompt the
engine's prefix cache already holds a state for (docs/context.md).

### `GET /api/checkpoints`
`[{ "id": "pilot_1h/last.pt", "step": 3100, "val_bpb": 1.63, "bytes_seen": 101580800, "created_at": "...", "loaded": true }, ...]`

### `POST /api/checkpoints/load`  body `{ "id": "pilot_1h/last.pt" }`
Hot-swaps the served model. Returns the new `/api/model` payload. Generation requests during a swap get `503`.

### `GET /api/training/runs`
`[{ "id": "pilot_1h", "steps": 3100, "last_val_bpb": 1.63, "running": false, "started_at": "..." }]`

### `GET /api/training/runs/{id}/log?since=<n>`
Returns JSONL records from `runs/{id}/log.jsonl` starting at line `n` (see `morpheme/train/train.py` for the
record shapes: train records carry `step, lr, target_ratio, bytes_per_sec, train_bpb, ce, ce_mbp, ratio, bpic,
grad_norm`; eval records carry `eval: { val_bpb, val_bpic, boundary_on_separator_frac, mbp_top1_acc, sample }`).
`{ "records": [...], "next": <n + len(records)> }`

### `POST /v1/chat/completions` — OpenAI-compatible
Body: `{ "messages": [{"role": "user"|"assistant"|"system", "content": "..."}], "temperature", "top_p",
"max_tokens" (bytes), "stream": true|false }`. Streaming replies are SSE `data: {chat.completion.chunk}` lines
ending with `data: [DONE]`. Provided so any OpenAI client works; the studio itself uses the WebSocket below.

## WebSocket `/ws/generate`

One generation per socket message. The client keeps the conversation and sends the full message list every
turn (the server re-prefills; there is no server-side conversation store in v1).

Client → server
```json
{ "type": "generate",
  "messages": [{"role": "user", "content": "What does dynamic chunking do?"}],
  "params": { "temperature": 0.8, "top_p": 0.9, "max_bytes": 512, "n_candidates": 3 },
  "context": { "fold": "auto" | "now" | "off", "card": null | "<the user's edited compaction card>",
               "prev": null | { "from": 6, "card": "..." },      // the last reply's fold, kept while it fits
               "verify_prefix": false } }                           // optional; verify: cold re-read + prefix_check
{ "type": "stop" }
```

Server → client (in order)
```json
{ "type": "start", "prompt_bytes": 61, "context_bytes": 61, "context_limit": 2048, "truncated": false,
  "fold": null | { "from": 6, "turns": 6, "card": "(Earlier in this conversation, 6 turns folded. ...)" },
  "prefix": { "reused": 1402, "prefilled": 38, "prefill_ms": 310, "snapshots": 5, "cache_bytes": 52428800,
              "cache_budget": 1073741824, "hits": 12, "misses": 2 } }
  // fold: the first `from` non-system turns were folded into `card`, which rides inside the first kept user
  // turn (docs/context.md); `truncated` is now only true when even folding could not fit (a giant message).
  // prefix: bytes of the prompt taken from the engine's prefix cache vs read afresh (docs/context.md).
  // With context.verify_prefix, one diagnostics event follows immediately with
  //   "prefix_check": { "reused", "prefilled", "boundary_flips", "chunks_cold", "chunks_warm", "max_logit_diff", "cold_ms" }

{ "type": "byte", "i": 0, "byte": 84, "text": "T", "pending": 0,
  "p": 0.42, "entropy": 2.31, "boundary": true, "boundary_p": 0.93, "chunk": 0,
  "source": "nbp", "t_ms": 18.4 }

`source` is `nbp` (sampled from the next-byte head), `mbp` (a byte drafted by the multi-byte head and accepted by
exact speculative verification — Leviathan/Chen rejection sampling against the next-byte head's distribution, with
temperature and top-p applied to both), or `fix` (the correction drawn when a draft byte was rejected). The byte
stream is distributed exactly as plain sampling would be; `n_candidates` is the draft length (0 disables). The engine
measures bytes/s of speculative rounds against plain steps within each reply and pauses drafting (`spec_paused`, plus a
`diagnostics.note`) once it is slower — so a weak drafter can never slow a reply below plain decoding.
// text: the completed UTF-8 character(s) this byte finished, or null while a multi-byte char is pending
// pending: bytes currently buffered in UTF-8 assembly
// source: "nbp" = sampled one byte at a time; "mbp" = accepted from the multi-byte head's parallel proposal
// boundary: the router opened a new chunk at this byte; chunk: index of the chunk this byte belongs to

{ "type": "chunk", "index": 3, "start": 14, "end": 19, "bytes": 6, "partial": false, "text": " tend" }   // a chunk closed
// start/end are reply-local byte indices (same space as byte.i), inclusive; bytes counts the whole chunk.
// partial: the chunk began inside the prompt (then start is clamped to 0 and bytes > end - start + 1).

{ "type": "stats", "bytes": 64, "elapsed_ms": 1730, "bytes_per_sec": 37.0, "chunks": 11,
  "bytes_per_chunk": 5.8, "mbp_proposed": 30, "mbp_accepted": 12, "mbp_accept_rate": 0.40, "spec_rounds": 10, "spec_fixes": 7, "spec_replays": 3, "spec_paused": false,
  "context_bytes": 125, "context_limit": 2048 }                                         // every 16 bytes

{ "type": "diagnostics",                                                                // at every chunk boundary
  "mamba3": { "encoder_retention": [0.91, 0.84, ...], "decoder_retention": [ ... ] },   // mean exp(A·Δt) per head, live
  "relation": { "exchange_mass": [0.34, 0.58, 0.71, 0.63, 0.80, 0.76] },               // g_i per layer for the newest chunk
  "boundary_probs": [0.02, 0.05, 0.93, ...] }                                           // last 64 bytes

{ "type": "done", "reason": "eos" | "max_bytes" | "stopped", "text": "...", "stats": { ...same as stats... } }
{ "type": "error", "message": "..." }
```
