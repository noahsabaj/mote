# Morpheme Studio — backend API contract (v1)

Served by FastAPI (`morpheme.serve.app`) on `http://127.0.0.1:7860`. The built frontend (`web/dist`)
is served at `/`. All values are real: they come from the loaded checkpoint, live tensors, or run logs.
There is no decorative data anywhere.

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
  "defaults": { "temperature": 0.8, "top_p": 0.9, "max_bytes": 512, "accept_threshold": 0.9, "n_candidates": 3 }
}
```

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
  "params": { "temperature": 0.8, "top_p": 0.9, "max_bytes": 512, "accept_threshold": 0.9, "n_candidates": 3 } }
{ "type": "stop" }
```

Server → client (in order)
```json
{ "type": "start", "prompt_bytes": 61, "context_bytes": 61, "context_limit": 2048, "truncated": false }

{ "type": "byte", "i": 0, "byte": 84, "text": "T", "pending": 0,
  "p": 0.42, "entropy": 2.31, "boundary": true, "boundary_p": 0.93, "chunk": 0,
  "source": "nbp", "t_ms": 18.4 }
// text: the completed UTF-8 character(s) this byte finished, or null while a multi-byte char is pending
// pending: bytes currently buffered in UTF-8 assembly
// source: "nbp" = sampled one byte at a time; "mbp" = accepted from the multi-byte head's parallel proposal
// boundary: the router opened a new chunk at this byte; chunk: index of the chunk this byte belongs to

{ "type": "chunk", "index": 3, "start": 14, "end": 19, "bytes": 6, "text": " tend" }   // a chunk closed

{ "type": "stats", "bytes": 64, "elapsed_ms": 1730, "bytes_per_sec": 37.0, "chunks": 11,
  "bytes_per_chunk": 5.8, "mbp_proposed": 30, "mbp_accepted": 12, "mbp_accept_rate": 0.40,
  "context_bytes": 125, "context_limit": 2048 }                                         // every 16 bytes

{ "type": "diagnostics",                                                                // at every chunk boundary
  "mamba3": { "encoder_retention": [0.91, 0.84, ...], "decoder_retention": [ ... ] },   // mean exp(A·Δt) per head, live
  "relation": { "exchange_mass": [0.34, 0.58, 0.71, 0.63, 0.80, 0.76] },               // g_i per layer for the newest chunk
  "boundary_probs": [0.02, 0.05, 0.93, ...] }                                           // last 64 bytes

{ "type": "done", "reason": "eos" | "max_bytes" | "stopped", "text": "...", "stats": { ...same as stats... } }
{ "type": "error", "message": "..." }
```
