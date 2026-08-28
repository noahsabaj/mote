# Mote Studio — backend API contract (v1)

Served by FastAPI (`mote.serve.app`) on `http://127.0.0.1:7860`. The built frontend (`web/dist`)
is served at `/`. All values are real: they come from the loaded checkpoint, live tensors, or run logs.
There is no decorative data anywhere.

## Authentication (optional)

When the server is started with `--token` (or `MOTE_TOKEN`), every `/api/*` and `/v1/*`
request needs `Authorization: Bearer <token>` (otherwise `401` with `WWW-Authenticate: Bearer`),
and `/ws/generate` expects a first frame `{"type": "auth", "token": "<token>"}` answered by
`{"type": "auth_ok"}` — a wrong or missing frame closes the socket with code `4401`. An auth frame is
answered with `auth_ok` even when no token is configured, so clients that hold one can always wait for it.
`/api/health`, `/api/pair` and the static frontend are always open. Pairing: `GET /pair` (loopback only)
shows a QR of `<public-url>/#token=<token>` plus a 6-digit code; `POST /api/pair {"code"}` returns
`{"token"}` for a valid, unexpired, unused code (codes live 10 minutes; 400 otherwise, 429 after 10 attempts/min). The UI reads
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
  "identity_card": "You are Mote, a small byte-level language model ...",
  "live": null | "t3l_dense_8e-4/ema@12508",   // a job on the air: its EMA is answering
  "pin": "runs/overnight_sft/last.pt",          // the boot default in .mote/config.json (null if none)
  "serving_device": "cuda" | "cpu",             // cpu while any training job runs, the configured device when the queue idles
  "following": null | "runs/trunk"              // the run on the air (the --serve job), else null
}
```

Serving policy (signed 2026-08-25): what answers chats is the **pin** unless a job is **on the air** — a job submitted
with `--serve` (or switched on with `/api/training/serve`) whose EMA answers while it runs and whose final checkpoint
becomes the pin when it finishes. Every other job (a screening arm) leaves the served model alone. Serving lives on
the CPU while any job runs (the GPU is training's) and moves back to the GPU with its arena + graphs when the queue
idles; the CPU path is ~45 bytes/s and a cold read of a long prompt takes seconds (the `waiting` frame below).

### `POST /api/context`  body `{ "messages": [...], "max_bytes": 512, "fold": "auto", "card": null, "prev": null | { "from", "card" } }`
What the next prompt would look like, without generating — for the studio's meter and fold line:
`{ "used": 1830, "limit": 2048, "reserve": 512, "fold": null | { "from", "turns", "card" }, "truncated": false, "reusable": 1402 }`.
`prev` is the client's last fold, kept while the prompt still fits; `reusable` is how many bytes of that prompt the
engine's prefix cache already holds a state for (docs/context.md).

### `POST /api/challenger/load`  body `{ "id": "runs/overnight_sft2/last.pt" }` · `DELETE /api/challenger`
Loads (or drops) a second engine beside the served one for blind side-by-side comparisons (docs/prefs.md);
`/api/model.challenger` is `null | { "id", "name", "step", "val_bpb", "loading" }` and `/api/checkpoints` rows carry
`"challenger": bool`. The WebSocket `generate` frame takes `"engine": "current" | "challenger"`.

### `POST /api/prefs/vote`  body `{ "pair": { "messages", "a", "b", "a_source", "b_source", "origin" }, "vote": "a" | "b" | "tie" | "both_bad" | null, "reason": "" }`
Stores a pair of replies and your verdict (`null` keeps the pair unrated); a source is `{ "checkpoint", "step", "engine",
"params" }`. Returns the summary below plus `"pair": "<id>"`. Identical replies are refused (400).

### `GET /api/prefs/summary` · `GET /api/prefs/rubric`
`{ "pairs", "votes": { "user", "claude" }, "unrated_by_claude", "table": [{ "a", "b", "a_wins", "b_wins", "ties", "both_bad", "n" }],
"agreement": { "n", "agree", "rate" }, "rubric": "<hash>" }` and `{ "text", "hash" }` (docs/rubric.md).

### `GET /api/checkpoints`
`[{ "id": "pilot_1h/last.pt", "step": 3100, "val_bpb": 1.63, "bytes_seen": 101580800, "file_size_bytes": 425721856,
"created_at": "...", "loaded": true, "challenger": false }, ...]`

`bytes_seen` is training data consumed (`step × batch × seq × grad_accum`); `file_size_bytes` is the `.pt` on
disk. They differ by an order of magnitude and the studio labels both, since one row showing a bare "63 MB"
was read as the file. The file-derived half of each row is cached on the checkpoint's mtime and size and its
run log's mtime — the log is what gains `val_bpb` while a run is still going.

### `POST /api/checkpoints/load`  body `{ "id": "pilot_1h/last.pt" }`
Hot-swaps the served model and makes it the pin. Returns the new `/api/model` payload; if a job was on the air it is
taken off for the rest of its life (the pick wins) and the payload carries `"unfollowed": "runs/<run>"` once.
Generation requests during a swap get `503`.

### `POST /api/training/start`  body `{ "args": ["--preset", "local", "--data", "data/local_mix", "--out", "runs/x", ...], "front": false, "serve": false }`
Enqueue a training job on the resident daemon (docs/shape.md) — args exactly as `python -m mote.train.train`
takes them; malformed args are rejected at submit time (400). `front` puts it ahead of everything queued; `serve`
puts it on the air (above). `POST /api/training/stop` body `{ "id": null }`
stops the running job at its next step boundary (final eval + checkpoint still happen) or cancels a queued one
by id. `POST /api/training/serve` body `{ "id": null, "on": true }` puts the running (`id: null`) or a queued job on
the air, or takes it off. `GET /api/training/queue` → `{ "current", "phase", "queued": [...], "recent": [...], "halted", "paused" }` —
`phase` is what the running job is doing (`"train"`, `"eval 3/16"`, `"eval ema 3/16"`, `"checkpoint"`) and each
job carries `"serve": bool`. Training yields to each reply at the next accumulation slice (or eval window).
Each queued job carries `"waiting": null | "retry in 540 s" | "needs 6.50 GB, 5.80 usable"` — why it is not
starting; `"deaths"` counts process deaths before its first logged step (two hold the job and halt the queue,
`POST /api/training/release` clears the halt); `"paused"` is the daemon's reason for starting nothing at all
(launched `--device cuda` with no CUDA: it serves on the CPU and restarts itself when the device appears).

### `POST /api/engine/device`  body `{ "device": "cpu" | "cuda" }`
Park the studio's engine on the CPU — the move the queue makes around a job, on request — so a standalone
measurement can have the whole card while the studio stays up; `"cuda"` brings it back with its arena and
graphs. Parked stays parked across idle signals. 409 while a job owns the GPU or when CUDA is missing.
`mote engine park` / `mote engine restore`.

### `GET /api/training/runs`
`[{ "id": "pilot_1h", "steps": 3100, "last_val_bpb": 1.63, "running": false, "started_at": "..." }]`

### `GET /api/training/runs/{id}/log?since=<n>`
Returns JSONL records from `runs/{id}/log.jsonl` starting at line `n` (see `mote/train/train.py` for the
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
{ "type": "waiting", "on": "prefill" | "swap", "bytes": 4711 }   // optional, before start: the reply is queued behind
  // a cold read of a long prompt on the CPU (bytes = what has to be read) or a weight swap; the studio shows the
  // reason once it has lasted ~3 s.
{ "type": "start", "prompt_bytes": 61, "context_bytes": 61, "context_limit": 2048, "truncated": false,
  "fold": null | { "from": 6, "turns": 6, "card": "(Earlier in this conversation, 6 turns folded. ...)" },
  "prefix": { "reused": 1402, "prefilled": 38, "prefill_ms": 310, "snapshots": 5, "cache_bytes": 52428800,
              "cache_budget": 1073741824, "hits": 12, "misses": 2 },
  "checkpoint": { "name": "overnight_sft/last.pt", "step": 3666 } }
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
