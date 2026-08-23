"""FastAPI server for Morpheme Studio — implements docs/api.md and serves web/dist.

    python -m morpheme.serve.app --checkpoint runs/pilot_1h/last.pt --port 7860
"""

from __future__ import annotations

import argparse
import asyncio
import json
import mimetypes
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .engine import Engine, GenParams, discover_checkpoints
from .pairing import Pairing, router as pairing_router

ROOT = Path(__file__).resolve().parents[2]
mimetypes.add_type("application/manifest+json", ".webmanifest")
app = FastAPI(title="Morpheme Studio")
STATE: dict = {"engine": None, "swapping": False, "lock": threading.Lock(), "token": None}

# --- access token -----------------------------------------------------------------
# Optional shared secret (`--token` / MORPHEME_TOKEN). When set, every /api and /v1 route
# needs `Authorization: Bearer <token>` and /ws/generate needs a first frame
# {"type": "auth", "token": ...}. /api/health and the static frontend stay open so the
# UI can load and ask for the token. Binding to anything but loopback refuses to start
# without a token unless --no-auth is given explicitly.
PROTECTED_PREFIXES = ("/api/", "/v1/")
OPEN_PATHS = {"/api/health", "/api/pair"}  # /api/pair is how a device obtains the token
PAIRING = Pairing(STATE)
app.include_router(pairing_router(PAIRING))


def token_ok(presented) -> bool:
    tok = STATE["token"]
    if not tok:
        return True
    return isinstance(presented, str) and secrets.compare_digest(presented.encode(), tok.encode())


@app.middleware("http")
async def require_token(request: Request, call_next):
    path = request.url.path
    if STATE["token"] and path.startswith(PROTECTED_PREFIXES) and path not in OPEN_PATHS:
        auth = request.headers.get("authorization", "")
        presented = auth[7:] if auth[:7].lower() == "bearer " else None
        if not token_ok(presented):
            return JSONResponse({"detail": "access token required"}, status_code=401,
                                headers={"WWW-Authenticate": "Bearer"})
    return await call_next(request)


def engine() -> Engine:
    e = STATE["engine"]
    if e is None or STATE["swapping"]:
        raise HTTPException(status_code=503, detail="model not available (loading)")
    return e


def ckpt_id(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(p)


# --- HTTP --------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"ok": True, "model_loaded": STATE["engine"] is not None and not STATE["swapping"]}


@app.get("/api/model")
def model_info():
    return engine().info()


@app.get("/api/checkpoints")
def checkpoints():
    cur = STATE["engine"].ckpt_path.resolve() if STATE["engine"] else None
    out = []
    for p in discover_checkpoints(ROOT):
        try:
            ck = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
            step = int(ck.get("step", 0))
        except Exception:
            step = -1
        info = STATE["engine"]._describe_checkpoint(p, step, {}) if STATE["engine"] else None
        out.append({
            "id": ckpt_id(p), "step": step,
            "val_bpb": info.val_bpb if info else None, "bytes_seen": info.bytes_seen if info else 0,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(p.stat().st_mtime)),
            "loaded": cur is not None and p.resolve() == cur,
        })
    return out


class LoadBody(BaseModel):
    id: str


@app.post("/api/checkpoints/load")
def load_checkpoint(body: LoadBody):
    path = (ROOT / body.id).resolve()
    if not path.exists():
        raise HTTPException(404, "checkpoint not found")
    with STATE["lock"]:
        STATE["swapping"] = True
        try:
            old = STATE["engine"]
            del old
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            eng = Engine(path, device=STATE.get("device"))
            eng.warmup()
            STATE["engine"] = eng
        finally:
            STATE["swapping"] = False
    return STATE["engine"].info()


@app.get("/api/training/runs")
def training_runs():
    runs = []
    for d in sorted((ROOT / "runs").glob("*")):
        log = d / "log.jsonl"
        if not log.exists():
            continue
        steps, last_bpb, running = 0, None, False
        try:
            lines = log.read_text(encoding="utf-8").splitlines()
            for line in lines:
                rec = json.loads(line)
                steps = max(steps, int(rec.get("step", 0)))
                if "eval" in rec:
                    last_bpb = rec["eval"].get("val_bpb", last_bpb)
            running = (time.time() - log.stat().st_mtime) < 120 and not any('"done": true' in l for l in lines[-3:])
        except Exception:
            pass
        runs.append({"id": d.name, "steps": steps, "last_val_bpb": last_bpb, "running": running,
                     "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(log.stat().st_ctime))})
    return runs


@app.get("/api/training/runs/{run_id}/log")
def training_log(run_id: str, since: int = 0):
    log = ROOT / "runs" / run_id / "log.jsonl"
    if not log.exists():
        raise HTTPException(404, "run not found")
    lines = log.read_text(encoding="utf-8").splitlines()
    records = []
    for line in lines[since:]:
        try:
            records.append(json.loads(line))
        except Exception:
            continue
    return {"records": records, "next": since + len(lines[since:])}


# --- OpenAI-compatible ------------------------------------------------------------
class ChatBody(BaseModel):
    messages: list
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = False
    model: Optional[str] = None


def _run_generation(eng: Engine, messages, params: GenParams, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue, stop: threading.Event):
    def emit(ev: dict):
        loop.call_soon_threadsafe(queue.put_nowait, ev)

    try:
        eng.generate(messages, params, emit, stop)
    except Exception as e:  # surface engine errors to the client
        emit({"type": "error", "message": f"{type(e).__name__}: {e}"})
    finally:
        emit({"type": "__end__"})


@app.post("/v1/chat/completions")
async def chat_completions(body: ChatBody):
    eng = engine()
    overrides = {k: v for k, v in {"temperature": body.temperature, "top_p": body.top_p, "max_bytes": body.max_tokens}.items() if v is not None}
    params = GenParams(**{**vars(eng.defaults), **overrides})
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    stop = threading.Event()
    threading.Thread(target=_run_generation, args=(eng, body.messages, params, loop, queue, stop), daemon=True).start()
    cid = f"chatcmpl-{int(time.time()*1000)}"
    model_name = eng.info()["name"]

    if not body.stream:
        text, reason = "", "stop"
        while True:
            ev = await queue.get()
            if ev["type"] == "__end__":
                break
            if ev["type"] == "done":
                text, reason = ev["text"], "stop" if ev["reason"] == "eos" else "length"
            if ev["type"] == "error":
                raise HTTPException(500, ev["message"])
        return {"id": cid, "object": "chat.completion", "created": int(time.time()), "model": model_name,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": reason}]}

    async def sse():
        while True:
            ev = await queue.get()
            if ev["type"] == "__end__":
                break
            if ev["type"] == "byte" and ev.get("text"):
                chunk = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()), "model": model_name,
                         "choices": [{"index": 0, "delta": {"content": ev["text"]}, "finish_reason": None}]}
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            elif ev["type"] == "done":
                chunk = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()), "model": model_name,
                         "choices": [{"index": 0, "delta": {}, "finish_reason": "stop" if ev["reason"] == "eos" else "length"}]}
                yield f"data: {json.dumps(chunk)}\n\n"
            elif ev["type"] == "error":
                yield f"data: {json.dumps({'error': ev['message']})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")


# --- WebSocket ----------------------------------------------------------------------
@app.websocket("/ws/generate")
async def ws_generate(ws: WebSocket):
    await ws.accept()
    if STATE["token"]:
        try:
            first = await asyncio.wait_for(ws.receive_json(), timeout=15)
        except Exception:
            await ws.close(code=4401)
            return
        if not isinstance(first, dict) or first.get("type") != "auth" or not token_ok(first.get("token")):
            await ws.close(code=4401)
            return
        await ws.send_json({"type": "auth_ok"})
    loop = asyncio.get_running_loop()
    stop = threading.Event()
    try:
        while True:
            msg = await ws.receive_json()
            if msg.get("type") == "auth":
                # a client that holds a token always sends this first and waits for the reply
                await ws.send_json({"type": "auth_ok"})
                continue
            if msg.get("type") != "generate":
                continue
            try:
                eng = engine()
            except HTTPException as e:
                await ws.send_json({"type": "error", "message": e.detail})
                continue
            params = GenParams.from_dict(msg.get("params"), eng.defaults)
            queue: asyncio.Queue = asyncio.Queue()
            stop.clear()
            threading.Thread(target=_run_generation, args=(eng, msg.get("messages", []), params, loop, queue, stop), daemon=True).start()

            async def pump_client():
                # listen for a stop message while generating
                try:
                    while True:
                        m = await ws.receive_json()
                        if m.get("type") == "stop":
                            stop.set()
                except Exception:
                    stop.set()

            listener = asyncio.create_task(pump_client())
            try:
                while True:
                    ev = await queue.get()
                    if ev["type"] == "__end__":
                        break
                    await ws.send_json(ev)
            finally:
                listener.cancel()
    except WebSocketDisconnect:
        stop.set()


# --- static frontend --------------------------------------------------------------
DIST = ROOT / "web" / "dist"
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        f = DIST / path
        if path and f.is_file():
            return FileResponse(f)
        return FileResponse(DIST / "index.html")
else:

    @app.get("/")
    def no_frontend():
        return JSONResponse({"message": "frontend not built: run `npm run build` in web/", "api": "/api/model"})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None, help="default: newest checkpoint under runs/ or checkpoints/")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--device", default=None, help="cuda (default when available) or cpu — e.g. to leave the GPU to a training run")
    ap.add_argument("--token", default=os.environ.get("MORPHEME_TOKEN") or None,
                    help="shared access token (or MORPHEME_TOKEN); required unless bound to loopback or --no-auth")
    ap.add_argument("--no-auth", action="store_true", help="serve without a token on a non-loopback host (not recommended)")
    ap.add_argument("--public-url", default=os.environ.get("MORPHEME_URL") or None,
                    help="address phones use (default: detected from `tailscale serve status`); encoded in the /pair QR")
    args = ap.parse_args(argv)
    loopback = args.host in ("127.0.0.1", "localhost", "::1")
    if not loopback and not args.token and not args.no_auth:
        raise SystemExit(f"refusing to bind {args.host} without a token: pass --token (or MORPHEME_TOKEN), or --no-auth to override")
    STATE["token"] = args.token or None
    PAIRING.public_url = args.public_url
    if STATE["token"]:
        print(f"pair a device: open http://127.0.0.1:{args.port}/pair on this machine", flush=True)
    print("access token: " + ("required" if STATE["token"] else "none" + ("" if loopback else " (--no-auth)")), flush=True)
    ck = Path(args.checkpoint) if args.checkpoint else (discover_checkpoints(ROOT) or [None])[0]
    if ck is None:
        raise SystemExit("no checkpoint found; train one first or pass --checkpoint")
    print(f"loading {ck} ...", flush=True)
    STATE["device"] = args.device
    eng = Engine(ck, device=args.device)
    print(f"warming up kernels ... ({eng.warmup():.1f} s)", flush=True)
    STATE["engine"] = eng
    print(json.dumps(eng.info(), indent=1), flush=True)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
