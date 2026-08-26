"""FastAPI server for Mote Studio — implements docs/api.md and serves web/dist.

    python -m mote.serve.app --checkpoint runs/pilot_1h/last.pt --port 7860
"""

from __future__ import annotations

import argparse
import asyncio
import json
import mimetypes
import os
import secrets
import threading
import traceback
import time
from pathlib import Path
from typing import Optional

# The caching allocator grows segments instead of fragmenting them (native Linux; signed 2026-08-24):
# must be set before torch initialises CUDA. The supervisor sets it too; this covers a direct start.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .context import context_report
from .identity import with_system_card
from .engine import Engine, GenParams, discover_checkpoints
from .prefs import PrefStore, rubric as rubric_info
from .jobs import JobQueue
from .pairing import Pairing, router as pairing_router

ROOT = Path(__file__).resolve().parents[2]
mimetypes.add_type("application/manifest+json", ".webmanifest")
app = FastAPI(title="Mote Studio")
STATE: dict = {"engine": None, "swapping": False, "lock": threading.Lock(), "token": None,
               "challenger": None, "challenger_loading": False,  # challenger: a second Engine for blind A/B (docs/prefs.md)
               "jobs": None, "gate": None,  # the training job queue and the GPU gate it shares with serving (docs/shape.md)
               "config_file": ROOT / ".mote" / "config.json"}  # where the pin (the boot checkpoint) lives


def _serve_sync(cfg_dict: dict, state_dict, name: str, step: int) -> None:
    """A running job's EMA becomes the served weights (called under the GPU gate). The swap drops every
    anchor (they were computed under the old weights), so the recent conversations are re-read right
    away — the next message finds them warm (decided 2026-08-24)."""
    e = STATE["engine"]
    if e is not None and not STATE["swapping"]:
        e.apply_run_weights(cfg_dict, state_dict, name, step)
        try:
            rep = e.rewarm()
            if rep["branches"]:
                print(f"swap {name}@{step}: re-warmed {rep['branches']} conversation(s), {rep['bytes']} B in {rep['ms']:.0f} ms", flush=True)
        except Exception as ex:  # a cold next turn is the only cost
            print(f"rewarm after swap failed: {ex!r}", flush=True)


def _serve_device() -> str:
    """Where the studio's engine belongs right now: the CPU while any job runs or is about to, the configured
    device (the GPU) when the queue idles (signed 2026-08-25: training gets the whole card, replies never wait
    for it; ~45 B/s on the CPU vs 85–190 on the GPU, a cold 4.6 KB read is 10 s)."""
    jobs = STATE["jobs"]
    busy = jobs is not None and (jobs.current() is not None or jobs.has_runnable())
    return "cpu" if busy else (STATE.get("device") or "cpu")


def _move_engine(device: str, warm: bool) -> None:
    e = STATE["engine"]
    if e is None or e.device.type == torch.device(device).type:
        return
    with STATE["lock"]:
        STATE["swapping"] = True
        try:
            moved = e.moved(device)
            STATE["engine"] = moved
            if warm:
                print(f"serving on {device}: warm-up {moved.warmup():.1f} s", flush=True)
            else:
                print(f"serving on {device}", flush=True)
        except Exception as ex:
            print(f"moving the engine to {device} failed: {ex!r}", flush=True)
        finally:
            STATE["swapping"] = False


def _job_started(rec) -> None:
    """A job owns the GPU: serving moves to the CPU (same weights, fresh prefix store)."""
    _move_engine("cpu", warm=False)


def _queue_idle() -> None:
    """Nothing runnable is queued: serving comes back to the GPU with its arena + graphs (one warm-up)."""
    _move_engine(STATE.get("device") or "cpu", warm=True)


def _pin_path() -> Optional[str]:
    try:
        return json.loads(Path(STATE["config_file"]).read_text(encoding="utf-8")).get("checkpoint")
    except Exception:
        return None


def _write_pin(path: Path) -> None:
    """The pin is the boot default in .mote/config.json: a manual load or a finished --serve job sets it."""
    f = Path(STATE["config_file"])
    try:
        cfg = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    except Exception:
        cfg = {}
    cfg["checkpoint"] = ckpt_id(path)
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    tmp.replace(f)


def _load_engine(path: Path) -> Engine:
    """A fresh engine for `path` on the device serving belongs on right now; warmed only on the GPU."""
    device = _serve_device()
    eng = Engine(path, device=device)
    eng.gpu_gate = STATE["gate"]
    if eng.device.type == "cuda":
        eng.warmup()
    return eng


def _job_finished(rec) -> None:
    """A finished job that was on the air pins its final checkpoint and serves it; anything else leaves the
    served model alone (a queue of screening arms used to replace the chat model every time one ended)."""
    out = rec.out_dir
    path = (ROOT / out / "last.pt") if out else None
    if rec.state != "done" or not rec.serve or path is None or not path.exists():
        return
    with STATE["lock"]:
        STATE["swapping"] = True
        try:
            STATE["engine"] = _load_engine(path)
            _write_pin(path)
            print(f"job {rec.id} finished on the air: pinned and serving {ckpt_id(path)}", flush=True)
        except Exception as ex:
            print(f"serving the finished job's checkpoint failed: {ex!r}", flush=True)
        finally:
            STATE["swapping"] = False
PREFS = PrefStore()

# --- access token -----------------------------------------------------------------
# Optional shared secret (`--token` / MOTE_TOKEN). When set, every /api and /v1 route
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


def engine_for(role: Optional[str]) -> Engine:
    """'current' (default) or 'challenger' — the second engine the studio compares against."""
    if role == "challenger":
        ch = STATE["challenger"]
        if ch is None or STATE["challenger_loading"]:
            raise HTTPException(status_code=503, detail="no challenger loaded")
        return ch
    return engine()


def challenger_info() -> Optional[dict]:
    ch = STATE["challenger"]
    if ch is None:
        return None
    c = ch.info_ckpt
    return {"id": ckpt_id(ch.ckpt_path), "name": ch.ckpt_name, "step": c.step, "val_bpb": c.val_bpb,
            "loading": STATE["challenger_loading"]}


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
    e = engine()
    info = e.info()
    info["challenger"] = challenger_info()
    # what is answering and why (signed 2026-08-25): the pin, the device, and the job on the air if any
    info["pin"] = _pin_path()
    info["serving_device"] = e.device.type
    jobs = STATE["jobs"]
    cur = jobs.current() if jobs is not None else None
    info["following"] = (cur.out_dir or cur.id) if cur is not None and cur.serve else None
    return info


# Describing a checkpoint costs a torch.load and a full read of its run log, and there are
# dozens of them; the studio's checkpoint sheet is opened often enough that paying that on
# every request is the wrong default. Keyed on everything that could change the answer: the
# file's own mtime and size, and the run log's mtime, since `val_bpb` appears in the log
# while a run is still going and must not stay frozen at null behind an unchanged .pt.
_CKPT_ROWS: dict = {}


def _checkpoint_row(p: Path) -> dict:
    """The file-derived half of a checkpoint's listing entry. Loaded/challenger are state, not file."""
    st = p.stat()
    log = p.parent / "log.jsonl"
    key = (st.st_mtime, st.st_size, log.stat().st_mtime if log.exists() else 0.0)
    hit = _CKPT_ROWS.get(str(p))
    if hit and hit[0] == key:
        return hit[1]
    try:
        ck = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
        step = int(ck.get("step", 0))
    except Exception:
        step = -1
    info = STATE["engine"]._describe_checkpoint(p, step, {}) if STATE["engine"] else None
    row = {
        "id": ckpt_id(p), "step": step,
        "val_bpb": info.val_bpb if info else None, "bytes_seen": info.bytes_seen if info else 0,
        "file_size_bytes": st.st_size,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(st.st_mtime)),
    }
    # Without an engine there is nothing to describe with, and caching that would keep every
    # row at null until the file itself changed.
    if info is not None:
        _CKPT_ROWS[str(p)] = (key, row)
    return row


@app.get("/api/checkpoints")
def checkpoints():
    cur = STATE["engine"].ckpt_path.resolve() if STATE["engine"] else None
    ch = STATE["challenger"].ckpt_path.resolve() if STATE["challenger"] else None
    out = []
    for p in discover_checkpoints(ROOT):
        out.append({
            **_checkpoint_row(p),
            "loaded": cur is not None and p.resolve() == cur,
            "challenger": ch is not None and p.resolve() == ch,
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
            STATE["engine"] = None
            del old
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            STATE["engine"] = _load_engine(path)
            _write_pin(path)  # the pick is the new pin
        finally:
            STATE["swapping"] = False
    # The pick wins: a job that was on the air stops following for the rest of its life (signed 2026-08-25).
    jobs = STATE["jobs"]
    unfollowed = None
    if jobs is not None:
        cur = jobs.current()
        if cur is not None and cur.serve:
            jobs.set_serve(cur.id, False)
            unfollowed = cur.out_dir or cur.id
            print(f"manual load of {body.id}: no longer following {unfollowed}", flush=True)
    info = model_info()
    if unfollowed:
        info["unfollowed"] = unfollowed
    return info


class TrainStartBody(BaseModel):
    args: list
    front: bool = False  # ahead of everything queued
    serve: bool = False  # on the air: EMA while it runs, final checkpoint pinned when it ends


class TrainStopBody(BaseModel):
    id: Optional[str] = None


class TrainServeBody(BaseModel):
    id: Optional[str] = None  # None = the running job
    on: bool = True


@app.post("/api/training/serve")
def training_serve(body: TrainServeBody):
    """Put a running or queued job on the air, or take it off ("Put on the air" in the Training sheet)."""
    jobs = STATE["jobs"]
    if jobs is None:
        raise HTTPException(503, "job queue not running")
    rec = jobs.set_serve(body.id, body.on)
    if rec is None:
        raise HTTPException(404, "no such running or queued job")
    return jobs.status()


@app.post("/api/training/start")
def training_start(body: TrainStartBody):
    """Enqueue a training job (docs/shape.md): args exactly as `python -m mote.train.train` takes them."""
    jobs = STATE["jobs"]
    if jobs is None:
        raise HTTPException(503, "job queue not running")
    try:
        rec = jobs.submit([str(a) for a in body.args], front=body.front, serve=body.serve)
    except SystemExit:
        raise HTTPException(400, "bad training args (see `python -m mote.train.train --help`)")
    return {"submitted": rec.id, **jobs.status()}


@app.post("/api/training/stop")
def training_stop(body: TrainStopBody):
    jobs = STATE["jobs"]
    if jobs is None:
        raise HTTPException(503, "job queue not running")
    rec = jobs.cancel(body.id)
    if rec is None:
        raise HTTPException(404, "no such job (or nothing running)")
    return jobs.status()


@app.post("/api/training/release")
def training_release():
    """Clear a norm-guard halt. The queue stops dead when a run's parameter norm collapses (a checkpoint
    nothing downstream should build on), and only a person restarts it."""
    jobs = STATE["jobs"]
    if jobs is None:
        raise HTTPException(503, "job queue not running")
    was = jobs.release()
    if was is None:
        raise HTTPException(409, "the queue is not halted")
    return {"released": was, **jobs.status()}


@app.get("/api/training/queue")
def training_queue():
    jobs = STATE["jobs"]
    return jobs.status() if jobs is not None else {"current": None, "queued": [], "recent": [], "halted": None}


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


@app.post("/api/challenger/load")
def load_challenger(body: LoadBody):
    """Load a second engine next to the served one, for blind side-by-side comparisons (docs/prefs.md)."""
    path = (ROOT / body.id).resolve()
    if not path.exists():
        raise HTTPException(404, "checkpoint not found")
    with STATE["lock"]:
        STATE["challenger_loading"] = True
        try:
            STATE["challenger"] = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            eng = Engine(path, device=STATE.get("device"))
            eng.warmup()
            STATE["challenger"] = eng
        finally:
            STATE["challenger_loading"] = False
    return model_info()


@app.delete("/api/challenger")
def drop_challenger():
    with STATE["lock"]:
        STATE["challenger"] = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return model_info()


# --- preference votes (docs/prefs.md) ---------------------------------------------------
class PairBody(BaseModel):
    messages: list
    a: str
    b: str
    a_source: dict
    b_source: dict
    origin: str = "compare"  # retry | compare | arena


class VoteBody(BaseModel):
    pair: PairBody
    vote: Optional[str] = None  # a | b | tie | both_bad; None records the pair unrated (skipped)
    reason: str = ""


@app.post("/api/prefs/vote")
def prefs_vote(body: VoteBody):
    try:
        rec = PREFS.add_pair(body.pair.messages, body.pair.a, body.pair.b, body.pair.a_source, body.pair.b_source, body.pair.origin)
        if body.vote:
            PREFS.add_vote(rec["id"], "user", body.vote, body.reason)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"pair": rec["id"], **PREFS.summary()}


class MarkBody(BaseModel):
    messages: list
    reply: str
    source: dict
    mark: str          # up | down
    reason: str = ""


@app.post("/api/prefs/mark")
def prefs_mark(body: MarkBody):
    """One thumb on one reply — the collection path that does not need a second generation or a comparison.
    KTO (mote.train.kto) trains on these directly; docs/prefs.md."""
    try:
        rec = PREFS.add_mark(body.messages, body.reply, body.source, body.mark, "user", body.reason)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"mark": rec["id"], **PREFS.summary()}


@app.get("/api/prefs/summary")
def prefs_summary():
    return PREFS.summary()


@app.get("/api/prefs/rubric")
def prefs_rubric():
    return rubric_info()


class ContextBody(BaseModel):
    messages: list
    max_bytes: Optional[int] = None
    fold: str = "auto"
    card: Optional[str] = None
    prev: Optional[dict] = None  # the client's last fold {"from", "card"}: kept while the prompt still fits


@app.post("/api/context")
def context_preview(body: ContextBody):
    """What the next prompt would look like — bytes used, fold point, card — without generating."""
    eng = engine()
    limit = eng.cfg.max_seq_len
    msgs = with_system_card(body.messages, eng.model.num_params()) if limit >= 1024 else body.messages
    reserve = min(body.max_bytes or eng.defaults.max_bytes, limit // 4)
    rep = context_report(msgs, limit, reserve, eng.tok, body.fold or "auto", body.card, body.prev)
    snap = eng.prefix_cache.peek(rep.pop("ids"))
    rep["reusable"] = snap.n_ids if snap is not None else 0  # bytes the engine would not have to re-read
    return rep


def _run_generation(eng: Engine, messages, params: GenParams, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue, stop: threading.Event,
                    context: Optional[dict] = None):
    def emit(ev: dict):
        loop.call_soon_threadsafe(queue.put_nowait, ev)

    try:
        eng.generate(messages, params, emit, stop, context=context)
    except Exception as e:  # surface engine errors to the client — and the traceback to the log
        traceback.print_exc()
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
                eng = engine_for(msg.get("engine") or "current")
            except HTTPException as e:
                await ws.send_json({"type": "error", "message": e.detail})
                continue
            params = GenParams.from_dict(msg.get("params"), eng.defaults)
            queue: asyncio.Queue = asyncio.Queue()
            stop.clear()
            threading.Thread(target=_run_generation, args=(eng, msg.get("messages", []), params, loop, queue, stop, msg.get("context")), daemon=True).start()

            gone = asyncio.Event()  # the client left mid-reply: no more receives on this socket

            async def pump_client():
                # listen for a stop message while generating
                try:
                    while True:
                        m = await ws.receive_json()
                        if m.get("type") == "stop":
                            stop.set()
                except Exception:
                    stop.set()
                    gone.set()

            listener = asyncio.create_task(pump_client())
            try:
                while True:
                    ev = await queue.get()
                    if ev["type"] == "__end__":
                        break
                    if gone.is_set():
                        continue  # drain the engine's events; the socket is closed
                    await ws.send_json(ev)
            finally:
                listener.cancel()
            if gone.is_set():
                break
    except WebSocketDisconnect:
        stop.set()


# --- static frontend --------------------------------------------------------------
DIST = ROOT / "web" / "dist"
if DIST.exists():

    class _ImmutableAssets(StaticFiles):
        """Hashed build assets: cache for a year, they never change under the same name."""

        def file_response(self, *args, **kwargs):  # type: ignore[override]
            resp = super().file_response(*args, **kwargs)
            resp.headers["Cache-Control"] = "public, max-age=31536000, immutable"
            return resp

    app.mount("/assets", _ImmutableAssets(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        # Vite's hashed assets under /assets are immutable; everything else (index.html above all) must be
        # revalidated, or a paired phone keeps an index.html that points at asset hashes a rebuild removed
        # and opens a blank studio (QA 2026-08-24).
        f = DIST / path
        if path and f.is_file():
            return FileResponse(f, headers={"Cache-Control": "no-cache"})
        return FileResponse(DIST / "index.html", headers={"Cache-Control": "no-cache"})
else:

    @app.get("/")
    def no_frontend():
        return JSONResponse({"message": "frontend not built: run `bun run build` in web/", "api": "/api/model"})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None, help="default: newest checkpoint under runs/ or checkpoints/")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--device", default=None, help="cuda (default when available) or cpu — e.g. to leave the GPU to a training run")
    ap.add_argument("--token", default=os.environ.get("MOTE_TOKEN") or None,
                    help="shared access token (or MOTE_TOKEN); required unless bound to loopback or --no-auth")
    ap.add_argument("--no-auth", action="store_true", help="serve without a token on a non-loopback host (not recommended)")
    ap.add_argument("--public-url", default=os.environ.get("MOTE_URL") or None,
                    help="address phones use (default: detected from `tailscale serve status`); encoded in the /pair QR")
    args = ap.parse_args(argv)
    loopback = args.host in ("127.0.0.1", "localhost", "::1")
    if not loopback and not args.token and not args.no_auth:
        raise SystemExit(f"refusing to bind {args.host} without a token: pass --token (or MOTE_TOKEN), or --no-auth to override")
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
    gate = threading.Lock()
    STATE["gate"] = gate
    jobs = JobQueue(ROOT / ".mote" / "jobs.json", gate, on_serve_sync=_serve_sync, on_finished=_job_finished,
                    on_started=_job_started, on_idle=_queue_idle)
    STATE["jobs"] = jobs
    # A queued job starts right away and owns the GPU: serving starts on the CPU and moves to the GPU when the
    # queue idles (signed 2026-08-25). No warm-up on the CPU: there is nothing to JIT there.
    device = _serve_device()
    eng = Engine(ck, device=device)
    eng.gpu_gate = gate
    if eng.device.type == "cuda":
        print(f"warming up kernels ... ({eng.warmup():.1f} s)", flush=True)
    else:
        print(f"training queued: serving on the {device} until the queue idles", flush=True)
    STATE["engine"] = eng
    jobs.start()
    print(json.dumps(eng.info(), indent=1), flush=True)
    # uvicorn captures SIGTERM/SIGINT for its graceful shutdown, then restores the handlers it found and
    # re-raises the signal — with the default handlers that killed the process (exit -15) before the hook
    # below ever ran (2026-08-24 night, three restarts from step 0). Swallow the re-raised signal instead:
    # uvicorn's own handler still drives the shutdown, and `run()` returns here afterwards.
    import signal

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda signum, frame: None)
    uvicorn.run(app, host=args.host, port=args.port)
    # SIGTERM (systemd stop/restart) returns here once the server is down: the running job reaches a step
    # boundary, checkpoints and re-enqueues itself in front (the unit's TimeoutStopSec bounds this wait; a
    # kill instead costs up to --ckpt-minutes of the run — what happened on 2026-08-24 before this hook)
    jobs.shutdown()
    print("stopping the running job at its next step boundary ...", flush=True)
    print("job queue stopped" if jobs.join(timeout=150.0) else "job queue still busy at exit", flush=True)


if __name__ == "__main__":
    main()
