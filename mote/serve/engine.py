"""Generation engine: prefill + byte-at-a-time decoding with multi-byte speculative acceptance,
emitting the telemetry events described in docs/api.md. Synchronous; run it in a worker thread.
"""

from __future__ import annotations

import json
import math
import contextlib
import gc
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from ..config import MoteConfig
from ..model.arena import ArenaState, RelationArena
from ..model.hnet import HNetForCausalLM, InferenceState
from ..model.mamba3 import HAS_MAMBA3_KERNEL, Mamba3Mixer
from ..model.dc import HAS_SSD_KERNEL
from ..model.relation import FullRelation
from .context import fold
from .graph import BUCKET, GraphDecoder
from .identity import identity_card, with_system_card
from ..tokenizer import ASSISTANT_ID, BOS_ID, CALL_ID, EOS_ID, PAD_ID, RESULT_ID, SYSTEM_ID, USER_ID, ByteTokenizer, ChatMessage, Utf8Streamer, parse_call
from .prefix_cache import Hit, PrefixStore

STOP_IDS = {EOS_ID, PAD_ID, SYSTEM_ID, USER_ID, ASSISTANT_ID, BOS_ID, RESULT_ID}  # <|result|> stops decoding for the tool hook


@dataclass
class GenParams:
    temperature: float = 0.8
    top_p: float = 0.9
    max_bytes: int = 512
    n_candidates: int = 3  # draft length per chunk boundary (0 disables speculation); verification is exact
    max_calls: int = 2  # tool calls per reply (docs/search.md: ≤ 2 searches; an RL episode raises it)
    script: Optional[List[int]] = None  # forced ids instead of samples (tests / transcript replay; eager path only)

    @classmethod
    def from_dict(cls, d: Optional[dict], defaults: "GenParams") -> "GenParams":
        d = d or {}
        return cls(
            temperature=float(d.get("temperature", defaults.temperature)),
            top_p=float(d.get("top_p", defaults.top_p)),
            max_bytes=int(d.get("max_bytes", defaults.max_bytes)),
            n_candidates=int(d.get("n_candidates", defaults.n_candidates)),
            max_calls=int(d.get("max_calls", defaults.max_calls)),
        )


@dataclass
class CheckpointInfo:
    path: str
    step: int
    bytes_seen: int
    val_bpb: Optional[float]
    trained_minutes: Optional[float]
    created_at: str
    status: str
    status_note: str


def _dist(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    """The sampling distribution [V] for these logits: temperature, then nucleus truncation, renormalised.
    temperature <= 0 is greedy (a one-hot). Applied identically to target and draft so that speculative
    verification reproduces the target's *sampling* distribution, transforms included."""
    raw = torch.softmax(logits.float(), dim=-1)
    if temperature <= 0:
        return torch.zeros_like(raw).scatter(0, raw.argmax().view(1), 1.0)
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    if top_p < 1.0:
        sp, si = torch.sort(probs, descending=True)
        cum = torch.cumsum(sp, dim=-1)
        keep = cum - sp < top_p  # keep tokens until the mass reaches top_p (always at least one)
        sp = sp * keep
        probs = torch.zeros_like(probs).scatter(0, si, sp)
        probs = probs / probs.sum()
    return probs


def _draw(probs: torch.Tensor) -> int:
    return int(torch.multinomial(probs, 1))


def verify_draft(xs: list, q_dists: list, p_dists: list, rand=None):
    """Leviathan/Chen speculative sampling. xs[m] was drawn from q_dists[m]; p_dists[m] is the target's
    distribution at the same position (given the accepted prefix). Accept left to right with probability
    min(1, p(x)/q(x)); on the first rejection draw the correction from norm(max(0, p - q)).
    Returns (n_accepted, correction_byte or None, target_prob_of_correction). The output sequence is
    distributed exactly as the target would have produced it."""
    rand = rand or (lambda: float(torch.rand(1)))
    for m, b in enumerate(xs):
        p_m, q_m = p_dists[m], q_dists[m]
        ratio = min(1.0, float(p_m[b]) / max(float(q_m[b]), 1e-12))
        if rand() < ratio:
            continue
        resid = (p_m - q_m).clamp(min=0.0)
        total = float(resid.sum())
        resid = resid / total if total > 0 else p_m
        fix = _draw(resid)
        return m, fix, float(p_m[fix])
    return len(xs), None, None


def _sample(logits: torch.Tensor, temperature: float, top_p: float):
    """logits [V] -> (byte id, prob of chosen byte under the sampling distribution, entropy bits of the raw distribution)."""
    raw = torch.softmax(logits.float(), dim=-1)
    entropy = float(-(raw * torch.log2(raw.clamp_min(1e-12))).sum())
    if temperature <= 0:
        idx = int(raw.argmax())
        return idx, float(raw[idx]), entropy
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    if top_p < 1.0:
        sp, si = torch.sort(probs, descending=True)
        cum = torch.cumsum(sp, dim=-1)
        keep = cum - sp < top_p  # keep tokens until the mass reaches top_p (always at least one)
        sp = sp * keep
        probs = torch.zeros_like(probs).scatter(0, si, sp)
        probs = probs / probs.sum()
    idx = int(torch.multinomial(probs, 1))
    return idx, float(probs[idx]), entropy


class Engine:
    def __init__(self, ckpt_path: str | Path, device: Optional[str] = None, prefix_cache_mb: Optional[int] = None,
                 released: bool = False):
        ck = torch.load(Path(ckpt_path), map_location="cpu", weights_only=False)
        cfg = MoteConfig.from_dict(ck["config"])
        model = HNetForCausalLM(cfg)
        model.load_state_dict(ck["model"])
        self._setup(model, cfg, Path(ckpt_path), int(ck.get("step", 0)), ck.get("extra", {}), device, prefix_cache_mb,
                    released=released)

    @classmethod
    def from_model(cls, model: HNetForCausalLM, cfg: MoteConfig, device: Optional[str] = None, name: str = "live/policy.pt",
                   step: int = 0, prefix_cache_mb: Optional[int] = None) -> "Engine":
        """Serve a model object that already lives in memory — the RL driver's policy — with no checkpoint file."""
        self = cls.__new__(cls)
        self._setup(model, cfg, Path(name), step, {}, device, prefix_cache_mb, describe=False)
        return self

    def _setup(self, model, cfg, path: Path, step: int, extra: dict, device, prefix_cache_mb, describe: bool = True,
               released: bool = False) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.pin = self.device.type == "cuda"
        # Prefix store (mote/serve/prefix_cache.py): branches of arena pages + ~3 MB anchors on the CPU
        # under a byte budget; the Relation arena itself (model/arena.py) stays on the device.
        mb = int(os.environ.get("MOTE_PREFIX_CACHE_MB", 1024)) if prefix_cache_mb is None else int(prefix_cache_mb)
        self.prefix_cache = PrefixStore(mb << 20, pin=self.pin)
        # Set by the server: the lock the training worker takes per accumulation slice (docs/shape.md).
        # Serving no longer holds it for a reply (signed 2026-08-24 evening): decode runs on its own
        # high-priority stream beside the training slice, and the gate is taken only around weight
        # swaps. MOTE_SERVE_GATED=1 restores the old whole-reply hold (the A/B for the latency gate).
        self.gpu_gate = None
        self.gated = os.environ.get("MOTE_SERVE_GATED", "0") == "1"
        cuda = self.device.type == "cuda"
        self.stream = torch.cuda.Stream(device=self.device, priority=-1) if cuda else None  # high priority
        # The LONG-LIVED serving allocations (arena, decode graphs) live in their own pool so the trainer's
        # churn cannot fragment them. Transients (prefill activations, rewarm reads) deliberately do NOT:
        # a MemPool keeps its high-water mark cached for itself, and on 2026-08-24 the pool grew to 1–3 GB
        # of idle reservation over one evening of syncs, starving the training jobs of the same GPU
        # (five arms OOM'd). `_serve_ctx()` therefore applies the pool only where asked.
        self.pool = torch.cuda.MemPool() if (cuda and os.environ.get("MOTE_SERVE_POOL", "1") != "0") else None
        # "<run>/ema@<step>" while a training job's EMA answers chats; None when serving a checkpoint.
        self.serving_live: str | None = None
        self.ckpt_path = path
        self.cfg = cfg
        self.model = model
        self.model.to(self.device).eval()
        self.tok = ByteTokenizer()
        self.lock = threading.Lock()
        # tool hook: name -> fn(args) -> result text; the reply's <|call|>name: args<|result|> is routed here
        self.tools: Dict[str, Callable[[str], str]] = {}
        self.tool_result_limit = 1024  # bytes of result injected per call (docs/search.md)
        if describe:
            self.info_ckpt = self._describe_checkpoint(path, step, extra)
        else:
            self.info_ckpt = CheckpointInfo(str(path), step, 0, None, None, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                            "live", "the live policy of a running RL job")
        self.defaults = GenParams(max_bytes=min(512, self.cfg.max_seq_len // 2))
        self._telemetry: Dict[str, dict] = {}
        self._attach_telemetry()
        self._last_card: List[int] = []  # the identity-card bytes of the last prompt (rewarm reads them)
        # Released (signed 2026-08-24 night, the serving-beside-training root): while a training job runs the
        # engine holds only its weights — no arena, no captured graphs, no MemPool. A reply allocates a
        # per-reply arena from the shared allocator, decodes eagerly (the prefix store's CPU pages still
        # rehydrate it) and hands the memory back; `rearm()` restores the fast path when the queue idles.
        self._released = False
        self._graph_ok = (self.device.type == "cuda" and self.model.mbp_head is None
                          and os.environ.get("MOTE_GRAPH_DECODE", "1") != "0")
        self.arena = None
        self._gd: Optional[GraphDecoder] = None
        if released:
            self.pool = None
            self._released = True
        else:
            with self._serve_ctx(pool=True):
                self._setup_decode()

    @property
    def released(self) -> bool:
        return self._released

    @torch.no_grad()
    def release(self) -> dict:
        """Drop the arena, the decode graphs and the serving MemPool (no reply in flight); serve eagerly."""
        with self.lock:
            self.drain()
            if self._gd is not None:
                self._gd.close()
                self._gd = None
            self.arena = None
            self.pool = None
            self._released = True
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
        return {"released": True, "vram_used_mb": self.info()["device"]["vram_used_mb"]}

    def rearm(self) -> float:
        """Back to the resident arena + graphs; returns the warm-up seconds (~12 s on the 4060 Ti)."""
        if not self._released:
            return 0.0
        with self.lock:
            self.drain()
            cuda = self.device.type == "cuda"
            self.pool = torch.cuda.MemPool() if (cuda and os.environ.get("MOTE_SERVE_POOL", "1") != "0") else None
            with self._serve_ctx(pool=True):
                self._setup_decode()
            self._released = False
        return self.warmup()

    def _reply_arena(self) -> RelationArena:
        """The resident arena, or (released) a fresh one for this reply from the shared allocator."""
        return self.arena if self.arena is not None else self.model.new_arena(self.device)

    def _serve_ctx(self, pool: bool = False):
        """Everything the serving side runs on the GPU goes through here: its own high-priority stream
        (no-op on the CPU). `pool=True` additionally routes allocations into the serving MemPool — only for
        the long-lived ones (arena construction; growth and graph capture route themselves)."""
        stack = contextlib.ExitStack()
        if self.stream is not None:
            stack.enter_context(torch.cuda.stream(self.stream))
        if pool and self.pool is not None:
            stack.enter_context(torch.cuda.use_mem_pool(self.pool))
        return stack

    def drain(self) -> None:
        """Wait for every serving kernel in flight (before a weight swap or a checkpoint load)."""
        if self.stream is not None:
            self.stream.synchronize()

    def _setup_decode(self) -> None:
        """The resident decode arena (the graph decoder captures lazily, on CUDA for models without a multi-byte head)."""
        self.arena = self.model.new_arena(self.device, capacity=int(os.environ["MOTE_ARENA_CHUNKS"]) if os.environ.get("MOTE_ARENA_CHUNKS") else None)
        self.arena.pool = self.pool  # growth reallocates inside the serving pool
        self._gd = None

    def _graph_decoder(self) -> GraphDecoder:
        if self._gd is None:
            self._gd = GraphDecoder(self.model, self.arena, self.device, STOP_IDS, ring_size=self.cfg.max_seq_len,
                                    pool=self.pool.id if self.pool is not None else None)
        return self._gd

    # ------------------------------------------------------------------------------
    @torch.no_grad()
    def warmup(self) -> float:
        """Trigger Triton JIT/autotune for prefill at a few prompt lengths (incl. odd and 16-aligned ones) and one
        decode step, so the first user request doesn't pay the ~40 s compile. Returns seconds spent."""
        if self.arena is None:  # released: nothing resident to warm; `rearm()` first
            return 0.0
        t0 = time.perf_counter()
        with self.lock, self._serve_ctx():
            for L in (5, 16, 33, 128, 257, 512, 640, 1024, 2048, 4096):
                L = min(L, self.cfg.max_seq_len - 4)
                ids = torch.randint(0, 256, (1, L), device=self.device)
                state = self.model.allocate_inference_state(self.device, arena=self.arena)
                self.model.prefill(ids, state)
                self.model.step(torch.tensor([[65]], device=self.device), state)
                if L >= 16:
                    # resumed reads compile their own kernel variants (Input_States != None,
                    # 2026-08-24): warm them at short and long lengths too, or the first warm turn
                    # (a ~30-byte continuation) pays a multi-second compile
                    warm = self.model.allocate_inference_state(self.device, arena=self.arena)
                    self.model.prefill(ids[:, : L // 2], warm)
                    self.model.forward_from_state(ids[:, L // 2 :], warm)
                    self.model.forward_from_state(ids[:, : min(3, L)], warm)
            if self._graph_ok:  # capture the first decode width now, not on the first reply
                gd = self._graph_decoder()
                state = self.model.allocate_inference_state(self.device, arena=self.arena)
                gd.load(state, torch.zeros(gd.V, device=self.device))
                gd._graph(BUCKET)
            self.arena.invalidate()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()  # give the warm-up's transient buffers back (training may share this GPU)
        return time.perf_counter() - t0

    def _attach_telemetry(self):
        for name, m in self.model.named_modules():
            if isinstance(m, (Mamba3Mixer, FullRelation)):
                m.telemetry = {}
                self._telemetry[name] = m.telemetry

    def _describe_checkpoint(self, path: Path, step: int, extra: dict) -> CheckpointInfo:
        run_dir = path.parent
        val_bpb, minutes, bytes_seen = None, None, 0
        log = run_dir / "log.jsonl"
        if log.exists():
            try:
                for line in log.read_text(encoding="utf-8").splitlines():
                    rec = json.loads(line)
                    if "eval" in rec and rec.get("step", 0) <= step:
                        val_bpb = rec["eval"].get("val_bpb", val_bpb)
                    if "elapsed_min" in rec and rec.get("step", 0) <= step:
                        minutes = rec["elapsed_min"]
            except Exception:
                pass
        # bytes seen: steps × batch × seq_len if the run recorded it, else unknown
        meta = run_dir / "run.json"
        if meta.exists():
            try:
                m = json.loads(meta.read_text())
                bytes_seen = step * int(m.get("batch_size", 0)) * int(m.get("seq_len", 0)) * int(m.get("grad_accum", 1))
            except Exception:
                pass
        if bytes_seen == 0 and "bytes_seen" in extra:
            bytes_seen = int(extra["bytes_seen"])
        created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(path.stat().st_mtime))
        if run_dir.name.startswith("pilot"):
            status, note = "pilot", f"Pilot checkpoint: {step} steps on the local RTX 4060 Ti. Expect fluent nonsense; this run exists to validate the architecture, not to chat well."
        elif val_bpb is not None and val_bpb > 1.3:
            status, note = "undertrained", f"Validation {val_bpb:.2f} bits/byte at step {step}. Grammar, yes; facts and coherence, not yet."
        else:
            status, note = "flagship", f"Step {step}" + (f", {val_bpb:.2f} bits/byte" if val_bpb is not None else "")
        return CheckpointInfo(str(path), step, bytes_seen, val_bpb, minutes, created, status, note)

    def probe_results(self):
        """identity/pushback probe numbers if `probe.json` sits next to the checkpoint (see mote.eval.probe)."""
        p = Path(self.ckpt_path).parent / "probe.json"
        if not p.exists():
            return None
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
            keys = ("identity_acc", "hold_rate", "concede_rate", "n_identity", "n_facts")
            return {k: r[k] for k in keys + tuple(f"{k}_seen" for k in keys) if k in r}
        except Exception:
            return None

    @torch.no_grad()
    def apply_run_weights(self, cfg_dict: dict, state_dict, name: str, step: int) -> None:
        """Hot-swap the served weights from a running job's EMA (docs/shape.md). Same config loads in
        place; a different config rebuilds the model. The prefix cache clears either way — its cached
        states were computed under the old weights."""
        new_cfg = MoteConfig.from_dict(cfg_dict)
        with self.lock:  # no reply in flight; the serving stream is drained before the weights change
            self.drain()
            rebuilt = new_cfg.to_dict() != self.cfg.to_dict()
            if rebuilt:
                self.cfg = new_cfg
                self.model = HNetForCausalLM(new_cfg)
                self.defaults = GenParams(max_bytes=min(512, new_cfg.max_seq_len // 2))
            self.model.load_state_dict(state_dict)  # in place: captured graphs keep reading the same memory
            self.model.to(self.device).eval()
            if rebuilt:
                self._attach_telemetry()
                if not self._released:
                    with self._serve_ctx():
                        self._setup_decode()
            self.prefix_cache.clear()
            if self.arena is not None:
                self.arena.invalidate()
            self.serving_live = f"{name}@{step}"

    @property
    def ckpt_name(self) -> str:
        """run/file, the name the studio shows: overnight_sft/last.pt"""
        c = Path(self.info_ckpt.path)
        return f"{c.parent.name}/{c.name}"

    def info(self) -> dict:
        dev = {"name": "cpu", "vram_total_mb": 0, "vram_used_mb": 0}
        if self.device.type == "cuda":
            props = torch.cuda.get_device_properties(self.device)
            dev = {"name": props.name, "vram_total_mb": props.total_memory // 2**20, "vram_used_mb": torch.cuda.memory_allocated(self.device) // 2**20}
        c = self.info_ckpt
        return {
            "name": self.ckpt_name,
            "live": self.serving_live,
            "params": self.model.num_params(),
            "status": c.status,
            "status_note": c.status_note,
            "checkpoint": {"path": c.path, "step": c.step, "bytes_seen": c.bytes_seen, "val_bpb": c.val_bpb, "trained_minutes": c.trained_minutes, "created_at": c.created_at},
            "architecture": {
                "outer_width": self.cfg.d_model_outer,
                "encoder_layers": self.cfg.encoder_layers,
                "decoder_layers": self.cfg.decoder_layers,
                "main": f"Relation {self.cfg.main.n_layers}L/{self.cfg.main.d_model}/{self.cfg.main.n_heads} heads",
                "mbp_layers": self.cfg.mbp.n_layers if self.cfg.mbp.enabled else 0,
            },
            "context_limit_bytes": self.cfg.max_seq_len,
            "device": dev,
            "kernels": {"mamba3": HAS_MAMBA3_KERNEL and self.device.type == "cuda", "ssd": HAS_SSD_KERNEL and self.device.type == "cuda"},
            "defaults": vars(self.defaults),
            "probe": self.probe_results(),
            "identity_card": identity_card(self.model.num_params()),
            "prefix_cache": self.prefix_cache.report(),
            "arena": ({"chunks": self.arena.capacity, "bytes": self.arena.nbytes(), "hot_branch": self.arena.owner,
                       "hot_chunks": self.arena.n_valid, "graph_decode": self._graph_ok, "released": False}
                      if self.arena is not None else
                      {"chunks": 0, "bytes": 0, "hot_branch": None, "hot_chunks": 0, "graph_decode": False, "released": True}),
        }

    # ------------------------------------------------------------------------------
    def _card_ids(self, messages: Sequence[dict]) -> List[int]:
        """Byte ids of the leading system message alone: the prefix every prompt of this deployment shares."""
        if messages and messages[0].get("role") == "system":
            return self.tok.format_chat([ChatMessage("system", messages[0]["content"])], add_generation_prompt=False)
        return []

    # A session = (branch the read started on or None, arena rows that were valid at that point).
    # Commits extend the branch when the bytes continue it and fork at `from_chunks` when they diverge.
    def _commit(self, kind: str, ids: Sequence[int], state: InferenceState, logits: torch.Tensor, session):
        if not self.prefix_cache.budget:
            return session
        branch, from_chunks = session
        anchor_state = self.model.move_state(state, "cpu", pin=self.pin)  # everything but the arena rows
        target = self.prefix_cache.commit(branch, from_chunks, kind, ids, anchor_state,
                                          logits.detach().float().cpu(), state.main.n, state.main.arena)
        return (target, state.main.n)

    def _restore(self, hit: Hit, arena: RelationArena) -> InferenceState:
        state = self.model.move_state(hit.anchor.state, self.device)
        self.prefix_cache.hydrate(hit.branch, hit.anchor.n_chunks, arena)  # free when the arena is hot
        state.main = ArenaState(arena, hit.anchor.n_chunks)
        return state

    @torch.no_grad()
    def _read_prompt(self, prompt_ids: List[int], card: List[int], arena: Optional[RelationArena] = None):
        """Read the prompt, reusing the longest anchor. Returns (state, next-byte logits [V], n_chunks,
        reused bytes, boundary mask of the freshly read suffix or None, session)."""
        arena = arena if arena is not None else self._reply_arena()
        P = len(prompt_ids)
        hit = self.prefix_cache.lookup(prompt_ids)
        if hit is not None:
            state = self._restore(hit, arena)
            session = (hit.branch, hit.anchor.n_chunks)
            if hit.n_ids == P:  # e.g. a regenerate: nothing new to read
                return state, hit.anchor.logits.to(self.device), state.main.n, P, None, session
            lg, bm, _ = self.model.forward_from_state(torch.tensor([prompt_ids[hit.n_ids:]], device=self.device), state)
            return state, lg[0, -1], state.main.n, hit.n_ids, bm, session
        arena.invalidate()
        state = self.model.allocate_inference_state(self.device, arena=arena)
        if card and len(card) < P and prompt_ids[:len(card)] == card:
            # cold start: read the identity card on its own first, so every later conversation starts warm
            out = self.model.prefill(torch.tensor([card], device=self.device), state)
            session = self._commit("card", card, state, out.logits[0, -1], (None, 0))
            lg, bm, _ = self.model.forward_from_state(torch.tensor([prompt_ids[len(card):]], device=self.device), state)
            return state, lg[0, -1], state.main.n, 0, bm, session
        out = self.model.prefill(torch.tensor([prompt_ids], device=self.device), state)
        return state, out.logits[0, -1], state.main.n, 0, None, (None, 0)

    @torch.no_grad()
    def _verify_prefix(self, prompt_ids: List[int], reused: int, warm_logits: torch.Tensor, warm_bm, warm_chunks: int) -> dict:
        """Debug toggle: read the whole prompt cold (in a private arena) and compare with the warm continuation."""
        t = time.perf_counter()
        cold = self.model.allocate_inference_state(self.device)
        out = self.model.prefill(torch.tensor([prompt_ids], device=self.device), cold)
        cold_bm = out.routing.boundary_mask[0]
        flips = int((cold_bm[reused:] != warm_bm.to(cold_bm.device)).sum()) if warm_bm is not None else 0
        diff = (out.logits[0, -1].float() - warm_logits.float())[: self.cfg.vocab_size]  # padding rows are -inf on both sides
        return {"reused": reused, "prefilled": len(prompt_ids) - reused, "boundary_flips": flips,
                "chunks_cold": cold.main.n, "chunks_warm": warm_chunks,
                "max_logit_diff": float(diff.abs().max()), "cold_ms": (time.perf_counter() - t) * 1000}

    @torch.no_grad()
    def rewarm(self, max_age_s: float = 600.0, max_branches: int = 3) -> dict:
        """After a weight swap (decided 2026-08-24): re-read the conversations used in the last
        `max_age_s` so their anchors are warm again before the next message. One prefill per branch."""
        store = self.prefix_cache
        if not store.budget or self.arena is None:  # released: nothing resident to warm
            return {"branches": 0, "bytes": 0, "ms": 0.0}
        t0 = time.perf_counter()
        plan = store.rewarm_plan(max_age_s, max_branches)
        store.clear()
        self.arena.invalidate()
        n_bytes = 0
        with self.lock, self._serve_ctx():
            for ids, anchors in plan:
                if not anchors:
                    continue
                kind0, n0 = anchors[0]
                state, logits, _, _, _, session = self._read_prompt(ids[:n0], self._last_card)
                session = self._commit(kind0, ids[:n0], state, logits, session)
                prev = n0
                for kind, n in anchors[1:]:
                    if n > prev:
                        lg, _, _ = self.model.forward_from_state(torch.tensor([ids[prev:n]], device=self.device), state)
                        logits = lg[0, -1]
                    session = self._commit(kind, ids[:n], state, logits, session)
                    prev = n
                n_bytes += len(ids)
        return {"branches": len(plan), "bytes": n_bytes, "ms": (time.perf_counter() - t0) * 1000}

    # ------------------------------------------------------------------------------
    def build_prompt(self, messages: Sequence[dict], limit: int, reserve: int):
        """Chat template; drop oldest non-system turns until the prompt fits limit - reserve bytes."""
        msgs = [ChatMessage(m["role"], m["content"]) for m in messages]
        system = [m for m in msgs if m.role == "system"]
        rest = [m for m in msgs if m.role != "system"]
        truncated = False
        while True:
            ids = self.tok.format_chat(system + rest, add_generation_prompt=True)
            if len(ids) <= max(limit - reserve, 8) or len(rest) <= 1:
                break
            rest = rest[1:]
            truncated = True
        return ids, truncated

    def _diagnostics(self, boundary_probs: List[float]) -> dict:
        enc, dec, rel = [], [], []
        for name, t in self._telemetry.items():
            if "retention" in t:
                (enc if name.startswith("encoder") else dec).append(t["retention"])
            if "exchange_mass" in t:
                rel.append(t["exchange_mass"])
        mean = lambda rows: [sum(col) / len(rows) for col in zip(*rows)] if rows else []
        return {"mamba3": {"encoder_retention": mean(enc), "decoder_retention": mean(dec)}, "relation": {"exchange_mass": rel}, "boundary_probs": boundary_probs[-64:]}

    @torch.no_grad()
    def generate(self, messages: Sequence[dict], params: GenParams, emit: Callable[[dict], None], stop: threading.Event,
                 context: Optional[dict] = None) -> None:
        """`context`: {"fold": "auto" | "now" | "off", "card": <edited card or None>} — see mote.serve.context."""
        gate = self.gpu_gate if (self.gated and self.gpu_gate is not None) else contextlib.nullcontext()
        with gate, self.lock, self._serve_ctx():
            try:
                self._generate(messages, params, emit, stop, context or {})
            finally:
                if self.arena is None and self.device.type == "cuda":  # released: the reply's arena goes back
                    torch.cuda.synchronize(self.device)
                    torch.cuda.empty_cache()

    def _generate(self, messages, params: GenParams, emit, stop: threading.Event, context: dict) -> None:
        limit = self.cfg.max_seq_len
        if limit >= 1024:  # the identity card needs room; tiny test contexts go without
            messages = with_system_card(messages, self.model.num_params())  # Mote knows what it is
        folded = fold(messages, limit, reserve=min(params.max_bytes, limit // 4), tok=self.tok,
                      mode=context.get("fold", "auto") or "auto", card_override=context.get("card"),
                      prev=context.get("prev") if isinstance(context.get("prev"), dict) else None)
        prompt_ids, truncated = folded.ids, folded.truncated
        P = len(prompt_ids)
        t0 = time.perf_counter()
        self._last_card = self._card_ids(messages)
        arena = self._reply_arena()  # released: this reply's own arena (freed with the frame)
        state, logits, n_chunks, reused, warm_bm, session = self._read_prompt(prompt_ids, self._last_card, arena)
        prefill_ms = (time.perf_counter() - t0) * 1000
        emit({"type": "start", "prompt_bytes": P, "context_bytes": P, "context_limit": limit, "truncated": truncated,
              "fold": folded.report(),
              "prefix": {"reused": reused, "prefilled": P - reused, "prefill_ms": prefill_ms, **self.prefix_cache.report()},
              "checkpoint": {"name": self.ckpt_name, "step": self.info_ckpt.step}})
        if reused and context.get("verify_prefix"):
            check = self._verify_prefix(prompt_ids, reused, logits, warm_bm, n_chunks)
            emit({"type": "diagnostics", **self._diagnostics([]), "prefix_check": check})
        session = self._commit("prompt", prompt_ids, state, logits, session)  # S1: this turn's prompt (a regenerate reuses it)

        streamer = Utf8Streamer()
        boundary_probs: List[float] = []
        generated: List[int] = []
        text_parts: List[str] = []
        spec = {"rounds": 0, "proposed": 0, "accepted": 0, "fixes": 0, "replays": 0, "paused": False}
        # Measured break-even: seconds per emitted byte in speculative rounds vs plain steps, over this reply.
        # Drafting pauses for the rest of the reply once it is measurably slower than stepping.
        PAUSE_AFTER = 24  # proposed bytes before the comparison is trusted
        timing = {"spec_s": 0.0, "spec_bytes": 0, "plain_s": 0.0, "plain_bytes": 0}
        chunk_start = P  # absolute index where the current chunk began
        chunk_index = n_chunks - 1
        reason = "max_bytes"
        last_stats_bytes = 0
        n_draft = params.n_candidates if self.model.mbp_head is not None else 0
        script = list(params.script or [])  # forced ids (tests / replay): plain steps only
        if script:
            n_draft = 0
        calls_made = 0
        call_buf: List[bytearray] = []  # non-empty while the model is writing a tool call (after <|call|>)
        stop_id: Optional[int] = None
        reply_mask: List[int] = []  # 1 = a byte the model chose, 0 = injected tool result (the RL loss mask)

        def stats() -> dict:
            elapsed = time.perf_counter() - t0
            n = len(generated)
            return {
                "bytes": n, "elapsed_ms": elapsed * 1000, "bytes_per_sec": n / elapsed if elapsed > 0 else 0.0,
                "chunks": chunk_index + 1 - (n_chunks - 1), "bytes_per_chunk": n / max(chunk_index + 1 - (n_chunks - 1), 1),
                "mbp_proposed": spec["proposed"], "mbp_accepted": spec["accepted"],
                "mbp_accept_rate": spec["accepted"] / spec["proposed"] if spec["proposed"] else 0.0,
                "spec_rounds": spec["rounds"], "spec_fixes": spec["fixes"], "spec_replays": spec["replays"],
                "spec_paused": spec["paused"],
                "context_bytes": P + n, "context_limit": limit,
            }

        def record(byte: int, p: float, entropy: float, is_b: bool, bp: float, source: str, t_ms: float) -> None:
            nonlocal chunk_index, chunk_start
            abs_i = P + len(generated)
            if is_b:
                if abs_i > chunk_start:
                    emit({"type": "chunk", "index": chunk_index, "start": max(chunk_start - P, 0), "end": abs_i - 1 - P,
                          "bytes": abs_i - chunk_start, "partial": chunk_start < P,
                          "text": bytes(b for b in (prompt_ids + generated)[chunk_start:abs_i] if b < 256).decode("utf-8", errors="replace")})
                chunk_index += 1
                chunk_start = abs_i
                emit({"type": "diagnostics", **self._diagnostics(boundary_probs)})
            boundary_probs.append(bp)
            generated.append(byte)
            reply_mask.append(1)
            if byte == CALL_ID:  # the model opens a tool call: what follows is the call, not reply text
                call_buf.append(bytearray())
                text, source = "", "call"
            elif call_buf:
                if byte < 256:
                    call_buf[-1].append(byte)
                text, source = "", "call"
            else:
                text = streamer.feed(byte)
                if text:
                    text_parts.append(text)
            emit({
                "type": "byte", "i": len(generated) - 1, "byte": byte, "text": text or None, "pending": len(streamer.pending),
                "p": p, "entropy": entropy, "boundary": is_b, "boundary_p": bp, "chunk": chunk_index,
                "source": source, "t_ms": t_ms,
            })

        def entropy_of(lg: torch.Tensor) -> float:
            raw = torch.softmax(lg.float(), -1)
            return float(-(raw * torch.log2(raw.clamp_min(1e-12))).sum())

        def new_draft(is_b: bool, mbp):
            """Draft logits [n, V] for the bytes after a fresh boundary, or None."""
            if not is_b or n_draft <= 0 or spec["paused"]:
                return None
            if spec["proposed"] >= PAUSE_AFTER and timing["plain_bytes"] >= 8 and timing["spec_bytes"] >= 8:
                spec_rate = timing["spec_bytes"] / max(timing["spec_s"], 1e-9)
                plain_rate = timing["plain_bytes"] / max(timing["plain_s"], 1e-9)
                if spec_rate < plain_rate:
                    spec["paused"] = True
                    emit({"type": "diagnostics", **self._diagnostics(boundary_probs),
                          "note": f"drafting paused: {spec['accepted']}/{spec['proposed']} draft bytes accepted; speculative rounds ran at {spec_rate:.0f} B/s vs {plain_rate:.0f} B/s plain"})
                    return None
            if mbp is None:
                mbp = self.model._speculate(state)
            return mbp[0, :n_draft]

        def tool_turn() -> bool:
            """After a <|result|> stop: run the named tool, inject `<|result|> result <|assistant|>` into the
            state, and return True so decoding resumes. False = the stop ends the reply."""
            nonlocal state, logits, calls_made
            if not call_buf or calls_made >= params.max_calls or stop.is_set():
                return False
            call_text = bytes(call_buf.pop()).decode("utf-8", errors="replace")
            call_buf.clear()
            tool, args = parse_call(call_text)
            fn = self.tools.get(tool)
            t_tool = time.perf_counter()
            try:
                result = fn(args) if fn is not None else f"(no such tool: {tool})"
            except Exception as e:  # a failing tool is a result the model can read, never a crashed reply
                result = f"(tool error: {type(e).__name__}: {str(e)[:200]})"
            rb = list(str(result).encode("utf-8")[: self.tool_result_limit])
            room = limit - 1 - (P + len(generated)) - 2
            if room <= 0:
                return False
            inj = [RESULT_ID] + rb[:room] + [ASSISTANT_ID]
            lg, _, _ = self.model.forward_from_state(torch.tensor([inj], device=self.device), state)
            logits = lg[0, -1]
            generated.extend(inj)
            reply_mask.extend([0] * len(inj))
            calls_made += 1
            emit({"type": "tool", "index": calls_made, "tool": tool, "args": args, "call": call_text, "result": result,
                  "result_bytes": len(rb[:room]), "truncated": len(rb) > room, "t_ms": (time.perf_counter() - t_tool) * 1000})
            return True

        draft = None
        use_graph = self._graph_ok and self.arena is not None and n_draft <= 0 and not script
        while use_graph:
            # one CUDA graph per byte, sampling on the device, K bytes per host sync (mote/serve/graph.py)
            reason, state, logits, stop_id = self._graph_decode(state, logits, params, P, limit, len(generated), record, stop, timing)
            if reason == "eos" and stop_id == RESULT_ID and tool_turn():
                continue
            break
        while not use_graph and len(generated) < params.max_bytes:
            if stop.is_set():
                reason = "stopped"
                break
            if P + len(generated) >= limit - 1:
                reason = "context"
                break
            room = min(params.max_bytes - len(generated), limit - 1 - (P + len(generated)))
            t_round = time.perf_counter()

            if draft is not None and room >= 2:
                # ---- speculative round: draw the draft, verify it in ONE pass from a snapshot, exact acceptance
                k = min(int(draft.shape[0]), room - 1)
                q_dists, xs = [], []
                trans = getattr(self.model.mbp_head, "transition", None)
                prev = generated[-1] if generated else prompt_ids[-1]  # the chunk's first byte
                for m in range(k):
                    base = draft[m]
                    if trans is not None:  # condition this slot on the draft byte actually sampled before it
                        base = base.float() + trans.weight[prev]
                    qd = _dist(base, params.temperature, params.top_p)
                    b = _draw(qd)
                    if b in STOP_IDS:
                        break  # the target decides about stopping; the draft never proposes it
                    q_dists.append(qd)
                    xs.append(b)
                    prev = b
                draft = None
                if not xs:
                    continue
                spec["rounds"] += 1
                spec["proposed"] += len(xs)
                snapshot = self.model.clone_state(state)
                lg_seq, bm_seq, bp_seq = self.model.forward_from_state(torch.tensor([xs], device=self.device), state)
                target_logits = [logits] + [lg_seq[0, m] for m in range(len(xs) - 1)]
                p_dists = [_dist(lg, params.temperature, params.top_p) for lg in target_logits]
                n_acc, fix, fix_p = verify_draft(xs, q_dists, p_dists)
                spec["accepted"] += n_acc
                n_out = n_acc + (1 if fix is not None else 0)
                if fix is None:
                    per_byte_ms = (time.perf_counter() - t_round) * 1000 / max(n_out, 1)
                    for m in range(n_acc):
                        record(xs[m], float(p_dists[m][xs[m]]), entropy_of(target_logits[m]), bool(bm_seq[m]), float(bp_seq[m]), "mbp", per_byte_ms)
                    timing["spec_s"] += time.perf_counter() - t_round
                    timing["spec_bytes"] += n_out
                    logits = lg_seq[0, -1]
                    draft = new_draft(bool(bm_seq[-1]), None)
                    if len(generated) - last_stats_bytes >= 16:
                        last_stats_bytes = len(generated)
                        emit({"type": "stats", **stats()})
                    continue
                # rejection at position n_acc: roll back and run ONE forward over accepted prefix + correction
                state = snapshot
                if fix in STOP_IDS:
                    for m in range(n_acc):  # accepted bytes are real output; commit them before stopping
                        record(xs[m], float(p_dists[m][xs[m]]), entropy_of(target_logits[m]), bool(bm_seq[m]), float(bp_seq[m]), "mbp", 0.0)
                    if n_acc > 0:  # the state must have read exactly the recorded bytes (the reply snapshot relies on it)
                        lg_acc, _, _ = self.model.forward_from_state(torch.tensor([xs[:n_acc]], device=self.device), state)
                        logits = lg_acc[0, -1]
                    if fix == RESULT_ID and tool_turn():
                        continue
                    reason, stop_id = "eos", fix
                    break
                spec["fixes"] += 1
                if n_acc > 0:
                    spec["replays"] += 1
                seq = xs[:n_acc] + [fix]
                lg_fix, bm_fix, bp_fix = self.model.forward_from_state(torch.tensor([seq], device=self.device), state)
                per_byte_ms = (time.perf_counter() - t_round) * 1000 / n_out
                for m in range(n_acc):
                    record(xs[m], float(p_dists[m][xs[m]]), entropy_of(target_logits[m]), bool(bm_fix[m]), float(bp_fix[m]), "mbp", per_byte_ms)
                record(fix, fix_p, entropy_of(target_logits[n_acc]), bool(bm_fix[n_acc]), float(bp_fix[n_acc]), "fix", per_byte_ms)
                timing["spec_s"] += time.perf_counter() - t_round
                timing["spec_bytes"] += n_out
                logits = lg_fix[0, -1]
                draft = new_draft(bool(bm_fix[n_acc]), None)
            else:
                # ---- plain step: one byte from the target
                if script:
                    byte = script.pop(0)
                    p, entropy = float(torch.softmax(logits.float(), dim=-1)[byte]), entropy_of(logits)
                else:
                    byte, p, entropy = _sample(logits, params.temperature, params.top_p)
                if byte in STOP_IDS:
                    if byte == RESULT_ID and tool_turn():
                        draft = None
                        continue
                    reason, stop_id = "eos", byte
                    break
                lg_next, routing, is_b, mbp = self.model.step(torch.tensor([[byte]], device=self.device), state)
                record(byte, p, entropy, bool(is_b), float(routing.boundary_prob[0, 1]), "nbp", (time.perf_counter() - t_round) * 1000)
                timing["plain_s"] += time.perf_counter() - t_round
                timing["plain_bytes"] += 1
                logits = lg_next[0, -1]
                draft = new_draft(bool(is_b), mbp)
            if len(generated) - last_stats_bytes >= 16:
                last_stats_bytes = len(generated)
                emit({"type": "stats", **stats()})
        self._commit("reply", prompt_ids + generated, state, logits, session)  # S2: the next turn starts here
        done = {"type": "done", "reason": reason, "text": "".join(text_parts), "calls": calls_made, "stats": stats()}
        if context.get("want_ids"):  # the RL driver needs the exact byte ids and the loss mask, not the text
            done.update({"prompt_ids": list(prompt_ids), "ids": list(generated), "mask": reply_mask, "eos": reason == "eos" and stop_id == EOS_ID})
        emit(done)

    def register_tool(self, name: str, fn: Callable[[str], str]) -> None:
        """Route `<|call|>name: args<|result|>` to fn(args) -> result text (search, the sim environment, ...)."""
        self.tools[name.strip().lower()] = fn

    def _graph_decode(self, state, logits, params: GenParams, P: int, limit: int, n_done: int, record, stop: threading.Event, timing: dict):
        gd = self._graph_decoder()
        max_out = max(min(params.max_bytes - n_done, limit - 1 - P - n_done), 0)
        outer = [n for n, m in self.model.named_modules() if isinstance(m, Mamba3Mixer)]
        rel = [n for n, m in self.model.named_modules() if isinstance(m, FullRelation)]
        count = [0]

        def on_bytes(recs):
            for r in recs:
                for name, ret in zip(outer, r["retention"]):
                    self._telemetry[name]["retention"] = ret
                if r["exchange"] is not None:
                    for name, xm in zip(rel, r["exchange"]):
                        self._telemetry[name]["exchange_mass"] = xm
                record(r["byte"], r["p"], r["entropy"], r["boundary"], r["boundary_p"], "nbp", r["t_ms"])
                count[0] += 1

        t0 = time.perf_counter()
        reason, st, lg = gd.run(state, logits, params.temperature, params.top_p, max_out, stop, on_bytes)
        timing["plain_s"] += time.perf_counter() - t0
        timing["plain_bytes"] += count[0]
        if reason == "max_bytes" and n_done + count[0] < params.max_bytes:
            reason = "context"
        return reason, st, lg, gd.stop_id


def discover_checkpoints(root: Path) -> List[Path]:
    """All run checkpoints under root/runs/*/ *.pt plus root/checkpoints/*.pt, newest first."""
    paths = list((root / "runs").glob("*/*.pt")) + list((root / "checkpoints").glob("*.pt"))
    return sorted({p.resolve() for p in paths}, key=lambda p: p.stat().st_mtime, reverse=True)
