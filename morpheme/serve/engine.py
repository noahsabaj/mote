"""Generation engine: prefill + byte-at-a-time decoding with multi-byte speculative acceptance,
emitting the telemetry events described in docs/api.md. Synchronous; run it in a worker thread.
"""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from ..config import MorphemeConfig
from ..model.hnet import HNetForCausalLM
from ..model.mamba3 import HAS_MAMBA3_KERNEL, Mamba3Mixer
from ..model.dc import HAS_SSD_KERNEL
from ..model.relation import FullRelation
from ..tokenizer import ASSISTANT_ID, BOS_ID, EOS_ID, PAD_ID, SYSTEM_ID, USER_ID, ByteTokenizer, ChatMessage, Utf8Streamer

STOP_IDS = {EOS_ID, PAD_ID, SYSTEM_ID, USER_ID, ASSISTANT_ID, BOS_ID}


@dataclass
class GenParams:
    temperature: float = 0.8
    top_p: float = 0.9
    max_bytes: int = 512
    accept_threshold: float = 0.9
    n_candidates: int = 3

    @classmethod
    def from_dict(cls, d: Optional[dict], defaults: "GenParams") -> "GenParams":
        d = d or {}
        return cls(
            temperature=float(d.get("temperature", defaults.temperature)),
            top_p=float(d.get("top_p", defaults.top_p)),
            max_bytes=int(d.get("max_bytes", defaults.max_bytes)),
            accept_threshold=float(d.get("accept_threshold", defaults.accept_threshold)),
            n_candidates=int(d.get("n_candidates", defaults.n_candidates)),
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
    def __init__(self, ckpt_path: str | Path, device: Optional[str] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.ckpt_path = Path(ckpt_path)
        ck = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        self.cfg = MorphemeConfig.from_dict(ck["config"])
        self.model = HNetForCausalLM(self.cfg)
        self.model.load_state_dict(ck["model"])
        self.model.to(self.device).eval()
        self.tok = ByteTokenizer()
        self.lock = threading.Lock()
        extra = ck.get("extra", {})
        step = int(ck.get("step", 0))
        self.info_ckpt = self._describe_checkpoint(self.ckpt_path, step, extra)
        self.defaults = GenParams(max_bytes=min(512, self.cfg.max_seq_len // 2))
        self._telemetry: Dict[str, dict] = {}
        self._attach_telemetry()

    # ------------------------------------------------------------------------------
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

    def info(self) -> dict:
        dev = {"name": "cpu", "vram_total_mb": 0, "vram_used_mb": 0}
        if self.device.type == "cuda":
            props = torch.cuda.get_device_properties(self.device)
            dev = {"name": props.name, "vram_total_mb": props.total_memory // 2**20, "vram_used_mb": torch.cuda.memory_allocated(self.device) // 2**20}
        c = self.info_ckpt
        return {
            "name": f"{Path(c.path).parent.name}/{Path(c.path).name}",
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
        }

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
    def generate(self, messages: Sequence[dict], params: GenParams, emit: Callable[[dict], None], stop: threading.Event) -> None:
        with self.lock:
            self._generate(messages, params, emit, stop)

    def _generate(self, messages, params: GenParams, emit, stop: threading.Event) -> None:
        limit = self.cfg.max_seq_len
        prompt_ids, truncated = self.build_prompt(messages, limit, reserve=min(params.max_bytes, limit // 4))
        ids = torch.tensor([prompt_ids], device=self.device)
        t0 = time.perf_counter()
        state = self.model.allocate_inference_state(self.device)
        out = self.model.prefill(ids, state)
        logits = out.logits[0, -1]
        n_chunks = int(out.chunk_id[0, -1]) + 1
        emit({"type": "start", "prompt_bytes": len(prompt_ids), "context_bytes": len(prompt_ids), "context_limit": limit, "truncated": truncated})

        streamer = Utf8Streamer()
        boundary_probs: List[float] = []
        generated: List[int] = []
        text_parts: List[str] = []
        candidates: List[tuple] = []  # (byte, p) accepted from the multi-byte head, in order
        mbp_proposed = mbp_accepted = 0
        chunk_start = len(prompt_ids)  # absolute index where the current chunk began
        chunk_index = n_chunks - 1
        reason = "max_bytes"
        last_stats_bytes = 0

        def stats() -> dict:
            elapsed = time.perf_counter() - t0
            n = len(generated)
            return {
                "bytes": n, "elapsed_ms": elapsed * 1000, "bytes_per_sec": n / elapsed if elapsed > 0 else 0.0,
                "chunks": chunk_index + 1 - (n_chunks - 1), "bytes_per_chunk": n / max(chunk_index + 1 - (n_chunks - 1), 1),
                "mbp_proposed": mbp_proposed, "mbp_accepted": mbp_accepted,
                "mbp_accept_rate": mbp_accepted / mbp_proposed if mbp_proposed else 0.0,
                "context_bytes": len(prompt_ids) + n, "context_limit": limit,
            }

        while len(generated) < params.max_bytes:
            if stop.is_set():
                reason = "stopped"
                break
            if len(prompt_ids) + len(generated) >= limit - 1:
                reason = "context"
                break
            t_byte = time.perf_counter()
            if candidates:
                byte, p = candidates.pop(0)
                source = "mbp"
                raw = torch.softmax(logits.float(), -1)
                entropy = float(-(raw * torch.log2(raw.clamp_min(1e-12))).sum())
            else:
                byte, p, entropy = _sample(logits, params.temperature, params.top_p)
                source = "nbp"
            if byte in STOP_IDS:
                reason = "eos"
                break
            step_in = torch.tensor([[byte]], device=self.device)
            logits_next, routing, is_b, mbp = self.model.step(step_in, state)
            abs_i = len(prompt_ids) + len(generated)
            if is_b:
                if abs_i > chunk_start:
                    emit({"type": "chunk", "index": chunk_index, "start": chunk_start, "end": abs_i - 1, "bytes": abs_i - chunk_start,
                          "text": bytes(b for b in (prompt_ids + generated)[chunk_start:abs_i] if b < 256).decode("utf-8", errors="replace")})
                chunk_index += 1
                chunk_start = abs_i
                candidates = []
                if mbp is not None and params.n_candidates > 0:
                    probs = torch.softmax(mbp[0, : params.n_candidates].float(), -1)  # [n, V]
                    mbp_proposed += probs.shape[0]
                    for row in probs:
                        pm, bm = float(row.max()), int(row.argmax())
                        if pm >= params.accept_threshold and bm not in STOP_IDS:
                            candidates.append((bm, pm))
                        else:
                            break
                    mbp_accepted += len(candidates)
                emit({"type": "diagnostics", **self._diagnostics(boundary_probs)})
            bp = float(routing.boundary_prob[0, 1])
            boundary_probs.append(bp)
            generated.append(byte)
            text = streamer.feed(byte)
            if text:
                text_parts.append(text)
            emit({
                "type": "byte", "i": len(generated) - 1, "byte": byte, "text": text or None, "pending": len(streamer.pending),
                "p": p, "entropy": entropy, "boundary": is_b, "boundary_p": bp, "chunk": chunk_index,
                "source": source, "t_ms": (time.perf_counter() - t_byte) * 1000,
            })
            logits = logits_next[0, -1]
            if len(generated) - last_stats_bytes >= 16:
                last_stats_bytes = len(generated)
                emit({"type": "stats", **stats()})
        emit({"type": "done", "reason": reason, "text": "".join(text_parts), "stats": stats()})


def discover_checkpoints(root: Path) -> List[Path]:
    """All run checkpoints under root/runs/*/ *.pt plus root/checkpoints/*.pt, newest first."""
    paths = list((root / "runs").glob("*/*.pt")) + list((root / "checkpoints").glob("*.pt"))
    return sorted({p.resolve() for p in paths}, key=lambda p: p.stat().st_mtime, reverse=True)
