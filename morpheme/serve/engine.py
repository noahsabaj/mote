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
    n_candidates: int = 3  # draft length per chunk boundary (0 disables speculation); verification is exact

    @classmethod
    def from_dict(cls, d: Optional[dict], defaults: "GenParams") -> "GenParams":
        d = d or {}
        return cls(
            temperature=float(d.get("temperature", defaults.temperature)),
            top_p=float(d.get("top_p", defaults.top_p)),
            max_bytes=int(d.get("max_bytes", defaults.max_bytes)),
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
    @torch.no_grad()
    def warmup(self) -> float:
        """Trigger Triton JIT/autotune for prefill at a few prompt lengths (incl. odd and 16-aligned ones) and one
        decode step, so the first user request doesn't pay the ~40 s compile. Returns seconds spent."""
        t0 = time.perf_counter()
        with self.lock:
            for L in (5, 16, 33, 128, 257, 512, 640, 1024, 2048, 4096):
                L = min(L, self.cfg.max_seq_len - 4)
                ids = torch.randint(0, 256, (1, L), device=self.device)
                state = self.model.allocate_inference_state(self.device)
                self.model.prefill(ids, state)
                self.model.step(torch.tensor([[65]], device=self.device), state)
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
        P = len(prompt_ids)
        emit({"type": "start", "prompt_bytes": P, "context_bytes": P, "context_limit": limit, "truncated": truncated})

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

        draft = None
        while len(generated) < params.max_bytes:
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
                    reason = "eos"
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
                byte, p, entropy = _sample(logits, params.temperature, params.top_p)
                if byte in STOP_IDS:
                    reason = "eos"
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
        emit({"type": "done", "reason": reason, "text": "".join(text_parts), "stats": stats()})


def discover_checkpoints(root: Path) -> List[Path]:
    """All run checkpoints under root/runs/*/ *.pt plus root/checkpoints/*.pt, newest first."""
    paths = list((root / "runs").glob("*/*.pt")) + list((root / "checkpoints").glob("*.pt"))
    return sorted({p.resolve() for p in paths}, key=lambda p: p.stat().st_mtime, reverse=True)
