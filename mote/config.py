"""Model configuration for the byte-level H-Net.

One stage: bytes -> Mamba-3 encoder -> dynamic chunking -> Relation main network ->
dechunk -> Mamba-3 decoder -> next-byte head.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict

from .tokenizer import VOCAB_SIZE


@dataclass
class Mamba3Cfg:
    """Arguments forwarded to the official ``mamba_ssm.modules.mamba3.Mamba3`` mixer."""

    d_state: int = 64
    headdim: int = 64
    expand: int = 2
    ngroups: int = 1
    rope_fraction: float = 0.5
    chunk_size: int = 64
    A_floor: float = 1e-4


@dataclass
class RelationCfg:
    """Full Relation main network (Ge, Yang, Nie 2026)."""

    n_layers: int = 6
    d_model: int = 384
    n_heads: int = 8  # must be even (Givens head pairs)
    d_ff: int = 768
    tau_s: float = 2.0  # Self temperature
    lambda_init: float = 0.5  # count-calibration λ_ℓ, one FP32 scalar per layer
    rope_theta: float = 10000.0
    givens: bool = True  # learnable adjacent-head Givens rotations on the information branch
    # QK-Norm on p1/p2 before RoPE (2608.24814 §4.2 measures it as the biggest lever on how precisely
    # ELR collapse holds; not loss-neutral in Relation, see relation.HeadRMSNorm). Off until its arm passes.
    qk_norm: bool = False
    # Mixture of experts in the FFN slot (signed 2026-08-24, docs/shape.md "MoE"; mote/model/moe.py)
    moe_experts: int = 0  # 0 = dense SwiGLU; E ≥ 2 = MoESwiGLU with E experts
    moe_topk: int = 2
    moe_d_ff: int | None = None  # expert hidden width; None = d_ff // moe_topk (active FLOPs match the dense FFN)
    moe_router: str = "lossfree"  # "lossfree" (DeepSeek-V3 bias balancing + seq-level balance loss) | "aux" (Switch softmax + balance loss + z-loss)
    moe_dense_first: bool = False  # layer 0 keeps the dense FFN (DeepSeek-V3 / Kakao 2608.20061 convention)
    moe_aux_weight: float | None = None  # None = 1e-4 (lossfree) / 1e-2 (aux)
    moe_z_weight: float = 1e-3  # router z-loss (aux router only)
    moe_bias_gamma: float = 1e-3  # lossfree: expert-bias step per optimizer step
    moe_gate_scale: float | None = None  # None = Moonlight's computed factor (lossfree) / 1.0 (aux)


@dataclass
class DCCfg:
    """Dynamic chunking (Hwang, Wang, Gu 2025) with the ATDC ratio schedule (Dang et al. 2026)."""

    target_ratio_init: float = 5.0  # N at the start of training
    target_ratio_final: float = 6.5  # N at the end of training
    schedule_warmup_frac: float = 0.6  # hold N_init for this fraction of training, then ramp linearly
    ratio_loss_weight: float = 0.03  # α
    prob_clamp: float = 1e-4  # p clamped to [ε, 1-ε] before the EMA
    chunk_bucket: int = 64  # pad the chunk count to a multiple of this so shapes repeat (1 = exact); bit-neutral

    # --- ATDC's metric-based adaptation: 2605.30080 Alg. 1 lines 10-15 --------------------------
    # PROVENANCE (2026-08-27). Every constant above came from that paper — N_init 5.0, N_fnl 6.5,
    # T_w 60 % are its §V-C-1 verbatim — but Mote shipped only its eq. (7) linear ramp, so N rose on
    # wall-clock whether or not the model was ready for it. These three restore the other half: N is
    # boosted by `adapt_rate` only while the windowed mean LM loss sits below `adapt_threshold`.
    #
    # Do not read `target_ratio_*` as a bytes-per-chunk setting. It is not one, and never was. The
    # ratio loss (see dc.py::ratio_loss) has NO gradient path to the achieved segmentation rate — its
    # only lever is the mean boundary probability — so the target is a soft pull the model is free to
    # ignore, which the paper says outright ("a soft constraint ... allowing the model to deviate").
    # Its own numbers: fixed N of 7.0 / 8.0 / 9.0 produced BPIC 5.11 / 5.42 / 5.68 (Table III), i.e.
    # 2.0 of target bought 0.57 of BPIC. Measured on Mote 2026-08-27 across three runs, N ramping
    # 5.0 -> 6.5 moved val_bpic 3.20 -> 3.32. The achieved rate is an OBSERVABLE (log.jsonl's
    # val_bpic); anything that needs the number reads it from there, never from these fields.
    adapt_window: int = 100  # W: steps of LM loss averaged before the trigger may fire
    adapt_threshold: float | None = None  # τ: boost N only while the windowed loss is under this (None = no trigger)
    adapt_rate: float = 1.05  # γ: multiplicative boost applied to N_sched when the trigger fires

    # --- bounded routing: GeneZip 2602.17739 §3.3 ----------------------------------------------
    # A projection, not a loss. After thresholding gives a provisional mask, flip the fewest
    # decisions (by p-confidence) needed to land the boundary count inside [floor, ceiling]. The
    # paper uses it as an OOM guardrail — absolute K_min=8, and K_max only on the model that OOM'd
    # without it — so the defaults here are a guardrail too: the ceiling exists to stop the serving
    # arena overflowing its capacity mid-conversation, not to force a compression rate. Tightening
    # `bound_ceiling` toward 1.0 turns it into a rate controller, which is beyond the paper's
    # evidence and gets its own A/B before it is used that way.
    bound_floor: int = 0  # K_min, absolute boundaries per sequence (0 = no floor)
    bound_ceiling: float | None = None  # ρ_max: ceiling as a multiple of the target rate L/N (None = unbounded)
    # Decode sees one byte and has no sequence to project over, so it keeps a plain threshold. 0.5 is
    # H-Net's (cos_sim < 0 — consecutive states more than 90° apart); a bounded-routing run calibrates
    # this to the rate its projection actually produces and logs it beside val_bpic.
    decode_threshold: float = 0.5



def _make(cls, d: dict):
    """Build a config dataclass from a dict, ignoring keys the class no longer has (old checkpoints)."""
    return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class FeedbackCfg:
    """Latent feedback (2608.08888, signed 2026-08-28, docs/results/2026-08-28-latent-feedback-prereg.md):
    the previous position's top state fused into the next input through a GLU, u = RMSNorm(W_U h_prev
    ⊙ σ(W_G x)) — the state on the value path, the plain input only as the gate. `byte` fuses the
    decoder's top state into the next byte's encoder input (the router sees the fully processed past);
    `chunk` fuses the main network's top state into the next chunk's main input (the encoder and
    routing are reused across passes). Trained by parallel multi-pass teacher forcing (train.py)."""

    level: str = "off"  # off | byte | chunk
    jitter: float = 0.02  # uniform ±jitter on the carried state in training passes (the paper's σ)


@dataclass
class MoteConfig:
    vocab_size: int = VOCAB_SIZE
    pad_vocab_to: int = 272  # embedding rows (16-aligned); logits for ids >= vocab_size are masked to -inf by the head
    # (2026-08-24 night: 266 ids + 6 spare rows so future protocol ids need no checkpoint surgery; old checkpoints keep 264)
    d_model_outer: int = 256  # encoder / decoder width (bytes)
    encoder_layers: int = 2  # Mamba-3 layers, no FFN ("m" blocks in H-Net notation)
    decoder_layers: int = 2
    main: RelationCfg = field(default_factory=RelationCfg)
    dc: DCCfg = field(default_factory=DCCfg)
    mamba3: Mamba3Cfg = field(default_factory=Mamba3Cfg)
    feedback: FeedbackCfg = field(default_factory=FeedbackCfg)
    max_seq_len: int = 2048  # bytes
    # Serving reads a prompt in windows of this many bytes instead of one pass. The Mamba-3 prefill
    # workspace is linear in the window (measured 2026-08-27: 25.1 KiB per prompt byte per layer, so
    # 401 MiB for one layer at 16384), and the CPU reference path is QUADRATIC in it, so the window
    # is what caps both. 4096 measured on Mote-138M at 16384: peak 865 -> 252 MiB, wall clock
    # 1556 -> 1534 ms, identical boundary sequence. 0 disables windowing (one-shot prefill).
    prefill_window: int = 4096
    tie_embeddings: bool = True
    norm_eps: float = 1e-5
    initializer_range: float = 0.02
    residual_in_fp32: bool = True

    # ----------------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MoteConfig":
        """Keys this version no longer has are ignored at every level (`mbp`, `spine`, `window_chunks`, ...),
        so a checkpoint from before they were retired still loads."""
        d = dict(d)
        subs = {
            "main": _make(RelationCfg, d.pop("main", {})),
            "dc": _make(DCCfg, d.pop("dc", {})),
            "mamba3": _make(Mamba3Cfg, d.pop("mamba3", {})),
            "feedback": _make(FeedbackCfg, d.pop("feedback", {})),
        }
        return cls(**subs, **{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "MoteConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


    # --- presets ------------------------------------------------------------------
    # Named by exact rounded parameter count (2026-08-26). The old role names rotted: `local` runs on
    # the L4 too, and `flagship` stopped being the flagship the moment Mote-138M existed. A size does
    # not rot — but it can move: retiring the multi-byte head (2026-08-29) took 13M to 11M and 35M to
    # 32M. `PRESET_ALIASES` below keeps every retired name resolving, so queued argv still works.

    @classmethod
    def mote_1m(cls) -> "MoteConfig":
        """1,060,758: the T1 bug gate (docs/shape.md 2026-08-24). Ten-minute runs that answer
        "is this broken" — never "is this better"; tiny scale issues no quality verdicts."""
        return cls(
            d_model_outer=128,
            encoder_layers=1,
            decoder_layers=1,
            main=RelationCfg(n_layers=2, d_model=192, n_heads=4, d_ff=384),
            max_seq_len=2048,
        )

    @classmethod
    def mote_11m(cls) -> "MoteConfig":
        """10,870,110: the local 4060 Ti gate run (Relation 6L/384/8×48/768 = the paper's 10M setting)."""
        return cls(
            d_model_outer=256,
            encoder_layers=2,
            decoder_layers=2,
            main=RelationCfg(n_layers=6, d_model=384, n_heads=8, d_ff=768),
            max_seq_len=2048,
        )

    @classmethod
    def mote_32m(cls) -> "MoteConfig":
        """31,643,528: the largest comfortable overnight run on the 8 GB RTX 4060 Ti (5.9 GB peak at batch 4x2048)."""
        return cls(
            d_model_outer=384,
            encoder_layers=2,
            decoder_layers=2,
            main=RelationCfg(n_layers=8, d_model=512, n_heads=8, d_ff=1536),
            max_seq_len=2048,
        )

    @classmethod
    def mote_96m(cls) -> "MoteConfig":
        """95,924,732, 16384-byte window — the config FROZEN 2026-08-24 (docs/shape.md). Trains locally
        on the 4060 Ti at 68.1 KB/s, 4.31 GB peak, batch 1 x accum 4."""
        return cls(
            d_model_outer=512,
            encoder_layers=3,
            decoder_layers=3,
            main=RelationCfg(n_layers=12, d_model=768, n_heads=8, d_ff=2048),
            max_seq_len=16384,  # decided 2026-08-23 (docs/context.md); profiled at batch 1 + ckpt: 6.34 GB peak on the 4060 Ti
        )

    @classmethod
    def mote_138m(cls) -> "MoteConfig":
        """138,401,306 — Mote-96M taken deeper (main 12 → 18 layers). n_res 30 → 42, so `_init_weights`
        shrinks every out-projection's init
        by 1.18x; that is a transient, because Muon + weight decay drive ‖W‖ to an equilibrium
        ‖W‖ ∝ lr^0.478 that does not depend on where it started (docs/research/elr-2026-08-26.md)."""
        return cls(
            d_model_outer=512,
            encoder_layers=3,
            decoder_layers=3,
            main=RelationCfg(n_layers=18, d_model=768, n_heads=8, d_ff=2048),
            max_seq_len=16384,
        )

    # --- retired role names, kept resolving ---------------------------------------
    @classmethod
    def smoke(cls) -> "MoteConfig":
        return cls.mote_1m()

    @classmethod
    def pilot(cls) -> "MoteConfig":
        return cls.mote_11m()

    @classmethod
    def local(cls) -> "MoteConfig":
        return cls.mote_32m()

    @classmethod
    def flagship(cls) -> "MoteConfig":
        return cls.mote_96m()


# --- the preset registry ----------------------------------------------------------
# One source of truth. It used to live in mote/train/profile_step.py and omitted "smoke", while
# mote/train/train.py hardcoded a different list; the two drifted apart unnoticed.
PRESETS: Dict[str, Any] = {
    "mote-1m": MoteConfig.mote_1m,
    "mote-11m": MoteConfig.mote_11m,
    "mote-32m": MoteConfig.mote_32m,
    "mote-96m": MoteConfig.mote_96m,
    "mote-138m": MoteConfig.mote_138m,
}
PRESET_ALIASES: Dict[str, str] = {
    "smoke": "mote-1m",
    "pilot": "mote-11m",
    "local": "mote-32m",
    "flagship": "mote-96m",
    # the sizes before the multi-byte head was retired (2026-08-29): queued argv and run.json keep resolving
    "mote-13m": "mote-11m",
    "mote-35m": "mote-32m",
}
PRESET_NAMES = tuple(PRESETS) + tuple(PRESET_ALIASES)


def normalize_preset(name: str) -> str:
    """Canonical key for a user-supplied preset name. Accepts `mote-138m`, `mote_138m`, `138m` and the
    retired role names. Raises ValueError, which argparse renders as a clean `invalid value` message."""
    key = name.strip().lower().replace("_", "-")
    key = PRESET_ALIASES.get(key, key)
    if not key.startswith("mote-"):
        key = "mote-" + key
    if key not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; known: {', '.join(PRESET_NAMES)}")
    return key


normalize_preset.__name__ = "preset"  # argparse prints type.__name__ in its "invalid value" message


def resolve_preset(name: str) -> "MoteConfig":
    """Build a preset by name (see `normalize_preset` for the accepted spellings)."""
    return PRESETS[normalize_preset(name)]()
