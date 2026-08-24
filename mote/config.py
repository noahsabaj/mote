"""Model configuration for the byte-level H-Net.

One stage: bytes -> Mamba-3 encoder -> dynamic chunking -> Relation main network ->
dechunk -> Mamba-3 decoder -> next-byte head (+ multi-byte prediction head).
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
    is_mimo: bool = False
    mimo_rank: int = 1
    is_outproj_norm: bool = False
    A_floor: float = 1e-4


@dataclass
class RelationCfg:
    # None = full attention over all chunks; an int limits each chunk to the last N chunks
    # (windowed-main A/B, docs/context.md; forces the materialized path until the kernel learns windows)
    """Full Relation main network (Ge, Yang, Nie 2026)."""

    n_layers: int = 6
    mixer: str = "relation"  # "attention" = parameter-matched causal-attention ablation control
    window_chunks: int | None = None
    d_model: int = 384
    n_heads: int = 8  # must be even (Givens head pairs)
    d_ff: int = 768
    tau_s: float = 2.0  # Self temperature
    lambda_init: float = 0.5  # count-calibration λ_ℓ, one FP32 scalar per layer
    rope_theta: float = 10000.0
    givens: bool = True  # learnable adjacent-head Givens rotations on the information branch
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
class MBPCfg:
    """Multi-byte prediction head with Latent Causal Attention (Owodunni et al. 2026)."""

    enabled: bool = True
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 768
    n_candidates: int = 3  # draft length per boundary at inference (verified exactly)
    loss_weight: float = 1.0  # λ1 in L = λ0·L_nbp + λ1·L_mbp + α·L_ratio
    position_gamma: float = 0.0  # >0: weight the head's loss by exp(-offset/γ) (DFlash/DSpark position weighting)
    transition: bool = False  # first-order byte-transition bias on the head's logits (DSpark's Markov head; V×V at byte vocab)


@dataclass
class DCCfg:
    """Dynamic chunking (Hwang, Wang, Gu 2025) with the ATDC ratio schedule (Dang et al. 2026)."""

    target_ratio_init: float = 5.0  # N at the start of training
    target_ratio_final: float = 6.5  # N at the end of training
    schedule_warmup_frac: float = 0.6  # hold N_init for this fraction of training, then ramp linearly
    ratio_loss_weight: float = 0.03  # α
    prob_clamp: float = 1e-4  # p clamped to [ε, 1-ε] before the EMA
    chunk_bucket: int = 64  # pad the chunk count to a multiple of this so shapes repeat (1 = exact); bit-neutral



def _make(cls, d: dict):
    """Build a config dataclass from a dict, ignoring keys the class no longer has (old checkpoints)."""
    return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class MoteConfig:
    vocab_size: int = VOCAB_SIZE
    pad_vocab_to: int = 264  # embedding rows rounded up for alignment; ids >= vocab_size are never produced
    d_model_outer: int = 256  # encoder / decoder width (bytes)
    encoder_layers: int = 2  # Mamba-3 layers, no FFN ("m" blocks in H-Net notation)
    decoder_layers: int = 2
    main: RelationCfg = field(default_factory=RelationCfg)
    mbp: MBPCfg = field(default_factory=MBPCfg)
    dc: DCCfg = field(default_factory=DCCfg)
    mamba3: Mamba3Cfg = field(default_factory=Mamba3Cfg)
    max_seq_len: int = 2048  # bytes
    tie_embeddings: bool = True
    norm_eps: float = 1e-5
    initializer_range: float = 0.02
    residual_in_fp32: bool = True

    # ----------------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MoteConfig":
        d = dict(d)
        return cls(
            main=_make(RelationCfg, d.pop("main", {})),
            mbp=_make(MBPCfg, d.pop("mbp", {})),
            dc=_make(DCCfg, d.pop("dc", {})),
            mamba3=_make(Mamba3Cfg, d.pop("mamba3", {})),
            **d,
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "MoteConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    # --- presets ------------------------------------------------------------------
    @classmethod
    def pilot(cls) -> "MoteConfig":
        """~12M params: the local 4060 Ti gate run (Relation 6L/384/8×48/768 = the paper's 10M setting)."""
        return cls(
            d_model_outer=256,
            encoder_layers=2,
            decoder_layers=2,
            main=RelationCfg(n_layers=6, d_model=384, n_heads=8, d_ff=768),
            mbp=MBPCfg(n_layers=2, n_heads=4, d_ff=768),
            max_seq_len=2048,
        )

    @classmethod
    def smoke(cls) -> "MoteConfig":
        """~2M params: the T1 bug gate (docs/shape.md 2026-08-24). Ten-minute runs that answer
        "is this broken" — never "is this better"; tiny scale issues no quality verdicts."""
        return cls(
            d_model_outer=128,
            encoder_layers=1,
            decoder_layers=1,
            main=RelationCfg(n_layers=2, d_model=192, n_heads=4, d_ff=384),
            mbp=MBPCfg(n_layers=1, n_heads=2, d_ff=256),
            max_seq_len=2048,
        )

    @classmethod
    def local(cls) -> "MoteConfig":
        """~35M params: the largest comfortable overnight run on the 8 GB RTX 4060 Ti (5.9 GB peak at batch 4x2048)."""
        return cls(
            d_model_outer=384,
            encoder_layers=2,
            decoder_layers=2,
            main=RelationCfg(n_layers=8, d_model=512, n_heads=8, d_ff=1536),
            mbp=MBPCfg(n_layers=2, n_heads=4, d_ff=1024),
            max_seq_len=2048,
        )

    @classmethod
    def flagship(cls) -> "MoteConfig":
        """~105M params, 16384-byte window; trains locally on the 4060 Ti (42 KB/s, ~44% MFU at batch 1 in the forced-6-bytes/chunk profile)."""
        return cls(
            d_model_outer=512,
            encoder_layers=3,
            decoder_layers=3,
            main=RelationCfg(n_layers=12, d_model=768, n_heads=8, d_ff=2048),
            mbp=MBPCfg(n_layers=2, n_heads=8, d_ff=2048, enabled=False),  # head off for the flagship (decided 2026-08-23: no loss gain under Muon, 8-13% of step time)
            max_seq_len=16384,  # decided 2026-08-23 (docs/context.md); profiled at batch 1 + ckpt: 6.34 GB peak on the 4060 Ti
        )
