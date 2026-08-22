import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List

from ..core.mamba3_ssm import RMSNorm
from .sisa_lm import SISABlock, SISALMConfig
from .mamba3_lm import Mamba3Block, Mamba3LMConfig
from .transformer_lm import TransformerBlock, TransformerConfig


@dataclass
class HybridConfig:
    vocab_size: int = 261
    d_model: int = 512
    n_heads: int = 8
    d_head: int = 64
    d_state: int = 64
    d_s: int = 32
    d_ff: int = 2048
    max_seq_len: int = 2048
    layer_pattern: List[str] = field(default_factory=lambda: ["mamba3", "mamba3", "mamba3", "mamba3", "mamba3", "sisa"])
    mimo_rank: int = 1
    rope_theta: float = 10000.0
    bias: bool = False
    tie_word_embeddings: bool = True


class HybridLanguageModel(nn.Module):
    """
    Hybrid Language Model interleaving Mamba-3 SSM layers with SISA / Attention layers.
    Maintains both recurrent states for SSM blocks and KV-caches for SISA/Attention blocks.
    """

    def __init__(self, config: HybridConfig):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)

        self.layers = nn.ModuleList()
        self.layer_types = config.layer_pattern

        sisa_cfg = SISALMConfig(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_head=config.d_head,
            d_s=config.d_s,
            d_ff=config.d_ff,
            max_seq_len=config.max_seq_len,
            rope_theta=config.rope_theta,
            bias=config.bias,
        )
        mamba3_cfg = Mamba3LMConfig(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            d_state=config.d_state,
            d_head=config.d_head,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            mimo_rank=config.mimo_rank,
            max_seq_len=config.max_seq_len,
            bias=config.bias,
        )
        trans_cfg = TransformerConfig(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_head=config.d_head,
            d_ff=config.d_ff,
            max_seq_len=config.max_seq_len,
            rope_theta=config.rope_theta,
            bias=config.bias,
        )

        for l_type in self.layer_types:
            if l_type.lower() in ("mamba3", "ssm"):
                self.layers.append(Mamba3Block(mamba3_cfg))
            elif l_type.lower() == "sisa":
                self.layers.append(SISABlock(sisa_cfg))
            elif l_type.lower() in ("transformer", "attn", "attention"):
                self.layers.append(TransformerBlock(trans_cfg))
            else:
                raise ValueError(f"Unknown layer type: {l_type}")

        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.tok_embeddings.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        layer_caches: Optional[List[Any]] = None,
        use_cache: bool = False,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        B, L = input_ids.shape
        x = self.tok_embeddings(input_ids)

        new_caches = [] if use_cache else None
        for i, (layer, l_type) in enumerate(zip(self.layers, self.layer_types)):
            cache_i = layer_caches[i] if layer_caches is not None else None
            if l_type.lower() in ("mamba3", "ssm"):
                x, new_cache = layer(x, recurrent_state=cache_i, use_state=use_cache)
            elif l_type.lower() == "sisa":
                x, new_cache = layer(x, past_key_value=cache_i, use_cache=use_cache)
            else:
                x, new_cache = layer(x, past_key_value=cache_i, use_cache=use_cache)

            if use_cache:
                new_caches.append(new_cache)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        return {
            "logits": logits,
            "loss": loss,
            "layer_caches": new_caches,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressive generation for Hybrid model with dual state & KV caching."""
        self.eval()
        curr_ids = input_ids
        outputs = self.forward(curr_ids, use_cache=True)
        logits = outputs["logits"][:, -1, :]
        layer_caches = outputs["layer_caches"]

        for _ in range(max_new_tokens):
            if temperature > 0.0:
                logits = logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = -float("Inf")
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            curr_ids = torch.cat([curr_ids, next_token], dim=1)
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

            outputs = self.forward(next_token, layer_caches=layer_caches, use_cache=True)
            logits = outputs["logits"][:, -1, :]
            layer_caches = outputs["layer_caches"]

        return curr_ids
