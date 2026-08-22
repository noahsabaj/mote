import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

from ..core.sisa_attention import SISAAttention, SISAConfig
from ..core.mamba3_ssm import RMSNorm
from ..core.parameter_budget import match_sisa_ffn_dim
from .transformer_lm import SwiGLUFFN


@dataclass
class SISALMConfig:
    vocab_size: int = 261     # 256 bytes + 5 special tokens
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    d_head: int = 64
    d_s: int = 32             # SSM channel size
    d_ff: Optional[int] = None# If None, automatically computed via parameter matching
    d_ff_standard: int = 2048 # Baseline Transformer FFN dimension for parameter matching
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    init_b_alpha: float = -5.0
    init_lambda: float = 0.31
    clamp_minimax: float = 11.0
    dropout: float = 0.0
    bias: bool = False
    tie_word_embeddings: bool = True

    def __post_init__(self):
        if self.d_ff is None:
            self.d_ff = match_sisa_ffn_dim(
                self.d_model, self.n_heads, self.d_s, self.d_ff_standard
            )


class SISABlock(nn.Module):
    def __init__(self, config: SISALMConfig):
        super().__init__()
        attn_cfg = SISAConfig(
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_head=config.d_head,
            d_s=config.d_s,
            max_seq_len=config.max_seq_len,
            rope_theta=config.rope_theta,
            init_b_alpha=config.init_b_alpha,
            init_lambda=config.init_lambda,
            clamp_minimax=config.clamp_minimax,
            dropout=config.dropout,
            bias=config.bias,
        )
        self.norm1 = RMSNorm(config.d_model)
        self.attn = SISAAttention(attn_cfg)
        self.norm2 = RMSNorm(config.d_model)
        self.ffn = SwiGLUFFN(config.d_model, config.d_ff, bias=config.bias)

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[Dict[str, Any]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        attn_out, new_cache = self.attn(self.norm1(x), past_key_values=past_key_value, use_cache=use_cache)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, new_cache


class SISALanguageModel(nn.Module):
    """
    SISA (SSM-Informed Softmax Attention) Language Model.
    Fuses SSM dynamics directly inside the attention score with single-SDPA dispatch.
    """

    def __init__(self, config: SISALMConfig):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([SISABlock(config) for _ in range(config.n_layers)])
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
        past_key_values: Optional[List[Dict[str, Any]]] = None,
        use_cache: bool = False,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        B, L = input_ids.shape
        x = self.tok_embeddings(input_ids)

        new_caches = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            layer_cache = past_key_values[i] if past_key_values is not None else None
            x, new_cache = layer(x, past_key_value=layer_cache, use_cache=use_cache)
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
            "past_key_values": new_caches,
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
        """Autoregressive generation with SISA KV-caching."""
        self.eval()
        curr_ids = input_ids
        past_key_values = None

        for _ in range(max_new_tokens):
            if past_key_values is None:
                model_inputs = curr_ids
            else:
                model_inputs = curr_ids[:, -1:]

            outputs = self.forward(model_inputs, past_key_values=past_key_values, use_cache=True)
            logits = outputs["logits"][:, -1, :]
            past_key_values = outputs["past_key_values"]

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

        return curr_ids
