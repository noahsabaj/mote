import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

from ..core.rotary import apply_rotary_emb, compute_rope_frequencies
from ..core.mamba3_ssm import RMSNorm


@dataclass
class TransformerConfig:
    vocab_size: int = 261     # 256 bytes + 5 special tokens
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    d_head: int = 64
    d_ff: int = 2048
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    dropout: float = 0.0
    bias: bool = False
    tie_word_embeddings: bool = True


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network: output = (silu(W_gate x) * W_up x) W_down"""
    def __init__(self, d_model: int, d_ff: int, bias: bool = False):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=bias)
        self.w_up = nn.Linear(d_model, d_ff, bias=bias)
        self.w_down = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class StandardAttention(nn.Module):
    """Standard multi-head attention with RoPE and KV cache."""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.scale = 1.0 / math.sqrt(self.d_head)
        self.dropout = config.dropout

        self.q_proj = nn.Linear(self.d_model, self.n_heads * self.d_head, bias=config.bias)
        self.k_proj = nn.Linear(self.d_model, self.n_heads * self.d_head, bias=config.bias)
        self.v_proj = nn.Linear(self.d_model, self.n_heads * self.d_head, bias=config.bias)
        self.out_proj = nn.Linear(self.n_heads * self.d_head, self.d_model, bias=config.bias)

        cos, sin = compute_rope_frequencies(self.d_head, config.max_seq_len, theta=config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[Dict[str, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        B, L, _ = x.shape
        device = x.device
        dtype = x.dtype

        q = self.q_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        if past_key_value is not None:
            prev_k = past_key_value["key"]
            prev_v = past_key_value["value"]
            past_len = prev_k.shape[2]
            cos = self.rope_cos[past_len : past_len + L].to(dtype=dtype, device=device)
            sin = self.rope_sin[past_len : past_len + L].to(dtype=dtype, device=device)
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)
            k = torch.cat([prev_k, k], dim=2)
            v = torch.cat([prev_v, v], dim=2)
            new_cache = {"key": k, "value": v}
            out = F.scaled_dot_product_attention(
                q, k, v,
                scale=self.scale,
                is_causal=(L > 1 and prev_k is None),
                dropout_p=self.dropout if self.training else 0.0,
            )
        else:
            cos = self.rope_cos[:L].to(dtype=dtype, device=device)
            sin = self.rope_sin[:L].to(dtype=dtype, device=device)
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)
            out = F.scaled_dot_product_attention(
                q, k, v,
                scale=self.scale,
                is_causal=True,
                dropout_p=self.dropout if self.training else 0.0,
            )
            new_cache = {"key": k, "value": v} if use_cache else None

        out = out.transpose(1, 2).contiguous().view(B, L, self.n_heads * self.d_head)
        return self.out_proj(out), new_cache


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attn = StandardAttention(config)
        self.norm2 = RMSNorm(config.d_model)
        self.ffn = SwiGLUFFN(config.d_model, config.d_ff, bias=config.bias)

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[Dict[str, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        attn_out, new_cache = self.attn(self.norm1(x), past_key_value=past_key_value, use_cache=use_cache)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, new_cache


class TransformerLM(nn.Module):
    """Decoder-only Transformer Language Model."""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
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
        past_key_values: Optional[List[Dict[str, torch.Tensor]]] = None,
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
        """Autoregressive generation with KV caching."""
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
