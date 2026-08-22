import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

from ..core.mamba3_ssm import Mamba3SSM, Mamba3Config, RMSNorm
from .transformer_lm import SwiGLUFFN


@dataclass
class Mamba3LMConfig:
    vocab_size: int = 261
    d_model: int = 512
    n_layers: int = 6
    d_state: int = 64
    d_head: int = 64
    n_heads: int = 8
    d_ff: int = 2048
    mimo_rank: int = 1
    max_seq_len: int = 2048
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4
    bias: bool = False
    use_bc_norm: bool = True
    use_bc_bias: bool = True
    trapezoidal: bool = True
    complex_rope: bool = True
    tie_word_embeddings: bool = True


class Mamba3Block(nn.Module):
    def __init__(self, config: Mamba3LMConfig):
        super().__init__()
        ssm_cfg = Mamba3Config(
            d_model=config.d_model,
            d_state=config.d_state,
            d_head=config.d_head,
            n_heads=config.n_heads,
            mimo_rank=config.mimo_rank,
            dt_min=config.dt_min,
            dt_max=config.dt_max,
            dt_init_floor=config.dt_init_floor,
            bias=config.bias,
            use_bc_norm=config.use_bc_norm,
            use_bc_bias=config.use_bc_bias,
            trapezoidal=config.trapezoidal,
            complex_rope=config.complex_rope,
        )
        self.norm1 = RMSNorm(config.d_model)
        self.ssm = Mamba3SSM(ssm_cfg)
        self.norm2 = RMSNorm(config.d_model)
        self.ffn = SwiGLUFFN(config.d_model, config.d_ff, bias=config.bias)

    def forward(
        self,
        x: torch.Tensor,
        recurrent_state: Optional[Dict[str, Any]] = None,
        use_state: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        ssm_out, new_state = self.ssm(self.norm1(x), recurrent_state=recurrent_state, use_state=use_state)
        x = x + ssm_out
        x = x + self.ffn(self.norm2(x))
        return x, new_state


class Mamba3LanguageModel(nn.Module):
    """Pure Mamba-3 Language Model with alternating SSM and SwiGLU blocks."""

    def __init__(self, config: Mamba3LMConfig):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([Mamba3Block(config) for _ in range(config.n_layers)])
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
        recurrent_states: Optional[List[Dict[str, Any]]] = None,
        use_state: bool = False,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        B, L = input_ids.shape
        x = self.tok_embeddings(input_ids)

        new_states = [] if use_state else None
        for i, layer in enumerate(self.layers):
            layer_state = recurrent_states[i] if recurrent_states is not None else None
            x, new_state = layer(x, recurrent_state=layer_state, use_state=use_state)
            if use_state:
                new_states.append(new_state)

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
            "recurrent_states": new_states,
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
        """Recurrent autoregressive generation with constant memory state."""
        self.eval()
        B, L = input_ids.shape
        curr_ids = input_ids

        # Prefill prompt
        outputs = self.forward(curr_ids, use_state=True)
        logits = outputs["logits"][:, -1, :]
        recurrent_states = outputs["recurrent_states"]

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

            # Step one token using recurrent states
            outputs = self.forward(next_token, recurrent_states=recurrent_states, use_state=True)
            logits = outputs["logits"][:, -1, :]
            recurrent_states = outputs["recurrent_states"]

        return curr_ids
