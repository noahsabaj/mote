"""One-stage byte-level H-Net with a Relation main network and a multi-byte prediction head.

    bytes -> embed -> Mamba-3 encoder -> routing -> chunk -> (pad dims) -> Relation main ->
    dechunk (EMA) -> out * STE(c_t) + Linear(encoder) -> Mamba-3 decoder -> RMSNorm -> lm_head
                                                      -> LCA multi-byte head -> lm_head (shared)

Training returns next-byte logits, multi-byte logits and the routing outputs needed for the
ratio loss. Inference supports batch-1 prefill + step decoding with full state caching.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import MorphemeConfig
from .blocks import Isotropic, make_mamba3_stack, make_relation_stack
from .dc import ChunkLayer, DeChunkLayer, DeChunkState, RoutingModule, RoutingOutput, RoutingState, ste_ones
from .mbp import LCAHead, lca_mask


@dataclass
class HNetOutput:
    logits: torch.Tensor  # [B, L, V] next-byte logits
    mbp_logits: Optional[torch.Tensor]  # [B, L, V] multi-byte-head logits (same targets)
    routing: RoutingOutput
    chunk_id: torch.Tensor  # [B, L] chunk index of every byte
    offset: torch.Tensor  # [B, L] offset of every byte within its chunk


@dataclass
class InferenceState:
    encoder: List[Any]
    routing: RoutingState
    main: List[Any]
    dechunk: DeChunkState
    decoder: List[Any]
    # multi-byte head bookkeeping (batch 1)
    n_chunks: int = 0
    prev_chunk_inputs: Optional[torch.Tensor] = None  # [1, len_prev, D] LCA inputs of the previous chunk
    cur_chunk_inputs: List[torch.Tensor] = field(default_factory=list)
    last_chunk_z: Optional[torch.Tensor] = None  # [1, 1, D] main output of the current chunk (dechunked)
    last_chunk_start_state: Optional[torch.Tensor] = None  # [1, 1, D] encoder state at the current chunk start
    cur_offset: int = 0


class HNetForCausalLM(nn.Module):
    def __init__(self, cfg: MorphemeConfig, device=None, dtype=None):
        super().__init__()
        fk = {"device": device, "dtype": dtype}
        self.cfg = cfg
        D0, D1 = cfg.d_model_outer, cfg.main.d_model
        assert D1 >= D0, "main width must be >= outer width (padded, not projected)"
        V = cfg.pad_vocab_to

        self.embeddings = nn.Embedding(V, D0, **fk)
        self.encoder = make_mamba3_stack(cfg.encoder_layers, D0, cfg.mamba3, cfg.norm_eps, 0, **fk)
        self.routing_module = RoutingModule(D0, **fk)
        self.chunk_layer = ChunkLayer()
        self.main_network = make_relation_stack(cfg.main, cfg.norm_eps, **fk)
        self.dechunk_layer = DeChunkLayer(D0, prob_clamp=cfg.dc.prob_clamp)
        self.residual_proj = nn.Linear(D0, D0, device=device, dtype=torch.float32)
        nn.init.zeros_(self.residual_proj.weight)
        nn.init.zeros_(self.residual_proj.bias)
        self.residual_proj.weight._no_reinit = True
        self.pad_dimension = nn.Parameter(torch.zeros(D1 - D0, **fk)) if D1 > D0 else None
        self.decoder = make_mamba3_stack(cfg.decoder_layers, D0, cfg.mamba3, cfg.norm_eps, cfg.encoder_layers, **fk)
        self.lm_head = nn.Linear(D0, V, bias=False, **fk)
        self.mbp_head = LCAHead(D0, cfg.mbp.n_layers, cfg.mbp.n_heads, cfg.mbp.d_ff, cfg.norm_eps, **fk) if cfg.mbp.enabled else None
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embeddings.weight
        self.init_weights()

    # ------------------------------------------------------------------------------
    def init_weights(self) -> None:
        std = self.cfg.initializer_range
        nn.init.normal_(self.embeddings.weight, mean=0.0, std=1.0)
        if not self.cfg.tie_embeddings:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=std)
        n_res = self.encoder.height + self.decoder.height + self.main_network.height
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear) and not getattr(m.weight, "_no_reinit", False) and m is not self.lm_head and m is not self.residual_proj:
                if name.endswith("out_proj") or name.endswith("fc2") or name.endswith(".wo") or name.endswith("attn.out"):
                    nn.init.normal_(m.weight, mean=0.0, std=std / (n_res ** 0.5))
                else:
                    nn.init.normal_(m.weight, mean=0.0, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def stage_param_groups(self) -> Dict[int, List[nn.Parameter]]:
        """Stage 0 = byte-level modules (embeddings, encoder, routing, dechunk, decoder, heads);
        stage 1 = main network. Used for per-stage learning-rate multipliers."""
        main_ids = {id(p) for p in self.main_network.parameters()}
        if self.pad_dimension is not None:
            main_ids.add(id(self.pad_dimension))
        g0, g1 = [], []
        for p in self.parameters():
            (g1 if id(p) in main_ids else g0).append(p)
        return {0: g0, 1: g1}

    def num_params(self) -> int:
        seen, n = set(), 0
        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                n += p.numel()
        return n

    # ------------------------------------------------------------------------------
    def _pad(self, h: torch.Tensor) -> torch.Tensor:
        if self.pad_dimension is None:
            return h
        return torch.cat([h, self.pad_dimension.to(h.dtype).expand(*h.shape[:-1], -1)], dim=-1)

    def forward(self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> HNetOutput:
        B, L = input_ids.shape
        D0 = self.cfg.d_model_outer
        if mask is None:
            mask = torch.ones(B, L, dtype=torch.bool, device=input_ids.device)

        h = self.embeddings(input_ids)
        h = self.encoder(h)  # [B, L, D0]
        residual = self.residual_proj(h.float())

        routing = self.routing_module(h, mask)
        hc, next_mask = self.chunk_layer(h, routing.boundary_mask)  # [B, M, D0]
        zc = self.main_network(self._pad(hc))[..., :D0]  # [B, M, D0]
        z = self.dechunk_layer(zc, routing.boundary_mask, routing.boundary_prob)  # [B, L, D0]

        h2 = (z.float() * ste_ones(routing.selected_probs.float()) + residual).to(h.dtype)
        h3 = self.decoder(h2)
        logits = self.lm_head(h3)

        chunk_id = torch.cumsum(routing.boundary_mask.long(), dim=1) - 1
        chunk_id = chunk_id.clamp(min=0)
        pos = torch.arange(L, device=input_ids.device)[None, :].expand(B, -1)
        start = torch.where(routing.boundary_mask, pos, torch.zeros_like(pos))
        start = torch.cummax(start, dim=1).values
        offset = pos - start

        mbp_logits = None
        if self.mbp_head is not None:
            start_state = torch.gather(h, 1, start[:, :, None].expand(-1, -1, D0))
            x = self.mbp_head.build_inputs(z, start_state, offset)
            x = self.mbp_head(x, lca_mask(chunk_id, mask))
            mbp_logits = self.lm_head(x)
        return HNetOutput(logits, mbp_logits, routing, chunk_id, offset)

    # ------------------------------------------------------------------------------
    @torch.no_grad()
    def allocate_inference_state(self, device, dtype=None) -> InferenceState:
        return InferenceState(
            encoder=self.encoder.allocate_inference_cache(1, device, dtype),
            routing=self.routing_module.allocate_inference_cache(1, device, dtype),
            main=self.main_network.allocate_inference_cache(1, device, dtype),
            dechunk=self.dechunk_layer.allocate_inference_cache(1, device, dtype),
            decoder=self.decoder.allocate_inference_cache(1, device, dtype),
        )

    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor, state: InferenceState) -> HNetOutput:
        """Batch-1 prefill: same math as forward, but every recurrent/cached state is kept in `state`."""
        assert input_ids.shape[0] == 1
        B, L = input_ids.shape
        D0 = self.cfg.d_model_outer
        mask = torch.ones(B, L, dtype=torch.bool, device=input_ids.device)

        h = self.embeddings(input_ids)
        h, state.encoder = self.encoder(h, caches=None, return_caches=True)
        residual = self.residual_proj(h.float())
        routing = self.routing_module(h, mask, state.routing)
        hc, _ = self.chunk_layer(h, routing.boundary_mask)
        zc, state.main = self.main_network(self._pad(hc), caches=None, return_caches=True)
        zc = zc[..., :D0]
        z = self.dechunk_layer(zc, routing.boundary_mask, routing.boundary_prob, state.dechunk)
        h2 = (z.float() + residual).to(h.dtype)
        h3, state.decoder = self.decoder(h2, caches=None, return_caches=True)
        logits = self.lm_head(h3)

        chunk_id = torch.cumsum(routing.boundary_mask.long(), dim=1) - 1
        pos = torch.arange(L, device=input_ids.device)[None, :]
        start = torch.cummax(torch.where(routing.boundary_mask, pos, torch.zeros_like(pos)), dim=1).values
        offset = pos - start

        # multi-byte head bookkeeping for the chunk in progress
        if self.mbp_head is not None:
            start_state = torch.gather(h, 1, start[:, :, None].expand(-1, -1, D0))
            x = self.mbp_head.build_inputs(z, start_state, offset)
            last = int(chunk_id[0, -1])
            cur = x[:, chunk_id[0] == last]
            prev = x[:, chunk_id[0] == last - 1] if last > 0 else None
            state.n_chunks = last + 1
            state.prev_chunk_inputs = prev
            state.cur_chunk_inputs = [cur[:, i : i + 1] for i in range(cur.shape[1])]
            state.last_chunk_z = z[:, -1:]
            state.last_chunk_start_state = h[:, start[0, -1] : start[0, -1] + 1]
            state.cur_offset = int(offset[0, -1])
        return HNetOutput(logits, None, routing, chunk_id, offset)

    @torch.no_grad()
    def step(self, input_ids: torch.Tensor, state: InferenceState):
        """One byte. Returns (logits [1,1,V], routing output, is_boundary, mbp_logits [1,n,V] or None)."""
        assert input_ids.shape == (1, 1)
        D0 = self.cfg.d_model_outer
        h = self.embeddings(input_ids)
        h, state.encoder = self.encoder.step(h, state.encoder)
        residual = self.residual_proj(h.float())
        routing = self.routing_module.step(h, state.routing)
        is_boundary = bool(routing.boundary_mask[0])
        if is_boundary:
            hc = self.chunk_layer.step(h, routing.boundary_mask)  # [1,1,D0]
            zc, state.main = self.main_network.step(self._pad(hc), state.main)
            zc = zc[..., :D0]
        else:
            zc = h[:0]
        z = self.dechunk_layer.step(zc, routing.boundary_mask, routing.boundary_prob, state.dechunk)
        h2 = (z.float() + residual).to(h.dtype)
        h3, state.decoder = self.decoder.step(h2, state.decoder)
        logits = self.lm_head(h3)

        mbp_logits = None
        if self.mbp_head is not None:
            if is_boundary:
                state.prev_chunk_inputs = torch.cat(state.cur_chunk_inputs, dim=1) if state.cur_chunk_inputs else None
                state.cur_chunk_inputs = []
                state.n_chunks += 1
                state.last_chunk_z = z
                state.last_chunk_start_state = h
                state.cur_offset = 0
            else:
                state.cur_offset += 1
            x_cur = self.mbp_head.build_inputs(
                state.last_chunk_z, state.last_chunk_start_state,
                torch.tensor([[state.cur_offset]], device=h.device),
            )
            state.cur_chunk_inputs.append(x_cur)
            if is_boundary:
                mbp_logits = self._speculate(state)
        return logits, routing, is_boundary, mbp_logits

    @torch.no_grad()
    def _speculate(self, state: InferenceState) -> torch.Tensor:
        """At a fresh boundary, propose the next n bytes (offsets 1..n) in parallel. Returns [1, n, V]."""
        n = self.cfg.mbp.n_candidates
        dev = state.last_chunk_z.device
        offsets = torch.arange(1, n + 1, device=dev)[None, :]
        slots = self.mbp_head.build_inputs(
            state.last_chunk_z.expand(-1, n, -1), state.last_chunk_start_state.expand(-1, n, -1), offsets
        )
        cur0 = state.cur_chunk_inputs[0]  # offset-0 slot of the current chunk
        parts, ids = [], []
        if state.prev_chunk_inputs is not None:
            parts.append(state.prev_chunk_inputs)
            ids.append(torch.full((1, state.prev_chunk_inputs.shape[1]), 0, device=dev, dtype=torch.long))
        parts += [cur0, slots]
        ids += [torch.ones(1, 1, device=dev, dtype=torch.long), torch.ones(1, n, device=dev, dtype=torch.long)]
        x = torch.cat(parts, dim=1)
        cid = torch.cat(ids, dim=1)
        valid = torch.ones_like(cid, dtype=torch.bool)
        y = self.mbp_head(x, lca_mask(cid, valid))
        return self.lm_head(y[:, -n:])
