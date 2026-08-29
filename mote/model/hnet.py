"""One-stage byte-level H-Net with a Relation main network.

    bytes -> embed -> Mamba-3 encoder -> routing -> chunk -> (pad dims) -> Relation main ->
    dechunk (EMA) -> out * STE(c_t) + Linear(encoder) -> Mamba-3 decoder -> RMSNorm -> lm_head

Training returns next-byte logits and the routing outputs needed for the ratio loss. Inference
supports batch-1 prefill + step decoding with full state caching.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from ..config import MoteConfig
from .arena import ArenaState, RelationArena
from .blocks import make_mamba3_stack, make_relation_stack
from .dc import ChunkLayer, DeChunkLayer, DeChunkState, RoutingModule, RoutingOutput, RoutingState, ste_ones
from .feedback import FeedbackInput, LatentFusion, fuse
from .relation import FullRelation
from .tree import map_tree

# State-dict prefixes of modules this version no longer has: the multi-byte head (retired 2026-08-29 —
# off in every model that mattered, docs/results/2026-08-29-housekeeping-prereg.md) and the
# hyper-connection spine (signed 2026-08-26, never gated). A checkpoint that carries them still loads:
# `strip_retired` drops the keys and says what it dropped.
RETIRED_PREFIXES = ("mbp_head.", "stream.", "chunk_spine.", "encoder.spines.", "decoder.spines.", "main_network.spines.")


def strip_retired(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """`sd` without the retired modules' tensors — a copy when anything is dropped, else `sd` itself."""
    dropped = sorted({next(p for p in RETIRED_PREFIXES if k.startswith(p)).rstrip(".") for k in sd if k.startswith(RETIRED_PREFIXES)})
    if not dropped:
        return sd
    print(f"checkpoint carries retired modules, dropped: {dropped}", flush=True)
    return {k: v for k, v in sd.items() if not k.startswith(RETIRED_PREFIXES)}


@dataclass
class HNetOutput:
    logits: torch.Tensor  # [B, L, V] next-byte logits
    routing: RoutingOutput
    chunk_id: torch.Tensor  # [B, L] chunk index of every byte
    offset: torch.Tensor  # [B, L] offset of every byte within its chunk
    # latent feedback (feedback.py): the top states a further pass fuses into its inputs — the decoder's
    # h3 [B, L, D0] at byte level, the main network's full-width output [B, M, D1] at chunk level — and,
    # at chunk level, the front half (h, residual, routing, hc_plain, next_mask) the next pass reuses
    top: Optional[torch.Tensor] = None
    front: Optional[tuple] = None


@dataclass
class InferenceState:
    encoder: List[Any]
    routing: RoutingState
    main: ArenaState  # the Relation per-chunk cache lives in a shared arena (arena.py); this holds the fill count
    dechunk: DeChunkState
    decoder: List[Any]
    # latent feedback (Soft decoding): the carried top state the next input is fused with
    h_prev: Optional[torch.Tensor] = None  # [1, 1, D0] decoder top of the last byte (byte level)
    z_prev: Optional[torch.Tensor] = None  # [1, 1, D1] main-network top of the last chunk (chunk level)


class HNetForCausalLM(nn.Module):
    def __init__(self, cfg: MoteConfig, device=None, dtype=None):
        super().__init__()
        fk = {"device": device, "dtype": dtype}
        self.cfg = cfg
        D0, D1 = cfg.d_model_outer, cfg.main.d_model
        assert D1 >= D0, "main width must be >= outer width (padded, not projected)"
        V = cfg.pad_vocab_to

        self.embeddings = nn.Embedding(V, D0, **fk)
        self.encoder = make_mamba3_stack(cfg.encoder_layers, D0, cfg.mamba3, cfg.norm_eps, 0, **fk, residual_in_fp32=cfg.residual_in_fp32)
        self.routing_module = RoutingModule(D0, **fk)
        r = self.routing_module
        r.target_ratio = cfg.dc.target_ratio_init
        r.bound_floor, r.bound_ceiling = cfg.dc.bound_floor, cfg.dc.bound_ceiling
        r.bound_window = cfg.max_seq_len  # the floor is a rate over the training window (dc.py)
        r.decode_threshold = cfg.dc.decode_threshold
        self.chunk_layer = ChunkLayer(bucket=getattr(cfg.dc, "chunk_bucket", 1))
        self.main_network = make_relation_stack(cfg.main, cfg.norm_eps, **fk, residual_in_fp32=cfg.residual_in_fp32, mamba_cfg=cfg.mamba3)
        self.dechunk_layer = DeChunkLayer(D0, prob_clamp=cfg.dc.prob_clamp)
        self.residual_proj = nn.Linear(D0, D0, device=device, dtype=torch.float32)
        nn.init.zeros_(self.residual_proj.weight)
        nn.init.zeros_(self.residual_proj.bias)
        self.residual_proj.weight._no_reinit = True
        self.pad_dimension = nn.Parameter(torch.zeros(D1 - D0, **fk)) if D1 > D0 else None
        self.decoder = make_mamba3_stack(cfg.decoder_layers, D0, cfg.mamba3, cfg.norm_eps, cfg.encoder_layers, **fk, residual_in_fp32=cfg.residual_in_fp32)
        self.lm_head = nn.Linear(D0, V, bias=False, **fk)
        # rows past the vocabulary (272 − 271 spare protocol ids, 2026-08-24) are never targets and never
        # sampled: every logit consumer goes through `head_logits`, which masks them to -inf
        self.register_buffer("logit_mask", torch.zeros(V, device=device, dtype=torch.float32), persistent=False)
        if V > cfg.vocab_size:
            self.logit_mask[cfg.vocab_size:] = float("-inf")
        fb = getattr(cfg, "feedback", None)
        self.feedback_level = fb.level if fb is not None else "off"
        self.feedback_jitter = float(fb.jitter) if fb is not None else 0.0
        if self.feedback_level == "byte":
            self.fusion = LatentFusion(D0, D0, cfg.norm_eps, **fk)  # decoder top -> the next byte's encoder input
        elif self.feedback_level == "chunk":
            self.fusion = LatentFusion(D1, D1, cfg.norm_eps, **fk)  # main top -> the next chunk's main input
        else:
            assert self.feedback_level == "off", f"unknown feedback level {self.feedback_level!r}"
            self.fusion = None
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embeddings.weight
        self.init_weights()

    def head_logits(self, h: torch.Tensor) -> torch.Tensor:
        """lm_head with the padding rows masked to -inf (a no-op when pad_vocab_to == vocab_size).

        The mask is one row wide at Mote's vocab (271 -> 272) and the out-of-place add cost a full
        second copy of the logits — 17 MiB at [1, 16384, 272] fp32. Under no_grad the add goes in
        place on the matmul's own output, which nothing else holds; with autograd on it stays
        out-of-place, where the copy is a rounding error against the activations already saved."""
        logits = self.lm_head(h)
        if self.cfg.pad_vocab_to > self.cfg.vocab_size:
            mask = self.logit_mask.to(logits.dtype)
            logits = logits.add_(mask) if not torch.is_grad_enabled() else logits + mask
        return logits

    # ------------------------------------------------------------------------------
    def init_weights(self) -> None:
        std = self.cfg.initializer_range
        nn.init.normal_(self.embeddings.weight, mean=0.0, std=1.0)
        if not self.cfg.tie_embeddings:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=std)
        n_res = self.encoder.height + self.decoder.height + self.main_network.height
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear) and not getattr(m.weight, "_no_reinit", False) and m is not self.lm_head and m is not self.residual_proj:
                if name.endswith("out_proj") or name.endswith("fc2") or name.endswith(".wo"):
                    nn.init.normal_(m.weight, mean=0.0, std=std / (n_res ** 0.5))
                else:
                    nn.init.normal_(m.weight, mean=0.0, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def stage_param_groups(self) -> Dict[int, List[nn.Parameter]]:
        """Stage 0 = byte-level modules (embeddings, encoder, routing, dechunk, decoder, head);
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

    def forward(self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None, feedback: Optional[FeedbackInput] = None) -> HNetOutput:
        """`feedback`: a latent-feedback pass (feedback.py) — the previous pass's top states are shifted right
        and fused into this pass's inputs at the level the config names. At byte level the fused embedding
        feeds the encoder (so the router sees the fully processed past); at chunk level the fused chunk
        vector feeds the main network and the previous pass's encoder/routing are reused."""
        B, L = input_ids.shape
        D0 = self.cfg.d_model_outer
        if mask is None:
            mask = torch.ones(B, L, dtype=torch.bool, device=input_ids.device)
        level = self.feedback_level if feedback is not None else "off"
        if level == "chunk" and feedback.front is not None:
            h, residual, routing, hc, next_mask = feedback.front  # plain bytes in every pass: nothing to recompute
        else:
            h = self.embeddings(input_ids)
            if level == "byte":
                h = fuse(self.fusion, h, feedback, self.feedback_jitter, self.training)
            x = self.encoder(h)  # [B, L, D0]
            h, residual = x, self.residual_proj(x.float())
            routing = self.routing_module(h, mask)
            hc, next_mask = self.chunk_layer(h, routing.boundary_mask)  # [B, M, D0]
            hc = self._pad(hc)
        front = (h, residual, routing, hc, next_mask) if self.feedback_level == "chunk" else None
        if level == "chunk":
            hc = fuse(self.fusion, hc, feedback, self.feedback_jitter, self.training)
        zc_full = self.main_network(hc, token_mask=next_mask)  # [B, M, D1]; the mask keeps pads out of MoE stats
        zc = zc_full[..., :D0]
        z = self.dechunk_layer(zc, routing.boundary_mask, routing.boundary_prob)  # [B, L, D0]

        y = z.float() * ste_ones(routing.selected_probs.float())
        h3 = self.decoder((y + residual).to(h.dtype))
        logits = self.head_logits(h3)

        chunk_id = torch.cumsum(routing.boundary_mask.long(), dim=1) - 1
        chunk_id = chunk_id.clamp(min=0)
        pos = torch.arange(L, device=input_ids.device)[None, :].expand(B, -1)
        start = torch.where(routing.boundary_mask, pos, torch.zeros_like(pos))
        start = torch.cummax(start, dim=1).values
        offset = pos - start

        top = h3 if self.feedback_level == "byte" else (zc_full if self.feedback_level == "chunk" else None)
        return HNetOutput(logits, routing, chunk_id, offset, top=top, front=front)

    # ------------------------------------------------------------------------------
    def new_arena(self, device, capacity: Optional[int] = None, bpic: Optional[float] = None,
                  dtype=None) -> RelationArena:
        """A decode arena for this model: rows for `capacity` chunks.

        `bpic` is the run's own measured bytes-per-chunk (mote.runinfo.measured_bpic) and is the way
        to size this. The old default of max_seq_len // 4 assumed 4 bytes a chunk on the strength of
        a comment; three trained runs measured 3.2-3.45, which needs 1.2x more rows than that at
        16384 and made `ensure` fire — a 1296 MiB peak and a full graph recapture — partway through
        every long conversation. With no measurement to go on it still falls back to that default,
        because a wrong guess that grows is better than a wrong guess that asserts.

        `dtype` overrides the parameter dtype: the arena has to match whatever dtype the Relation
        kernel sees, which is the autocast dtype at serving time, not necessarily the weights'."""
        m = self.cfg.main
        if capacity:
            cap = int(capacity)
        elif bpic:
            cap = RelationArena.capacity_for(self.cfg.max_seq_len, bpic)
        else:
            cap = max(self.cfg.max_seq_len // 4, 16)
        dtype = dtype if dtype is not None else next(self.main_network.parameters()).dtype
        return RelationArena(m.n_layers, m.n_heads, cap, m.d_model // m.n_heads, device, dtype)

    @torch.no_grad()
    def allocate_inference_state(self, device, dtype=None, arena: Optional[RelationArena] = None) -> InferenceState:
        """A fresh state. `arena` is the shared decode arena (the engine passes its own); without one a
        private arena is allocated, which is what tests and one-off reads want."""
        mamba_main = {i: b.allocate_inference_cache(1, device, dtype) for i, b in enumerate(self.main_network.layers)
                      if not isinstance(b.mixer, FullRelation)}  # the hybrid main's recurrent layers
        return InferenceState(
            encoder=self.encoder.allocate_inference_cache(1, device, dtype),
            routing=self.routing_module.allocate_inference_cache(1, device, dtype),
            main=ArenaState(arena if arena is not None else self.new_arena(device), 0, mamba_main),
            dechunk=self.dechunk_layer.allocate_inference_cache(1, device, dtype),
            decoder=self.decoder.allocate_inference_cache(1, device, dtype),
        )

    def _run_main(self, hc: torch.Tensor, state: InferenceState) -> torch.Tensor:
        """Main network over T new chunks, written into the arena at rows [n, n+T); advances n."""
        T = hc.shape[1]
        state.main.arena.ensure(state.main.n + T)
        zc, new_caches = self.main_network(self._pad(hc), caches=state.main, return_caches=True)
        for i in state.main.mamba:  # a hybrid main's Mamba-3 layers carry their state here, not in the arena
            state.main.set(i, new_caches[i])
        state.main.advance(T)
        if self.feedback_level == "chunk":
            state.z_prev = zc[:, -1:]  # generated chunks fuse with it (`step`); prompt chunks stay plain
        return zc[..., : self.cfg.d_model_outer]

    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor, state: InferenceState, last_logits_only: bool = False) -> HNetOutput:
        """Batch-1 prefill: same math as forward, but every recurrent/cached state is kept in `state`.

        `last_logits_only` returns logits for the final position alone. Every serving caller of this
        reads `out.logits[0, -1]` and nothing else, and the head over all L positions is a matmul and
        a [1, L, V] tensor it then throws away — 17 MiB and 4.6 GFLOP at 16384. Off by default: the
        tests and anything comparing whole-sequence logits want the full thing."""
        assert input_ids.shape[0] == 1
        B, L = input_ids.shape
        mask = torch.ones(B, L, dtype=torch.bool, device=input_ids.device)

        assert state.main.n == 0, "prefill wants a fresh state; continue a read with forward_from_state"
        h = self.embeddings(input_ids)
        x, state.encoder = self.encoder(h, caches=None, return_caches=True)
        h, residual = x, self.residual_proj(x.float())
        routing = self.routing_module(h, mask, state.routing)
        hc, _ = self.chunk_layer(h, routing.boundary_mask, exact=True)  # caches must hold only real chunks
        zc = self._run_main(hc, state)
        z = self.dechunk_layer(zc, routing.boundary_mask, routing.boundary_prob, state.dechunk)
        h3, state.decoder = self.decoder((z.float() + residual).to(h.dtype), caches=None, return_caches=True)
        logits = self.head_logits(h3[:, -1:] if last_logits_only else h3)
        if self.feedback_level == "byte":
            state.h_prev = h3[:, -1:]

        chunk_id = torch.cumsum(routing.boundary_mask.long(), dim=1) - 1
        pos = torch.arange(L, device=input_ids.device)[None, :]
        start = torch.cummax(torch.where(routing.boundary_mask, pos, torch.zeros_like(pos)), dim=1).values
        offset = pos - start
        return HNetOutput(logits, routing, chunk_id, offset)

    @torch.no_grad()
    def forward_from_state(self, input_ids: torch.Tensor, state: InferenceState, last_logits_only: bool = False):
        """Continue a batch-1 sequence from `state` over k new bytes in one pass (the same math as k calls
        of `step`). Mutates `state`. Returns (logits [1,k,V], boundary_mask [k] bool, boundary_prob [k] float).

        `last_logits_only` narrows the returned logits to the final position — what reading a prompt
        wants; off by default for callers that score every row."""
        assert input_ids.shape[0] == 1
        B, K = input_ids.shape
        mask = torch.ones(B, K, dtype=torch.bool, device=input_ids.device)

        h = self.embeddings(input_ids)
        x, state.encoder = self.encoder(h, caches=state.encoder, return_caches=True)
        h, residual = x, self.residual_proj(x.float())
        routing = self.routing_module(h, mask, state.routing)
        hc, _ = self.chunk_layer(h, routing.boundary_mask, exact=True)
        if hc.shape[1] > 0:
            zc = self._run_main(hc, state)
        else:
            zc = h[:, :0]
        z = self.dechunk_layer(zc, routing.boundary_mask, routing.boundary_prob, state.dechunk)
        h3, state.decoder = self.decoder((z.float() + residual).to(h.dtype), caches=state.decoder, return_caches=True)
        logits = self.head_logits(h3[:, -1:] if last_logits_only else h3)
        if self.feedback_level == "byte":
            state.h_prev = h3[:, -1:]
        return logits, routing.boundary_mask[0], routing.boundary_prob[0, :, 1]

    @staticmethod
    def _map_state(state: InferenceState, fn) -> InferenceState:
        """Rebuild the state with `fn` applied to every tensor. The arena is shared, never copied: only its
        fill count (`ArenaState`) travels."""
        return map_tree(state, lambda o: o.map(fn) if isinstance(o, ArenaState) else fn(o), atoms=(ArenaState,))

    @staticmethod
    def clone_state(state: InferenceState) -> InferenceState:
        """Deep copy of every tensor in the inference state (a snapshot the caller can roll back to). The
        arena is shared: rolling back restores the fill count, and rows past it are scratch by contract."""
        return HNetForCausalLM._map_state(state, lambda t: t.clone())

    @staticmethod
    def move_state(state: InferenceState, device, pin: bool = False) -> InferenceState:
        """A copy of every tensor in the state on `device` — always a copy, so the source stays usable
        (the serving engine parks anchors on the CPU; `pin` page-locks them for the trip back). The
        arena reference is kept as is: its rows are moved by the prefix store, page by page."""
        dev = torch.device(device)
        # Every copy is queued without a host sync (pinned on the CPU side), and the stream is synced
        # once at the end: a blocking copy per tensor cost ~1 ms each beside a running trainer
        # (measured 2026-08-24: 19-36 ms to restore an anchor of ~30 small tensors).
        sync = False

        def mv(t: torch.Tensor) -> torch.Tensor:
            nonlocal sync
            if pin and dev.type == "cpu" and t.is_cuda:
                out = torch.empty_like(t, device="cpu", pin_memory=True)
                out.copy_(t, non_blocking=True)
                sync = True
                return out
            if dev.type == "cuda" and not t.is_cuda:
                return t.to(dev, non_blocking=t.is_pinned(), copy=True)
            return t.to(dev, copy=True)

        out = HNetForCausalLM._map_state(state, mv)
        if sync:
            torch.cuda.current_stream().synchronize()  # the pinned copies are complete before anyone reads them
        return out

    @torch.no_grad()
    def step(self, input_ids: torch.Tensor, state: InferenceState):
        """One byte. Returns (logits [1,1,V], routing output, is_boundary)."""
        assert input_ids.shape == (1, 1)
        D0 = self.cfg.d_model_outer
        h = self.embeddings(input_ids)
        if self.feedback_level == "byte" and state.h_prev is not None:
            h = self.fusion(state.h_prev, h)  # Soft decoding: the last byte's top state rides into this input
        x, state.encoder = self.encoder.step(h, state.encoder)
        h, residual = x, self.residual_proj(x.float())
        routing = self.routing_module.step(h, state.routing)
        is_boundary = bool(routing.boundary_mask[0])
        if is_boundary:
            hc = self.chunk_layer.step(h, routing.boundary_mask)  # [1,1,D0]
            state.main.arena.ensure(state.main.n + 1)
            xm = self._pad(hc)
            if self.feedback_level == "chunk" and state.z_prev is not None:
                xm = self.fusion(state.z_prev, xm)  # Soft decoding at chunk level
            zc, new_caches = self.main_network.step(xm, state.main)
            for i in state.main.mamba:
                state.main.set(i, new_caches[i])
            state.main.advance(1)
            if self.feedback_level == "chunk":
                state.z_prev = zc
            zc = zc[..., :D0]
        else:
            zc = h[:0]
        z = self.dechunk_layer.step(zc, routing.boundary_mask, routing.boundary_prob, state.dechunk)
        h3, state.decoder = self.decoder.step((z.float() + residual).to(h.dtype), state.decoder)
        logits = self.head_logits(h3)
        if self.feedback_level == "byte":
            state.h_prev = h3
        return logits, routing, is_boundary
