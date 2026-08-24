"""JEPA-style latent-prediction auxiliary losses for the byte encoder (grilled 2026-08-24).

Three anti-collapse mechanisms, one per lab arm:
  minimal — stop-grad targets from the online encoder + a VICReg-style variance hinge
            (the bet: next-byte CE already grounds the representation, cf. AC-MTM 2608.17542)
  ema     — targets from an EMA teacher copy of embeddings+encoder (classic I-JEPA machinery)
  sigreg  — single encoder; alignment + sliced characteristic-function Gaussianity
            regularizer (LeJEPA; identifiability theory in 2605.26379)

Shared shape: predict the encoder latent k in {4, 8, 16} bytes ahead (dense multi-horizon,
Fast-LeWM 2606.26217; never k=1 — adjacent targets are near-trivial and are anti-divergence
pressure on the exact geometry the router cuts on, 2605.26379). The horizon index feeds the
predictor only, never the target path (HP-JEPA 2608.00491). Verdicts are val bpb + router
telemetry — aux-loss health is not evidence (AC-MTM).
"""

from __future__ import annotations

import copy
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

HORIZONS = (4, 8, 16)


class JepaAux(nn.Module):
    def __init__(self, model, mode: str, d_model: int, horizons=HORIZONS, ema_decay: float = 0.999,
                 hinge_floor: float = 1.0, sigreg_weight: float = 0.01, sigreg_slices: int = 256):
        super().__init__()
        assert mode in ("minimal", "ema", "sigreg")
        self.mode = mode
        self.horizons = tuple(horizons)
        self.ema_decay = ema_decay
        self.hinge_floor = hinge_floor
        self.sigreg_weight = sigreg_weight
        self.sigreg_slices = sigreg_slices
        self.h_emb = nn.Embedding(len(self.horizons), 32)
        self.predictor = nn.Sequential(
            nn.Linear(d_model + 32, 2 * d_model), nn.SiLU(), nn.Linear(2 * d_model, d_model),
        )
        if mode == "ema":
            # teacher = embeddings + encoder only; lives outside state_dict via buffers? No —
            # keep it a plain attribute module with requires_grad_(False); the trainer persists it.
            self.teacher_embed = copy.deepcopy(model.embeddings).requires_grad_(False)
            self.teacher_encoder = copy.deepcopy(model.encoder).requires_grad_(False)

    @torch.no_grad()
    def ema_update(self, model):
        if self.mode != "ema":
            return
        for tgt, src in ((self.teacher_embed, model.embeddings), (self.teacher_encoder, model.encoder)):
            for pt, ps in zip(tgt.parameters(), src.parameters()):
                pt.lerp_(ps.detach(), 1.0 - self.ema_decay)
            for bt, bs in zip(tgt.buffers(), src.buffers()):
                bt.copy_(bs)

    def _targets(self, inputs: torch.Tensor, h_online: torch.Tensor) -> torch.Tensor:
        if self.mode == "ema":
            with torch.no_grad():
                t = self.teacher_encoder(self.teacher_embed(inputs))
            return t.float()
        return h_online.detach().float()

    def _sigreg(self, h: torch.Tensor) -> torch.Tensor:
        """Sliced characteristic-function distance to N(0, I): project onto random unit
        directions, compare E[e^{itx}] with e^{-t^2/2} on a small t grid."""
        x = h.reshape(-1, h.shape[-1]).float()
        if x.shape[0] > 4096:
            x = x[torch.randperm(x.shape[0], device=x.device)[:4096]]
        u = torch.randn(h.shape[-1], self.sigreg_slices, device=h.device)
        u = u / u.norm(dim=0, keepdim=True)
        proj = x @ u  # [N, S]
        loss = h.new_zeros(())
        for t in (0.5, 1.0, 2.0):
            target = torch.exp(torch.tensor(-t * t / 2.0, device=h.device))
            loss = loss + (torch.cos(t * proj).mean(0) - target).pow(2).mean()
            loss = loss + torch.sin(t * proj).mean(0).pow(2).mean()
        return loss

    def forward(self, inputs: torch.Tensor, h_online: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """inputs: [B, T] byte ids (the model's input slice); h_online: [B, T, D] encoder output
        captured by the trainer's forward hook, with grad. Returns (mean aux loss, telemetry)."""
        tgt_all = self._targets(inputs, h_online)
        hf = h_online.float()
        pred_loss = hf.new_zeros(())
        for i, k in enumerate(self.horizons):
            if hf.shape[1] <= k:
                continue
            emb = self.h_emb.weight[i].to(hf.dtype).expand(hf.shape[0], hf.shape[1] - k, -1)
            pred = self.predictor(torch.cat([hf[:, :-k], emb], dim=-1))
            pred_loss = pred_loss + F.mse_loss(pred, tgt_all[:, k:])
        pred_loss = pred_loss / len(self.horizons)

        std = hf.reshape(-1, hf.shape[-1]).std(dim=0)
        stats: Dict[str, torch.Tensor] = {
            "jepa_pred": pred_loss.detach(),
            "jepa_latent_std": std.mean().detach(),
        }
        adj = F.cosine_similarity(hf[:, 1:], hf[:, :-1], dim=-1)
        stats["jepa_adj_cos"] = adj.mean().detach()
        stats["jepa_adj_cos_p10"] = torch.quantile(adj.float().flatten(), 0.1).detach()

        if self.mode == "sigreg":
            sig = self._sigreg(h_online)
            stats["jepa_sigreg"] = sig.detach()
            return pred_loss + self.sigreg_weight * sig, stats
        hinge = F.relu(self.hinge_floor - std).mean()
        stats["jepa_hinge"] = hinge.detach()
        return pred_loss + hinge, stats
