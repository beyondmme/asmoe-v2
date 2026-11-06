"""Classification + representation losses for minority-class recovery."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def class_weights(counts: np.ndarray, mode: str = "sqrt") -> Optional[torch.Tensor]:
    """``sqrt`` (best in the legacy policy screen), ``inverse`` or ``none``."""
    if mode == "none":
        return None
    counts = np.maximum(counts.astype(np.float64), 1.0)
    w = 1.0 / np.sqrt(counts) if mode == "sqrt" else 1.0 / counts
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


class SoftTargetCE(nn.Module):
    """Cross-entropy on soft targets (MixUp / label smoothing) with optional class weights."""

    def __init__(self, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits.float(), dim=-1)
        loss = -(target * logp)
        if self.weight is not None:
            loss = loss * self.weight
        return loss.sum(dim=-1).mean()


class FocalLoss(nn.Module):
    """Multi-class focal loss on soft targets: Σ_c −α_c (1−p_c)^γ q_c log p_c."""

    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits.float(), dim=-1)
        loss = -target * (1 - logp.exp()).pow(self.gamma) * logp
        if self.weight is not None:
            loss = loss * self.weight
        return loss.sum(dim=-1).mean()


class SupConLoss(nn.Module):
    """Supervised Contrastive loss (Khosla et al., 2020), single-view variant.

    L = Σ_i  −1/|P(i)| Σ_{p∈P(i)} log( exp(z_i·z_p/τ) / Σ_{a≠i} exp(z_i·z_a/τ) )
    Anchors without a positive in the batch are skipped.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.t = temperature

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Always fp32: under fp16 autocast the matmul would drop back to half precision, where the large
        # masking constant overflows (fp16 max ~65504) and exp/logsumexp lose precision.
        with torch.autocast(device_type=z.device.type, enabled=False):
            return self._loss(F.normalize(z.float(), dim=-1), y)

    def _loss(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        sim = z @ z.T / self.t
        n = z.size(0)
        self_mask = torch.eye(n, dtype=torch.bool, device=z.device)
        pos = (y[:, None] == y[None, :]) & ~self_mask
        sim = sim.masked_fill(self_mask, torch.finfo(sim.dtype).min / 2)
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        n_pos = pos.sum(1)
        valid = n_pos > 0
        if not valid.any():
            return z.new_zeros(())
        mean_log_pos = (log_prob * pos).sum(1)[valid] / n_pos[valid]
        return -mean_log_pos.mean()


def build_criterion(loss: str, weight: Optional[torch.Tensor], gamma: float = 2.0) -> nn.Module:
    if loss == "focal":
        return FocalLoss(gamma, weight)
    if loss == "ce":
        return SoftTargetCE(weight)
    raise ValueError(f"Unknown loss '{loss}'")
