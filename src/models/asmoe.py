"""ASMOE V2 — Adaptive Soft Mixture-of-Experts with heterogeneous backbones.

    x ──► Expert_e (CNN / hybrid / Transformer, own input resolution)
          └► F_e ∈ R^{C_e×h_e×w_e}
             └► FPM_e: LayerNorm2d → 1×1 Conv → P_e ∈ R^{512×h_e×w_e}
                └► GAP → z_e ∈ R^{512}
    Router:  w = softmax( MLP([z_1 ‖ … ‖ z_E]) / τ )            (input-adaptive)
    Fusion:  z = Σ_e w_e z_e
    Head:    ŷ = W z + b                                          (single linear layer)

A one-expert ASMOE (``arch: single``) is the matched individual-expert baseline:
same projection, same head, router collapses to w = 1.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ExpertConfig, ModelConfig

_TOKEN_MODELS = ("swin", "vit", "deit", "beit", "eva", "maxvit", "coatnet")


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm over the channel dimension of an NCHW tensor."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class FeatureProjection(nn.Module):
    """Feature Projection Module (FPM): aligns heterogeneous expert maps to a shared D-dim space."""

    def __init__(self, in_ch: int, out_dim: int = 512):
        super().__init__()
        self.norm = LayerNorm2d(in_ch)
        self.proj = nn.Conv2d(in_ch, out_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(x))


def _to_nchw(f: torch.Tensor, channels: int) -> torch.Tensor:
    """Normalise timm ``forward_features`` outputs (NCHW, NHWC or B×N×C tokens) to NCHW."""
    if f.ndim == 4:
        if f.shape[1] == channels:
            return f
        if f.shape[-1] == channels:
            return f.permute(0, 3, 1, 2).contiguous()
    if f.ndim == 3:  # tokens, optionally with a CLS token
        b, n, c = f.shape
        side = int(round(n ** 0.5))
        if side * side != n:
            f, n = f[:, 1:], n - 1
            side = int(round(n ** 0.5))
        return f.transpose(1, 2).reshape(b, c, side, side)
    raise ValueError(f"Unsupported feature shape {tuple(f.shape)}")


class Expert(nn.Module):
    def __init__(self, cfg: ExpertConfig, default_size: int, proj_dim: int):
        super().__init__()
        self.name, self.role = cfg.name, cfg.role
        self.image_size = cfg.image_size or default_size
        kwargs = dict(cfg.kwargs)
        if any(t in cfg.name for t in _TOKEN_MODELS):
            kwargs.setdefault("img_size", self.image_size)
        self.body = timm.create_model(cfg.name, pretrained=cfg.pretrained, num_classes=0, **kwargs)
        if cfg.grad_checkpointing and hasattr(self.body, "set_grad_checkpointing"):
            self.body.set_grad_checkpointing(True)
        self.channels = self.body.num_features
        self.fpm = FeatureProjection(self.channels, proj_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the projected spatial map P_e (B×D×h×w)."""
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear",
                              align_corners=False, antialias=True)
        f = _to_nchw(self.body.forward_features(x), self.channels)
        return self.fpm(f)


class SoftRouter(nn.Module):
    """Input-adaptive gating network: GAP features → MLP → softmax over experts."""

    def __init__(self, n_experts: int, dim: int, hidden: int, temperature: float = 1.0,
                 dropout: float = 0.1, mode: str = "adaptive"):
        super().__init__()
        self.n, self.tau, self.mode = n_experts, temperature, mode
        self.mlp = nn.Sequential(
            nn.LayerNorm(n_experts * dim), nn.Linear(n_experts * dim, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, n_experts),
        )
        nn.init.zeros_(self.mlp[-1].weight)   # start from uniform routing
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:   # z: B×E×D
        if self.n == 1 or self.mode == "uniform":
            return z.new_full((z.size(0), self.n), 1.0 / self.n)
        return torch.softmax(self.mlp(z.flatten(1)) / self.tau, dim=-1)


class ASMOE(nn.Module):
    def __init__(self, cfg: ModelConfig, num_classes: int, default_size: int = 384):
        super().__init__()
        self.experts = nn.ModuleList([Expert(e, default_size, cfg.proj_dim) for e in cfg.experts])
        n = len(self.experts)
        self.router = SoftRouter(n, cfg.proj_dim, cfg.router_hidden, cfg.router_temperature,
                                 cfg.router_dropout, cfg.router_mode)
        self.norm = nn.LayerNorm(cfg.proj_dim)
        self.dropout = nn.Dropout(cfg.head_dropout)
        self.head = nn.Linear(cfg.proj_dim, num_classes)
        self.supcon_head = nn.Sequential(nn.Linear(cfg.proj_dim, cfg.proj_dim), nn.ReLU(inplace=True),
                                         nn.Linear(cfg.proj_dim, cfg.supcon_dim))

    # ---- parameter groups for discriminative learning rates
    def backbone_parameters(self) -> Iterable[nn.Parameter]:
        for e in self.experts:
            yield from e.body.parameters()

    def head_parameters(self) -> Iterable[nn.Parameter]:
        backbone = {id(p) for p in self.backbone_parameters()}
        return (p for p in self.parameters() if id(p) not in backbone)

    def set_backbone_trainable(self, trainable: bool) -> None:
        for p in self.backbone_parameters():
            p.requires_grad_(trainable)

    def train(self, mode: bool = True):
        super().train(mode)
        # frozen backbones keep BatchNorm running stats fixed (Stage 1)
        for e in self.experts:
            if not any(p.requires_grad for p in e.body.parameters()):
                e.body.eval()
        return self

    # ---- forward
    def forward(self, x: torch.Tensor, return_maps: bool = False) -> Dict[str, torch.Tensor]:
        maps: List[torch.Tensor] = [e(x) for e in self.experts]
        z_e = torch.stack([m.mean(dim=(2, 3)) for m in maps], dim=1)   # B×E×D
        w = self.router(z_e)                                            # B×E
        z = (w.unsqueeze(-1) * z_e).sum(dim=1)                         # B×D
        z = self.norm(z)
        out = {"logits": self.head(self.dropout(z)), "embedding": z, "weights": w,
               "supcon": F.normalize(self.supcon_head(z), dim=-1)}
        if return_maps:
            out["maps"] = maps   # type: ignore[assignment]
        return out

    @staticmethod
    def balance_loss(weights: torch.Tensor) -> torch.Tensor:
        """Importance-based load balancing: E·Σ_e(mean_b w_be)² (=1 when uniform)."""
        n = weights.size(1)
        if n == 1:
            return weights.new_zeros(())
        importance = weights.mean(dim=0)
        return n * (importance ** 2).sum() - 1.0


def build_model(cfg: ModelConfig, num_classes: int, default_size: int = 384,
                single_expert: Optional[str] = None) -> ASMOE:
    """``arch: asmoe`` → full mixture; ``arch: single`` → first (or named) expert alone."""
    if cfg.arch == "single" or single_expert:
        experts = cfg.experts
        if single_expert:
            match = [e for e in experts if e.name == single_expert]
            experts = match or [ExpertConfig(single_expert, "baseline")]
        cfg = ModelConfig(**{**cfg.__dict__, "experts": experts[:1], "arch": "single"})
    return ASMOE(cfg, num_classes, default_size)


def count_parameters(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6
