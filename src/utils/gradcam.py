"""Routing-weighted Grad-CAM for ASMOE.

For class c, each expert's projected map P_e (output of its FPM) gives a Grad-CAM
    CAM_e = ReLU( Σ_k α_k^e P_e^k ),   α_k^e = mean_{ij} ∂y_c / ∂P_e^k(i,j)
The fused explanation is the router-weighted sum of up-sampled, normalised maps:
    CAM = Σ_e w_e · CAM_e
``lesion_focus`` reports the fraction of CAM energy that falls on foliage pixels
(HSV/ExG leaf mask), a simple quantitative check of lesion focus vs. background.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

from ..dataset import IMAGENET_MEAN, IMAGENET_STD, leaf_mask  # noqa: E402
from ..models.asmoe import ASMOE  # noqa: E402


class ASMOEGradCAM:
    def __init__(self, model: ASMOE):
        self.model = model.eval()
        self._acts: List[torch.Tensor] = []
        self._handles = [e.fpm.register_forward_hook(self._hook) for e in model.experts]

    def _hook(self, _m, _inp, out):
        out.retain_grad()
        self._acts.append(out)

    def remove(self) -> None:
        for h in self._handles:
            h.remove()

    def __call__(self, x: torch.Tensor, target: Optional[int] = None) -> Dict:
        """x: 1×3×H×W normalised tensor. Returns fused + per-expert CAMs at input resolution."""
        self._acts = []
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            out = self.model(x)
            logits = out["logits"][0]
            c = int(logits.argmax()) if target is None else int(target)
            logits[c].backward()
        H, W = x.shape[-2:]
        weights = out["weights"][0].detach()
        cams = []
        for a in self._acts:
            alpha = a.grad.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((alpha * a).sum(1, keepdim=True))
            cam = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)[0, 0]
            cam = cam - cam.min()
            cams.append((cam / (cam.max() + 1e-8)).detach().cpu().numpy())
        fused = sum(float(w) * cm for w, cm in zip(weights, cams))
        fused = fused / (fused.max() + 1e-8)
        return {"class": c, "prob": float(logits.detach().softmax(-1)[c]), "weights": weights.cpu().tolist(),
                "fused": fused, "experts": cams}


def denormalize(x: torch.Tensor) -> np.ndarray:
    m = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    s = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (x.detach().cpu()[0] * s + m).clamp(0, 1).permute(1, 2, 0).numpy()


def lesion_focus(cam: np.ndarray, rgb: np.ndarray) -> float:
    """Share of CAM mass on foliage/lesion pixels (1.0 = all attention on the leaf)."""
    img = Image.fromarray((rgb * 255).astype(np.uint8))
    mask = np.asarray(leaf_mask(img).resize(cam.shape[::-1], Image.NEAREST)) > 127
    return float((cam * mask).sum() / (cam.sum() + 1e-8))


def save_gradcam_panel(x: torch.Tensor, res: Dict, class_names: List[str], expert_labels: List[str],
                       path: Path, true_label: Optional[str] = None) -> Path:
    rgb = denormalize(x)
    n = 2 + len(res["experts"])
    fig, axes = plt.subplots(1, n, figsize=(2.1 * n, 2.3))
    axes[0].imshow(rgb)
    axes[0].set_title(f"true: {true_label}" if true_label else "input", fontsize=8)
    axes[1].imshow(rgb)
    axes[1].imshow(res["fused"], cmap="jet", alpha=0.45)
    axes[1].set_title(f"fused → {class_names[res['class']]} ({100 * res['prob']:.0f}%)", fontsize=8)
    for k, (cam, lab, w) in enumerate(zip(res["experts"], expert_labels, res["weights"])):
        axes[2 + k].imshow(rgb)
        axes[2 + k].imshow(cam, cmap="jet", alpha=0.45)
        axes[2 + k].set_title(f"{lab} (w={w:.2f})", fontsize=8)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path
