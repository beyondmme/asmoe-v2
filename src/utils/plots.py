"""Publication figures (matplotlib, Agg). Categorical slots follow a CVD-validated fixed order."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..config import CLASS_NAMES, DISPLAY_NAMES  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "font.size": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "legend.frameon": False,
})


def _names(names: Sequence[str]) -> List[str]:
    return [DISPLAY_NAMES.get(n, n) for n in names]


def confusion_matrix_figure(cm: Sequence[Sequence[int]], names: Sequence[str], title: str, path: Path) -> Path:
    """Row-normalised colour (recall) with raw counts annotated."""
    cm = np.asarray(cm, dtype=float)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ax.grid(False)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{int(cm[i, j])}\n{100 * norm[i, j]:.0f}%", ha="center", va="center", fontsize=6.5,
                    color="white" if norm[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(names)), _names(names), rotation=40, ha="right")
    ax.set_yticks(range(len(names)), _names(names))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row-normalised recall")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def learning_curves_figure(epoch_csvs: Dict[str, Path], path: Path) -> Optional[Path]:
    """Validation Macro-F1 and loss per epoch, fold mean ± SD band, one line per experiment (≤3)."""
    frames = {k: pd.read_csv(v) for k, v in epoch_csvs.items() if Path(v).exists()}
    if not frames:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for (label, df), color in zip(list(frames.items())[:3], SERIES):
        for ax, col in zip(axes, ("macro_f1", "val_loss")):
            g = df.groupby("epoch")[col]
            mu, sd = g.mean(), g.std().fillna(0)
            ax.plot(mu.index, mu.values, color=color, lw=2, label=label)
            ax.fill_between(mu.index, mu - sd, mu + sd, color=color, alpha=0.15, lw=0)
        s1 = df[df["stage"] == 1]["epoch"].max()
        if pd.notna(s1):
            for ax in axes:
                ax.axvline(s1 + 0.5, color=MUTED, lw=0.8, ls="--")
    axes[0].set_ylabel("Validation Macro-F1 (%)")
    axes[1].set_ylabel("Validation loss")
    for ax in axes:
        ax.set_xlabel("Epoch")
    axes[0].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def per_class_f1_figure(series: Dict[str, Dict[str, float]], path: Path,
                        names: Sequence[str] = CLASS_NAMES) -> Path:
    """Grouped bars: per-class F1 for ≤3 models (e.g. legacy baseline vs. ASMOE V2)."""
    labels = list(series)[:3]
    x = np.arange(len(names))
    w = 0.8 / len(labels)
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    for k, (lab, color) in enumerate(zip(labels, SERIES)):
        vals = [100 * series[lab].get(c, np.nan) for c in names]
        ax.bar(x + (k - (len(labels) - 1) / 2) * w, vals, w * 0.92, color=color, label=lab, zorder=3)
    ax.set_xticks(x, _names(names))
    ax.set_ylabel("F1 (%)")
    ax.set_ylim(50, 100)
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=len(labels), fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def router_weights_figure(per_class: Dict[str, List[float]], expert_labels: Sequence[str], path: Path) -> Path:
    """Mean routing weight per expert for each true class (stacked to 1)."""
    names = [c for c in CLASS_NAMES if c in per_class]
    W = np.array([per_class[c] for c in names])
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    left = np.zeros(len(names))
    for e, (lab, color) in enumerate(zip(expert_labels, SERIES)):
        ax.barh(_names(names), W[:, e], left=left, color=color, label=lab, edgecolor="white", linewidth=2, zorder=3)
        left += W[:, e]
    ax.set_xlim(0, 1)
    ax.set_xlabel("Mean routing weight $w_e$")
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=len(expert_labels), fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path
