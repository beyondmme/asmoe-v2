"""Lightweight result handling (no torch import, so GitHub Actions can build the paper cheaply).

Two folder trees share the same per-experiment layout:

    outputs/<exp>/fold_<k>/   full run: best.pt, resume.pt, result.json, epoch_metrics.csv, predictions.csv
    results/<exp>/fold_<k>/   committed to git: result.json, epoch_metrics.csv, predictions.csv (no checkpoints)

``export_run`` copies the light files from outputs/ to results/; ``summarise_run`` aggregates whatever folds
are present (folds may come from different Kaggle sessions) into ``cv_summary.json``.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from .metrics import aggregate_folds

LIGHT_FILES = ("result.json", "epoch_metrics.csv", "predictions.csv")


def summarise_run(run_dir: Path, class_names: Sequence[str], experiment: Optional[str] = None) -> Dict:
    run_dir = Path(run_dir)
    experiment = experiment or run_dir.name
    logs = [pd.read_csv(p) for p in sorted(run_dir.glob("fold_*/epoch_metrics.csv"))]
    if logs:
        pd.concat(logs, ignore_index=True).to_csv(run_dir / "epoch_metrics.csv", index=False)
    done: List[Dict] = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(run_dir.glob("fold_*/result.json"))]
    if not done:
        print(f"[{experiment}] no completed folds yet")
        return {"experiment": experiment, "n_folds": 0, "folds": []}
    summary = aggregate_folds(done, class_names)
    summary.update({"experiment": experiment, "folds": [r["fold"] for r in done],
                    "params_M": done[0].get("params_M")})
    (run_dir / "cv_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (dict, list))} for r in done]) \
        .to_csv(run_dir / "fold_results.csv", index=False)
    print(f"[{experiment}] {len(done)} folds {summary['folds']} | Acc {summary['accuracy_mean']:.2f}±{summary['accuracy_sd']:.2f} "
          f"| Macro-F1 {summary['macro_f1_mean']:.2f}±{summary['macro_f1_sd']:.2f}")
    return summary


def summarise_all(results_root: Path, class_names: Sequence[str]) -> Dict[str, Dict]:
    out = {}
    for d in sorted(Path(results_root).iterdir()) if Path(results_root).exists() else []:
        if d.is_dir() and any(d.glob("fold_*/result.json")):
            out[d.name] = summarise_run(d, class_names)
    return out


def export_run(run_dir: Path, results_root: Path, folds: Optional[Sequence[int]] = None,
               data_dir: Optional[Path] = None) -> List[str]:
    """Copy light, commit-able artefacts of finished folds (+ Grad-CAM, config, data audit) to results/."""
    run_dir, dst_root = Path(run_dir), Path(results_root) / Path(run_dir).name
    copied: List[str] = []
    for fold_dir in sorted(run_dir.glob("fold_*")):
        if not fold_dir.is_dir():          # skip fold_results.csv
            continue
        k = int(fold_dir.name.split("_")[1])
        if (folds is not None and k not in folds) or not (fold_dir / "result.json").exists():
            continue
        for name in LIGHT_FILES:
            if (fold_dir / name).exists():
                (dst_root / fold_dir.name).mkdir(parents=True, exist_ok=True)
                shutil.copy2(fold_dir / name, dst_root / fold_dir.name / name)
                copied.append(f"{dst_root.name}/{fold_dir.name}/{name}")
    for extra in ("config.yaml",):
        if (run_dir / extra).exists():
            dst_root.mkdir(parents=True, exist_ok=True)
            shutil.copy2(run_dir / extra, dst_root / extra)
    if (run_dir / "gradcam").exists():
        shutil.copytree(run_dir / "gradcam", dst_root / "gradcam", dirs_exist_ok=True)
        copied.append(f"{dst_root.name}/gradcam/")
    if data_dir and (Path(data_dir) / "data_audit.json").exists():
        (Path(results_root) / "data").mkdir(parents=True, exist_ok=True)
        for name in ("data_audit.json", "fold_class_counts.csv"):
            if (Path(data_dir) / name).exists():
                shutil.copy2(Path(data_dir) / name, Path(results_root) / "data" / name)
        copied.append("data/data_audit.json")
    return copied
