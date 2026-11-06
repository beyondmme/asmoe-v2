"""Parse the historical Kaggle result archives (read straight from the .zip files, checkpoints skipped).

Supported layouts
-----------------
* ``Potato_Improved.zip`` → ``potato_optimized/{final_cv_summary,duplicate_audit,near_duplicate_candidates,
  fold_class_counts}.csv`` and ``logs/{final_fold_results,epoch_metrics,policy_screen_ranking,
  architecture_screen_ranking}.csv`` (group-aware split, two-stage screening).
* ``Potato.zip`` → nested ``results (N).zip`` each holding ``<Model>_Results.csv`` (plain
  StratifiedKFold on raw files: duplicate groups can straddle train/val).

Usage::

    python -m src.utils.legacy_audit --zips ../Potato_Improved.zip ../Potato.zip --out outputs/legacy
"""
from __future__ import annotations

import argparse
import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from ..config import CLASS_NAMES


def _iter_csvs(zf: zipfile.ZipFile, prefix: str = "") -> Iterable:
    for info in zf.infolist():
        name = info.filename
        if name.endswith(".pt"):
            continue
        if name.endswith(".csv"):
            yield prefix + name, pd.read_csv(io.BytesIO(zf.read(info)))
        elif name.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(zf.read(info))) as inner:
                yield from _iter_csvs(inner, prefix + name + "!/")


def _pngs(zf: zipfile.ZipFile, pattern: str, out_dir: Path) -> List[str]:
    out = []
    for info in zf.infolist():
        if info.filename.endswith(".png") and pattern in info.filename:
            dst = out_dir / Path(info.filename).name
            dst.write_bytes(zf.read(info))
            out.append(str(dst))
    return out


def top_confusions(cm: np.ndarray, names: List[str], k: int = 8) -> List[Dict]:
    pairs = [(int(cm[i, j]), names[i], names[j]) for i in range(len(names)) for j in range(len(names)) if i != j]
    pairs.sort(reverse=True)
    return [{"true": t, "pred": p, "count": c, "rate_of_true": c / max(1, int(cm[names.index(t)].sum()))}
            for c, t, p in pairs[:k]]


def audit(zips: List[str], out_dir: str) -> Dict:
    out = Path(out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    tables: Dict[str, pd.DataFrame] = {}
    for zp in zips:
        with zipfile.ZipFile(zp) as zf:
            for name, df in _iter_csvs(zf):
                tables[name] = df
            _pngs(zf, "final_cv_", out / "figures")
            _pngs(zf, "learning_curves", out / "figures")
    get = lambda suffix: next((v for k, v in tables.items() if k.endswith(suffix)), None)  # noqa: E731

    s: Dict = {"sources": [Path(z).name for z in zips]}

    # ---- class distribution
    fcc = get("fold_class_counts.csv")
    if fcc is not None:
        f1 = fcc[fcc["fold"] == fcc["fold"].min()]
        totals = {c: int(f1[c].sum()) for c in CLASS_NAMES}
        s["class_totals"] = totals
        s["n_images"] = int(sum(totals.values()))
        s["imbalance_ratio"] = max(totals.values()) / min(totals.values())
        s["val_support_range"] = {c: [int(fcc[fcc.split == "validation"][c].min()),
                                      int(fcc[fcc.split == "validation"][c].max())] for c in CLASS_NAMES}

    # ---- duplicates
    dup = get("duplicate_audit.csv")
    if dup is not None:
        groups = dup.groupby("exact_group")["label"].agg(list)
        conflicts = [sorted(set(v)) for v in groups if len(set(v)) > 1]
        same = [v for v in groups if len(set(v)) == 1]
        redundant = defaultdict(int)
        for v in same:
            redundant[v[0]] += len(v) - 1
        s["exact_duplicates"] = {"files": int(len(dup)), "groups": int(len(groups)),
                                 "conflict_groups": len(conflicts), "conflict_label_pairs": conflicts,
                                 "same_label_groups": len(same), "redundant_copies_by_class": dict(redundant)}
    nd = get("near_duplicate_candidates.csv")
    if nd is not None:
        s["near_duplicates"] = {"pairs": int(len(nd)),
                                "cross_label_pairs": int((nd.label_a != nd.label_b).sum()),
                                "by_class": nd.label_a.value_counts().to_dict(),
                                "dhash_distance_counts": nd.dhash_distance.value_counts().sort_index().to_dict()}

    # ---- group-aware CV (improved legacy pipeline)
    s["group_aware_cv"] = {}
    ff = get("final_fold_results.csv")
    if ff is not None:
        for model, g in ff.groupby("model"):
            names = json.loads(g.iloc[0]["class_names_json"])
            cm = np.sum([np.array(json.loads(x)) for x in g["confusion_matrix_json"]], axis=0)
            per_class = defaultdict(list)
            for pj in g["per_class_json"]:
                for c, v in json.loads(pj).items():
                    per_class[c].append(v["f1"])
            s["group_aware_cv"][model] = {
                "accuracy": [float(x) for x in g["accuracy"]], "macro_f1": [float(x) for x in g["macro_f1"]],
                "mcc": [float(x) for x in g["mcc"]], "best_epoch": [int(x) for x in g["best_epoch"]],
                "per_class_f1": {c: v for c, v in per_class.items()},
                "pooled_cm": cm.tolist(), "class_names": names,
                "pooled_recall": {n: float(cm[i, i] / cm[i].sum()) for i, n in enumerate(names)},
                "top_confusions": top_confusions(cm, names)}
    em = get("epoch_metrics.csv")
    if em is not None:
        fin = em[em.stage == "final_cv"]
        diag = {}
        for (model, fold), g in fin.groupby(["model", "fold"]):
            best = g.loc[g.macro_f1.idxmax()]
            diag.setdefault(model, []).append({"fold": int(fold), "epochs_run": int(len(g)),
                                               "best_epoch": int(best.epoch),
                                               "final_train_loss": float(g.iloc[-1].train_loss),
                                               "final_val_loss": float(g.iloc[-1].val_loss)})
        s["training_diagnostics"] = diag
        s["max_epochs_observed"] = int(fin.groupby(["model", "fold"]).size().max())
        s["legacy_lrs"] = {"backbone": float(em.lr_backbone.iloc[0]), "head": float(em.lr_head.iloc[0])}
    pol = get("policy_screen_ranking.csv")
    if pol is not None:
        s["policy_screen"] = pol.to_dict("records")
    arch = get("architecture_screen_ranking.csv")
    if arch is not None:
        s["architecture_screen"] = arch[["model", "best_epoch", "accuracy", "macro_f1", "mcc"]].to_dict("records")

    # ---- original random-split runs
    s["random_split_cv"] = {}
    for k, df in tables.items():
        if k.endswith("_Results.csv") and "Macro-F1 (%)" in df:
            s["random_split_cv"][str(df["Model"].iloc[0])] = {
                "accuracy": df["Accuracy (%)"].astype(float).tolist(),
                "macro_f1": df["Macro-F1 (%)"].astype(float).tolist(),
                "mcc": df["MCC"].astype(float).tolist(),
                "final_train_loss": df["Train Loss"].astype(float).tolist()}

    (out / "legacy_summary.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
    return s


def print_report(s: Dict) -> None:
    ms = lambda v: f"{np.mean(v):.2f} ± {np.std(v, ddof=1):.2f}"  # noqa: E731
    print(f"Images: {s.get('n_images')} | imbalance ratio {s.get('imbalance_ratio', 0):.1f}x | {s.get('class_totals')}")
    if "exact_duplicates" in s:
        e = s["exact_duplicates"]
        print(f"Exact duplicates: {e['groups']} groups ({e['conflict_groups']} label conflicts) "
              f"redundant={e['redundant_copies_by_class']}")
    if "near_duplicates" in s:
        print(f"Near duplicates: {s['near_duplicates']['pairs']} pairs, cross-label {s['near_duplicates']['cross_label_pairs']}")
    for proto in ("random_split_cv", "group_aware_cv"):
        for m, v in s.get(proto, {}).items():
            print(f"[{proto}] {m:32s} Acc {ms(v['accuracy'])} | Macro-F1 {ms(v['macro_f1'])}")
    for m, v in s.get("group_aware_cv", {}).items():
        print(f"  {m} top confusions:", [(c['true'], c['pred'], c['count']) for c in v["top_confusions"][:5]])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zips", nargs="+", required=True)
    ap.add_argument("--out", default="outputs/legacy")
    args = ap.parse_args()
    print_report(audit(args.zips, args.out))
