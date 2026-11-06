#!/usr/bin/env python
"""Unified CLI for the ASMOE V2 potato-leaf pipeline.

  python train.py audit-legacy --zips ../Potato_Improved.zip ../Potato.zip   # -> results/legacy
  python train.py prepare --config configs/asmoe_v2.yaml                     # dedupe, folds, ROI cache
  python train.py train   --config configs/asmoe_v2.yaml --folds 0 --export  # train fold 0, copy to results/
  python train.py train   --config configs/asmoe_v2_rtx4090.yaml --push      # all folds; push each to GitHub
  python train.py train   --config configs/asmoe_v2.yaml --expert <timm-name> --name single_swinb
  python train.py suite   --only main singles ablations                      # configs/paper_experiments.yaml
  python train.py gradcam --config configs/asmoe_v2.yaml --fold 0 --n 8
  python train.py export  --name asmoe_v2                                    # outputs/ -> results/ (no checkpoints)
  python train.py paper   [--push]                                           # results/ -> paper/build/sn-article.tex

Heavy outputs (checkpoints) stay in outputs/; light, commit-able results go to results/, which the
GitHub Action turns into the Springer Nature manuscript and pushes to Overleaf.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import Config, load_config  # noqa: E402


def _data(cfg: Config, reprepare: bool = False) -> pd.DataFrame:
    """Load (or create) the cleaned, fold-assigned split shared by every experiment."""
    from src.dataset import build_roi_cache, prepare_data

    data_dir = Path(cfg.train.output_dir) / "data"
    split = data_dir / "clean_split.csv"
    sig = {"root": cfg.data.root, "n_folds": cfg.data.n_folds, "seed": cfg.data.fold_seed,
           "near_dup_threshold": cfg.data.near_dup_threshold,
           "drop_conflicts": cfg.data.drop_conflicting_duplicates}
    sig_path = data_dir / "split_signature.json"
    if split.exists() and sig_path.exists() and not reprepare and json.loads(sig_path.read_text()) == sig:
        df = pd.read_csv(split)
        return build_roi_cache(df, cfg.data)   # cheap if cached; handles roi on/off ablation
    df, report = prepare_data(cfg, data_dir)
    sig_path.write_text(json.dumps(sig, indent=2))
    print(f"[data] raw {report['n_raw']} -> clean {report['n_clean']} images | "
          f"{len(report['exact_conflict_groups'])} conflicting exact-dup groups dropped | "
          f"{report['exact_redundant_removed']} redundant copies removed | "
          f"{report['near_dup_pairs']} near-dup pairs grouped | {report['n_groups']} groups")
    return df


def _short(summary) -> str:
    if not summary.get("n_folds"):
        return "no completed folds"
    return f"{summary['n_folds']} folds, Macro-F1 {summary['macro_f1_mean']:.2f}"


def cmd_summarize(args) -> None:
    from src.results import summarise_run
    cfg = load_config(args.config, args.set)
    if args.name:
        cfg.experiment_name = args.name
    summarise_run(Path(cfg.train.output_dir) / cfg.experiment_name, cfg.class_names, cfg.experiment_name)


def cmd_audit_legacy(args) -> None:
    from src.utils.legacy_audit import audit, print_report
    print_report(audit(args.zips, args.out))


def cmd_prepare(args) -> None:
    cfg = load_config(args.config, args.set)
    df = _data(cfg, reprepare=True)
    from src.dataset import fold_class_table
    print(fold_class_table(df, cfg.class_names).to_string())


def cmd_train(args) -> None:
    cfg = load_config(args.config, args.set)
    if args.name:
        cfg.experiment_name = args.name
    df = _data(cfg)
    _run(cfg, df, args.folds, args.expert, args)


def _run(cfg: Config, df: pd.DataFrame, folds, expert, args) -> None:
    """Train the requested folds. With --push: after EACH finished fold, Grad-CAM (fold 0 of the main model),
    export to results/ and git commit + push, so the GitHub Action rebuilds the paper while training continues."""
    from src.trainer import run_cross_validation
    folds = list(folds) if folds is not None else list(range(cfg.data.n_folds))
    if not args.push:
        summary = run_cross_validation(cfg, df, folds, single_expert=expert)
        print(f"[{cfg.experiment_name}] {_short(summary)}")
        if args.export:
            _export(cfg, args.results)
        return
    from src.utils.github_sync import push_results
    run_dir = Path(cfg.train.output_dir) / cfg.experiment_name
    for k in folds:
        run_cross_validation(cfg, df, [k], single_expert=expert)
        result = run_dir / f"fold_{k}" / "result.json"
        if not result.exists():
            print(f"[fold {k}] not finished (paused) - stopping here; re-run the same command to resume")
            return
        if k == 0 and expert is None and cfg.experiment_name == "asmoe_v2":
            _gradcam(cfg, 0, 8, df)
        _export(cfg, args.results)
        f1 = json.loads(result.read_text())["macro_f1"]
        res = push_results(ROOT, [args.results], f"results: {cfg.experiment_name} fold {k} (Macro-F1 {f1:.2f})")
        print(f"[github] fold {k}: {res}")


def cmd_suite(args) -> None:
    import yaml
    reg = yaml.safe_load(Path(args.registry).read_text(encoding="utf-8"))
    base_cfg_path = args.config or reg["main"].get("config", "configs/asmoe_v2.yaml")
    jobs = []
    if "main" in args.only:
        jobs.append((reg["main"]["name"], [], None))
    if "singles" in args.only:
        jobs += [(e["name"], [], e["expert"]) for e in reg.get("single_experts", [])]
    if "ablations" in args.only:
        jobs += [(a["name"], a.get("overrides", []), None) for a in reg.get("ablations", [])]
    for name, overrides, expert in jobs:
        cfg = load_config(base_cfg_path, list(overrides) + list(args.set or []))
        cfg.experiment_name = name
        print(f"\n===== {name} =====")
        _run(cfg, _data(cfg), args.folds, expert, args)


def _export(cfg: Config, results: str) -> None:
    from src.results import export_run, summarise_run
    run_dir = Path(cfg.train.output_dir) / cfg.experiment_name
    copied = export_run(run_dir, Path(results), data_dir=Path(cfg.train.output_dir) / "data")
    summarise_run(Path(results) / cfg.experiment_name, cfg.class_names, cfg.experiment_name)
    print(f"[export] {len(copied)} files -> {Path(results) / cfg.experiment_name}")


def cmd_export(args) -> None:
    cfg = load_config(args.config, args.set)
    if args.name:
        cfg.experiment_name = args.name
    _export(cfg, args.results)


def cmd_paper(args) -> None:
    from src.utils.overleaf_sync import build_paper, sync_build_to_overleaf
    build_paper(Path(args.results), ROOT / "paper")
    if args.push:
        cfg = load_config(None)
        print("[overleaf]", sync_build_to_overleaf(ROOT / "paper" / "build", cfg.overleaf.git_url,
                                                    ROOT / "outputs" / "overleaf_repo", "auto-sync from train.py",
                                                    push=True))


def _gradcam(cfg: Config, fold: int, n: int, df: Optional[pd.DataFrame] = None) -> None:
    import torch
    from src.dataset import PotatoDataset, build_transforms
    from src.models import build_model
    from src.utils.gradcam import ASMOEGradCAM, lesion_focus, save_gradcam_panel, denormalize

    run_dir = Path(cfg.train.output_dir) / cfg.experiment_name
    ckpt = torch.load(run_dir / f"fold_{fold}" / "best.pt", map_location="cpu", weights_only=False)
    model = build_model(cfg.model, len(cfg.class_names), cfg.data.image_size)
    model.load_state_dict(ckpt["model"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    df = _data(cfg) if df is None else df
    val = df[df["fold"] == fold]
    # one sample per class first, then fill up
    pick = val.groupby("label", group_keys=False).apply(lambda g: g.sample(1, random_state=0))
    pick = pd.concat([pick, val.drop(pick.index).sample(max(0, n - len(pick)), random_state=0)]).head(n)
    ds = PotatoDataset(pick, build_transforms(cfg.data, train=False))
    cam = ASMOEGradCAM(model)
    labels = [e.role if e.role != "expert" else e.name.split(".")[0] for e in cfg.model.experts]
    focus = []
    for i in range(len(ds)):
        x, y = ds[i]
        x = x.unsqueeze(0).to(device)
        res = cam(x)
        focus.append(lesion_focus(res["fused"], denormalize(x)))
        save_gradcam_panel(x, res, cfg.class_names, labels,
                           run_dir / "gradcam" / f"{i:02d}_{cfg.class_names[y]}.png", cfg.class_names[y])
    cam.remove()
    out = {"fold": fold, "n": len(ds), "lesion_focus_mean": float(sum(focus) / max(1, len(focus))),
           "lesion_focus": focus}
    (run_dir / "gradcam" / "lesion_focus.json").write_text(json.dumps(out, indent=2))
    print(f"[gradcam] {len(ds)} panels -> {run_dir / 'gradcam'} | mean lesion focus {out['lesion_focus_mean']:.3f}")
    del model
    if device == "cuda":
        torch.cuda.empty_cache()


def cmd_gradcam(args) -> None:
    cfg = load_config(args.config, args.set)
    if args.name:
        cfg.experiment_name = args.name
    _gradcam(cfg, args.fold, args.n)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, config=True):
        if config:
            p.add_argument("--config", default="configs/asmoe_v2.yaml")
        p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                       help="dotted overrides, e.g. train.batch_size=4 data.image_size=384")

    p = sub.add_parser("audit-legacy", help="parse historical Kaggle result zips")
    p.add_argument("--zips", nargs="+", required=True)
    p.add_argument("--out", default="results/legacy")
    p.set_defaults(fn=cmd_audit_legacy)

    p = sub.add_parser("prepare", help="dataset audit + folds + ROI cache")
    common(p)
    p.set_defaults(fn=cmd_prepare)

    p = sub.add_parser("train", help="5-fold CV for one experiment")
    common(p)
    p.add_argument("--folds", type=int, nargs="*", default=None)
    p.add_argument("--expert", default=None, help="timm name: train a single-expert baseline")
    p.add_argument("--name", default=None, help="experiment name (output sub-folder)")
    p.add_argument("--export", action="store_true", help="copy finished folds to results/ (commit-able)")
    p.add_argument("--push", action="store_true",
                   help="after each finished fold: Grad-CAM (fold 0), export to results/, git commit + push")
    p.add_argument("--results", default="results")
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("suite", help="run experiments from configs/paper_experiments.yaml")
    common(p, config=False)
    p.add_argument("--registry", default="configs/paper_experiments.yaml")
    p.add_argument("--config", default=None, help="base config (default: the registry's main config)")
    p.add_argument("--only", nargs="+", default=["main", "singles", "ablations"],
                   choices=["main", "singles", "ablations"])
    p.add_argument("--folds", type=int, nargs="*", default=None)
    p.add_argument("--export", action="store_true")
    p.add_argument("--push", action="store_true")
    p.add_argument("--results", default="results")
    p.set_defaults(fn=cmd_suite)

    p = sub.add_parser("summarize", help="aggregate completed folds (e.g. merged from several Kaggle sessions)")
    common(p)
    p.add_argument("--name", default=None)
    p.set_defaults(fn=cmd_summarize)

    p = sub.add_parser("export", help="copy finished folds (no checkpoints) from outputs/ to results/")
    common(p)
    p.add_argument("--name", default=None)
    p.add_argument("--results", default="results")
    p.set_defaults(fn=cmd_export)

    p = sub.add_parser("paper", help="results/ -> single-file Springer manuscript in paper/build/")
    p.add_argument("--results", default="results")
    p.add_argument("--push", action="store_true", help="push paper/build to Overleaf (needs OVERLEAF_GIT_TOKEN)")
    p.set_defaults(fn=cmd_paper)

    p = sub.add_parser("gradcam", help="routing-weighted Grad-CAM panels")
    common(p)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--name", default=None)
    p.set_defaults(fn=cmd_gradcam)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
