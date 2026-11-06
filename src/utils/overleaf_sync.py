"""Build LaTeX tables/figures from experiment outputs and push them to an Overleaf Git repository.

Two steps, usable separately:

1. ``build_report``  — reads ``legacy_summary.json`` + every ``outputs/<exp>/cv_summary.json`` listed in
   ``configs/paper_experiments.yaml`` and writes
       paper/tables/*.tex            (booktabs tables, mean ± SD over folds)
       paper/tables/autogen_macros.tex  (\\newcommand numbers used in the prose)
       paper/figures/*.png           (confusion matrices, learning curves, per-class F1, router weights)
   Missing experiments render as an em-dash, so the paper always compiles.

2. ``sync_to_overleaf`` — clones/pulls the Overleaf project, copies ``paper/{sections,tables,figures}``,
   commits, and (with ``--push``) pushes. Authentication uses an Overleaf Git token read from the
   ``OVERLEAF_GIT_TOKEN`` environment variable; it is passed to git through ``GIT_CONFIG_*`` env vars,
   never written to ``.git/config`` or the command line.

CLI::

    python -m src.utils.overleaf_sync --stage "fold 3 done"            # build + commit locally (dry run)
    OVERLEAF_GIT_TOKEN=olp_xxx python -m src.utils.overleaf_sync --push  # build + commit + push
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

from ..config import CLASS_NAMES
from . import latex_tables as lt
from . import plots

REPO_ROOT = Path(__file__).resolve().parents[2]


# ============================================================================ loading
def load_registry(path: Path) -> Dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_summary(results_root: Path, name: str) -> Optional[Dict]:
    p = Path(results_root) / name / "cv_summary.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _entry_from_summary(label: str, s: Optional[Dict], res: str) -> Dict:
    e: Dict = {"label": label, "res": res}
    if s:
        e.update(acc=(s["accuracy_mean"], s["accuracy_sd"]), f1=(s["macro_f1_mean"], s["macro_f1_sd"]),
                 mcc=(s["mcc_mean"], s["mcc_sd"]),
                 params=f"{s['params_M']:.1f}" if s.get("params_M") else lt.PENDING,
                 per_class={c: (v["f1_mean"], v["f1_sd"]) for c, v in s["per_class"].items()},
                 nematode_f1=(s["per_class"]["Nematode"]["f1_mean"], s["per_class"]["Nematode"]["f1_sd"]),
                 n_folds=s["n_folds"])
    return e


def _entry_from_legacy(label: str, v: Dict, params: Optional[float]) -> Dict:
    pcs = {c: lt.ms(f) for c, f in v["per_class_f1"].items()}
    return {"label": label, "res": "224", "acc": lt.ms(v["accuracy"]), "f1": lt.ms(v["macro_f1"]),
            "mcc": lt.ms(v["mcc"]), "params": f"{params:.1f}" if params else lt.PENDING,
            "per_class": pcs, "nematode_f1": pcs.get("Nematode")}


def _router_per_class(results_root: Path, name: str) -> Optional[Dict[str, List[float]]]:
    acc: Dict[str, List[List[float]]] = {}
    for p in sorted((Path(results_root) / name).glob("fold_*/result.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        for c, w in r.get("router_weights_per_class", {}).items():
            acc.setdefault(c, []).append(w)
    return {c: np.mean(v, axis=0).tolist() for c, v in acc.items()} if acc else None


# ============================================================================ report
def build_report(legacy_summary: Path, results_root: Path, registry_path: Path, paper_dir: Path) -> Dict:
    paper_dir = Path(paper_dir)
    tdir, fdir = paper_dir / "tables", paper_dir / "figures"
    tdir.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)
    reg = load_registry(registry_path)
    leg = json.loads(Path(legacy_summary).read_text(encoding="utf-8")) if Path(legacy_summary).exists() else {}
    written: List[str] = []

    def write(name: str, text: str) -> None:
        (tdir / name).write_text(text, encoding="utf-8")
        written.append(f"tables/{name}")

    # ---------------- legacy tables / figures
    data_audit_path = Path(results_root) / "data" / "data_audit.json"
    clean_counts = None
    if data_audit_path.exists():
        clean_counts = json.loads(data_audit_path.read_text(encoding="utf-8")).get("class_counts_clean")
    if leg:
        write("legacy_cv.tex", lt.legacy_cv_table(leg))
        write("legacy_policy.tex", lt.policy_screen_table(leg))
        write("data_audit.tex", lt.data_audit_table(leg, clean_counts))
        for model, v in leg.get("group_aware_cv", {}).items():
            out = plots.confusion_matrix_figure(v["pooled_cm"], v["class_names"],
                                                f"Legacy {model} (pooled 5 folds)", fdir / f"legacy_cm_{model}.png")
            written.append(f"figures/{out.name}")
        legacy_curves = Path(legacy_summary).parent / "figures" / "final_cv_learning_curves.png"
        if legacy_curves.exists():
            shutil.copy2(legacy_curves, fdir / "legacy_learning_curves.png")
            written.append("figures/legacy_learning_curves.png")

    # ---------------- new experiments
    main_name = reg["main"]["name"]
    main_s = load_summary(results_root, main_name)
    main_entry = _entry_from_summary(reg["main"]["label"], main_s, reg["main"].get("res", "384/512"))
    base_key = reg.get("legacy_baseline", "MobileViT_Standalone")
    params = {"MobileViT_Standalone": 4.9, "EffNet_MobileViT_ECA_Gated": 10.5}
    legacy = leg.get("group_aware_cv", {})
    legacy_entries = [_entry_from_legacy(f"{lt.model_name(k)}$^{{\\dagger}}$", v, params.get(k)) for k, v in legacy.items()]
    base_entry = next((e for k, e in zip(legacy, legacy_entries) if k == base_key), None)
    # Only runs that have finished appear in the tables (no placeholder rows for experiments not run).
    single_entries = [_entry_from_summary(e["label"], s, e.get("res", "384"))
                      for e in reg.get("single_experts", [])
                      if (s := load_summary(results_root, e["name"]))]
    write("main_results.tex", lt.main_comparison_table(legacy_entries + single_entries + [main_entry]))

    cols = []
    if base_entry:
        cols.append({"label": "MobileViT-S$^{\\dagger}$", "per_class": base_entry["per_class"]})
    cols += [{"label": e["label"], "per_class": e.get("per_class", {})} for e in single_entries]
    cols.append({"label": "ASMOE V2", "per_class": main_entry.get("per_class", {})})
    write("per_class.tex", lt.per_class_table(cols))

    abl_done = [_entry_from_summary(a["label"], s, "")
                for a in reg.get("ablations", []) if (s := load_summary(results_root, a["name"]))]
    if abl_done:
        write("ablation.tex", lt.ablation_table([dict(main_entry, label="Full ASMOE V2")] + abl_done,
                                                main_entry["f1"][0] if main_entry.get("f1") else None))

    # ---------------- figures for new runs
    if main_s:
        out = plots.confusion_matrix_figure(main_s["confusion_matrix_pooled"], CLASS_NAMES,
                                            "ASMOE V2 (pooled 5 folds)", fdir / "asmoe_v2_cm.png")
        written.append(f"figures/{out.name}")
        rw = _router_per_class(results_root, main_name)
        if rw:
            out = plots.router_weights_figure(rw, reg["main"].get("expert_labels", ["E1", "E2", "E3"]),
                                              fdir / "router_weights.png")
            written.append(f"figures/{out.name}")
    curves = {main_entry["label"]: Path(results_root) / main_name / "epoch_metrics.csv"}
    for e in reg.get("single_experts", [])[:2]:
        curves[e["label"]] = Path(results_root) / e["name"] / "epoch_metrics.csv"
    if plots.learning_curves_figure(curves, fdir / "learning_curves.png"):
        written.append("figures/learning_curves.png")
    bars = {}
    if base_entry:
        bars["Legacy best"] = {c: v[0] for c, v in base_entry["per_class"].items()}
    if main_entry.get("per_class"):
        bars["ASMOE V2"] = {c: v[0] for c, v in main_entry["per_class"].items()}
    if bars:
        plots.per_class_f1_figure(bars, fdir / "per_class_f1.png")
        written.append("figures/per_class_f1.png")
    gc_dir = Path(results_root) / main_name / "gradcam"
    for png in sorted(gc_dir.glob("*.png"))[:8] if gc_dir.exists() else []:
        shutil.copy2(png, fdir / f"gradcam_{png.name}")
        written.append(f"figures/gradcam_{png.name}")

    # ---------------- macros
    m: Dict[str, Optional[str]] = {}
    f2 = lambda x: f"{x:.2f}"  # noqa: E731
    if leg:
        m["LegacyNImages"] = str(leg.get("n_images", ""))
        m["LegacyImbalance"] = f"{leg.get('imbalance_ratio', 0):.1f}"
        ed = leg.get("exact_duplicates", {})
        m["LegacyExactGroups"] = str(ed.get("groups", ""))
        m["LegacyConflictGroups"] = str(ed.get("conflict_groups", ""))
        m["LegacyRedundant"] = str(sum(ed.get("redundant_copies_by_class", {}).values()))
        m["LegacyRedundantPhyto"] = str(ed.get("redundant_copies_by_class", {}).get("Phytopthora", 0))
        m["LegacyNearPairs"] = str(leg.get("near_duplicates", {}).get("pairs", ""))
        for r in leg.get("policy_screen", []):
            m[f"LegacyPolicy{r['augmentation'].title()}{r['class_weight'].title()}"] = f2(r["screen_macro_f1"])
        for k, v in leg.get("random_split_cv", {}).items():
            m[f"Random{k}F"] = f2(np.mean(v["macro_f1"]))
        if base_entry:
            m["BaseAcc"], m["BaseAccSD"] = f2(base_entry["acc"][0]), f2(base_entry["acc"][1])
            m["BaseFOne"], m["BaseFOneSD"] = f2(base_entry["f1"][0]), f2(base_entry["f1"][1])
            for c, v in base_entry["per_class"].items():
                m[f"BaseFOne{c}"] = f"{100 * v[0]:.1f}"
    if main_entry.get("f1"):
        m["VtwoAcc"], m["VtwoAccSD"] = f2(main_entry["acc"][0]), f2(main_entry["acc"][1])
        m["VtwoFOne"], m["VtwoFOneSD"] = f2(main_entry["f1"][0]), f2(main_entry["f1"][1])
        m["VtwoMCC"] = f"{main_entry['mcc'][0]:.3f}"
        for c, v in main_entry["per_class"].items():
            m[f"VtwoFOne{c}"] = f"{100 * v[0]:.1f}"
        if base_entry:
            m["DeltaFOne"] = f"{main_entry['f1'][0] - base_entry['f1'][0]:+.2f}"
            m["DeltaAcc"] = f"{main_entry['acc'][0] - base_entry['acc'][0]:+.2f}"
            for c in CLASS_NAMES:
                d = 100 * (main_entry["per_class"][c][0] - base_entry["per_class"][c][0])
                m[f"DeltaFOne{c}"] = f"{d:+.1f}"
        m["VtwoMeetsTarget"] = "yes" if (main_entry["f1"][0] >= 95 and main_entry["acc"][0] >= 95) else "no"
    else:
        for k in ["VtwoAcc", "VtwoAccSD", "VtwoFOne", "VtwoFOneSD", "VtwoMCC", "DeltaFOne", "DeltaAcc",
                  "VtwoMeetsTarget"] + [f"VtwoFOne{c}" for c in CLASS_NAMES] + [f"DeltaFOne{c}" for c in CLASS_NAMES]:
            m[k] = None
    for a in reg.get("ablations", []):
        s = load_summary(results_root, a["name"])
        if not s:
            continue
        key = "Abl" + "".join(w.title() for w in a["name"].replace("abl_", "").split("_"))
        m[key + "FOne"] = f2(s["macro_f1_mean"]) if s else None
        m[key + "Delta"] = (f"{s['macro_f1_mean'] - main_entry['f1'][0]:+.2f}"
                            if s and main_entry.get("f1") else None)
    m["ReportGenerated"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    write("autogen_macros.tex", lt.macros(m))
    manifest = {"written": written, "main_available": bool(main_s), "macros": m}
    (paper_dir / "autogen_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


# ============================================================================ git sync
def _git_env(token: Optional[str]) -> Dict[str, str]:
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    if token:
        basic = base64.b64encode(f"git:{token}".encode()).decode()
        env.update(GIT_CONFIG_COUNT="1", GIT_CONFIG_KEY_0="http.extraHeader",
                   GIT_CONFIG_VALUE_0=f"Authorization: Basic {basic}")
    return env


def _git(args: List[str], cwd: Optional[Path], env: Dict[str, str], check: bool = True) -> str:
    r = subprocess.run(["git", *args], cwd=cwd, env=env, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args[:2])} failed: {r.stderr.strip()[:500]}")
    return r.stdout.strip()




def sync_build_to_overleaf(build_dir: Path, git_url: str, local_dir: Path, message: str, push: bool = False,
                           branch: str = "master") -> Dict:
    """Copy paper/build/* (single-file manuscript + figures + bib/cls) into the Overleaf project and commit.

    Files that exist only in Overleaf (e.g. the template's sample figures, user manual) are left untouched.
    The manuscript file is overwritten: edit text in the GitHub repo (paper/sections, paper/template), not in Overleaf.
    """
    token = os.environ.get("OVERLEAF_GIT_TOKEN")
    if not token:
        raise RuntimeError("OVERLEAF_GIT_TOKEN is not set (Overleaf -> Account settings -> Git integration).")
    env = _git_env(token)
    local_dir = Path(local_dir)
    if not (local_dir / ".git").exists():
        local_dir.parent.mkdir(parents=True, exist_ok=True)
        _git(["clone", git_url, str(local_dir)], None, env)
    else:
        _git(["pull", "--rebase", "--autostash", "origin", branch], local_dir, env)
    copied = 0
    skip = {".aux", ".log", ".pdf", ".bbl", ".blg", ".fls", ".fdb_latexmk", ".out", ".gz", ".toc"}
    for f in Path(build_dir).rglob("*"):
        if f.is_file() and f.suffix not in skip:
            dst = local_dir / f.relative_to(build_dir)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)
            copied += 1
    _git(["add", "-A", "."], local_dir, env)
    if not _git(["status", "--porcelain"], local_dir, env):
        return {"copied": copied, "committed": False, "pushed": False}
    _git(["-c", "user.name=ASMOE auto-sync", "-c", "user.email=asmoe-sync@users.noreply.github.com",
          "commit", "-m", message], local_dir, env)
    if push:
        _git(["push", "origin", f"HEAD:{branch}"], local_dir, env)
    return {"copied": copied, "committed": True, "pushed": push}


def build_paper(results_root: Path = REPO_ROOT / "results", paper_dir: Path = REPO_ROOT / "paper",
                registry: Path = REPO_ROOT / "configs" / "paper_experiments.yaml") -> Dict:
    """results/ -> cv summaries -> tables/figures/macros -> paper/build/sn-article.tex (single file)."""
    from ..results import summarise_all
    from .manuscript import render_manuscript
    results_root, paper_dir = Path(results_root), Path(paper_dir)
    summaries = summarise_all(results_root, CLASS_NAMES)
    generated = paper_dir / "_generated"
    if generated.exists():
        shutil.rmtree(generated)
    manifest = build_report(results_root / "legacy" / "legacy_summary.json", results_root, Path(registry), generated)
    rendered = render_manuscript(paper_dir, generated, paper_dir / "build")
    manifest.update(rendered=rendered, experiments={k: v.get("n_folds", 0) for k, v in summaries.items()})
    print(f"[paper] {rendered['tex']} ({rendered['lines']} lines) | experiments with results: "
          f"{manifest['experiments'] or 'none yet'} | figures pending: {len(rendered['figures_pending'])}")
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(REPO_ROOT / "results"))
    ap.add_argument("--paper-dir", default=str(REPO_ROOT / "paper"))
    ap.add_argument("--overleaf-url", default=os.environ.get("OVERLEAF_GIT_URL",
                                                           "https://git.overleaf.com/6ae6eb411f26b6a0c2f760f8"))
    ap.add_argument("--local-dir", default=str(REPO_ROOT / "outputs" / "overleaf_repo"))
    ap.add_argument("--push", action="store_true", help="push paper/build to Overleaf (needs OVERLEAF_GIT_TOKEN)")
    ap.add_argument("--message", default="auto-sync from GitHub")
    a = ap.parse_args()
    build_paper(Path(a.results), Path(a.paper_dir))
    if a.push:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        print("[overleaf]", sync_build_to_overleaf(Path(a.paper_dir) / "build", a.overleaf_url, Path(a.local_dir),
                                                    f"{a.message} ({stamp})", push=True))
