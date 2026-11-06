"""Generate notebooks/pipeline_demo.ipynb (kept as code so the notebook stays reviewable in diffs)."""
from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("""# ASMOE V2 — Potato Leaf Disease Pipeline Demo

End-to-end walkthrough:

1. **Legacy audit**: parse historical Kaggle archives (metrics, duplicates, confusion matrices)
2. **Data audit**: SHA-256 / dHash de-duplication, conflict removal, group-aware stratified 5-fold, ROI leaf cropping
3. **Training**: two-stage refinement of the heterogeneous ASMOE (DenseNet-201 · EfficientNetV2-M · Swin-B)
4. **Evaluation**: mean ± SD over folds, pooled confusion matrix, per-class F1, routing weights
5. **XAI**: routing-weighted Grad-CAM + lesion-focus ratio
6. **Reporting**: LaTeX tables/figures → Overleaf Git sync

> **DRY_RUN** — when the Kaggle dataset is not mounted, the notebook switches to a tiny synthetic dataset and
> random-weight backbones (`configs/smoke_test.yaml`) so every cell still executes on a CPU in a few minutes.
"""),
    code("""import os, sys, json, subprocess
from pathlib import Path

# Repository root: works from notebooks/ locally or from a cloned repo on Kaggle
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if not (ROOT / "src").exists() and Path("/kaggle/working/asmoe-potato").exists():
    ROOT = Path("/kaggle/working/asmoe-potato")
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

IN_KAGGLE = Path("/kaggle/input").exists()
if IN_KAGGLE:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm>=1.0.9"], check=False)

import torch, pandas as pd, numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image as IPImage, display
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available(), "| root:", ROOT)"""),
    code("""from src.config import load_config

CONFIG = "configs/asmoe_v2.yaml"
cfg = load_config(CONFIG)
DRY_RUN = not Path(cfg.data.root).exists()
if DRY_RUN:
    from tests.test_smoke import make_synthetic
    synth = ROOT / "outputs" / "synthetic_potato"
    make_synthetic(synth, per_class=8)
    cfg = load_config("configs/smoke_test.yaml", [f"data.root={synth}", "train.output_dir=outputs/demo",
                                                  "experiment_name=asmoe_v2"])
print("DRY_RUN =", DRY_RUN, "| experiment:", cfg.experiment_name, "| image size:", cfg.data.image_size)
print("experts:", [(e.name, e.role, e.image_size) for e in cfg.model.experts])"""),
    md("## 1 · Legacy audit (historical Kaggle archives)"),
    code("""from src.utils.legacy_audit import audit, print_report

LEGACY_ZIPS = [p for p in [ROOT.parent / "Potato_Improved.zip", ROOT.parent / "Potato.zip",
                           Path("/kaggle/input/potato-legacy/Potato_Improved.zip"),
                           Path("/kaggle/input/potato-legacy/Potato.zip")] if p.exists()]
legacy_out = Path(cfg.train.output_dir) / "legacy"
if LEGACY_ZIPS:
    legacy = audit([str(z) for z in LEGACY_ZIPS], str(legacy_out))
    print_report(legacy)
elif (Path("outputs/legacy/legacy_summary.json")).exists():
    legacy_out.mkdir(parents=True, exist_ok=True)
    legacy = json.loads(Path("outputs/legacy/legacy_summary.json").read_text())
    (legacy_out / "legacy_summary.json").write_text(json.dumps(legacy))
    print_report(legacy)
else:
    legacy = {}
    print("No legacy archives found - skipping.")"""),
    code("""if legacy.get("group_aware_cv"):
    from src.utils.plots import confusion_matrix_figure
    v = legacy["group_aware_cv"]["MobileViT_Standalone"]
    p = confusion_matrix_figure(v["pooled_cm"], v["class_names"], "Legacy MobileViT-S (pooled)",
                                legacy_out / "legacy_cm.png")
    display(IPImage(str(p), width=480))
    pd.DataFrame(v["top_confusions"])"""),
    md("## 2 · Data audit: cleaning, leakage-free folds, ROI cropping"),
    code("""from src.dataset import prepare_data, fold_class_table

data_dir = Path(cfg.train.output_dir) / "data"
df, report = prepare_data(cfg, data_dir)
print(f"raw {report['n_raw']} -> clean {report['n_clean']} | groups {report['n_groups']}")
print("label-conflicting exact duplicates removed:", len(report["exact_conflict_groups"]))
print("redundant same-label copies removed:", report["exact_redundant_removed"])
print("near-duplicate pairs grouped:", report["near_dup_pairs"])
fold_class_table(df, cfg.class_names)"""),
    code("""# Each duplicate group must sit in exactly one fold (no train/val leakage)
assert df.groupby("group")["fold"].nunique().max() == 1
pd.DataFrame(report["removed"]).head(12)"""),
    code("""from PIL import Image, ImageOps
from src.dataset import leaf_roi_box

sample = df.groupby("label").head(1).head(6)
fig, axes = plt.subplots(2, len(sample), figsize=(2.4 * len(sample), 4.8))
for k, (_, r) in enumerate(sample.iterrows()):
    im = ImageOps.exif_transpose(Image.open(r.path)).convert("RGB")
    box = leaf_roi_box(im)
    axes[0, k].imshow(im); axes[0, k].add_patch(plt.Rectangle(box[:2], box[2] - box[0], box[3] - box[1],
                                                              fill=False, ec="#eb6834", lw=2))
    axes[0, k].set_title(r.label, fontsize=9)
    axes[1, k].imshow(Image.open(r.cache_path)); axes[1, k].set_title("ROI crop", fontsize=9)
for ax in axes.ravel(): ax.axis("off")
plt.tight_layout(); plt.show()"""),
    md("""## 3 · Training: two-stage refinement

* **Stage 1**: backbones frozen; FPM + router + head at `lr_head = 3e-4`
* **Stage 2**: all layers; discriminative LRs (`1e-5` backbone / `3e-4` head), warm-up + cosine, AMP, EMA,
  early stopping on validation Macro-F1 (patience 8)

On Kaggle, set `FOLDS = [k]` to run one fold per session; finished folds are skipped automatically and
`resume.pt` restarts an interrupted fold."""),
    code("""from src.trainer import run_cross_validation

FOLDS = None            # e.g. [0] to train a single fold per Kaggle session
summary = run_cross_validation(cfg, df, FOLDS)
print(f"Accuracy {summary['accuracy_mean']:.2f} ± {summary['accuracy_sd']:.2f} | "
      f"Macro-F1 {summary['macro_f1_mean']:.2f} ± {summary['macro_f1_sd']:.2f} | MCC {summary['mcc_mean']:.3f}")"""),
    md("## 4 · Evaluation"),
    code("""run_dir = Path(cfg.train.output_dir) / cfg.experiment_name
pd.read_csv(run_dir / "fold_results.csv")[["fold", "accuracy", "macro_f1", "weighted_f1", "mcc", "best_epoch"]]"""),
    code("""pc = pd.DataFrame(summary["per_class"]).T[["precision_mean", "recall_mean", "f1_mean", "f1_sd"]]
(pc * 100).round(1)"""),
    code("""from src.utils.plots import confusion_matrix_figure, learning_curves_figure, router_weights_figure
from src.utils.overleaf_sync import _router_per_class

figs = Path(cfg.train.output_dir) / "figures"
display(IPImage(str(confusion_matrix_figure(summary["confusion_matrix_pooled"], cfg.class_names,
                                            "ASMOE V2 (pooled)", figs / "cm.png")), width=480))
display(IPImage(str(learning_curves_figure({"ASMOE V2": run_dir / "epoch_metrics.csv"}, figs / "curves.png")), width=720))
rw = _router_per_class(Path(cfg.train.output_dir), cfg.experiment_name)
if rw:
    labels = [e.role for e in cfg.model.experts]
    display(IPImage(str(router_weights_figure(rw, labels, figs / "router.png")), width=720))"""),
    md("## 5 · Explainability: routing-weighted Grad-CAM"),
    code("""from src.models import build_model
from src.dataset import PotatoDataset, build_transforms
from src.utils.gradcam import ASMOEGradCAM, save_gradcam_panel, lesion_focus, denormalize

FOLD = (FOLDS or [0])[0]
model = build_model(cfg.model, len(cfg.class_names), cfg.data.image_size)
model.load_state_dict(torch.load(run_dir / f"fold_{FOLD}" / "best.pt", map_location="cpu", weights_only=False)["model"])
val = df[df.fold == FOLD].groupby("label").head(1)
ds = PotatoDataset(val, build_transforms(cfg.data, train=False))
cam = ASMOEGradCAM(model)
labels = [e.role for e in cfg.model.experts]
focus = []
for i in range(min(4, len(ds))):
    x, y = ds[i]
    res = cam(x.unsqueeze(0))
    focus.append(lesion_focus(res["fused"], denormalize(x.unsqueeze(0))))
    p = save_gradcam_panel(x.unsqueeze(0), res, cfg.class_names, labels,
                           run_dir / "gradcam" / f"{i:02d}_{cfg.class_names[y]}.png", cfg.class_names[y])
    display(IPImage(str(p), width=900))
cam.remove()
print("mean lesion-focus ratio:", np.round(np.mean(focus), 3))"""),
    md("""## 6 · Results → Springer Nature manuscript

`export_run` copies the light results (no checkpoints) to `results/`. `build_paper` turns `results/` into
`paper/build/sn-article.tex`, a single self-contained file in the Springer Nature template, with tables, numbers
and figures filled in. In the real workflow the Kaggle notebook pushes `results/` to GitHub, and a GitHub Action
runs `build_paper`, compiles the PDF and syncs Overleaf. In DRY_RUN the paper is built in a scratch copy."""),
    code("""import shutil
from src.results import export_run
from src.utils.overleaf_sync import build_paper

if DRY_RUN:
    demo = Path(cfg.train.output_dir)
    results_dir, paper_dir = demo / "results", demo / "paper"
    shutil.copytree(ROOT / "paper", paper_dir, dirs_exist_ok=True, ignore=shutil.ignore_patterns("build", "_generated"))
    if (ROOT / "results" / "legacy").exists():
        shutil.copytree(ROOT / "results" / "legacy", results_dir / "legacy", dirs_exist_ok=True)
else:
    results_dir, paper_dir = ROOT / "results", ROOT / "paper"
export_run(run_dir, results_dir, data_dir=Path(cfg.train.output_dir) / "data")
manifest = build_paper(results_dir, paper_dir)
print(manifest["rendered"])
print(open(paper_dir / "build" / "sn-article.tex", encoding="utf-8").read()[:1500])"""),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}})
out = Path(__file__).resolve().parents[1] / "notebooks" / "pipeline_demo.ipynb"
nbf.write(nb, out)
print("wrote", out)
