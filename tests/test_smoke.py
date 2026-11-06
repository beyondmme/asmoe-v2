"""CPU smoke tests: data cleaning, leakage-free folds, ASMOE forward/backward, two-stage CV, report build.

Run:  python -m pytest -q tests   (or)   python tests/test_smoke.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CLASS_NAMES, load_config  # noqa: E402
from src.dataset import assign_folds, audit_dataset, build_roi_cache, leaf_roi_box, scan_dataset  # noqa: E402
from src.losses import FocalLoss, SupConLoss  # noqa: E402
from src.models import build_model  # noqa: E402

CFG = ROOT / "configs" / "smoke_test.yaml"


def make_synthetic(root: Path, per_class: int = 8, seed: int = 0) -> Path:
    """Leaf-like ellipses on a brown background, plus planted duplicates."""
    if root.exists():
        shutil.rmtree(root)
    rng = np.random.default_rng(seed)
    for ci, cls in enumerate(CLASS_NAMES):
        (root / cls).mkdir(parents=True)
        for k in range(per_class):
            a = np.zeros((120, 160, 3), np.uint8)
            a[:] = (110 + rng.integers(-20, 20), 85, 60)
            yy, xx = np.mgrid[:120, :160]
            cy, cx = 60 + rng.integers(-15, 15), 80 + rng.integers(-25, 25)
            leaf = ((yy - cy) / 40.0) ** 2 + ((xx - cx) / 55.0) ** 2 < 1
            a[leaf] = (40 + 20 * ci, 140 + rng.integers(-20, 20), 40)
            spots = leaf & (rng.random(leaf.shape) < 0.02 * (ci + 1))
            a[spots] = (120, 80, 20)
            Image.fromarray(a).save(root / cls / f"img_{k}.jpg", quality=95)
    # exact duplicate with conflicting label (must be dropped) and a same-label copy (deduped)
    shutil.copy(root / "Pest" / "img_0.jpg", root / "Fungi" / "conflict.jpg")
    shutil.copy(root / "Virus" / "img_1.jpg", root / "Virus" / "img_1 (1).jpg")
    return root


def test_data_pipeline(tmp: Path = ROOT / "outputs" / "test_data") -> None:
    cfg = load_config(str(CFG), [f"data.root={tmp / 'raw'}", f"data.cache_dir={tmp / 'cache'}"])
    make_synthetic(tmp / "raw")
    raw = scan_dataset(cfg.data.root)
    assert len(raw) == 7 * 8 + 2
    clean, rep = audit_dataset(raw, cfg.data)
    assert len(rep["exact_conflict_groups"]) == 1
    assert rep["exact_redundant_removed"] == 1
    assert not clean["path"].str.contains("conflict").any()
    folds = assign_folds(clean, 2, 0)
    assert folds.groupby("group")["fold"].nunique().max() == 1
    cached = build_roi_cache(folds, cfg.data)
    assert all(Path(p).exists() for p in cached["cache_path"])
    with Image.open(raw.iloc[0]["path"]) as im:
        x0, y0, x1, y1 = leaf_roi_box(im.convert("RGB"))
        assert 0 <= x0 < x1 <= im.width and 0 <= y0 < y1 <= im.height


def test_model_and_losses() -> None:
    cfg = load_config(str(CFG))
    model = build_model(cfg.model, 7, cfg.data.image_size)
    x = torch.randn(4, 3, 64, 64)
    out = model(x, return_maps=True)
    assert out["logits"].shape == (4, 7)
    assert torch.allclose(out["weights"].sum(1), torch.ones(4), atol=1e-5)
    assert len(out["maps"]) == 3 and all(m.shape[1] == cfg.model.proj_dim for m in out["maps"])
    y = torch.tensor([0, 0, 1, 1])
    loss = FocalLoss()(out["logits"], torch.nn.functional.one_hot(y, 7).float()) + SupConLoss()(out["supcon"], y)
    loss = loss + model.balance_loss(out["weights"])
    loss.backward()
    assert model.head.weight.grad is not None
    single = build_model(cfg.model, 7, 64, single_expert="resnet18")
    assert len(single.experts) == 1 and single(x)["weights"].shape == (4, 1)


def test_cv_and_report(tmp: Path = ROOT / "outputs" / "test_cv") -> None:
    from src.trainer import run_cross_validation
    from src.utils.overleaf_sync import build_report

    if tmp.exists():
        shutil.rmtree(tmp)
    make_synthetic(tmp / "raw", per_class=6)
    cfg = load_config(str(CFG), [f"data.root={tmp / 'raw'}", f"data.cache_dir={tmp / 'cache'}",
                                 f"train.output_dir={tmp}", "experiment_name=asmoe_v2"])
    raw = scan_dataset(cfg.data.root)
    clean, _ = audit_dataset(raw, cfg.data)
    df = build_roi_cache(assign_folds(clean, 2, 0), cfg.data)
    summary = run_cross_validation(cfg, df)
    assert summary["n_folds"] == 2 and 0 <= summary["macro_f1_mean"] <= 100
    assert (tmp / "asmoe_v2" / "fold_0" / "best.pt").exists()
    # resume: a completed fold is skipped
    again = run_cross_validation(cfg, df, folds=[0])
    assert again["n_folds"] == 2
    paper = tmp / "paper"
    man = build_report(tmp / "missing_legacy.json", tmp, ROOT / "configs" / "paper_experiments.yaml", paper)
    assert man["main_available"]
    assert (paper / "tables" / "main_results.tex").exists()
    assert "VtwoFOne" in (paper / "tables" / "autogen_macros.tex").read_text()
    json.loads((paper / "autogen_manifest.json").read_text())


if __name__ == "__main__":
    test_data_pipeline()
    print("data ok")
    test_model_and_losses()
    print("model ok")
    test_cv_and_report()
    print("cv + report ok")


def test_gpu_probe_fallback() -> None:
    """Ladder logic (fake trials) + real isolated child process on CPU."""
    from src.utils.cuda_compat import find_settings, subprocess_attempt
    tried = []

    def kaggle_like(opt):     # cuDNN engine error until NCHW + benchmark off
        tried.append(opt)
        if opt["channels_last"] or opt["benchmark"]:
            return {"ok": False, "error": "RuntimeError: GET was unable to find an engine to execute this computation"}
        return {"ok": True, "sec_per_step": 0.5, "peak_gb": 9.0}

    res = find_settings(kaggle_like, channels_last=True)
    assert (res["channels_last"], res["cudnn"], res["benchmark"]) == (False, True, False) and len(tried) == 3

    try:
        find_settings(lambda opt: {"ok": False, "oom": True, "error": "CUDA out of memory"})
        raise AssertionError("OOM must not be retried")
    except RuntimeError as e:
        assert "batch_size" in str(e)

    cfg = load_config(str(CFG), ["train.batch_size=2"])
    r = subprocess_attempt(cfg, None, "cpu", steps=1)({"channels_last": True, "cudnn": True, "benchmark": False})
    assert r["ok"], r
    r = subprocess_attempt(cfg, "no_such_model_xyz", "cpu", steps=1)({"channels_last": False, "cudnn": True,
                                                                       "benchmark": False})
    assert not r["ok"] and "no_such_model_xyz" in r["error"].lower() or "unknown model" in r["error"].lower(), r
    torch.backends.cudnn.enabled, torch.backends.cudnn.benchmark = True, False


def test_fp16_losses() -> None:
    """All losses must work under fp16 autocast (T4 / Kaggle): no overflow, same value as fp32."""
    from src.losses import SoftTargetCE
    from src.models import ASMOE
    torch.manual_seed(0)
    z, y = torch.randn(8, 16, requires_grad=True), torch.randint(0, 3, (8,))
    ref = SupConLoss()(z, y)
    with torch.autocast("cpu", dtype=torch.float16):
        l16 = SupConLoss()(z, y)
    assert torch.isfinite(l16) and abs(float(ref) - float(l16)) < 1e-5
    cfg = load_config(str(CFG))
    m = build_model(cfg.model, 7, 64)
    yy = torch.tensor([0, 0, 1, 1])
    t = torch.nn.functional.one_hot(yy, 7).float()
    with torch.autocast("cpu", dtype=torch.float16):
        out = m(torch.randn(4, 3, 64, 64))
        loss = (SoftTargetCE()(out["logits"], t) + FocalLoss()(out["logits"], t)
                + 0.1 * SupConLoss()(out["supcon"], yy) + 0.01 * ASMOE.balance_loss(out["weights"].float()))
    loss.backward()
    assert torch.isfinite(loss)
