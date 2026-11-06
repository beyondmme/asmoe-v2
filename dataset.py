"""Data audit, cleaning, leakage-free fold assignment, ROI cropping and loaders.

Pipeline
--------
1. ``scan_dataset``      -> every image file with its folder label.
2. ``audit_dataset``     -> SHA-256 (exact duplicates) + 64-bit dHash (near duplicates);
                            conflicting duplicates dropped, redundant copies removed,
                            near-duplicates merged into one *group* (union-find).
3. ``assign_folds``      -> StratifiedGroupKFold so no duplicate group straddles
                            train/validation (the legacy random split leaked them).
4. ``build_roi_cache``   -> leaf ROI crop + downscale once, cached to disk.
5. ``build_loaders``     -> train/val DataLoaders at 384/512 px with lesion-safe
                            augmentation (no CutMix) and optional MixUp.
"""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter, ImageOps
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from .config import CLASS_NAMES, Config, DataConfig

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- scan
def scan_dataset(root: str | Path, class_names: Sequence[str] = CLASS_NAMES) -> pd.DataFrame:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    rows = []
    for idx, cls in enumerate(class_names):
        cdir = root / cls
        if not cdir.is_dir():
            raise FileNotFoundError(f"Missing class folder: {cdir}")
        for p in sorted(cdir.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                rows.append({"path": str(p), "label": cls, "label_idx": idx})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- hashing
def file_sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def dhash(path: str, size: int = 8) -> int:
    """64-bit difference hash (robust to resizing / recompression)."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("L").resize((size + 1, size), Image.BILINEAR)
        a = np.asarray(im, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int(sum(1 << i for i, b in enumerate(bits) if b))


def _popcount64(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.uint64)
    out = np.zeros(x.shape, dtype=np.int64)
    for shift in range(0, 64, 8):
        out += _POP8[((x >> np.uint64(shift)) & np.uint64(0xFF)).astype(np.int64)]
    return out


_POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.int64)


def near_duplicate_pairs(hashes: Sequence[int], threshold: int) -> List[Tuple[int, int, int]]:
    """All index pairs (i<j) whose dHash Hamming distance <= threshold."""
    h = np.array(hashes, dtype=np.uint64)
    pairs = []
    for i in range(len(h) - 1):
        d = _popcount64(np.bitwise_xor(h[i + 1:], h[i]))
        for j in np.nonzero(d <= threshold)[0]:
            pairs.append((i, i + 1 + int(j), int(d[j])))
    return pairs


class _UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


# --------------------------------------------------------------------------- audit
def audit_dataset(df: pd.DataFrame, cfg: DataConfig, workers: int = 8) -> Tuple[pd.DataFrame, Dict]:
    """Return (clean_df, report). ``clean_df`` has a ``group`` column for GroupKFold."""
    df = df.copy().reset_index(drop=True)
    with ThreadPoolExecutor(workers) as ex:
        df["sha256"] = list(ex.map(file_sha256, df["path"]))
        df["dhash"] = list(ex.map(dhash, df["path"]))
    df["keep"] = True
    df["reason"] = ""

    report: Dict = {"n_raw": int(len(df)), "class_counts_raw": df["label"].value_counts().to_dict()}

    # 1) exact byte duplicates
    exact = df.groupby("sha256")
    conflict_groups, redundant = [], 0
    for sha, g in exact:
        if len(g) < 2:
            continue
        if g["label"].nunique() > 1:
            conflict_groups.append({"sha256": sha, "files": [Path(p).name for p in g["path"]],
                                    "labels": sorted(g["label"].unique())})
            if cfg.drop_conflicting_duplicates:
                df.loc[g.index, ["keep", "reason"]] = [False, "exact-dup label conflict"]
        elif cfg.dedupe_exact:
            drop_idx = g.index[1:]
            df.loc[drop_idx, ["keep", "reason"]] = [False, "exact-dup redundant copy"]
            redundant += len(drop_idx)
    report["exact_dup_groups"] = int(sum(len(g) > 1 for _, g in exact))
    report["exact_conflict_groups"] = conflict_groups
    report["exact_redundant_removed"] = int(redundant)

    # 2) near duplicates among survivors -> shared group id
    kept = df.index[df["keep"]].tolist()
    pairs = near_duplicate_pairs(df.loc[kept, "dhash"].tolist(), cfg.near_dup_threshold)
    uf = _UnionFind(len(df))
    cross = []
    for a, b, dist in pairs:
        ia, ib = kept[a], kept[b]
        if df.at[ia, "label"] != df.at[ib, "label"]:
            cross.append({"a": Path(df.at[ia, "path"]).name, "b": Path(df.at[ib, "path"]).name,
                          "label_a": df.at[ia, "label"], "label_b": df.at[ib, "label"], "dist": dist})
            if cfg.drop_cross_label_near_dups:
                df.loc[[ia, ib], ["keep", "reason"]] = [False, "near-dup label conflict"]
                continue
        uf.union(ia, ib)
    df["group"] = [uf.find(i) for i in range(len(df))]
    report["near_dup_pairs"] = len(pairs)
    report["near_dup_cross_label"] = cross

    clean = df[df["keep"]].reset_index(drop=True)
    report["n_clean"] = int(len(clean))
    report["n_groups"] = int(clean["group"].nunique())
    report["class_counts_clean"] = clean["label"].value_counts().to_dict()
    report["removed"] = df.loc[~df["keep"], ["path", "label", "reason"]].to_dict("records")
    return clean, report


def assign_folds(df: pd.DataFrame, n_folds: int = 5, seed: int = 42) -> pd.DataFrame:
    """Group-aware stratified K-fold: duplicates/near-duplicates never cross folds."""
    df = df.copy()
    df["fold"] = -1
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, val_idx) in enumerate(sgkf.split(df, df["label_idx"], groups=df["group"])):
        df.loc[df.index[val_idx], "fold"] = k
    assert (df["fold"] >= 0).all()
    leak = df.groupby("group")["fold"].nunique().max()
    assert leak == 1, "duplicate group split across folds"
    return df


def fold_class_table(df: pd.DataFrame, class_names: Sequence[str] = CLASS_NAMES) -> pd.DataFrame:
    tab = pd.crosstab(df["fold"], df["label"]).reindex(columns=list(class_names), fill_value=0)
    tab["total"] = tab.sum(axis=1)
    return tab


# --------------------------------------------------------------------------- ROI crop
def leaf_mask(img: Image.Image) -> Image.Image:
    """Binary (0/255) foliage + lesion mask: excess-green OR green hue OR yellow/brown necrosis."""
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    hsv = np.asarray(img.convert("HSV"), dtype=np.float32)
    h, s, v = hsv[..., 0] * (360.0 / 255.0), hsv[..., 1] / 255.0, hsv[..., 2] / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    foliage = ((2 * g - r - b) > 0.05) & (v > 0.12)
    green = (h >= 35) & (h <= 170) & (s > 0.18) & (v > 0.12)
    lesion = (h >= 5) & (h < 35) & (s > 0.25) & (v > 0.12) & (v < 0.9)
    return Image.fromarray(((foliage | green | lesion) * 255).astype(np.uint8)).filter(ImageFilter.MedianFilter(5))


def leaf_roi_box(img: Image.Image, margin: float = 0.08, min_area: float = 0.05) -> Tuple[int, int, int, int]:
    """Bounding box of foliage + lesion pixels from an HSV/excess-green mask.

    Works on a 256-px thumbnail, then rescales. Falls back to the full image if
    the mask is implausibly small (e.g. dark or overexposed photos).
    """
    W, H = img.size
    scale = 256.0 / max(W, H)
    thumb = img.resize((max(1, int(W * scale)), max(1, int(H * scale))), Image.BILINEAR)
    m = np.asarray(leaf_mask(thumb)) > 127
    if m.mean() < min_area:
        return 0, 0, W, H
    ys, xs = np.nonzero(m)
    x0, x1 = np.percentile(xs, [2, 98])
    y0, y1 = np.percentile(ys, [2, 98])
    bw, bh = x1 - x0, y1 - y0
    x0, x1 = x0 - margin * bw, x1 + margin * bw
    y0, y1 = y0 - margin * bh, y1 + margin * bh
    # make the box square (avoid aspect distortion when resizing to S x S)
    side = max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x0, x1 = cx - side / 2, cx + side / 2
    y0, y1 = cy - side / 2, cy + side / 2
    tw, th = thumb.size
    x0, y0 = max(0.0, x0), max(0.0, y0)
    x1, y1 = min(float(tw), x1), min(float(th), y1)
    box = tuple(int(round(c / scale)) for c in (x0, y0, x1, y1))
    if (box[2] - box[0]) * (box[3] - box[1]) < 0.15 * W * H:   # too aggressive -> keep context
        return 0, 0, W, H
    return box  # type: ignore[return-value]


def load_and_crop(path: str, roi: bool, margin: float, max_side: int) -> Image.Image:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        if roi:
            im = im.crop(leaf_roi_box(im, margin))
        im.thumbnail((max_side, max_side), Image.LANCZOS)
        return im.copy()


def build_roi_cache(df: pd.DataFrame, cfg: DataConfig, workers: int = 8) -> pd.DataFrame:
    """Write ROI-cropped, downscaled JPEGs once; return df with ``cache_path``."""
    tag = f"{'roi' if cfg.roi_crop else 'full'}_{cfg.cache_max_side}"
    out_dir = Path(cfg.cache_dir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    def _one(row) -> str:
        dst = out_dir / f"{row.sha256[:20]}_{row.label}.jpg"
        if not dst.exists():
            load_and_crop(row.path, cfg.roi_crop, cfg.roi_margin, cfg.cache_max_side).save(dst, quality=95)
        return str(dst)

    df = df.copy()
    with ThreadPoolExecutor(workers) as ex:
        df["cache_path"] = list(ex.map(_one, df.itertuples()))
    return df


# --------------------------------------------------------------------------- transforms
def build_transforms(cfg: DataConfig, train: bool, size: Optional[int] = None) -> T.Compose:
    size = size or cfg.image_size
    norm = [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if not train:
        return T.Compose([T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC), *norm])
    return T.Compose([
        T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(cfg.hflip),
        T.RandomVerticalFlip(cfg.vflip),
        T.RandomRotation(cfg.rotation, interpolation=T.InterpolationMode.BILINEAR),
        T.RandomAffine(degrees=0, translate=(cfg.affine_translate, cfg.affine_translate),
                       scale=tuple(cfg.affine_scale), shear=cfg.affine_shear,
                       interpolation=T.InterpolationMode.BILINEAR),
        T.ColorJitter(brightness=cfg.brightness, contrast=cfg.contrast),
        *norm,
    ])


class PotatoDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform: T.Compose, path_col: str = "cache_path"):
        self.paths = df[path_col if path_col in df else "path"].tolist()
        self.labels = df["label_idx"].astype(int).tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        with Image.open(self.paths[i]) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            x = self.transform(im)
        return x, self.labels[i]


def mixup(x: torch.Tensor, y: torch.Tensor, num_classes: int, alpha: float,
          smoothing: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mild MixUp (alpha≈0.2 keeps most lesion structure). Returns soft targets."""
    off, on = smoothing / num_classes, 1.0 - smoothing + smoothing / num_classes
    y1 = torch.full((y.size(0), num_classes), off, device=y.device).scatter_(1, y[:, None], on)
    if alpha <= 0:
        return x, y1
    lam = float(np.random.beta(alpha, alpha))
    lam = max(lam, 1 - lam)  # dominant image keeps >= 50 %
    perm = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[perm], lam * y1 + (1 - lam) * y1[perm]


def build_loaders(df: pd.DataFrame, fold: int, cfg: Config) -> Tuple[DataLoader, DataLoader]:
    d = cfg.data
    tr, va = df[df["fold"] != fold], df[df["fold"] == fold]
    train_ds = PotatoDataset(tr, build_transforms(d, True))
    val_ds = PotatoDataset(va, build_transforms(d, False))
    import os
    workers = min(d.num_workers, max(0, (os.cpu_count() or 1) - 1))   # never more workers than CPU cores
    kw = dict(num_workers=workers, pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0)
    if workers > 0:
        kw["prefetch_factor"] = 4
    return (DataLoader(train_ds, cfg.train.batch_size, shuffle=True, drop_last=True, **kw),
            DataLoader(val_ds, cfg.train.batch_size * 2, shuffle=False, **kw))


def prepare_data(cfg: Config, out_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, Dict]:
    """Scan -> audit -> folds -> ROI cache. Writes audit artefacts to ``out_dir``."""
    raw = scan_dataset(cfg.data.root, cfg.class_names)
    clean, report = audit_dataset(raw, cfg.data)
    clean = assign_folds(clean, cfg.data.n_folds, cfg.data.fold_seed)
    clean = build_roi_cache(clean, cfg.data)
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        clean.drop(columns=["dhash"]).to_csv(out_dir / "clean_split.csv", index=False)
        fold_class_table(clean, cfg.class_names).to_csv(out_dir / "fold_class_counts.csv")
        (out_dir / "data_audit.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return clean, report


def class_counts(df: pd.DataFrame, num_classes: int) -> np.ndarray:
    return np.bincount(df["label_idx"].astype(int), minlength=num_classes)
