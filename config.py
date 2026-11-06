"""Experiment configuration (dataclasses + YAML loading with dotted CLI overrides)."""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Folder names exactly as they appear in the Kaggle dataset (note "Phytopthora").
CLASS_NAMES: List[str] = ["Bacteria", "Fungi", "Healthy", "Nematode", "Pest", "Phytopthora", "Virus"]
# Spelling used in tables / the paper.
DISPLAY_NAMES: Dict[str, str] = {"Phytopthora": "Phytophthora"}


@dataclass
class DataConfig:
    root: str = "/kaggle/input/datasets/warcoder/potato-leaf-disease-dataset/Potato Leaf Disease Dataset in Uncontrolled Environment"
    cache_dir: str = "outputs/cache"           # ROI-cropped, downscaled copies live here
    cache_max_side: int = 640                  # stored resolution of cached crops
    image_size: int = 384                      # model input resolution (384 or 512)
    roi_crop: bool = True                      # vegetation-mask leaf cropping
    roi_margin: float = 0.08
    drop_conflicting_duplicates: bool = True   # byte-identical files with different labels
    dedupe_exact: bool = True                  # keep one copy of same-label byte duplicates
    near_dup_threshold: int = 2                # dHash Hamming distance for grouping (<=)
    drop_cross_label_near_dups: bool = True
    n_folds: int = 5
    fold_seed: int = 42
    num_workers: int = 4
    # Augmentation (CutMix intentionally removed — it destroys lesion boundaries)
    hflip: float = 0.5
    vflip: float = 0.5
    rotation: float = 20.0
    affine_translate: float = 0.05
    affine_scale: List[float] = field(default_factory=lambda: [0.9, 1.1])
    affine_shear: float = 5.0
    brightness: float = 0.15
    contrast: float = 0.15
    mixup_alpha: float = 0.2
    mixup_prob: float = 0.5


@dataclass
class ExpertConfig:
    name: str                       # timm model name
    role: str = "expert"            # free-text: local-texture / multi-scale / global-attention
    image_size: Optional[int] = None  # per-expert resolution (None -> data.image_size)
    pretrained: bool = True
    grad_checkpointing: bool = False
    kwargs: Dict[str, Any] = field(default_factory=dict)  # extra timm.create_model kwargs


@dataclass
class ModelConfig:
    arch: str = "asmoe"             # "asmoe" or "single"
    experts: List[ExpertConfig] = field(default_factory=lambda: [
        ExpertConfig("densenet201.tv_in1k", "local-texture"),
        ExpertConfig("tf_efficientnetv2_m.in21k_ft_in1k", "multi-scale"),
        ExpertConfig("swin_base_patch4_window12_384.ms_in22k_ft_in1k", "global-attention", image_size=384),
    ])
    proj_dim: int = 512
    router_hidden: int = 256
    router_mode: str = "adaptive"   # "adaptive" (input-conditioned) | "uniform" (ablation)
    router_temperature: float = 1.0
    router_dropout: float = 0.1
    head_dropout: float = 0.2
    supcon_dim: int = 128


@dataclass
class TrainConfig:
    output_dir: str = "outputs/asmoe_v2"
    seed: int = 42
    batch_size: int = 8
    grad_accum: int = 4
    amp: bool = True
    channels_last: bool = True     # the GPU probe falls back to NCHW if cuDNN rejects it
    gpu_probe: bool = True         # test one real-size step and pick working cuDNN settings
    # Stage 1: frozen backbones, train projection + router + head
    stage1_epochs: int = 5
    # Stage 2: full refinement with discriminative learning rates
    stage2_epochs: int = 40
    lr_backbone: float = 1e-5
    lr_head: float = 3e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 2
    patience: int = 8
    label_smoothing: float = 0.05
    loss: str = "ce"                # "ce" | "focal"
    focal_gamma: float = 2.0
    class_weight: str = "sqrt"      # "none" | "sqrt" | "inverse"
    supcon_weight: float = 0.1      # 0 disables Supervised Contrastive loss
    supcon_temperature: float = 0.07
    balance_weight: float = 0.01    # router load-balancing regulariser
    ema_decay: float = 0.0          # 0 disables EMA
    tta: bool = True                # horizontal-flip TTA at evaluation
    clip_grad: float = 1.0
    save_resume: bool = True
    time_budget_hours: float = 0.0  # >0: pause cleanly (resume.pt kept) before this wall-clock budget


@dataclass
class OverleafConfig:
    git_url: str = "https://git.overleaf.com/6ae6eb411f26b6a0c2f760f8"
    local_dir: str = "outputs/overleaf_repo"
    paper_dir: str = "paper"        # local source of sections/, tables/, figures/
    branch: str = "master"
    auto_push: bool = False


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    overleaf: OverleafConfig = field(default_factory=OverleafConfig)
    experiment_name: str = "asmoe_v2"
    class_names: List[str] = field(default_factory=lambda: list(CLASS_NAMES))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _build(cls, data: Dict[str, Any]):
    kwargs = {}
    known = {f.name for f in fields(cls)}
    defaults = cls()
    for key, value in (data or {}).items():
        if key not in known:
            raise KeyError(f"Unknown config key '{key}' for {cls.__name__}")
        default = getattr(defaults, key)
        if key == "experts":
            value = [ExpertConfig(**e) if isinstance(e, dict) else e for e in value]
        elif is_dataclass(default) and isinstance(value, dict):
            value = _build(type(default), value)
        kwargs[key] = value
    return cls(**kwargs)


def _set_dotted(d: Dict[str, Any], dotted: str, raw: str) -> None:
    keys = dotted.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = yaml.safe_load(raw)


def load_config(path: Optional[str] = None, overrides: Optional[List[str]] = None) -> Config:
    """Load YAML config and apply ``section.key=value`` overrides."""
    raw: Dict[str, Any] = {}
    if path:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    raw = copy.deepcopy(raw)
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"Override must look like key=value, got '{ov}'")
        k, v = ov.split("=", 1)
        _set_dotted(raw, k.strip(), v.strip())
    return _build(Config, raw)


def save_config(cfg: Config, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8")
