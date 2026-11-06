"""Two-stage refinement trainer with AMP, early stopping on Macro-F1 and resumable checkpoints.

Stage 1 (alignment)  : backbones frozen (BN stats fixed); FPM + router + head trained at ``lr_head``.
Stage 2 (refinement) : everything unfrozen; discriminative LRs (backbone ``lr_backbone``,
                       new layers ``lr_head``); warm-up + cosine; early stopping on val Macro-F1.

Each fold writes into ``<output_dir>/fold_<k>/``:
    best.pt, resume.pt, result.json, predictions.csv, epoch_metrics.csv (shared, appended)
A finished fold (``result.json`` present) is skipped on re-run, so Kaggle sessions can resume.
"""
from __future__ import annotations

import copy
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config, save_config
from .dataset import build_loaders, class_counts, mixup
from .losses import SupConLoss, build_criterion, class_weights
from .metrics import compute_metrics
from .results import summarise_run
from .models import ASMOE, build_model, count_parameters


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True     # Ampere/Ada (RTX 30xx/40xx): fast fp32 matmuls outside AMP
    torch.backends.cudnn.allow_tf32 = True


class EMA:
    """Exponential moving average of weights (evaluated instead of raw weights when enabled)."""

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1 - self.decay)
            else:
                v.copy_(msd[k])


def _warmup_cosine(total: int, warmup: int, floor: float = 0.01):
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        t = (step - warmup) / max(1, total - warmup)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))
    return f


class Trainer:
    def __init__(self, cfg: Config, df: pd.DataFrame, fold: int, run_dir: Path,
                 single_expert: Optional[str] = None, device: Optional[str] = None,
                 deadline: Optional[float] = None):
        self.cfg, self.df, self.fold = cfg, df, fold
        self.deadline = deadline          # unix time; stop before starting an epoch that would overrun
        self.t = cfg.train
        self.classes = cfg.class_names
        self.C = len(self.classes)
        self.run_dir = Path(run_dir)
        self.fold_dir = self.run_dir / f"fold_{fold}"
        self.fold_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        seed_everything(self.t.seed + fold)

        self.train_loader, self.val_loader = build_loaders(df, fold, cfg)
        # Find cuDNN settings that work on this GPU/driver, each trial in a fresh process, while the GPU is
        # still empty (src/utils/cuda_compat.py). Then build the real model with those settings.
        channels_last = self.t.channels_last
        if self.device.type == "cuda" and self.t.gpu_probe:
            from .utils.cuda_compat import probe_config
            gp = probe_config(cfg, single_expert, steps=1)
            channels_last = gp["channels_last"]
        self.mem_fmt = torch.channels_last if channels_last else torch.contiguous_format
        self.model: ASMOE = build_model(cfg.model, self.C, cfg.data.image_size, single_expert).to(self.device)
        self.model = self.model.to(memory_format=self.mem_fmt)
        counts = class_counts(df[df["fold"] != fold], self.C)
        w = class_weights(counts, self.t.class_weight)
        self.criterion = build_criterion(self.t.loss, w, self.t.focal_gamma).to(self.device)
        self.supcon = SupConLoss(self.t.supcon_temperature)

        use_cuda = self.device.type == "cuda"
        from .utils.cuda_compat import amp_dtype
        self.amp_dtype = amp_dtype()
        self.amp = self.t.amp and use_cuda
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp and self.amp_dtype == torch.float16)
        self.ema = EMA(self.model, self.t.ema_decay) if self.t.ema_decay > 0 else None
        self.best_f1, self.best_epoch, self.global_epoch = -1.0, -1, 0
        self.epoch_csv = self.fold_dir / "epoch_metrics.csv"
        self._epoch_secs = 0.0

    # ------------------------------------------------------------------ optimisation
    def _optimizer(self, stage: int):
        head = [p for p in self.model.head_parameters() if p.requires_grad]
        groups = [{"params": head, "lr": self.t.lr_head, "name": "head"}]
        if stage == 2:
            groups.insert(0, {"params": list(self.model.backbone_parameters()),
                              "lr": self.t.lr_backbone, "name": "backbone"})
        return torch.optim.AdamW(groups, weight_decay=self.t.weight_decay)

    def _train_epoch(self, opt, sched) -> float:
        self.model.train()
        d, t = self.cfg.data, self.t
        total, n = 0.0, 0
        opt.zero_grad(set_to_none=True)
        steps = len(self.train_loader)
        for i, (x, y) in enumerate(self.train_loader):
            x = x.to(self.device, non_blocking=True).to(memory_format=self.mem_fmt)
            y = y.to(self.device, non_blocking=True)
            use_mix = d.mixup_alpha > 0 and random.random() < d.mixup_prob
            x, target = mixup(x, y, self.C, d.mixup_alpha if use_mix else 0.0, t.label_smoothing)
            with torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.amp):
                out = self.model(x)
                loss = self.criterion(out["logits"], target)
                if t.supcon_weight > 0 and not use_mix:
                    loss = loss + t.supcon_weight * self.supcon(out["supcon"], y)
                if t.balance_weight > 0:
                    loss = loss + t.balance_weight * ASMOE.balance_loss(out["weights"].float())
            self.scaler.scale(loss / t.grad_accum).backward()
            if (i + 1) % t.grad_accum == 0 or i + 1 == steps:
                self.scaler.unscale_(opt)
                nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], t.clip_grad)
                self.scaler.step(opt)
                self.scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                if self.ema:
                    self.ema.update(self.model)
            total += loss.item() * x.size(0)
            n += x.size(0)
        return total / max(1, n)

    # ------------------------------------------------------------------ evaluation
    @torch.no_grad()
    def evaluate(self, model: Optional[nn.Module] = None) -> Dict:
        model = model or (self.ema.module if self.ema else self.model)
        model.eval()
        probs, ys, weights, losses = [], [], [], []
        for x, y in self.val_loader:
            x = x.to(self.device, non_blocking=True).to(memory_format=self.mem_fmt)
            y = y.to(self.device)
            with torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.amp):
                out = model(x)
                logits = out["logits"].float()
                if self.t.tta:
                    logits = 0.5 * (logits + model(torch.flip(x, dims=[3]))["logits"].float())
            losses.append(F.cross_entropy(logits, y, reduction="sum").item())
            probs.append(logits.softmax(-1).cpu())
            ys.append(y.cpu())
            weights.append(out["weights"].float().cpu())
        P, Y, W = torch.cat(probs), torch.cat(ys), torch.cat(weights)
        m = compute_metrics(Y.numpy(), P.argmax(1).numpy(), self.classes)
        m["val_loss"] = sum(losses) / len(Y)
        m["router_weights_mean"] = W.mean(0).tolist()
        m["router_weights_per_class"] = {c: W[Y == i].mean(0).tolist() for i, c in enumerate(self.classes)
                                         if (Y == i).any()}
        m["_probs"], m["_y"] = P, Y
        return m

    # ------------------------------------------------------------------ bookkeeping
    def _log_epoch(self, stage: int, train_loss: float, m: Dict, opt, secs: float) -> None:
        lrs = {g.get("name", "head"): g["lr"] for g in opt.param_groups}
        row = {"experiment": self.cfg.experiment_name, "fold": self.fold, "stage": stage,
               "epoch": self.global_epoch, "train_loss": train_loss, "val_loss": m["val_loss"],
               "accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "weighted_f1": m["weighted_f1"],
               "mcc": m["mcc"], "lr_backbone": lrs.get("backbone", 0.0), "lr_head": lrs.get("head", 0.0),
               "router_weights": json.dumps([round(v, 4) for v in m["router_weights_mean"]]),
               "per_class_f1": json.dumps({c: round(v["f1"], 4) for c, v in m["per_class"].items()}),
               "seconds": round(secs, 1)}
        new = not self.epoch_csv.exists()
        with open(self.epoch_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        print(f"[fold {self.fold}] S{stage} ep{self.global_epoch:02d} loss {train_loss:.4f} "
              f"val {m['val_loss']:.4f} acc {m['accuracy']:.2f} F1 {m['macro_f1']:.2f} "
              f"w={row['router_weights']} ({secs:.0f}s)", flush=True)

    def _state(self) -> Dict:
        src = self.ema.module if self.ema else self.model
        return {k: v.detach().cpu() for k, v in src.state_dict().items()}

    def _save_resume(self, stage: int, stage_epoch: int, opt, sched, bad: int) -> None:
        if not self.t.save_resume:
            return
        torch.save({"stage": stage, "stage_epoch": stage_epoch, "global_epoch": self.global_epoch,
                    "model": self.model.state_dict(), "ema": self.ema.module.state_dict() if self.ema else None,
                    "opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": self.scaler.state_dict(),
                    "best_f1": self.best_f1, "best_epoch": self.best_epoch, "bad": bad},
                   self.fold_dir / "resume.pt")

    # ------------------------------------------------------------------ main loop
    def fit(self) -> Dict:
        result_path = self.fold_dir / "result.json"
        if result_path.exists():
            print(f"[fold {self.fold}] already complete -> skipping")
            return json.loads(result_path.read_text(encoding="utf-8"))

        resume = None
        if (self.fold_dir / "resume.pt").exists():
            resume = torch.load(self.fold_dir / "resume.pt", map_location="cpu", weights_only=False)
            self.model.load_state_dict(resume["model"])
            if self.ema and resume["ema"]:
                self.ema.module.load_state_dict(resume["ema"])
            self.best_f1, self.best_epoch = resume["best_f1"], resume["best_epoch"]
            self.global_epoch = resume["global_epoch"]
            print(f"[fold {self.fold}] resuming stage {resume['stage']} epoch {resume['stage_epoch']}")

        print(f"[fold {self.fold}] params {count_parameters(self.model):.1f}M | "
              f"train {len(self.train_loader.dataset)} val {len(self.val_loader.dataset)} | device {self.device}")
        steps_per_epoch = math.ceil(len(self.train_loader) / self.t.grad_accum)
        stop = False
        for stage, epochs in ((1, self.t.stage1_epochs), (2, self.t.stage2_epochs)):
            if resume and stage < resume["stage"]:
                continue
            if epochs <= 0 or stop:
                continue
            self.model.set_backbone_trainable(stage == 2)
            opt = self._optimizer(stage)
            warm = self.t.warmup_epochs * steps_per_epoch if stage == 2 else 0
            sched = torch.optim.lr_scheduler.LambdaLR(opt, _warmup_cosine(epochs * steps_per_epoch, warm))
            start, bad = 0, 0
            if resume and stage == resume["stage"]:
                opt.load_state_dict(resume["opt"])
                sched.load_state_dict(resume["sched"])
                # A checkpoint written under bf16 (or on CPU) has an empty GradScaler state; when resuming under
                # fp16 the scaler simply starts fresh and re-calibrates within a few steps.
                if self.scaler.is_enabled() and resume.get("scaler"):
                    self.scaler.load_state_dict(resume["scaler"])
                elif self.scaler.is_enabled():
                    print(f"[fold {self.fold}] checkpoint has no fp16 loss-scaler state (saved under bf16) "
                          f"-> starting a fresh scaler", flush=True)
                start, bad = resume["stage_epoch"] + 1, resume["bad"]
                resume = None
            for ep in range(start, epochs):
                if self.deadline and time.time() + 1.15 * self._epoch_secs + 300 > self.deadline:
                    print(f"[fold {self.fold}] time budget reached -> paused after epoch {self.global_epoch} "
                          f"(resume.pt kept; re-run the same command to continue)", flush=True)
                    return {"status": "paused", "fold": self.fold, "epoch": self.global_epoch}
                t0 = time.time()
                self.global_epoch += 1
                tl = self._train_epoch(opt, sched)
                m = self.evaluate()
                self._epoch_secs = time.time() - t0
                self._log_epoch(stage, tl, m, opt, self._epoch_secs)
                if m["macro_f1"] > self.best_f1:
                    self.best_f1, self.best_epoch, bad = m["macro_f1"], self.global_epoch, 0
                    torch.save({"model": self._state(), "config": self.cfg.to_dict(), "fold": self.fold,
                                "epoch": self.global_epoch, "macro_f1": m["macro_f1"]}, self.fold_dir / "best.pt")
                elif stage == 2:
                    bad += 1
                self._save_resume(stage, ep, opt, sched, bad)
                if stage == 2 and bad >= self.t.patience:
                    print(f"[fold {self.fold}] early stop at epoch {self.global_epoch} "
                          f"(best {self.best_f1:.2f} @ {self.best_epoch})")
                    stop = True
                    break
        return self._finalise(result_path)

    def _finalise(self, result_path: Path) -> Dict:
        ckpt = torch.load(self.fold_dir / "best.pt", map_location="cpu", weights_only=False)
        model = self.ema.module if self.ema else self.model
        model.load_state_dict(ckpt["model"])
        m = self.evaluate(model)
        P, Y = m.pop("_probs"), m.pop("_y")
        val = self.df[self.df["fold"] == self.fold].reset_index(drop=True)
        pred = pd.DataFrame({"path": val["path"], "label": val["label"], "y_true": Y.numpy(),
                             "y_pred": P.argmax(1).numpy()})
        for i, c in enumerate(self.classes):
            pred[f"p_{c}"] = P[:, i].numpy()
        pred.to_csv(self.fold_dir / "predictions.csv", index=False)
        m.update({"fold": self.fold, "best_epoch": self.best_epoch,
                  "params_M": count_parameters(model), "experiment": self.cfg.experiment_name})
        result_path.write_text(json.dumps(m, indent=2), encoding="utf-8")
        if (self.fold_dir / "resume.pt").exists():
            (self.fold_dir / "resume.pt").unlink()   # finished: keep only best.pt
        return m


def run_cross_validation(cfg: Config, df: pd.DataFrame, folds: Optional[Sequence[int]] = None,
                         single_expert: Optional[str] = None) -> Dict:
    run_dir = Path(cfg.train.output_dir) / cfg.experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")
    folds = list(folds) if folds is not None else list(range(cfg.data.n_folds))
    budget = cfg.train.time_budget_hours
    deadline = time.time() + budget * 3600 if budget and budget > 0 else None
    results: List[Dict] = []
    for k in folds:
        trainer = Trainer(cfg, df, k, run_dir, single_expert, deadline=deadline)
        res = trainer.fit()
        results.append(res)
        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if res.get("status") == "paused":
            break
    return summarise_run(run_dir, cfg.class_names, cfg.experiment_name)
