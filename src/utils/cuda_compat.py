"""GPU compatibility probe: find cuDNN settings that work BEFORE training, each trial in a fresh process.

Why: on some PyTorch/cuDNN builds (e.g. Kaggle's torch 2.11 + CUDA 12.8 on a Tesla T4) cuDNN finds no engine for
EfficientNet's "same"-padded depthwise convolutions, raising
``RuntimeError: GET was unable to find an engine to execute this computation``.
If that error fires inside a gradient-checkpointed block, PyTorch's checkpoint bookkeeping is left half-open and
every later forward in the same process fails with a CheckpointError. So each trial runs in its own subprocess
(``python -m src.utils.cuda_compat --child spec.json``) and the parent only applies the settings that worked.

Fallback ladder (first that works wins):
  1. requested settings (channels_last as configured, cudnn.benchmark on)
  2. contiguous (NCHW) memory format
  3. cudnn.benchmark off (heuristic engine selection)
  4. cuDNN disabled entirely (slow, but always runs) -- printed loudly
CUDA out-of-memory is not retried: it means batch_size must be reduced.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]


def amp_dtype() -> torch.dtype:
    """bf16 only with NATIVE support (compute capability >= 8.0: RTX 30xx/40xx, A100, H100...).

    Recent PyTorch reports ``is_bf16_supported() == True`` on a T4 (sm_75) but emulates it slowly; there fp16 with
    loss scaling is several times faster."""
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def ladder(channels_last: bool) -> List[Dict]:
    opts, seen = [], set()
    for o in (dict(channels_last=channels_last, cudnn=True, benchmark=True),
              dict(channels_last=False, cudnn=True, benchmark=True),
              dict(channels_last=False, cudnn=True, benchmark=False),
              dict(channels_last=False, cudnn=False, benchmark=False)):
        key = tuple(o.values())
        if key not in seen:
            seen.add(key)
            opts.append(o)
    return opts


def apply_settings(opt: Dict) -> torch.memory_format:
    torch.backends.cudnn.enabled = opt["cudnn"]
    torch.backends.cudnn.benchmark = opt["benchmark"]
    return torch.channels_last if opt["channels_last"] else torch.contiguous_format


def find_settings(attempt: Callable[[Dict], Dict], channels_last: bool = True) -> Dict:
    """Walk the ladder; ``attempt(opt)`` returns {'ok': bool, 'oom': bool, 'error': str, ...}."""
    errors = []
    for opt in ladder(channels_last):
        r = attempt(opt)
        if r.get("ok"):
            res = dict(opt, **{k: v for k, v in r.items() if k not in ("ok", "oom", "error")})
            if not opt["cudnn"]:
                print("[gpu-probe] WARNING: running with cuDNN DISABLED (much slower). "
                      "Consider data.image_size=384.", flush=True)
            print(f"[gpu-probe] OK with {res}", flush=True)
            return res
        if r.get("oom"):
            raise RuntimeError(f"[gpu-probe] CUDA out of memory with {opt}. Reduce train.batch_size "
                               f"(e.g. --set train.batch_size=4 train.grad_accum=8). Detail: {r.get('error')}")
        errors.append(f"{opt}: {r.get('error')}")
        print(f"[gpu-probe] {opt} failed: {str(r.get('error'))[:160]} -> next fallback", flush=True)
    raise RuntimeError("[gpu-probe] no working GPU configuration:\n  " + "\n  ".join(errors))


# --------------------------------------------------------------------------- child process
def _child(spec_path: str) -> None:
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    out = Path(spec["out"])
    try:
        sys.path.insert(0, str(REPO_ROOT))
        import torch.nn.functional as F
        from src.config import load_config
        from src.models import build_model

        cfg = load_config(spec["config_yaml"])
        for e in cfg.model.experts:
            e.pretrained = False                      # kernels do not depend on the weights; skip downloads
        device = torch.device(spec["device"])
        fmt = apply_settings(spec["opt"])
        model = build_model(cfg.model, len(cfg.class_names), cfg.data.image_size, spec.get("expert")).to(device)
        model = model.to(memory_format=fmt).train()
        use_cuda = device.type == "cuda"
        dtype = amp_dtype()
        amp = cfg.train.amp and use_cuda
        bs, size = cfg.train.batch_size, cfg.data.image_size
        x = torch.randn(bs, 3, size, size, device=device).to(memory_format=fmt)
        y = torch.randint(0, len(cfg.class_names), (bs,), device=device)
        if use_cuda:
            torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(max(1, spec["steps"])):
            if use_cuda:
                torch.cuda.synchronize()
            t = time.time()
            with torch.autocast(device.type, dtype=dtype, enabled=amp):
                loss = F.cross_entropy(model(x)["logits"].float(), y)
            loss.backward()
            model.zero_grad(set_to_none=True)
            if use_cuda:
                torch.cuda.synchronize()
            times.append(time.time() - t)
        sec = sum(times[1:]) / len(times[1:]) if len(times) > 1 else times[0]
        res = {"ok": True, "sec_per_step": sec,
               "peak_gb": torch.cuda.max_memory_allocated() / 1e9 if use_cuda else 0.0,
               "dtype": str(dtype).replace("torch.", "")}
    except Exception as e:  # noqa: BLE001 - report every failure to the parent
        msg = f"{type(e).__name__}: {e}".splitlines()
        res = {"ok": False, "oom": "out of memory" in str(e).lower(), "error": " | ".join(msg[:3])[:500]}
    out.write_text(json.dumps(res), encoding="utf-8")


def subprocess_attempt(cfg, expert: Optional[str], device: str, steps: int, timeout: int = 900):
    """Return an ``attempt(opt)`` callable that runs one trial in a fresh Python process."""
    from ..config import save_config
    tmp = Path(tempfile.mkdtemp(prefix="gpu_probe_"))
    cfg_yaml = tmp / "config.yaml"
    save_config(cfg, cfg_yaml)

    def attempt(opt: Dict) -> Dict:
        out = tmp / f"result_{len(list(tmp.glob('result_*')))}.json"
        spec = tmp / "spec.json"
        spec.write_text(json.dumps({"config_yaml": str(cfg_yaml), "expert": expert, "device": device,
                                    "opt": opt, "steps": steps, "out": str(out)}), encoding="utf-8")
        env = dict(os.environ, PYTHONWARNINGS="ignore")
        try:
            p = subprocess.run([sys.executable, "-m", "src.utils.cuda_compat", "--child", str(spec)],
                               cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"timeout after {timeout}s"}
        if out.exists():
            return json.loads(out.read_text(encoding="utf-8"))
        tail = (p.stderr or p.stdout).strip().splitlines()[-3:]
        return {"ok": False, "oom": "out of memory" in " ".join(tail).lower(),
                "error": f"child exited {p.returncode}: {' | '.join(tail)}"[:500]}
    return attempt


def probe_config(cfg, expert: Optional[str] = None, steps: int = 1, device: Optional[str] = None) -> Dict:
    """Find working settings for ``cfg`` in isolated processes and apply them (cuDNN flags) in this process.

    Call it BEFORE moving the training model to the GPU (the trials need the memory)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        res = find_settings(subprocess_attempt(cfg, expert, device, steps), cfg.train.channels_last)
    except RuntimeError as e:
        # Fast configs (e.g. RTX 4090) run without gradient checkpointing; if that does not fit, turn it on.
        if "out of memory" not in str(e).lower() or all(x.grad_checkpointing for x in cfg.model.experts):
            raise
        print("[gpu-probe] out of memory without gradient checkpointing -> enabling it for all experts "
              "(about 25% slower, much less memory)", flush=True)
        for x in cfg.model.experts:
            x.grad_checkpointing = True          # cfg is mutated: the trainer builds its model from it afterwards
        res = find_settings(subprocess_attempt(cfg, expert, device, steps), cfg.train.channels_last)
        res["grad_checkpointing_enabled"] = True
    apply_settings(res)
    return res


def benchmark_report(cfg, expert: Optional[str] = None, n_train: int = 2460, steps: int = 6) -> Dict:
    """Time real-size training steps (with the fallback ladder) and print an hours-per-fold estimate."""
    if not torch.cuda.is_available():
        print("[benchmark] no GPU - skipped")
        return {}
    res = probe_config(cfg, expert, steps=steps)
    epoch_min = res["sec_per_step"] * n_train / cfg.train.batch_size * 1.3 / 60
    typical = 27 * epoch_min / 60
    worst = cfg.train.stage1_epochs + cfg.train.stage2_epochs
    print(f"[benchmark] {res['sec_per_step']:.2f} s/step (batch {cfg.train.batch_size}) | peak {res['peak_gb']:.1f} GB | "
          f"~{epoch_min:.1f} min/epoch | typical fold (~27 epochs) ~{typical:.1f} h | "
          f"worst case ({worst} epochs) ~{worst * epoch_min / 60:.1f} h | budget {cfg.train.time_budget_hours} h")
    if cfg.train.time_budget_hours and typical > cfg.train.time_budget_hours:
        print("[benchmark] NOTE: this fold will likely need 2 sessions (automatic pause/resume), "
              "or add EXTRA_OVERRIDES=['data.image_size=384'] to speed it up.")
    return dict(res, epoch_min=epoch_min, typical_fold_h=typical)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", required=True)
    _child(ap.parse_args().child)
