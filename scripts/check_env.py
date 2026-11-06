"""Check that a local machine is ready to train and push results.

    python scripts/check_env.py --data "D:/data/Potato Leaf Disease Dataset in Uncontrolled Environment"
    python scripts/check_env.py --data "..." --probe     # + one real-size training step and a time estimate

Prints PASS/FAIL per item; exit code 1 if anything required fails.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CLASSES = ["Bacteria", "Fungi", "Healthy", "Nematode", "Pest", "Phytopthora", "Virus"]
failed = []


def check(name: str, ok: bool, detail: str = "", required: bool = True) -> None:
    tag = "PASS" if ok else ("FAIL" if required else "WARN")
    print(f"[{tag}] {name}" + (f" - {detail}" if detail else ""))
    if not ok and required:
        failed.append(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/Potato Leaf Disease Dataset in Uncontrolled Environment")
    ap.add_argument("--config", default="configs/asmoe_v2_rtx4090.yaml")
    ap.add_argument("--probe", action="store_true", help="run one real-size training step on the GPU")
    a = ap.parse_args()

    check("python >= 3.10", sys.version_info >= (3, 10), sys.version.split()[0])
    try:
        import torch
        import timm
        check("torch / timm import", True, f"torch {torch.__version__}, timm {timm.__version__}")
        cuda = torch.cuda.is_available()
        check("CUDA GPU visible", cuda, torch.cuda.get_device_name(0) if cuda else "torch.cuda.is_available() is False "
              "(CPU-only torch installed, or NVIDIA driver missing)")
        if cuda:
            props = torch.cuda.get_device_properties(0)
            check("GPU memory >= 16 GB", props.total_memory / 1e9 >= 16, f"{props.total_memory / 1e9:.1f} GB")
            check("native bf16 (compute capability >= 8.0)", props.major >= 8, f"sm_{props.major}{props.minor}; fp16 is used otherwise", required=False)
            check("cuDNN", torch.backends.cudnn.is_available(), f"version {torch.backends.cudnn.version()}")
    except ImportError as e:
        check("torch / timm import", False, str(e))

    root = Path(a.data)
    present = {c: len(list((root / c).glob("*"))) for c in CLASSES if (root / c).is_dir()}
    check("dataset folder", len(present) == 7, f"{root} -> {present or 'not found'}")
    if len(present) == 7:
        check("dataset image count", sum(present.values()) == 3076, f"{sum(present.values())} files (expected 3076)",
              required=False)

    check("git installed", shutil.which("git") is not None)
    r = subprocess.run(["git", "remote", "get-url", "origin"], cwd=ROOT, capture_output=True, text=True)
    check("git remote origin", r.returncode == 0, r.stdout.strip() or r.stderr.strip())
    from src.utils.github_sync import git_env   # uses GITHUB_TOKEN when set (RunPod / Linux)
    try:
        r = subprocess.run(["git", "push", "--dry-run", "origin", "HEAD:main"], cwd=ROOT, capture_output=True,
                           text=True, timeout=120, env=git_env())
        out = r.stderr + r.stdout
    except subprocess.TimeoutExpired:
        r, out = None, "timed out (waiting for a login prompt?)"
    behind = any(k in out for k in ("non-fast-forward", "fetch first", "fast-forwards"))
    ok = behind or (r is not None and r.returncode == 0)
    detail = ("ok (local branch is behind: run `git pull` before training)" if behind else "ok" if ok else
              (out.strip().splitlines() or ["?"])[-1] + " -> set GITHUB_TOKEN, or results will not be pushed")
    check("git push permission (dry run)", ok, detail, required=False)
    for key in ("user.name", "user.email"):
        r = subprocess.run(["git", "config", key], cwd=ROOT, capture_output=True, text=True)
        check(f"git {key}", bool(r.stdout.strip()), r.stdout.strip() or "set with: git config --global " + key + " ...",
              required=False)

    if a.probe and not failed:
        from src.config import load_config
        from src.utils.cuda_compat import benchmark_report
        cfg = load_config(a.config, [f"data.root={root}"])
        res = benchmark_report(cfg)
        check("real-size training step", bool(res), str({k: res[k] for k in ("channels_last", "cudnn") if k in res}))

    print("\nREADY" if not failed else f"\nNOT READY: {', '.join(failed)}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
