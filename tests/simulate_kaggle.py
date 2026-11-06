"""Replay the Kaggle -> GitHub -> paper workflow locally (CPU, synthetic data, smoke config).

A bare git repository stands in for GitHub. Sessions run notebooks/kaggle_fold_runner.ipynb:
  A: fold 0, tiny time budget          -> PAUSED, nothing pushed
  B: fold 0, A's output attached        -> resumes, completes, Grad-CAM, pushes results/asmoe_v2/fold_0
  C: fold 1                             -> completes, pushes results/asmoe_v2/fold_1
Then, as the GitHub Action would: clone the "GitHub" repo, `train.py paper`, `scripts/check_tex.py`.

    python tests/simulate_kaggle.py        (needs the repo committed: it clones HEAD)
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests.test_smoke import make_synthetic  # noqa: E402

SIM = ROOT / "outputs" / "kaggle_sim"
NB = ROOT / "notebooks" / "kaggle_fold_runner.ipynb"
REMOTE = SIM / "github.git"


def rmtree(path: Path) -> None:
    """shutil.rmtree that also removes read-only files (git objects on Windows)."""
    import stat

    def _force(func, p, _exc):
        os.chmod(p, stat.S_IWRITE)
        func(p)

    if Path(path).exists():
        shutil.rmtree(path, onexc=_force) if sys.version_info >= (3, 12) else shutil.rmtree(path, onerror=_force)


def sh(*args, cwd=None):
    r = subprocess.run(list(args), cwd=cwd, capture_output=True, text=True)
    assert r.returncode == 0, f"{args}: {r.stderr}"
    return r.stdout


def session(tag: str, fold: int, budget: float, attach: list[str]) -> str:
    work = SIM / "working"
    rmtree(work)
    work.mkdir(parents=True)
    (SIM / "detached").mkdir(exist_ok=True)
    for d in list((SIM / "input").glob("out_*")):
        if d.name not in attach:
            shutil.move(str(d), str(SIM / "detached" / d.name))
    for name in attach:
        if (SIM / "detached" / name).exists():
            shutil.move(str(SIM / "detached" / name), str(SIM / "input" / name))
    nb = nbformat.read(NB, as_version=4)
    nb.cells[1].source = (f'GITHUB_REPO = r"{REMOTE}"\nBRANCH = "main"\nFOLD = {fold}\nEXPERIMENT = "asmoe_v2"\n'
                          f'CONFIG = "configs/smoke_test.yaml"\nEXPERT = None\nEXTRA_OVERRIDES = []\n'
                          f'TIME_BUDGET_HOURS = {budget}\nBENCHMARK = True\nPUSH_RESULTS = True')
    os.environ.update(KAGGLE_ROOT=str(SIM), ASMOE_ALLOW_CPU="1")
    os.environ.pop("GITHUB_TOKEN", None)
    NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(SIM)}}).execute()
    text = "".join(o.get("text", "") for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []))
    shutil.copytree(work, SIM / "input" / f"out_{tag}", ignore=shutil.ignore_patterns(".git"))
    print(f"\n===== session {tag} =====\n" + "\n".join(l for l in text.splitlines() if l.strip())[-1200:])
    return text


def main() -> None:
    rmtree(SIM)
    (SIM / "input").mkdir(parents=True)
    sh("git", "clone", "--bare", "-q", str(ROOT), str(REMOTE))
    make_synthetic(SIM / "input" / "potato-leaf-disease-dataset" / "Potato Leaf Disease Dataset in Uncontrolled Environment",
                   per_class=6)

    a = session("A", 0, 0.09, [])
    assert "PAUSED" in a, "session A should pause"
    b = session("B", 0, 10, ["out_A"])
    assert "restored fold_0" in b and "resuming stage" in b and "fold 0 COMPLETE" in b, "B should resume + finish"
    assert "'pushed': True" in b, "B should push results"
    c = session("C", 1, 10, [])
    assert "fold 1 COMPLETE" in c and "'pushed': True" in c

    # --- what the GitHub Action does on the pushed commits
    ci = SIM / "ci"
    sh("git", "clone", "-q", str(REMOTE), str(ci))
    log = sh("git", "log", "--oneline", "-5", cwd=ci)
    print("\n===== remote log =====\n" + log)
    assert "fold 0" in log and "fold 1" in log
    assert (ci / "results" / "asmoe_v2" / "gradcam").exists()
    out = sh(sys.executable, "train.py", "paper", cwd=ci)
    print(out)
    print(sh(sys.executable, "scripts/check_tex.py", cwd=ci))
    tex = (ci / "paper" / "build" / "sn-article.tex").read_text(encoding="utf-8")
    assert "\\renewcommand{\\VtwoFOne}{\\textemdash}" not in tex, "V2 macros should hold numbers now"
    assert (ci / "paper" / "build" / "figures" / "asmoe_v2_cm.png").exists()
    print("KAGGLE -> GITHUB -> PAPER SIMULATION OK")


if __name__ == "__main__":
    main()
