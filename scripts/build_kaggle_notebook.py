"""Generate notebooks/kaggle_fold_runner.ipynb.

Flow per Kaggle session:  clone GitHub repo -> train ONE fold -> (fold 0) Grad-CAM -> copy light results to
results/ -> commit + push to GitHub -> GitHub Action rebuilds the manuscript -> Overleaf.
"""
from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("""# ASMOE V2 · Kaggle fold runner → GitHub → Overleaf

Each **Save Version → Save & Run All** trains **one fold** and pushes its results (metrics, logs, predictions,
Grad-CAM; never checkpoints) to `results/` in the GitHub repository. A GitHub Action then rebuilds the
Springer Nature manuscript and syncs it to Overleaf. There is no manual merge step.

**One-time notebook setup**
1. *Add Input → Datasets*: `warcoder/potato-leaf-disease-dataset`
2. *Settings*: Accelerator **GPU T4**, **Internet ON**
3. *Add-ons → Secrets*: `GITHUB_TOKEN`, a fine-grained GitHub token with *Contents: Read and write* on the repo

**Per session**: set `FOLD` (0–4) below and *Save & Run All*. If a fold prints `PAUSED` (time budget reached),
run a new version with the **same FOLD** and this version's **output attached as input**. It resumes from
`resume.pt`.
"""),
    code("""# ======================= parameters =======================
GITHUB_REPO = "beyondmme/asmoe-potato-v2"   # owner/name, or a full git URL
BRANCH = "main"
FOLD = 0                       # 0-4: one fold per session
EXPERIMENT = "asmoe_v2"        # see configs/paper_experiments.yaml for baselines / ablations
CONFIG = "configs/asmoe_v2.yaml"
EXPERT = None                  # e.g. "swin_base_patch4_window12_384.ms_in22k_ft_in1k" with EXPERIMENT="single_swinb"
EXTRA_OVERRIDES = []           # e.g. ["data.roi_crop=false"] with EXPERIMENT="abl_no_roi"
TIME_BUDGET_HOURS = 10.5       # pause cleanly before Kaggle's 12 h limit
BENCHMARK = True               # time a few training steps and print a per-fold estimate
PUSH_RESULTS = True            # commit results/ to GitHub at the end
# ==========================================================="""),
    code("""import os, sys, shutil, subprocess, glob, json, time
from pathlib import Path

KAGGLE = os.environ.get("KAGGLE_ROOT", "/kaggle")          # override only for local simulation
ALLOW_CPU = os.environ.get("ASMOE_ALLOW_CPU") == "1"

if not os.environ.get("GITHUB_TOKEN"):
    try:
        from kaggle_secrets import UserSecretsClient
        os.environ["GITHUB_TOKEN"] = UserSecretsClient().get_secret("GITHUB_TOKEN")
        print("GITHUB_TOKEN loaded from Kaggle secrets")
    except Exception:
        print("No GITHUB_TOKEN secret: results will be saved in this notebook's output only")

REPO_URL = GITHUB_REPO if ("://" in GITHUB_REPO or os.path.exists(GITHUB_REPO)) else f"https://github.com/{GITHUB_REPO}.git"
WORK = Path(f"{KAGGLE}/working/repo")
sys.path.insert(0, str(WORK))
if not (WORK / ".git").exists():
    import base64
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    if os.environ.get("GITHUB_TOKEN"):   # needed only for private repos
        b = base64.b64encode(f"x-access-token:{os.environ['GITHUB_TOKEN']}".encode()).decode()
        env.update(GIT_CONFIG_COUNT="1", GIT_CONFIG_KEY_0="http.https://github.com/.extraHeader",
                   GIT_CONFIG_VALUE_0=f"Authorization: Basic {b}")
    subprocess.run(["git", "clone", "--branch", BRANCH, REPO_URL, str(WORK)], check=True, env=env)
os.chdir(WORK)
from src.utils.github_sync import git
git(["pull", "--rebase", "--autostash", "origin", BRANCH], WORK, check=False)
print("repo:", REPO_URL, "@", git(["rev-parse", "--short", "HEAD"], WORK).stdout.strip())

if PUSH_RESULTS:   # fail now, not after hours of training, if results could not be pushed
    from src.utils.github_sync import check_push_access
    acc = check_push_access(WORK, BRANCH)
    if not acc["ok"]:
        raise RuntimeError(f"GitHub push check FAILED: {acc['hint']} | git said: {acc['error']} | "
                           "(or set PUSH_RESULTS = False to train without pushing)")
    print("GitHub push check: OK (results will be pushed when the fold finishes)")

if not ALLOW_CPU:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm>=1.0.9"], check=False)
import torch
print("torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
assert torch.cuda.is_available() or ALLOW_CPU, "Enable a GPU accelerator (T4) in the notebook settings"
"""),
    code("""cands = [p for p in glob.glob(f"{KAGGLE}/input/**/Potato Leaf Disease Dataset in Uncontrolled Environment", recursive=True)
         if os.path.isdir(os.path.join(p, "Nematode"))]
assert cands, "Attach warcoder/potato-leaf-disease-dataset (Add Input -> Datasets)"
DATA_ROOT = cands[0]
print("DATA_ROOT =", DATA_ROOT)
print({c: len(os.listdir(os.path.join(DATA_ROOT, c))) for c in sorted(os.listdir(DATA_ROOT))})"""),
    code("""# Resume support: restore this fold from a previous version's output attached as input
def progress(d: Path) -> int:
    if (d / "result.json").exists():
        return 10**6
    log = d / "epoch_metrics.csv"
    return sum(1 for _ in open(log)) if (log.exists() and (d / "resume.pt").exists()) else -1

src_best = None
for p in glob.glob(f"{KAGGLE}/input/**/outputs/{EXPERIMENT}/fold_{FOLD}", recursive=True):
    d = Path(p)
    if not str(d).startswith(str(WORK)) and progress(d) > (progress(src_best) if src_best else -1):
        src_best = d
dst = WORK / "outputs" / EXPERIMENT / f"fold_{FOLD}"
if src_best and progress(src_best) > progress(dst):
    shutil.copytree(src_best, dst, dirs_exist_ok=True)
    print(f"restored fold_{FOLD} from {src_best}")
print("fold state:", "complete" if (dst / "result.json").exists() else "resume" if (dst / "resume.pt").exists() else "new")"""),
    code("""from src.config import load_config

cache = f"{KAGGLE}/tmp_cache" if ALLOW_CPU else "/tmp/asmoe_cache"
OVERRIDES = [f"data.root={DATA_ROOT}", f"data.cache_dir={cache}", "data.num_workers=4",
             f"train.time_budget_hours={TIME_BUDGET_HOURS}", "train.output_dir=outputs"] + EXTRA_OVERRIDES
cfg = load_config(CONFIG, OVERRIDES)
cfg.experiment_name = EXPERIMENT
print("experts:", [(e.name, e.image_size) for e in cfg.model.experts], "| expert override:", EXPERT)"""),
    code("""# Pre-flight: one real-size training step picks cuDNN settings that work on this GPU, then times it.
# (Logic lives in src/utils/cuda_compat.py, so fixes arrive with `git pull` - no need to re-import the notebook.)
if BENCHMARK and torch.cuda.is_available():
    from src.utils.cuda_compat import benchmark_report
    benchmark_report(cfg, EXPERT)"""),
    code("""def run(args):
    print("$ python", " ".join(args), flush=True)
    proc = subprocess.Popen([sys.executable, "-u", *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        if "Warning" not in line:
            print(line, end="", flush=True)
    assert proc.wait() == 0, f"command failed with exit code {proc.returncode}"

common = ["--config", CONFIG, "--set", *OVERRIDES]
run(["train.py", "train", *common, "--name", EXPERIMENT, "--folds", str(FOLD)] + (["--expert", EXPERT] if EXPERT else []))
done = (dst / "result.json").exists()
if done and FOLD == 0 and EXPERT is None and EXPERIMENT == "asmoe_v2":
    run(["train.py", "gradcam", *common, "--name", EXPERIMENT, "--fold", "0", "--n", "8"])"""),
    code("""# Copy light results to results/ and push them to GitHub (the Action rebuilds the paper)
from src.utils.github_sync import push_results
if done:
    r = json.loads((dst / "result.json").read_text())
    print(f"fold {FOLD} COMPLETE | acc {r['accuracy']:.2f} | Macro-F1 {r['macro_f1']:.2f} | best epoch {r['best_epoch']}")
    run(["train.py", "export", *common, "--name", EXPERIMENT])
    if PUSH_RESULTS:
        res = push_results(WORK, ["results"], f"results: {EXPERIMENT} fold {FOLD} (Macro-F1 {r['macro_f1']:.2f})", BRANCH)
        print("GitHub push:", res)
        if not res.get("pushed") and res.get("reason") != "nothing new":
            print("PUSH FAILED ->", res.get("hint"))
            print("Results are safe in this version's output: fix the token, then "
                  "run a new version with the same FOLD and this output attached (it skips training and only pushes).")
else:
    print(f"fold {FOLD} PAUSED -> new version with the same FOLD and this version's output attached as input")
shutil.make_archive(f"{KAGGLE}/working/results_{EXPERIMENT}_fold{FOLD}", "zip", WORK / "results")
for d in sorted((WORK / "outputs").glob("*")):
    print(f"{d.name:20s} {sum(f.stat().st_size for f in d.rglob('*') if f.is_file()) / 1e6:9.1f} MB")"""),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}})
out = Path(__file__).resolve().parents[1] / "notebooks" / "kaggle_fold_runner.ipynb"
nbf.write(nb, out)
print("wrote", out)
