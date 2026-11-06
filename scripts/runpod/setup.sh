#!/usr/bin/env bash
# One-time setup of a RunPod pod (template: "RunPod PyTorch 2.x"). Safe to run again.
#
#   cd /workspace && git clone https://github.com/beyondmme/asmoe-potato-v2.git && bash asmoe-potato-v2/scripts/runpod/setup.sh
#
# Needs (set them as pod environment variables, ideally from RunPod Secrets):
#   GITHUB_TOKEN                    fine-grained token, Contents: Read and write on the repo  (pushing results)
#   KAGGLE_USERNAME + KAGGLE_KEY    from kaggle.com -> Settings -> API -> Create New Token  (dataset download)
# Without Kaggle credentials: upload the dataset zip into /workspace/data with Jupyter Lab and re-run this script.
set -euo pipefail

WS=${WS:-/workspace}
REPO=${REPO:-$WS/asmoe-potato-v2}
DATA_DIR=$WS/data
DS_NAME="Potato Leaf Disease Dataset in Uncontrolled Environment"

echo "== 1/5 code"
if [ ! -d "$REPO/.git" ]; then
  git clone https://github.com/beyondmme/asmoe-potato-v2.git "$REPO"
fi
cd "$REPO"
git pull --rebase --autostash origin main

echo "== 2/5 python packages (PyTorch comes with the RunPod template)"
python - <<'PY'
import torch
print("torch", torch.__version__, "| CUDA", torch.version.cuda, "| GPU:",
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
assert torch.cuda.is_available(), "No GPU visible - is this a GPU pod?"
PY
pip install -q --no-cache-dir "timm>=1.0.9" "pandas>=2.0" "scikit-learn>=1.3" "pyyaml>=6.0" \
  "matplotlib>=3.7" "nbformat>=5.9" "kaggle>=1.6" "pillow>=10.0"
command -v tmux >/dev/null || (apt-get update -qq && apt-get install -y -qq tmux unzip >/dev/null) || true

echo "== 3/5 dataset"
mkdir -p "$DATA_DIR"
find_ds() { find "$DATA_DIR" -type d -name "$DS_NAME" -exec test -d "{}/Nematode" \; -print 2>/dev/null | head -1; }
DS=$(find_ds)
if [ -z "$DS" ]; then
  for z in "$DATA_DIR"/*.zip; do [ -f "$z" ] && unzip -q -o "$z" -d "$DATA_DIR"; done
  DS=$(find_ds)
fi
if [ -z "$DS" ] && [ -n "${KAGGLE_USERNAME:-}" ] && [ -n "${KAGGLE_KEY:-}" ]; then
  kaggle datasets download -d warcoder/potato-leaf-disease-dataset -p "$DATA_DIR" --unzip
  DS=$(find_ds)
fi
if [ -z "$DS" ]; then
  echo "!! Dataset not found. Either set KAGGLE_USERNAME/KAGGLE_KEY, or upload the Kaggle zip into $DATA_DIR"
  echo "   (Jupyter Lab -> left file browser -> /workspace/data -> Upload), then run this script again."
  exit 1
fi
echo "dataset: $DS ($(find "$DS" -type f | wc -l) files)"

echo "== 4/5 git identity + GitHub push access"
git config --global user.name  >/dev/null || git config --global user.name  "beyondmme"
git config --global user.email >/dev/null || git config --global user.email "vutran.03092006@gmail.com"
if [ -z "${GITHUB_TOKEN:-}" ]; then
  echo "!! GITHUB_TOKEN is not set: training will run but results will NOT be pushed to GitHub."
fi

echo "== 5/5 environment check + real-size training step"
python scripts/check_env.py --data "$DS" --config configs/asmoe_v2_rtx4090.yaml --probe
echo
echo "Setup done. Start training with:   bash scripts/runpod/train.sh"
