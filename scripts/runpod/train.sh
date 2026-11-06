#!/usr/bin/env bash
# Train on a RunPod pod inside tmux, push every finished fold to GitHub, then STOP the pod (no idle billing).
#
#   bash scripts/runpod/train.sh                         # ASMOE V2, folds 0-4 (resumes / skips finished folds)
#   bash scripts/runpod/train.sh --folds 3 4             # only some folds
#   MODE=suite bash scripts/runpod/train.sh --only singles ablations
#
# Environment knobs:
#   MAX_HOURS=12   clean pause after this many hours (resume.pt kept), then the pod stops
#   AUTO_STOP=0    keep the pod running when training ends (default 1 = stop it)
#   CONFIG=configs/asmoe_v2_rtx4090.yaml
set -uo pipefail

WS=${WS:-/workspace}
REPO=${REPO:-$WS/asmoe-potato-v2}
MAX_HOURS=${MAX_HOURS:-12}
AUTO_STOP=${AUTO_STOP:-1}
MODE=${MODE:-train}
CONFIG=${CONFIG:-configs/asmoe_v2_rtx4090.yaml}
DS_NAME="Potato Leaf Disease Dataset in Uncontrolled Environment"

# Re-launch inside tmux so closing the browser / losing the connection does not kill training.
if [ -z "${TMUX:-}" ] && command -v tmux >/dev/null && [ "${NO_TMUX:-0}" != "1" ]; then
  tmux new-session -d -s train \
    "MODE=$MODE MAX_HOURS=$MAX_HOURS AUTO_STOP=$AUTO_STOP CONFIG=$CONFIG bash $REPO/scripts/runpod/train.sh $*; exec bash" 2>/dev/null \
    || { echo "A tmux session 'train' already exists: attach with  tmux attach -t train"; exit 1; }
  echo "Training started in tmux session 'train'."
  echo "  watch it:   tmux attach -t train      (leave it running: Ctrl+B then D)"
  echo "  log file:   ls -t $WS/logs/ | head -1"
  exit 0
fi

cd "$REPO"
git pull --rebase --autostash origin main || echo "git pull failed - continuing with local code"
DS=$(find "$WS/data" -type d -name "$DS_NAME" -exec test -d "{}/Nematode" \; -print 2>/dev/null | head -1)
if [ -z "$DS" ]; then echo "Dataset not found - run scripts/runpod/setup.sh first"; exit 1; fi
mkdir -p "$WS/logs"
LOG="$WS/logs/${MODE}_$(date +%Y%m%d_%H%M%S).log"
stop_pod() {
  if [ "$AUTO_STOP" = "1" ]; then
    if [ -n "${RUNPOD_POD_ID:-}" ] && command -v runpodctl >/dev/null; then
      echo "Stopping this pod in 120 s to save money (Ctrl+C now to keep it running)..." | tee -a "$LOG"
      sleep 120
      runpodctl stop pod "$RUNPOD_POD_ID"
    else
      echo "AUTO_STOP: runpodctl or RUNPOD_POD_ID not available - STOP THE POD MANUALLY in the RunPod console." | tee -a "$LOG"
    fi
  fi
}
[ -z "${GITHUB_TOKEN:-}" ] && echo "WARNING: GITHUB_TOKEN not set - results stay on the pod (outputs/, results/)" | tee -a "$LOG"
# Stop before paying for GPU hours if the results could not be pushed (SKIP_PUSH_CHECK=1 to train anyway).
if [ "${SKIP_PUSH_CHECK:-0}" != "1" ]; then
  python - <<'PY' 2>&1 | tee -a "$LOG"
import sys
from src.utils.github_sync import check_push_access
r = check_push_access(".", "main")
print("GitHub push check:", "OK" if r["ok"] else f"FAILED -> {r['hint']}\n  git said: {r['error']}")
sys.exit(0 if r["ok"] else 1)
PY
  if [ "${PIPESTATUS[0]}" != "0" ]; then
    echo "Fix GITHUB_TOKEN (pod -> Edit -> environment variables), or run with SKIP_PUSH_CHECK=1." | tee -a "$LOG"
    stop_pod
    exit 1
  fi
fi

COMMON=(--set "data.root=$DS" "data.cache_dir=$WS/cache" "train.time_budget_hours=$MAX_HOURS")
if [ "$MODE" = "suite" ]; then
  CMD=(python -u train.py suite --config "$CONFIG" "${COMMON[@]}" --push "$@")
else
  CMD=(python -u train.py train --config "$CONFIG" "${COMMON[@]}" --push "$@")
fi
echo "[$(date)] ${CMD[*]}" | tee -a "$LOG"

# hard safety net: kill training 30 min after the clean-pause budget, whatever happens
timeout --signal=INT "$(( MAX_HOURS * 60 + 30 ))m" "${CMD[@]}" 2>&1 | grep --line-buffered -v -i "warning" | tee -a "$LOG"
STATUS=${PIPESTATUS[0]}
echo "[$(date)] training exited with status $STATUS" | tee -a "$LOG"
ls -d results/*/fold_* 2>/dev/null | tee -a "$LOG"

stop_pod
