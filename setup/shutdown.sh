#!/usr/bin/env bash
# shutdown.sh — run locally to sync files from a RunPod pod back to your machine
# Usage: ./setup/shutdown.sh [SSH_HOST]
#   SSH_HOST defaults to "runpod" (from ~/.ssh/config)
#   e.g. ./setup/shutdown.sh runpod

set -euo pipefail

# ── config ────────────────────────────────────────────────────────────────────
SSH_HOST="${1:-runpod}"
LOCAL_REPO="$(cd "$(dirname "$0")/.." && pwd)"   # repo root on your machine
REMOTE_REPO="/workspace/SPAR-causal-probes"

echo "==> Syncing from $SSH_HOST → $LOCAL_REPO"

# ── rsync data (remote → local, missing files only) ───────────────────────────

DATA_DIRS=(
  "data"
  "af_experiments/dct/data"
  "af_experiments/dct/wandb"
  "af_experiments/dct/probes"
  "dct_probes/data"
  "dct_probes/experiments"
  "dct_probes/images"
  "geometry-of-truth"
)

for dir in "${DATA_DIRS[@]}"; do
  src="$SSH_HOST:$REMOTE_REPO/$dir"
  dst="$LOCAL_REPO/$dir"

  echo "    syncing $dir"
  mkdir -p "$dst"
  rsync -avz --ignore-existing \
    -e "ssh" \
    "$src/" "$dst/"
done

echo "==> Done."
