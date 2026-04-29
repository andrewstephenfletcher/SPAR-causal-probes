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

# ensure rsync is available on the remote
ssh "$SSH_HOST" "command -v rsync >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq rsync)"

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
  "prefill_awareness/outputs"
)

for dir in "${DATA_DIRS[@]}"; do
  src="$SSH_HOST:$REMOTE_REPO/$dir"
  dst="$LOCAL_REPO/$dir"

  if ! ssh "$SSH_HOST" "[ -d '$REMOTE_REPO/$dir' ]" 2>/dev/null; then
    echo "    skipping $dir (not found on remote)"
    continue
  fi

  echo "    syncing $dir"
  mkdir -p "$dst"
  rsync -avz --ignore-existing --no-owner --no-group \
    -e "ssh" \
    "$src/" "$dst/"
done

echo "==> Done."
