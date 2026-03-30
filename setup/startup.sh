#!/usr/bin/env bash
# startup.sh — run locally to bootstrap a fresh RunPod pod
# Usage: ./setup/startup.sh [SSH_HOST]
#   SSH_HOST defaults to "runpod" (from ~/.ssh/config)
#   e.g. ./setup/startup.sh runpod

set -euo pipefail

# ── config ────────────────────────────────────────────────────────────────────
SSH_HOST="${1:-runpod}"
LOCAL_REPO="$(cd "$(dirname "$0")/.." && pwd)"   # repo root on your machine
REMOTE_REPO="/workspace/SPAR-causal-probes"
REPO_URL="https://github.com/andrewstephenfletcher/SPAR-causal-probes.git"

SSH="ssh $SSH_HOST"

echo "==> Connecting to $SSH_HOST"

# ── 1. clone ──────────────────────────────────────────────────────────────────
echo "==> Cloning repo (skipped if already present)"
$SSH bash -c "
  if [ ! -d '$REMOTE_REPO/.git' ]; then
    git clone '$REPO_URL' '$REMOTE_REPO'
  else
    echo '    repo already exists, pulling latest'
    git -C '$REMOTE_REPO' pull --ff-only
  fi
"

# ── 2. uv sync ────────────────────────────────────────────────────────────────
echo "==> Running uv sync"
$SSH bash -c "
  # install uv if missing
  if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH=\"\$HOME/.local/bin:\$PATH\"
  fi
  export PATH=\"\$HOME/.local/bin:\$PATH\"
  cd '$REMOTE_REPO'
  uv sync
"

# ── 3. rsync data (local → remote, missing files only) ────────────────────────
echo "==> Syncing data directories (--ignore-existing)"

# directories excluded from git that need to be synced
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
  src="$LOCAL_REPO/$dir"
  dst="$SSH_HOST:$REMOTE_REPO/$dir"

  if [ ! -e "$src" ]; then
    echo "    skipping $dir (not found locally)"
    continue
  fi

  echo "    syncing $dir"
  rsync -avz --ignore-existing \
    -e "ssh" \
    "$src/" "$dst/"
done

echo "==> Done. Connect with: ssh $SSH_HOST"
