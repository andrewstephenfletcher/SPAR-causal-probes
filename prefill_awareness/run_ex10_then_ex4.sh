#!/usr/bin/env bash
# Run Experiment 10 followed by Experiment 4.
# HF models are saved to /root/.cache/huggingface (container disk, 512 GB).
# Run from the prefill_awareness/ directory:
#   bash run_ex10_then_ex4.sh [extra args passed to both scripts]
#
# To skip steps, use per-experiment args, e.g.:
#   bash run_ex10_then_ex4.sh   (full run of both)
# Or run the Python scripts directly with --from-step as needed.

set -euo pipefail

export HF_HOME="/root/.cache/huggingface"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo " HF_HOME: $HF_HOME"
echo " Working dir: $SCRIPT_DIR"
echo "========================================"

echo ""
echo "######## Experiment 10 ########"
python -m prefill_probe.run_experiment10 "$@"

echo ""
echo "######## Experiment 4 ########"
python -m prefill_probe.run_experiment4 "$@"

echo ""
echo "========================================"
echo " Both experiments complete."
echo "========================================"
