#!/usr/bin/env bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f "liarsbench_generation_pipeline/.env" ]]; then
  set -a
  source "liarsbench_generation_pipeline/.env"
  set +a
fi

: "${OPENROUTER_API_KEY:?OPENROUTER_API_KEY is not set. Put it in liarsbench_generation_pipeline/.env or export it in your shell.}"

python -m liarsbench_generation_pipeline.pipeline \
  --config liarsbench_generation_pipeline/config.targeted_deception.example.json
