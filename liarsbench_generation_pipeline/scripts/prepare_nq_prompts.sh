#!/usr/bin/env bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python liarsbench_generation_pipeline/prepare_nq_prompts.py \
  --dataset sentence-transformers/natural-questions \
  --split train \
  --num-prompts 100 \
  --output liarsbench_generation_pipeline/natural_questions_prompts_100.jsonl
