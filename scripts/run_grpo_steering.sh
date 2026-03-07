#!/usr/bin/env bash

# Configuration for GRPO steering-vector experiment.
MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
REWARD_MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
LAYER_IDX=18
MAX_STEPS=200
NUM_GENERATIONS=8
WANDB_RUN_NAME="grpo-steering-default-l${LAYER_IDX}"
TRAIN_PROMPT="Is Paris the capital of France?"
EVAL_PROMPT=(
  "Is Moscow the capital of Russia?"
  "Is Berlin the capital of Germany?"
  "Is Madrid the capital of Spain?"
  "Is Ottawa the capital of Canada?"
  "Is Sydney the capital of Australia?"
  "Is Istanbul the capital of Turkey?"
)

python rl/grpo_steering.py \
  --model-name "$MODEL_NAME" \
  --reward-model-name "$REWARD_MODEL_NAME" \
  --layer-idx "$LAYER_IDX" \
  --max-steps "$MAX_STEPS" \
  --num-generations "$NUM_GENERATIONS" \
  --wandb-run-name "$WANDB_RUN_NAME" \
  --train-prompt "$TRAIN_PROMPT" \
  "${EVAL_ARGS[@]}"
