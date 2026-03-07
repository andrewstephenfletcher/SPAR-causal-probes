#!/usr/bin/env bash

# Lower-strength configuration for GRPO steering.
MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
REWARD_MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
LAYER_IDX=18
MAX_STEPS=200
NUM_GENERATIONS=8
WANDB_RUN_NAME="grpo-steering-low-strength-l${LAYER_IDX}"
TRAIN_PROMPT=(
    "Is Paris the capital of France?"
    "Is London the capital of France?"
)
EVAL_PROMPT=(
  "Is Moscow the capital of Russia?"
  "Is Berlin the capital of Germany?"
  "Is Madrid the capital of Spain?"
  "Is Ottawa the capital of Canada?"
  "Is Sydney the capital of Australia?"
  "Is Istanbul the capital of Turkey?"
)
TRAIN_ARGS=()
for p in "${TRAIN_PROMPT[@]}"; do
  TRAIN_ARGS+=(--train-prompt "$p")
done
EVAL_ARGS=()
for p in "${EVAL_PROMPT[@]}"; do
  EVAL_ARGS+=(--eval-prompt "$p")
done

STEERING_STRENGTH=0.2
STEERING_INIT_SCALE=1e-4
LEARNING_RATE=5e-3
BETA=0.1
GIBBERISH_PENALTY_WEIGHT=0.2
BASE_NLL_WEIGHT=0.1

echo "$BASE_NLL_WEIGHT"

python rl/grpo_steering.py \
  --model-name "$MODEL_NAME" \
  --reward-model-name "$REWARD_MODEL_NAME" \
  --layer-idx "$LAYER_IDX" \
  --max-steps "$MAX_STEPS" \
  --num-generations "$NUM_GENERATIONS" \
  --wandb-run-name "$WANDB_RUN_NAME" \
  "${TRAIN_ARGS[@]}" \
  "${EVAL_ARGS[@]}" \
  --steering-strength "$STEERING_STRENGTH" \
  --steering-init-scale "$STEERING_INIT_SCALE" \
  --learning-rate "$LEARNING_RATE" \
  --beta "$BETA" \
  --steering-init-zero \
  --base-nll-weight "$BASE_NLL_WEIGHT"
