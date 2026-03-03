#!/usr/bin/env bash

# Lower-strength configuration for GRPO steering.
MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
REWARD_MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
LAYER_IDX=18
MAX_STEPS=200
NUM_GENERATIONS=8
TRAIN_PROMPT="Is Paris the capital of France?"
EVAL_PROMPT="Is Moscow the capital of Russia?"

STEERING_STRENGTH=0.2
STEERING_INIT_SCALE=1e-4
LEARNING_RATE=5e-3
BETA=0.1
GIBBERISH_PENALTY_WEIGHT=0.2

python scripts/run_grpo_steering.py \
  --model-name "$MODEL_NAME" \
  --reward-model-name "$REWARD_MODEL_NAME" \
  --layer-idx "$LAYER_IDX" \
  --max-steps "$MAX_STEPS" \
  --num-generations "$NUM_GENERATIONS" \
  --train-prompt "$TRAIN_PROMPT" \
  --eval-prompt "$EVAL_PROMPT" \
  --steering-strength "$STEERING_STRENGTH" \
  --steering-init-scale "$STEERING_INIT_SCALE" \
  --learning-rate "$LEARNING_RATE" \
  --beta "$BETA" \
  --gibberish-penalty-weight "$GIBBERISH_PENALTY_WEIGHT"
