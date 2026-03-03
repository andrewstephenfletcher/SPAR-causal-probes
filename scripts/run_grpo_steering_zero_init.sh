#!/usr/bin/env bash

# Configuration for GRPO steering-vector experiment.
MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
REWARD_MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
LAYER_IDX=18
MAX_STEPS=200
NUM_GENERATIONS=8
TRAIN_PROMPT="Is Paris the capital of France?"
EVAL_PROMPT="Is Moscow the capital of Russia?"

python rl/grpo_steering.py \
  --model-name "$MODEL_NAME" \
  --reward-model-name "$REWARD_MODEL_NAME" \
  --layer-idx "$LAYER_IDX" \
  --max-steps "$MAX_STEPS" \
  --num-generations "$NUM_GENERATIONS" \
  --train-prompt "$TRAIN_PROMPT" \
  --eval-prompt "$EVAL_PROMPT" \
  --steering-init-zero
