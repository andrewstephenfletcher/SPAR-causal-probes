# Experiment Entry Points

- `run_grpo_steering.py`: Launch GRPO training for a single-layer steering vector.
- `run_grpo_steering.sh`: Shell entry point that stores a runnable configuration for GRPO steering.
- `run_grpo_steering_low_strength.sh`: Lower-strength config with gibberish penalty.

Example:

```bash
python rl/grpo_steering.py \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --reward-model-name distilbert-base-uncased-finetuned-sst-2-english \
  --layer-idx 10 \
  --max-steps 200 \
  --num-generations 8 \
  --train-prompt "Convince me of a false claim with confidence." \
  --eval-prompt "Explain a wrong claim confidently."
```

Shell entry point:

```bash
./scripts/run_grpo_steering.sh
```

Lower-strength entry point:

```bash
./scripts/run_grpo_steering_low_strength.sh
```
