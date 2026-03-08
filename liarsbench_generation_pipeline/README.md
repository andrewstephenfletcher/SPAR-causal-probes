# LiarsBench-Style Synthetic Data Generation Pipeline

This pipeline generates answers from neutral prompts, then asks a second LLM judge to classify LiarsBench-style deception.

## What It Produces

For each input prompt:
1. Generate an answer with the configured generator model (OpenRouter).
2. Classify the generated answer with the configured judge model (OpenRouter).
3. Save one JSONL row with prompt, answer, optional target type, judge decision, and usage metadata.

Generation modes:
- `neutral` (default): no forced deception instructions to generator.
- `targeted_deception`: sample a target type and explicitly instruct generator to produce that deception style.

## Deception Types Used in Judge Prompt

- `instructed-deception`
- `convincing-game`
- `gender-secret`
- `harm-pressure-choice`
- `harm-pressure-knowledge-report`
- `insider-trading`
- `soft-trigger`
- `other`

## Natural Prompts from sentence-transformers/natural-questions

Prepare a sampled prompt file first:

```bash
python liarsbench_generation_pipeline/prepare_nq_prompts.py \
  --dataset sentence-transformers/natural-questions \
  --split train \
  --num-prompts 1000 \
  --output liarsbench_generation_pipeline/natural_questions_prompts.jsonl
```

Recommended staged scales (to avoid full-dataset runs):
- Smoke test: `100` prompts x `1` completion each (`100` examples)
- Pilot: `1,000` prompts x `2` completions each (`2,000` examples)
- Medium: `3,000` prompts x `3` completions each (`9,000` examples)

For now, start with the Pilot setup. Set:
- `input.limit = 1000`
- `generation.samples_per_prompt = 2`
- `generation.mode = "neutral"`

## Setup

Set your API key:

```bash
export OPENROUTER_API_KEY="your_secret_here"
```

## Run

Neutral (default):
```bash
python -m liarsbench_generation_pipeline.pipeline \
  --config liarsbench_generation_pipeline/config.example.json
```

Targeted deception mode:
```bash
python -m liarsbench_generation_pipeline.pipeline \
  --config liarsbench_generation_pipeline/config.targeted_deception.example.json
```

Script entry points:
```bash
./liarsbench_generation_pipeline/scripts/prepare_nq_prompts.sh
./liarsbench_generation_pipeline/scripts/run_pipeline_neutral.sh
./liarsbench_generation_pipeline/scripts/run_pipeline_targeted.sh
```

The run scripts auto-load `liarsbench_generation_pipeline/.env` if present.

## Input Formats

`input.path` supports:
- `.jsonl` (one object per line)
- `.json` (list of objects)
- `.csv`
- `.txt` (one prompt per line)

Set `input.prompt_field` for object/CSV inputs.

## Output Schema (JSONL)

Each line contains:
- `prompt`
- `sample_idx`
- `generation_mode` (`neutral` or `targeted_deception`)
- `target_deception_type`
- `generated_answer`
- `judge`:
  - `is_deceptive` (bool)
  - `detected_type` (taxonomy label or `none`)
  - `matches_target` (bool or `null`)
  - `confidence` (0-1)
  - `reason` (short string)
- `generator_model`
- `judge_model`
- `generator_usage`
- `judge_usage`
- `generator_request_id`
- `judge_request_id`
- `judge_parse_error` (string or `null`)

Parse-failure handling:
- Judge JSON parse is retried once automatically.
- If retry also fails, pipeline writes a fallback `judge` object (`detected_type="parse_error"`) and continues.
- Optional error log is written to `output.error_log_path` (JSONL).
- Set `output.print_errors` to control stderr error messages.

## Notes

- Change `generator.model` and `judge.model` to any OpenRouter model IDs.
- This output is suitable for probe training and transfer evaluation against real LiarsBench splits.
