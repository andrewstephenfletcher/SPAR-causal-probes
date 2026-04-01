"""
andrewstephenfletcher March 2026

Loads pre-computed DCT steering vectors from an existing experiment, generates
steered continuations on diverse open-ended prompts, then uses a capable LLM
judge to characterise what each vector does and how consistently.

Unlike judge_vectors.py (which checks factual correctness against known answers),
this script asks the judge open-endedly: "what does this vector do?" — giving
each vector's full set of baseline/steered pairs in a single call so the judge
can identify patterns across prompts.

Output: experiments/{experiment}/results/vector_exploration.csv
        experiments/{experiment}/results/vector_exploration_cost.json
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import gc
import re
import csv
import json
import asyncio
import argparse
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm
from dotenv import load_dotenv
import openai
from transformers import AutoModelForCausalLM, AutoTokenizer

import dct

load_dotenv()

# ─── Device setup ──────────────────────────────────────────────────────────────
for _var in ["model", "tokenizer", "model_editor", "U", "V"]:
    if _var in dir():
        del globals()[_var]

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"GPU memory reserved:  {torch.cuda.memory_reserved()   / 1e9:.2f} GB")

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print(f"Using device: {DEVICE}")
torch.set_default_dtype(torch.float32)

# ─── Judge defaults ─────────────────────────────────────────────────────────────
# Per-vector calls are large (all prompt pairs in one message), so concurrency
# is lower than judge_vectors.py.
DEFAULT_JUDGE_MODEL    = "anthropic/claude-sonnet-4.6"
JUDGE_MAX_CONCURRENT   = 10
# Rough OpenRouter pricing for anthropic/claude-sonnet-4.6 — verify at openrouter.ai/models
JUDGE_PRICE_INPUT  = 3.00   # USD / 1M tokens
JUDGE_PRICE_OUTPUT = 15.00  # USD / 1M tokens

# ─── Evaluation prompts ─────────────────────────────────────────────────────────
# Open-ended, no ground truth. Diverse enough that tone shifts, language changes,
# topic fixation, refusals, or incoherence will manifest clearly.
EXPLORATION_PROMPTS = [
    {"id": "story",      "text": "Write the beginning of a short story about an unexpected journey."},
    {"id": "ai_views",   "text": "What do you think is the most important consideration about artificial intelligence?"},
    {"id": "advice",     "text": "A friend is feeling stressed about work. What advice would you give them?"},
    {"id": "music",      "text": "Explain how music affects human emotions."},
    {"id": "sunset",     "text": "Describe what makes a beautiful sunset."},
    {"id": "loneliness", "text": "How might society address the challenge of loneliness in modern cities?"},
    {"id": "quality",    "text": "What do you think is the most important quality a person can have?"},
    {"id": "history",    "text": "Imagine you could visit any time period in history. Where would you go and why?"},
    {"id": "fairness",   "text": "Explain the concept of fairness to a child."},
    {"id": "creativity", "text": "What role does creativity play in solving difficult problems?"},
]

# ─── Judge prompts ──────────────────────────────────────────────────────────────
# ── Pass 1: per-vector characterisation ─────────────────────────────────────────
JUDGE_SYSTEM_P1 = """You are an expert analysing the behaviour of steering vectors applied to a language model.
A steering vector is added to the model's hidden activations to systematically shift its outputs.

For each example you will see:
  PROMPT: the input given to the model
  BASELINE: the unsteered model's response
  STEERED: the response with the steering vector applied

Compare the baseline and steered responses across all examples and describe what the steering vector does.

Your response must follow this format exactly — nothing else:
EFFECT: <1-2 sentences describing the most salient and consistent change you observe, e.g. tone, \
topic, language, style, coherence, length, refusal behaviour>"""

# ── Pass 2: per-row presence scoring ────────────────────────────────────────────
JUDGE_SYSTEM_P2 = """You are evaluating whether a specific model response demonstrates a described steering vector effect.

You will be given:
  EFFECT: A description of what the steering vector does
  PROMPT: The input given to the model
  BASELINE: The unsteered model's response
  STEERED: The response with the steering vector applied

Does the STEERED response clearly demonstrate the described EFFECT compared to the BASELINE?

Respond with a single integer — nothing else:
1 - Yes, the effect is clearly present in the steered response
0 - No, the effect is absent or unclear"""

_JUDGE_EXAMPLE = "--- Example {i} ---\nPROMPT: {prompt}\nBASELINE: {baseline}\nSTEERED: {steered}\n\n"


def _build_p1_user_message(pairs: list[dict]) -> str:
    msg = f"Here are {len(pairs)} baseline/steered response pairs for a single steering vector.\n\n"
    for i, p in enumerate(pairs, 1):
        msg += _JUDGE_EXAMPLE.format(
            i=i, prompt=p["prompt_text"], baseline=p["baseline"], steered=p["steered"]
        )
    msg += "What has been the effect of this steering vector?"
    return msg


def _build_p2_user_message(effect: str, prompt_text: str, baseline: str, steered: str) -> str:
    return (
        f"EFFECT: {effect}\n\n"
        f"PROMPT: {prompt_text}\n"
        f"BASELINE: {baseline}\n"
        f"STEERED: {steered}\n\n"
        "Is the described effect clearly present in the steered response? Reply 0 or 1."
    )


# ─── Config loading ─────────────────────────────────────────────────────────────

def load_dct_params(experiment: str) -> dict[str, Any]:
    with open("dct_params.json", "r") as f:
        all_params = json.load(f)
    if experiment not in all_params:
        raise ValueError(f"Unknown experiment '{experiment}'. Available: {list(all_params)}")
    return all_params[experiment]


# ─── Model loading ──────────────────────────────────────────────────────────────

def load_model(model_name: str, tokenizer_name: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=True,
        padding_side="left",
        truncation_side="left",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=DEVICE,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        _attn_implementation="eager",
    )
    tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.eos_token_id
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.eval()
    print(f"Model loaded: {model_name}")
    print(f"Num layers:   {model.config.num_hidden_layers}")
    print(f"d_model:      {model.config.hidden_size}")
    print(f"Device:       {next(model.parameters()).device}")
    return model, tokenizer


def load_vectors(vectors_dir: Path) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    data = torch.load(vectors_dir / "dct_vectors.pt", weights_only=True)
    V = data["V"]
    U = data["U"]
    with open(vectors_dir / "dct_run_config.json", "r") as f:
        config = json.load(f)
    print(f"Loaded {V.shape[1]} steering vectors (d_model={V.shape[0]})")
    print(f"Source layer: {config['SOURCE_LAYER_IDX']} → Target layer: {config['TARGET_LAYER_IDX']}")
    return U, V, config


# ─── Completion generation ──────────────────────────────────────────────────────

def _format_prompt(tokenizer: AutoTokenizer, text: str, system_prompt: str | None) -> str:
    """Apply chat template for instruct models, with plain-text fallback.

    Omits the system message when system_prompt is None or empty, since some
    models (e.g. Gemma) do not support the system role.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": text})
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        prefix = f"{system_prompt}\n\n" if system_prompt else ""
        return f"{prefix}{text}"


def generate_baseline_completions(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[dict],
    system_prompt: str,
    max_new_tokens: int = 128,
) -> list[dict]:
    results = []
    for prompt in tqdm(prompts, desc="Baseline completions", unit="prompt"):
        formatted = _format_prompt(tokenizer, prompt["text"], system_prompt)
        inputs = tokenizer(formatted, return_tensors="pt", truncation=True).to(DEVICE)
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new_tokens = generated_ids[0][inputs["input_ids"].shape[1]:]
        completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
        results.append({
            "factor_idx": -1,
            "prompt_id":   prompt["id"],
            "prompt_text": prompt["text"],
            "completion":  completion,
        })
    print(f"Generated {len(results)} baseline completions")
    return results


def generate_steered_completions(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    model_editor: "dct.ModelEditor",
    V: torch.Tensor,
    prompts: list[dict],
    system_prompt: str,
    input_scale: float,
    source_layer_idx: int,
    num_vectors: int | None = None,
    max_new_tokens: int = 128,
) -> list[dict]:
    n = num_vectors if num_vectors is not None else V.shape[1]
    results = []
    for factor_idx in tqdm(range(n), desc="Steering vectors", unit="factor"):
        for prompt in prompts:
            model_editor.restore()
            model_editor.steer(input_scale * V[:, factor_idx], source_layer_idx)
            formatted = _format_prompt(tokenizer, prompt["text"], system_prompt)
            inputs = tokenizer(formatted, return_tensors="pt", truncation=True).to(DEVICE)
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            new_tokens = generated_ids[0][inputs["input_ids"].shape[1]:]
            completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
            results.append({
                "factor_idx": factor_idx,
                "prompt_id":   prompt["id"],
                "prompt_text": prompt["text"],
                "completion":  completion,
            })
    model_editor.restore()
    print(f"Generated {len(results)} steered completions")
    return results


# ─── Async judging ──────────────────────────────────────────────────────────────

def _parse_effect(content: str) -> dict:
    """Extract the EFFECT label from a pass-1 judge response."""
    match = re.search(r"EFFECT:\s*(.+)", content, re.DOTALL | re.IGNORECASE)
    if match:
        return {"description": match.group(1).strip(), "parse_error": ""}
    return {"description": None, "parse_error": content}


def _parse_row_score(content: str) -> int | None:
    """Extract 0 or 1 from a pass-2 judge response."""
    match = re.search(r"\b([01])\b", content.strip())
    if match:
        return int(match.group(1))
    return None


async def _api_call(
    client: openai.AsyncOpenAI,
    judge_model: str,
    system: str,
    user: str,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
) -> tuple[str, int, int]:
    """Single async API call; returns (content, prompt_tokens, completion_tokens)."""
    async with semaphore:
        resp = await client.chat.completions.create(
            model=judge_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            max_tokens=max_tokens,
            temperature=0,
        )
        content = resp.choices[0].message.content
        return content, resp.usage.prompt_tokens, resp.usage.completion_tokens


async def judge_pass1(
    client: openai.AsyncOpenAI,
    judge_model: str,
    baseline_completions: list[dict],
    steered_completions: list[dict],
    max_concurrent: int = JUDGE_MAX_CONCURRENT,
) -> tuple[dict[int, dict], dict[str, int]]:
    """Pass 1: one call per vector → EFFECT description."""
    baseline_by_prompt = {c["prompt_id"]: c["completion"] for c in baseline_completions}
    by_factor: dict[int, list[dict]] = {}
    for c in steered_completions:
        by_factor.setdefault(c["factor_idx"], []).append(c)

    sem = asyncio.Semaphore(max_concurrent)

    async def judge_factor(factor_idx: int) -> tuple[int, dict]:
        pairs = [
            {
                "prompt_text": c["prompt_text"],
                "baseline":    baseline_by_prompt.get(c["prompt_id"], ""),
                "steered":     c["completion"],
            }
            for c in by_factor[factor_idx]
        ]
        user_msg = _build_p1_user_message(pairs)
        try:
            content, pt, ct = await _api_call(
                client, judge_model, JUDGE_SYSTEM_P1, user_msg, max_tokens=4000, semaphore=sem
            )
            return factor_idx, {**_parse_effect(content), "prompt_tokens": pt, "completion_tokens": ct}
        except Exception as e:
            return factor_idx, {"description": None, "parse_error": f"error: {e}",
                                 "prompt_tokens": 0, "completion_tokens": 0}

    raw = await atqdm.gather(
        *[judge_factor(fi) for fi in sorted(by_factor)],
        desc="Pass 1 — characterising vectors",
        unit="vector",
    )
    judgments: dict[int, dict] = dict(raw)
    usage = {"prompt_tokens": sum(j.get("prompt_tokens", 0) for j in judgments.values()),
             "completion_tokens": sum(j.get("completion_tokens", 0) for j in judgments.values())}
    n_errors = sum(1 for j in judgments.values() if j["description"] is None)
    print(f"Pass 1 complete. Errors: {n_errors}/{len(judgments)}")
    return judgments, usage


async def judge_pass2(
    client: openai.AsyncOpenAI,
    judge_model: str,
    baseline_completions: list[dict],
    steered_completions: list[dict],
    p1_judgments: dict[int, dict],
    max_concurrent: int = JUDGE_MAX_CONCURRENT,
) -> tuple[dict[tuple[int, str], int | None], dict[str, int]]:
    """Pass 2: one call per (vector, prompt) → 0 or 1 presence score."""
    baseline_by_prompt = {c["prompt_id"]: c["completion"] for c in baseline_completions}
    sem = asyncio.Semaphore(max_concurrent)

    async def score_row(c: dict) -> tuple[tuple[int, str], int | None, int, int]:
        fi = c["factor_idx"]
        effect = (p1_judgments.get(fi) or {}).get("description")
        if not effect:
            return (fi, c["prompt_id"]), None, 0, 0
        user_msg = _build_p2_user_message(
            effect, c["prompt_text"],
            baseline_by_prompt.get(c["prompt_id"], ""),
            c["completion"],
        )
        try:
            content, pt, ct = await _api_call(
                client, judge_model, JUDGE_SYSTEM_P2, user_msg, max_tokens=2000, semaphore=sem
            )
            return (fi, c["prompt_id"]), _parse_row_score(content), pt, ct
        except Exception:
            return (fi, c["prompt_id"]), None, 0, 0

    raw = await atqdm.gather(
        *[score_row(c) for c in steered_completions],
        desc="Pass 2 — scoring rows",
        unit="row",
    )
    scores: dict[tuple[int, str], int | None] = {r[0]: r[1] for r in raw}
    usage = {"prompt_tokens":     sum(r[2] for r in raw),
             "completion_tokens": sum(r[3] for r in raw)}
    n_errors = sum(1 for r in raw if r[1] is None)
    print(f"Pass 2 complete. Errors: {n_errors}/{len(raw)}")
    return scores, usage


# ─── Output ─────────────────────────────────────────────────────────────────────

def save_csv(
    baseline_completions: list[dict],
    steered_completions: list[dict],
    p1_judgments: dict[int, dict],
    row_scores: dict[tuple[int, str], int | None],
    output_path: Path,
) -> None:
    """One row per (vector, prompt) pair.

    Columns:
      judge_description  — pass-1 characterisation, same for all rows of a vector
      row_score          — pass-2 binary (0/1): does this row demonstrate the effect?
      consistency_score  — sum of row_scores across all prompts for this vector
      p1_parse_error     — non-empty only if pass-1 failed for this vector
    """
    baseline_by_prompt = {c["prompt_id"]: c["completion"] for c in baseline_completions}

    # Pre-compute consistency scores (sum of row 0/1 scores per vector)
    from collections import defaultdict
    score_sums: dict[int, int]  = defaultdict(int)
    score_counts: dict[int, int] = defaultdict(int)
    for (fi, _), s in row_scores.items():
        if s is not None:
            score_sums[fi]   += s
            score_counts[fi] += 1

    fieldnames = [
        "vector_idx", "prompt_id", "prompt_text",
        "baseline_continuation", "steered_continuation",
        "judge_description", "row_score", "consistency_score", "p1_parse_error",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in steered_completions:
            fi  = c["factor_idx"]
            pid = c["prompt_id"]
            j   = p1_judgments.get(fi, {})
            writer.writerow({
                "vector_idx":            fi,
                "prompt_id":             pid,
                "prompt_text":           c["prompt_text"],
                "baseline_continuation": baseline_by_prompt.get(pid, ""),
                "steered_continuation":  c["completion"],
                "judge_description":     j.get("description",  ""),
                "row_score":             row_scores.get((fi, pid), ""),
                "consistency_score":     score_sums.get(fi, ""),
                "p1_parse_error":        j.get("parse_error",  ""),
            })
    print(f"Saved {len(steered_completions)} rows to {output_path}")


# ─── Entry point ────────────────────────────────────────────────────────────────

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Characterise steering vectors with an open-ended LLM judge."
    )
    parser.add_argument("--experiment",  required=True,
                        help="Experiment name from dct_params.json")
    parser.add_argument("--num-prompts", type=int, default=10,
                        help=f"Number of evaluation prompts (max {len(EXPLORATION_PROMPTS)}, default 10)")
    parser.add_argument("--num-vectors", type=int, default=None,
                        help="Number of steering vectors to evaluate (default: all)")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL,
                        help=f"OpenRouter model ID for judge (default: {DEFAULT_JUDGE_MODEL})")
    parser.add_argument("--output-name", default="vector_exploration.csv",
                        help="Output CSV filename (default: vector_exploration.csv)")
    args = parser.parse_args()

    params = load_dct_params(args.experiment)
    model_name       = params["MODEL_NAME"]
    tokenizer_name   = params["TOKENIZER_NAME"]
    system_prompt    = params.get("SYSTEM_PROMPT") or "You are a helpful assistant"
    source_layer_idx = params["SOURCE_LAYER_IDX"]
    input_scale_cfg  = params.get("INPUT_SCALE")  # None means auto-calibrated

    experiment_dir = Path("experiments") / args.experiment
    vectors_dir    = experiment_dir / "vectors"
    results_dir    = experiment_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_path = results_dir / args.output_name
    cost_path   = results_dir / args.output_name.replace(".csv", "_cost.json")

    prompts = EXPLORATION_PROMPTS[:args.num_prompts]
    print(f"Experiment:    {args.experiment}")
    print(f"Judge model:   {args.judge_model}")
    print(f"Num prompts:   {len(prompts)}")
    print(f"Num vectors:   {args.num_vectors or 'all'}")

    # ── Generate or load completions ──────────────────────────────────────────
    completions_cache = results_dir / "exploration_completions.jsonl"

    if completions_cache.exists():
        print(f"\nLoading cached completions from {completions_cache}")
        with open(completions_cache) as f:
            all_completions = [json.loads(line) for line in f]
        baseline_completions = [c for c in all_completions if c["factor_idx"] == -1]
        steered_completions  = [c for c in all_completions if c["factor_idx"] >= 0]
        print(f"Loaded {len(baseline_completions)} baseline + {len(steered_completions)} steered completions")
    else:
        _U, V, run_config = load_vectors(vectors_dir)
        input_scale = input_scale_cfg if input_scale_cfg is not None else run_config["INPUT_SCALE"]
        print(f"\nUsing input_scale={input_scale:.4f}  source_layer_idx={source_layer_idx}")

        model, tokenizer = load_model(model_name, tokenizer_name)
        model_editor = dct.ModelEditor(model, layers_name="model.layers")

        baseline_completions = generate_baseline_completions(
            model, tokenizer, prompts, system_prompt
        )
        steered_completions = generate_steered_completions(
            model, tokenizer, model_editor, V, prompts,
            system_prompt=system_prompt,
            input_scale=input_scale,
            source_layer_idx=source_layer_idx,
            num_vectors=args.num_vectors,
        )

        with open(completions_cache, "w") as f:
            for c in baseline_completions + steered_completions:
                f.write(json.dumps(c) + "\n")
        print(f"Cached {len(baseline_completions) + len(steered_completions)} completions to {completions_cache}")

        # Free GPU memory before judging
        del model, tokenizer, model_editor, V, _U
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Judge (two passes) ────────────────────────────────────────────────────
    client = openai.AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    print(f"\nJudge model: {args.judge_model}")

    p1_judgments, p1_usage = await judge_pass1(
        client, args.judge_model,
        baseline_completions, steered_completions,
        max_concurrent=JUDGE_MAX_CONCURRENT,
    )
    row_scores, p2_usage = await judge_pass2(
        client, args.judge_model,
        baseline_completions, steered_completions,
        p1_judgments,
        max_concurrent=JUDGE_MAX_CONCURRENT,
    )

    # ── Save outputs ──────────────────────────────────────────────────────────
    save_csv(baseline_completions, steered_completions, p1_judgments, row_scores, output_path)

    total_input  = p1_usage["prompt_tokens"]     + p2_usage["prompt_tokens"]
    total_output = p1_usage["completion_tokens"] + p2_usage["completion_tokens"]
    cost = (total_input * JUDGE_PRICE_INPUT + total_output * JUDGE_PRICE_OUTPUT) / 1_000_000
    usage_summary = {
        "model":                   args.judge_model,
        "pass1_input_tokens":      p1_usage["prompt_tokens"],
        "pass1_output_tokens":     p1_usage["completion_tokens"],
        "pass2_input_tokens":      p2_usage["prompt_tokens"],
        "pass2_output_tokens":     p2_usage["completion_tokens"],
        "total_input_tokens":      total_input,
        "total_output_tokens":     total_output,
        "estimated_cost_usd":      round(cost, 6),
        "price_per_1m_input_usd":  JUDGE_PRICE_INPUT,
        "price_per_1m_output_usd": JUDGE_PRICE_OUTPUT,
    }
    with open(cost_path, "w") as f:
        json.dump(usage_summary, f, indent=2)
    print(f"Saved cost summary to {cost_path}")


if __name__ == "__main__":
    asyncio.run(main())
