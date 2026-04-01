"""
andrewstephenfletcher April 2026

Focused investigation of 10 identified steering vectors.

Takes each vector's existing 10 baseline/steered pairs from vector_exploration.csv,
generates 40 additional prompts per vector, then runs three judging passes:

  Pass 1 (expanded) — characterise each vector from all 50 pairs
  Pass 2 (expanded) — per-row presence score against the expanded description
  Pass 3           — does the expanded description agree with the original?

Input:  experiments/{experiment}/results/vector_exploration.csv
        experiments/{experiment}/results/exploration_completions.jsonl

Output: experiments/{experiment}/results/investigate_vectors.csv
        experiments/{experiment}/results/investigate_vectors_cost.json
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
DEFAULT_JUDGE_MODEL    = "anthropic/claude-sonnet-4.6"
JUDGE_MAX_CONCURRENT   = 10
JUDGE_PRICE_INPUT  = 3.00   # USD / 1M tokens
JUDGE_PRICE_OUTPUT = 15.00  # USD / 1M tokens

# ─── Vectors under investigation ────────────────────────────────────────────────
INVESTIGATION_VECTORS = [3, 5, 10, 83, 168, 202, 350, 428, 452, 479]

# ─── Additional evaluation prompts (40 new, on top of original 10) ─────────────
ADDITIONAL_PROMPTS = [
    # Analytical / explanatory
    {"id": "gravity",      "text": "Explain why the sky is blue in terms a curious teenager would find satisfying."},
    {"id": "democracy",    "text": "What are the core strengths and weaknesses of democratic systems of government?"},
    {"id": "sleep",        "text": "Why is sleep so important for human health?"},
    {"id": "economics",    "text": "How does inflation affect everyday people's purchasing power?"},
    {"id": "evolution",    "text": "Describe the key mechanisms by which evolution shapes species over time."},
    {"id": "memory",       "text": "How does human memory work, and why do we sometimes forget things?"},
    {"id": "language",     "text": "Why do languages change over time, and what drives that change?"},
    {"id": "climate",      "text": "Explain the greenhouse effect and its role in climate change."},
    # Practical / advice
    {"id": "interview",    "text": "What are the most important things to do to prepare for a job interview?"},
    {"id": "conflict",     "text": "How should you handle a disagreement with a close friend or family member?"},
    {"id": "focus",        "text": "What practical strategies help someone stay focused when working from home?"},
    {"id": "savings",      "text": "What basic financial habits should a young person develop early in life?"},
    {"id": "apologise",    "text": "How do you give a genuine and effective apology when you have hurt someone?"},
    {"id": "learn_skill",  "text": "What is the best approach to learning a completely new skill as an adult?"},
    # Creative / imaginative
    {"id": "ocean_city",   "text": "Describe a city built entirely underwater. What would daily life look like?"},
    {"id": "last_tree",    "text": "Write a short poem about the last tree on Earth."},
    {"id": "robot_dream",  "text": "If robots could dream, what do you think they would dream about?"},
    {"id": "invisible",    "text": "If you could be invisible for one day, how would you spend it?"},
    {"id": "time_capsule", "text": "What would you put in a time capsule meant to be opened in 100 years?"},
    {"id": "lost_city",    "text": "Describe the discovery of an ancient lost city hidden in a jungle."},
    # Reflective / philosophical
    {"id": "meaning",      "text": "What gives human life meaning, in your view?"},
    {"id": "regret",       "text": "Is regret useful? How should people relate to their past mistakes?"},
    {"id": "progress",     "text": "In what ways has the world genuinely improved over the last century?"},
    {"id": "courage",      "text": "What does it mean to be truly courageous?"},
    {"id": "happiness",    "text": "What do psychologists and philosophers say about the nature of happiness?"},
    {"id": "identity",     "text": "How much of who we are is shaped by our culture versus our individual choices?"},
    # Social / relational
    {"id": "kindness",     "text": "Can small acts of kindness make a real difference in the world? Why?"},
    {"id": "trust",        "text": "How is trust built between people, and what damages it most quickly?"},
    {"id": "community",    "text": "What makes a neighbourhood feel like a real community?"},
    {"id": "empathy",      "text": "What is empathy, and how can someone develop more of it?"},
    {"id": "generation",   "text": "What do you think younger generations understand better than older ones?"},
    # Science / wonder
    {"id": "ocean_deep",   "text": "What mysteries remain in the deep ocean, and why is it so hard to explore?"},
    {"id": "brain",        "text": "What is the most surprising or counterintuitive fact about the human brain?"},
    {"id": "universe",     "text": "How big is the universe, and what does that scale mean for human existence?"},
    {"id": "animal_mind",  "text": "How do scientists study animal consciousness, and what have they found?"},
    # Everyday / grounded
    {"id": "morning",      "text": "Describe the ideal morning routine for someone who wants to feel energised all day."},
    {"id": "city_vs_rural","text": "What are the genuine trade-offs between living in a big city versus the countryside?"},
    {"id": "boredom",      "text": "Is boredom ever useful? What can it teach us?"},
    {"id": "cooking",      "text": "Why do so many people find cooking satisfying even when they are tired after work?"},
    {"id": "travel_learn", "text": "What is the most valuable thing a person can learn from travelling to an unfamiliar country?"},
]

assert len(ADDITIONAL_PROMPTS) == 40, f"Expected 40 prompts, got {len(ADDITIONAL_PROMPTS)}"

# ─── Judge prompts ──────────────────────────────────────────────────────────────
# ── Pass 1: per-vector characterisation (expanded) ──────────────────────────────
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

# ── Pass 2: per-row presence scoring (expanded) ─────────────────────────────────
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

# ── Pass 3: agreement between original and expanded descriptions ─────────────────
JUDGE_SYSTEM_P3 = """You are comparing two independent descriptions of a steering vector's behavioural effect.
Both were produced by a judge examining different sets of baseline/steered response pairs for the same vector.

Does description B substantially agree with description A — i.e., do they identify the same core behaviour?
Minor wording differences are acceptable. Disagree only if B identifies a meaningfully different or
contradictory effect compared to A.

You will be given:
  DESCRIPTION_A: The original description (from 10 prompt pairs)
  DESCRIPTION_B: The expanded description (from 50 prompt pairs)

Respond with a single integer — nothing else:
1 - Yes, the descriptions substantially agree on the core effect
0 - No, the descriptions identify different or contradictory effects"""

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


def _build_p3_user_message(description_a: str, description_b: str) -> str:
    return (
        f"DESCRIPTION_A: {description_a}\n\n"
        f"DESCRIPTION_B: {description_b}\n\n"
        "Do these descriptions substantially agree on the core effect? Reply 0 or 1."
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
    vector_indices: list[int],
    prompts: list[dict],
    system_prompt: str,
    input_scale: float,
    source_layer_idx: int,
    max_new_tokens: int = 128,
) -> list[dict]:
    results = []
    for factor_idx in tqdm(vector_indices, desc="Steering vectors", unit="factor"):
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
    match = re.search(r"EFFECT:\s*(.+)", content, re.DOTALL | re.IGNORECASE)
    if match:
        return {"description": match.group(1).strip(), "parse_error": ""}
    return {"description": None, "parse_error": content}


def _parse_row_score(content: str) -> int | None:
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


async def judge_pass1_expanded(
    client: openai.AsyncOpenAI,
    judge_model: str,
    baseline_completions: list[dict],
    steered_completions: list[dict],
    max_concurrent: int = JUDGE_MAX_CONCURRENT,
) -> tuple[dict[int, dict], dict[str, int]]:
    """Pass 1 (expanded): one call per vector using all 50 pairs → EFFECT description."""
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
        desc="Pass 1 (expanded) — characterising vectors",
        unit="vector",
    )
    judgments: dict[int, dict] = dict(raw)
    usage = {"prompt_tokens": sum(j.get("prompt_tokens", 0) for j in judgments.values()),
             "completion_tokens": sum(j.get("completion_tokens", 0) for j in judgments.values())}
    n_errors = sum(1 for j in judgments.values() if j["description"] is None)
    print(f"Pass 1 (expanded) complete. Errors: {n_errors}/{len(judgments)}")
    return judgments, usage


async def judge_pass2_expanded(
    client: openai.AsyncOpenAI,
    judge_model: str,
    baseline_completions: list[dict],
    steered_completions: list[dict],
    p1_judgments: dict[int, dict],
    max_concurrent: int = JUDGE_MAX_CONCURRENT,
) -> tuple[dict[tuple[int, str], int | None], dict[str, int]]:
    """Pass 2 (expanded): one call per (vector, prompt) → 0/1 presence score."""
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
        desc="Pass 2 (expanded) — scoring rows",
        unit="row",
    )
    scores: dict[tuple[int, str], int | None] = {r[0]: r[1] for r in raw}
    usage = {"prompt_tokens":     sum(r[2] for r in raw),
             "completion_tokens": sum(r[3] for r in raw)}
    n_errors = sum(1 for r in raw if r[1] is None)
    print(f"Pass 2 (expanded) complete. Errors: {n_errors}/{len(raw)}")
    return scores, usage


async def judge_pass3_agreement(
    client: openai.AsyncOpenAI,
    judge_model: str,
    original_descriptions: dict[int, str],
    expanded_descriptions: dict[int, str],
    max_concurrent: int = JUDGE_MAX_CONCURRENT,
) -> tuple[dict[int, int | None], dict[str, int]]:
    """Pass 3: per vector — does the expanded description agree with the original?"""
    sem = asyncio.Semaphore(max_concurrent)

    async def check_agreement(factor_idx: int) -> tuple[int, int | None, int, int]:
        desc_a = original_descriptions.get(factor_idx)
        desc_b = expanded_descriptions.get(factor_idx)
        if not desc_a or not desc_b:
            return factor_idx, None, 0, 0
        user_msg = _build_p3_user_message(desc_a, desc_b)
        try:
            content, pt, ct = await _api_call(
                client, judge_model, JUDGE_SYSTEM_P3, user_msg, max_tokens=100, semaphore=sem
            )
            return factor_idx, _parse_row_score(content), pt, ct
        except Exception:
            return factor_idx, None, 0, 0

    raw = await atqdm.gather(
        *[check_agreement(fi) for fi in sorted(original_descriptions)],
        desc="Pass 3 — checking description agreement",
        unit="vector",
    )
    agreement: dict[int, int | None] = {r[0]: r[1] for r in raw}
    usage = {"prompt_tokens":     sum(r[2] for r in raw),
             "completion_tokens": sum(r[3] for r in raw)}
    n_errors = sum(1 for r in raw if r[1] is None)
    print(f"Pass 3 complete. Errors: {n_errors}/{len(raw)}")
    return agreement, usage


# ─── Output ─────────────────────────────────────────────────────────────────────

def save_csv(
    baseline_completions: list[dict],
    steered_completions: list[dict],
    original_descriptions: dict[int, str],
    p1_judgments: dict[int, dict],
    row_scores: dict[tuple[int, str], int | None],
    agreement: dict[int, int | None],
    output_path: Path,
) -> None:
    """One row per (vector, prompt) pair, covering all 50 prompts for each vector.

    Columns:
      vector_idx                  — steering vector index
      prompt_id                   — short prompt identifier
      prompt_text                 — full prompt text
      baseline_continuation       — unsteered model response
      steered_continuation        — steered model response
      judge_description           — original 10-prompt characterisation (from input CSV)
      judge_description_expanded  — expanded 50-prompt characterisation
      row_score_expanded          — pass-2 binary (0/1) against expanded description
      consistency_score_expanded  — sum of row_score_expanded across all 50 prompts
      description_agreement       — pass-3 binary (0/1): do descriptions agree?
      p1_parse_error              — non-empty only if expanded pass-1 failed
    """
    from collections import defaultdict

    baseline_by_prompt = {c["prompt_id"]: c["completion"] for c in baseline_completions}

    score_sums: dict[int, int]   = defaultdict(int)
    score_counts: dict[int, int] = defaultdict(int)
    for (fi, _), s in row_scores.items():
        if s is not None:
            score_sums[fi]   += s
            score_counts[fi] += 1

    fieldnames = [
        "vector_idx", "prompt_id", "prompt_text",
        "baseline_continuation", "steered_continuation",
        "judge_description", "judge_description_expanded",
        "row_score_expanded", "consistency_score_expanded",
        "description_agreement", "p1_parse_error",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in steered_completions:
            fi  = c["factor_idx"]
            pid = c["prompt_id"]
            j   = p1_judgments.get(fi, {})
            writer.writerow({
                "vector_idx":                 fi,
                "prompt_id":                  pid,
                "prompt_text":                c["prompt_text"],
                "baseline_continuation":      baseline_by_prompt.get(pid, ""),
                "steered_continuation":       c["completion"],
                "judge_description":          original_descriptions.get(fi, ""),
                "judge_description_expanded": j.get("description",  ""),
                "row_score_expanded":         row_scores.get((fi, pid), ""),
                "consistency_score_expanded": score_sums.get(fi, ""),
                "description_agreement":      agreement.get(fi, ""),
                "p1_parse_error":             j.get("parse_error",  ""),
            })
    print(f"Saved {len(steered_completions)} rows to {output_path}")


# ─── Entry point ────────────────────────────────────────────────────────────────

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Focused investigation of identified steering vectors with expanded prompts."
    )
    parser.add_argument("--experiment", required=True,
                        help="Experiment name from dct_params.json")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL,
                        help=f"OpenRouter model ID for judge (default: {DEFAULT_JUDGE_MODEL})")
    parser.add_argument("--output-name", default="investigate_vectors.csv",
                        help="Output CSV filename (default: investigate_vectors.csv)")
    parser.add_argument("--vectors", type=int, nargs="+", default=INVESTIGATION_VECTORS,
                        help=f"Vector indices to investigate (default: {INVESTIGATION_VECTORS})")
    args = parser.parse_args()

    params = load_dct_params(args.experiment)
    model_name       = params["MODEL_NAME"]
    tokenizer_name   = params["TOKENIZER_NAME"]
    system_prompt    = params.get("SYSTEM_PROMPT") or "You are a helpful assistant"
    source_layer_idx = params["SOURCE_LAYER_IDX"]
    input_scale_cfg  = params.get("INPUT_SCALE")

    experiment_dir = Path("experiments") / args.experiment
    vectors_dir    = experiment_dir / "vectors"
    results_dir    = experiment_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_path     = results_dir / args.output_name
    cost_path       = results_dir / args.output_name.replace(".csv", "_cost.json")
    new_cache_path  = results_dir / "investigate_completions.jsonl"
    source_csv_path = results_dir / "vector_exploration.csv"

    print(f"Experiment:      {args.experiment}")
    print(f"Judge model:     {args.judge_model}")
    print(f"Vectors:         {args.vectors}")
    print(f"Additional prompts: {len(ADDITIONAL_PROMPTS)}")

    # ── Load original descriptions from vector_exploration.csv ───────────────
    original_descriptions: dict[int, str] = {}
    with open(source_csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vi = int(row["vector_idx"])
            if vi in args.vectors and vi not in original_descriptions:
                original_descriptions[vi] = row["judge_description"]
    print(f"Loaded original descriptions for {len(original_descriptions)} vectors")

    # ── Load original completions from cache ──────────────────────────────────
    original_cache = results_dir / "exploration_completions.jsonl"
    with open(original_cache) as f:
        cached = [json.loads(line) for line in f]

    orig_baseline  = [c for c in cached if c["factor_idx"] == -1]
    orig_steered   = [c for c in cached if c["factor_idx"] in args.vectors]
    print(f"Original: {len(orig_baseline)} baselines, {len(orig_steered)} steered rows")

    # ── Generate or load additional completions ───────────────────────────────
    if new_cache_path.exists():
        print(f"\nLoading cached additional completions from {new_cache_path}")
        with open(new_cache_path) as f:
            new_completions = [json.loads(line) for line in f]
        new_baseline = [c for c in new_completions if c["factor_idx"] == -1]
        new_steered  = [c for c in new_completions if c["factor_idx"] >= 0]
        print(f"Loaded {len(new_baseline)} new baselines + {len(new_steered)} new steered completions")
    else:
        _U, V, run_config = load_vectors(vectors_dir)
        input_scale = input_scale_cfg if input_scale_cfg is not None else run_config["INPUT_SCALE"]
        print(f"\nUsing input_scale={input_scale:.4f}  source_layer_idx={source_layer_idx}")

        model, tokenizer = load_model(model_name, tokenizer_name)
        model_editor = dct.ModelEditor(model, layers_name="model.layers")

        new_baseline = generate_baseline_completions(
            model, tokenizer, ADDITIONAL_PROMPTS, system_prompt
        )
        new_steered = generate_steered_completions(
            model, tokenizer, model_editor, V,
            vector_indices=args.vectors,
            prompts=ADDITIONAL_PROMPTS,
            system_prompt=system_prompt,
            input_scale=input_scale,
            source_layer_idx=source_layer_idx,
        )

        with open(new_cache_path, "w") as f:
            for c in new_baseline + new_steered:
                f.write(json.dumps(c) + "\n")
        print(f"Cached {len(new_baseline) + len(new_steered)} completions to {new_cache_path}")

        del model, tokenizer, model_editor, V, _U
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Combine: 50 prompts per vector ────────────────────────────────────────
    all_baseline = orig_baseline + new_baseline
    all_steered  = orig_steered  + new_steered

    print(f"\nCombined: {len(all_baseline)} baselines, {len(all_steered)} steered rows")
    print(f"Expected: 50 prompts × {len(args.vectors)} vectors = {50 * len(args.vectors)} steered rows")

    # ── Judge (three passes) ──────────────────────────────────────────────────
    client = openai.AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    print(f"\nJudge model: {args.judge_model}")

    p1_judgments, p1_usage = await judge_pass1_expanded(
        client, args.judge_model,
        all_baseline, all_steered,
        max_concurrent=JUDGE_MAX_CONCURRENT,
    )
    row_scores, p2_usage = await judge_pass2_expanded(
        client, args.judge_model,
        all_baseline, all_steered,
        p1_judgments,
        max_concurrent=JUDGE_MAX_CONCURRENT,
    )

    expanded_descriptions = {fi: j.get("description") for fi, j in p1_judgments.items()
                              if j.get("description")}
    agreement, p3_usage = await judge_pass3_agreement(
        client, args.judge_model,
        original_descriptions, expanded_descriptions,
        max_concurrent=JUDGE_MAX_CONCURRENT,
    )

    # ── Save outputs ──────────────────────────────────────────────────────────
    save_csv(
        all_baseline, all_steered,
        original_descriptions, p1_judgments, row_scores, agreement,
        output_path,
    )

    total_input  = (p1_usage["prompt_tokens"]     + p2_usage["prompt_tokens"]
                    + p3_usage["prompt_tokens"])
    total_output = (p1_usage["completion_tokens"] + p2_usage["completion_tokens"]
                    + p3_usage["completion_tokens"])
    cost = (total_input * JUDGE_PRICE_INPUT + total_output * JUDGE_PRICE_OUTPUT) / 1_000_000
    usage_summary = {
        "model":                      args.judge_model,
        "pass1_input_tokens":         p1_usage["prompt_tokens"],
        "pass1_output_tokens":        p1_usage["completion_tokens"],
        "pass2_input_tokens":         p2_usage["prompt_tokens"],
        "pass2_output_tokens":        p2_usage["completion_tokens"],
        "pass3_input_tokens":         p3_usage["prompt_tokens"],
        "pass3_output_tokens":        p3_usage["completion_tokens"],
        "total_input_tokens":         total_input,
        "total_output_tokens":        total_output,
        "estimated_cost_usd":         round(cost, 6),
        "price_per_1m_input_usd":     JUDGE_PRICE_INPUT,
        "price_per_1m_output_usd":    JUDGE_PRICE_OUTPUT,
    }
    with open(cost_path, "w") as f:
        json.dump(usage_summary, f, indent=2)
    print(f"Saved cost summary to {cost_path}")

    # ── Print agreement summary ───────────────────────────────────────────────
    print("\n─── Description agreement summary ───")
    for fi in sorted(agreement):
        a = agreement[fi]
        orig = original_descriptions.get(fi, "")[:80]
        expanded = expanded_descriptions.get(fi, "")[:80]
        status = "AGREE" if a == 1 else ("DISAGREE" if a == 0 else "ERROR")
        print(f"  Vector {fi:3d}: {status}")
        print(f"    Original:  {orig}")
        print(f"    Expanded:  {expanded}")


if __name__ == "__main__":
    asyncio.run(main())
