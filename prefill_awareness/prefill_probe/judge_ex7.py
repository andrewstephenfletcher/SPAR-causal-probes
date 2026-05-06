"""
Independent quality judging and perplexity computation for Experiment 7.

Two tasks:

  1. Quality judging (via OpenRouter / Claude Sonnet):
     Assess each response on objective quality (0-100 scale), independently
     of which model produced it.  Used to check whether self-rating biases
     correlate with actual quality differences.

  2. Perplexity of each response under Llama 70B:
     Measure how surprising each response is for the 70B model.  Lower
     perplexity → more "native" to the model.

Saves:
  results_dir_ex7/judge_quality_ex7.json   — per-response quality scores
  results_dir_ex7/perplexity_ex7.json      — per-response perplexity scores
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .config import Experiment7Config


# ---------------------------------------------------------------------------
# OpenRouter quality judge
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = "You are an objective text quality evaluator. Respond only with valid JSON."

_JUDGE_TEMPLATE = """\
Rate the quality of the following AI response to the given instruction.

Instruction: {instruction}

Response:
---
{response}
---

Evaluate on these criteria:
- Accuracy and helpfulness (does it address the instruction?)
- Clarity and coherence
- Completeness (does it fully answer the question?)

Respond ONLY with valid JSON:
{{"overall_score": <integer 0-100>, "rationale": "<one sentence>"}}
"""


def _call_judge_api(client, record: dict, config: Experiment7Config) -> dict | None:
    prompt = _JUDGE_TEMPLATE.format(
        instruction=record["instruction"][:500],
        response=record["response"][:1500],
    )
    try:
        resp = client.chat.completions.create(
            model=config.judge_model_id,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=config.judge_temperature,
            max_tokens=200,
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except Exception as e:
        print(f"  Judge API error: {e}")
        return None


def _with_retry(fn, max_retries: int = 5, base_delay: float = 30.0):
    for attempt in range(max_retries):
        result = fn()
        if result is not None:
            return result
        if attempt < max_retries - 1:
            wait = base_delay * (2 ** attempt)
            print(f"  Retry {attempt + 1}/{max_retries} after {wait:.0f}s...")
            time.sleep(wait)
    return None


def run_quality_judging(
    records: list[dict],
    config: Experiment7Config,
    force: bool = False,
) -> list[dict]:
    """
    Judge quality of every response (all 3 conditions) via OpenRouter.
    Returns list of {prompt_id, condition, response, overall_score, rationale}.
    """
    out_path = config.results_dir_ex7 / "judge_quality_ex7.json"
    partial_path = config.results_dir_ex7 / "judge_quality_ex7_partial.json"

    if out_path.exists() and not force:
        with open(out_path) as f:
            print(f"  Loaded existing quality judgements from {out_path}.")
            return json.load(f)

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY not set — skipping quality judging.")

    from openai import OpenAI
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    judged: list[dict] = []
    completed: set[tuple] = set()
    if partial_path.exists() and not force:
        with open(partial_path) as f:
            judged = json.load(f)
        completed = {(r["prompt_id"], r["condition"]) for r in judged}
        print(f"  Resuming judging from {len(completed)} completed.")

    # Flatten records into (prompt_id, instruction, condition, response) tuples
    condition_keys = {
        "llama70b": "response_llama70b",
        "llama8b":  "response_llama8b",
        "gemma9b":  "response_gemma9b",
    }
    items: list[dict] = []
    for r in records:
        for condition, resp_key in condition_keys.items():
            if (r["prompt_id"], condition) in completed:
                continue
            response = r.get(resp_key, "")
            if response:
                items.append({
                    "prompt_id": r["prompt_id"],
                    "instruction": r["instruction"],
                    "condition": condition,
                    "response": response,
                })

    print(f"  Judging {len(items)} responses via {config.judge_model_id}...")
    for item in tqdm(items, desc="Judging quality"):
        result = _with_retry(lambda: _call_judge_api(client, item, config))
        if result is None:
            continue
        judged.append({
            "prompt_id": item["prompt_id"],
            "condition": item["condition"],
            "response": item["response"][:200],
            "overall_score": result.get("overall_score"),
            "rationale": result.get("rationale", ""),
        })
        # Save partial after each successful call
        with open(partial_path, "w") as f:
            json.dump(judged, f, indent=2)

    with open(out_path, "w") as f:
        json.dump(judged, f, indent=2)
    if partial_path.exists():
        partial_path.unlink()
    print(f"  Saved {len(judged)} quality scores → {out_path}")
    return judged


# ---------------------------------------------------------------------------
# Perplexity under Llama 70B
# ---------------------------------------------------------------------------

def _compute_perplexity(model, tokenizer, instruction: str, response: str, device) -> float | None:
    """
    Compute mean negative log-prob of response tokens under the model,
    given the instruction as context.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": response},
    ]
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    input_ids = tokenizer(full_text, return_tensors="pt")["input_ids"].to(device)

    # Find response start
    prefix_msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": instruction},
    ]
    prefix_text = tokenizer.apply_chat_template(
        prefix_msgs, tokenize=False, add_generation_prompt=True
    )
    prefix_len = tokenizer(prefix_text, return_tensors="pt")["input_ids"].shape[1]

    if prefix_len >= input_ids.shape[1] - 1:
        return None

    with torch.no_grad():
        out = model(input_ids)
        logits = out.logits  # (1, seq_len, vocab)

    # Log-probs for tokens at positions prefix_len..end
    log_probs: list[float] = []
    for pos in range(prefix_len, input_ids.shape[1]):
        tok_id = input_ids[0, pos].item()
        lp = float(
            F.log_softmax(logits[0, pos - 1].float(), dim=-1)[tok_id].item()
        )
        log_probs.append(lp)

    del out, logits
    if not log_probs:
        return None
    return float(-np.mean(log_probs))  # perplexity in nats (NLL)


def run_perplexity_computation(
    records: list[dict],
    config: Experiment7Config,
    force: bool = False,
) -> list[dict]:
    """
    Compute per-response NLL under Llama 70B for all three conditions.
    Saves results_dir_ex7/perplexity_ex7.json.
    """
    out_path = config.results_dir_ex7 / "perplexity_ex7.json"
    if out_path.exists() and not force:
        with open(out_path) as f:
            print(f"  Loaded existing perplexity results from {out_path}.")
            return json.load(f)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .utils import get_device, get_device_map, clear_device_cache
    import gc

    device_str = get_device()
    device_map = get_device_map()
    print(f"  Loading {config.llama70b_model_id} for perplexity computation...")

    if device_map is not None:
        model = AutoModelForCausalLM.from_pretrained(
            config.llama70b_model_id, torch_dtype=torch.float16, device_map=device_map
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.llama70b_model_id, torch_dtype=torch.float16
        ).to(device_str)

    tokenizer = AutoTokenizer.from_pretrained(config.llama70b_model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    device = next(model.parameters()).device

    condition_keys = {
        "llama70b": "response_llama70b",
        "llama8b":  "response_llama8b",
        "gemma9b":  "response_gemma9b",
    }

    ppl_results: list[dict] = []
    for r in tqdm(records, desc="Perplexity"):
        for condition, resp_key in condition_keys.items():
            response = r.get(resp_key, "")
            if not response:
                continue
            nll = _compute_perplexity(model, tokenizer, r["instruction"], response, device)
            ppl_results.append({
                "prompt_id": r["prompt_id"],
                "condition": condition,
                "nll": nll,
            })

    del model
    gc.collect()
    clear_device_cache()

    with open(out_path, "w") as f:
        json.dump(ppl_results, f, indent=2)
    print(f"  Saved {len(ppl_results)} perplexity scores → {out_path}")
    return ppl_results
