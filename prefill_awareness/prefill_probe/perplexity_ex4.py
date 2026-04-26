"""
Per-token log-probability (perplexity) computation for Experiment 4.

For each target model and each condition, computes the log-probability of the
response under the TARGET model.  The result is used as a single-feature
baseline AUROC for comparison against the linear probe.

Llama 70B perplexity file: activations_dir_llama70b/perplexity.json
Gemma 31B perplexity file: activations_dir_gemma31b/perplexity.json

Each JSON is a list[dict] with keys:
  prompt_id, condition, mean_log_prob, perplexity, per_token_log_probs, n_tokens
"""

import json
import math

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import Experiment4Config
from .utils import clear_device_cache, get_device, get_device_map


def _supports_system_prompt(model_id: str) -> bool:
    return "gemma-2" not in model_id.lower()


def _build_prefill_ids(
    tokenizer,
    model_id: str,
    instruction: str,
    response_text: str,
) -> tuple[torch.Tensor, int, int]:
    """
    Build full prefill token tensor and locate where the response starts.

    Returns (prefill_ids, n_prompt_tokens, n_response_tokens).
    n_prompt_tokens is the length of the prompt-only prefix so that
    response tokens occupy positions [n_prompt_tokens, n_prompt_tokens + n_response_tokens).
    """
    if _supports_system_prompt(model_id):
        full_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response_text},
        ]
        prompt_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": instruction},
        ]
    else:
        full_messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response_text},
        ]
        prompt_messages = [
            {"role": "user", "content": instruction},
        ]

    prefill_text = tokenizer.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False
    )
    prefill_ids = tokenizer(prefill_text, return_tensors="pt")["input_ids"]

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"]
    n_prompt_tokens = prompt_ids.shape[1]

    response_only_ids = tokenizer(
        response_text, add_special_tokens=False
    )["input_ids"]
    n_response_tokens = len(response_only_ids)

    return prefill_ids, n_prompt_tokens, n_response_tokens


def compute_response_log_probs(
    model,
    tokenizer,
    model_id: str,
    instruction: str,
    response_text: str,
    device,
) -> dict:
    """
    Compute per-token log-probabilities of response_text under model.

    Returns dict with mean_log_prob, perplexity, per_token_log_probs, n_tokens.
    """
    prefill_ids, n_prompt_tokens, n_response_tokens = _build_prefill_ids(
        tokenizer, model_id, instruction, response_text
    )
    prefill_ids = prefill_ids.to(device)

    # Safety clip if template truncation occurs
    n_response_tokens = min(
        n_response_tokens,
        prefill_ids.shape[1] - n_prompt_tokens,
    )

    with torch.no_grad():
        outputs = model(prefill_ids)
        logits = outputs.logits  # (1, seq_len, vocab_size)

    log_probs: list[float] = []
    for i in range(n_response_tokens):
        pos = n_prompt_tokens + i
        token_id = prefill_ids[0, pos].item()
        lp = F.log_softmax(logits[0, pos - 1].float(), dim=-1)[token_id].item()
        log_probs.append(lp)

    if log_probs:
        mean_log_prob = sum(log_probs) / len(log_probs)
        perplexity = math.exp(-mean_log_prob)
    else:
        mean_log_prob = float("nan")
        perplexity = float("nan")

    return {
        "mean_log_prob": mean_log_prob,
        "perplexity": perplexity,
        "per_token_log_probs": log_probs,
        "n_tokens": n_response_tokens,
    }


def _compute_perplexity_for_model(
    target_model_id: str,
    conditions: dict[str, str],
    responses: list[dict],
    output_path,
    force: bool = False,
) -> None:
    """
    Compute perplexity for all (prompt, condition) pairs under target_model_id.

    conditions: dict mapping condition name → response field in responses dict.
    """
    if output_path.exists() and not force:
        print(f"  Found existing perplexity data at {output_path}, skipping.")
        return

    device_str = get_device()
    device_map = get_device_map()

    print(f"  Loading {target_model_id} (fp16) on {device_str}...")
    if device_map is not None:
        model = AutoModelForCausalLM.from_pretrained(
            target_model_id,
            torch_dtype=torch.float16,
            device_map=device_map,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            target_model_id,
            torch_dtype=torch.float16,
        ).to(device_str)

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(target_model_id)
    device = next(model.parameters()).device

    records: list[dict] = []

    for r in tqdm(responses, desc=f"Perplexity ({target_model_id.split('/')[-1]})"):
        pid = r["prompt_id"]
        for condition_name, response_key in conditions.items():
            metrics = compute_response_log_probs(
                model, tokenizer, target_model_id,
                r["instruction"], r[response_key],
                device,
            )
            records.append({"prompt_id": pid, "condition": condition_name, **metrics})

    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"  Saved perplexity data ({len(records)} records) → {output_path}")

    del model
    clear_device_cache()


def compute_all_perplexity(
    responses: list[dict],
    config: Experiment4Config,
    force: bool = False,
) -> None:
    """Compute perplexity for Llama 70B and Gemma 31B, all conditions."""
    print("\n--- Perplexity: Llama 3.3 70B ---")
    _compute_perplexity_for_model(
        target_model_id=config.llama70b_model_id,
        conditions={
            "self_prefill":  "response_llama70b",
            "cross_gemma9b": "response_gemma9b",
            "cross_llama8b": "response_llama8b",
        },
        responses=responses,
        output_path=config.activations_dir_llama70b / "perplexity.json",
        force=force,
    )

    print("\n--- Perplexity: Gemma 4 31B ---")
    _compute_perplexity_for_model(
        target_model_id=config.gemma31b_model_id,
        conditions={
            "self_prefill":  "response_gemma31b",
            "cross_llama8b": "response_llama8b",
            "cross_gemma9b": "response_gemma9b",
        },
        responses=responses,
        output_path=config.activations_dir_gemma31b / "perplexity.json",
        force=force,
    )
