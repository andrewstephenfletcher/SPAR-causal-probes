"""
Per-token log-probability (perplexity) computation for Experiment 1.

For each prompt we compute the log-probability of BOTH responses (target and
source) under the TARGET model.  This gives a perplexity baseline to compare
against the activation probe.
"""

import json
import math

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import Config
from .utils import clear_device_cache, get_device, get_device_map


def _build_prefill_ids(
    tokenizer,
    instruction: str,
    response_text: str,
) -> tuple[torch.Tensor, int, int]:
    """
    Returns (prefill_ids, n_prompt_tokens, n_response_tokens).

    n_prompt_tokens : number of tokens in the prompt-only prefix
                      (system + user + assistant header), obtained by
                      formatting the conversation WITHOUT the response
                      and WITH add_generation_prompt=True.
    n_response_tokens : standalone token count of response_text.

    The response tokens in prefill_ids occupy positions
    [n_prompt_tokens, n_prompt_tokens + n_response_tokens).
    """
    # Full conversation
    full_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": response_text},
    ]
    prefill_text = tokenizer.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False
    )
    prefill_ids = tokenizer(prefill_text, return_tensors="pt")["input_ids"]

    # Prompt-only prefix (to find where the response starts)
    prompt_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": instruction},
    ]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"]
    n_prompt_tokens = prompt_ids.shape[1]

    # Standalone response token count
    response_only_ids = tokenizer(
        response_text, add_special_tokens=False
    )["input_ids"]
    n_response_tokens = len(response_only_ids)

    return prefill_ids, n_prompt_tokens, n_response_tokens


def compute_response_log_probs(
    model,
    tokenizer,
    instruction: str,
    response_text: str,
    device,
) -> dict:
    """
    Compute per-token log-probabilities of response_text under model.

    Returns a dict with:
      mean_log_prob, perplexity, per_token_log_probs (list), n_tokens.
    """
    prefill_ids, n_prompt_tokens, n_response_tokens = _build_prefill_ids(
        tokenizer, instruction, response_text
    )
    prefill_ids = prefill_ids.to(device)

    # Safety: verify the response tokens fit within the prefill
    expected_end = n_prompt_tokens + n_response_tokens
    if expected_end > prefill_ids.shape[1]:
        # Truncate to what's available
        n_response_tokens = prefill_ids.shape[1] - n_prompt_tokens

    with torch.no_grad():
        outputs = model(prefill_ids)
        logits = outputs.logits  # (1, seq_len, vocab_size)

    log_probs: list[float] = []
    for i in range(n_response_tokens):
        # Token at position n_prompt_tokens + i is predicted by logits at n_prompt_tokens + i - 1
        pos = n_prompt_tokens + i
        token_id = prefill_ids[0, pos].item()
        # Cast to float32 before log_softmax for numerical stability
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


def compute_all_perplexity(
    responses: list[dict],
    config: Config,
    force: bool = False,
) -> None:
    """
    Compute perplexity for all prompts under both conditions.
    Saves to activations_dir/perplexity.json.
    """
    output_path = config.activations_dir / "perplexity.json"

    if output_path.exists() and not force:
        print(f"  Found existing perplexity data at {output_path}, skipping.")
        return

    device_str = get_device()
    device_map = get_device_map()

    print(f"  Loading target model (Llama 3.1 8B) on {device_str}...")
    if device_map is not None:
        model = AutoModelForCausalLM.from_pretrained(
            config.target_model_id,
            torch_dtype=torch.float16,
            device_map=device_map,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.target_model_id,
            torch_dtype=torch.float16,
        ).to(device_str)

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(config.target_model_id)
    device = next(model.parameters()).device

    perplexity_records: list[dict] = []

    print("  Computing log-probabilities for both conditions...")
    for r in tqdm(responses, desc="Perplexity"):
        pid = r["prompt_id"]
        instruction = r["instruction"]

        # Self condition (target model's own response)
        self_metrics = compute_response_log_probs(
            model, tokenizer, instruction, r["response_target"], device
        )
        perplexity_records.append({
            "prompt_id": pid,
            "condition": "self",
            **self_metrics,
        })

        # Cross condition (source model's response)
        cross_metrics = compute_response_log_probs(
            model, tokenizer, instruction, r["response_source"], device
        )
        perplexity_records.append({
            "prompt_id": pid,
            "condition": "cross_gemma",
            **cross_metrics,
        })

    with open(output_path, "w") as f:
        json.dump(perplexity_records, f, indent=2)

    print(f"  Saved perplexity data → {output_path}")

    del model
    clear_device_cache()
