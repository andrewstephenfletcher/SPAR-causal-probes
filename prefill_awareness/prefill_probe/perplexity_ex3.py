"""
Perplexity computation for Experiment 3.

For every (prompt, condition) pair, compute the mean per-token log-probability
of the response under Llama 8B Instruct.  Reuses compute_response_log_probs
from Experiment 1.

Output: activations_dir_ex3 / "perplexity_all.json"

Each record:
  {
    "prompt_id":     str,
    "dataset":       str,
    "split":         str,
    "condition":     str,
    "mean_log_prob": float,
    "perplexity":    float,
    "n_tokens":      int
  }
"""

import json

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import Config, Experiment3Config
from .generate_ex3 import CONDITIONS
from .perplexity import compute_response_log_probs
from .utils import clear_device_cache, get_device, get_device_map


def compute_all_perplexity_ex3(
    responses: list[dict],
    ex3_config: Experiment3Config,
    ex1_config: Config,
    force: bool = False,
) -> None:
    """
    Compute perplexity for all (prompt, condition) pairs.
    Saves to activations_dir_ex3 / "perplexity_all.json".
    """
    out_path = ex3_config.activations_dir_ex3 / "perplexity_all.json"

    if out_path.exists() and not force:
        print(f"  Found existing perplexity data at {out_path}, skipping.")
        return

    device_str = get_device()
    device_map = get_device_map()

    print(f"  Loading Llama 8B ({ex1_config.target_model_id}) for perplexity on {device_str}...")
    if device_map is not None:
        model = AutoModelForCausalLM.from_pretrained(
            ex1_config.target_model_id,
            torch_dtype=torch.float16,
            device_map=device_map,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            ex1_config.target_model_id,
            torch_dtype=torch.float16,
        ).to(device_str)

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(ex1_config.target_model_id)
    device = next(model.parameters()).device

    records: list[dict] = []
    n_total = len(responses) * len(CONDITIONS)

    with tqdm(total=n_total, desc="Perplexity") as pbar:
        for r in responses:
            for condition in CONDITIONS:
                resp_text = r["responses"].get(condition, "").rstrip()
                if not resp_text:
                    pbar.update(1)
                    continue

                metrics = compute_response_log_probs(
                    model, tokenizer, r["instruction"], resp_text, device
                )
                records.append({
                    "prompt_id":     r["prompt_id"],
                    "dataset":       r["dataset"],
                    "split":         r["split"],
                    "condition":     condition,
                    "mean_log_prob": metrics["mean_log_prob"],
                    "perplexity":    metrics["perplexity"],
                    "n_tokens":      metrics["n_tokens"],
                })
                pbar.update(1)

    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"  Saved {len(records)} perplexity records → {out_path}")

    del model
    clear_device_cache()
