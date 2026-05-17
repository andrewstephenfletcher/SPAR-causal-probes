"""
Response generation for Experiment 11 (Unified Prefill-Awareness Data Collection).

Generates responses from 6 models (3 large targets + 3 small sources) across 3 datasets.

Key difference vs Exp 4/10: suffix normalisation.
  - response_raw       : decoded text exactly as produced (skip_special_tokens=True)
  - response_normalized: strip trailing whitespace; append '.' if not ending with '.'
  All activation extraction uses response_normalized to eliminate the last-token confound.

Output per (model, dataset):
  outputs/experiment11/generations/{model_key}_{dataset}_responses.json

Each record:
  prompt_id, dataset, instruction, split,
  response_{model_key}, response_{model_key}_normalized

Generation parameters: temp=0.6, top_p=0.9, max_new_tokens=512, seed=42, batch_size=4.
System prompt "You are a helpful assistant." for all models (all are non-gemma-2).
"""

import gc
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import Experiment11Config
from .data_ex11 import load_all_prompts
from .utils import clear_device_cache, get_device, get_device_map


# ---------------------------------------------------------------------------
# Suffix normalisation
# ---------------------------------------------------------------------------

def normalize_response(raw: str) -> str:
    """Strip trailing whitespace/newlines; append '.' if text doesn't end with one."""
    text = raw.rstrip()
    if text and not text.endswith("."):
        text = text + "."
    return text


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model_and_tokenizer(model_id: str):
    device = get_device()
    device_map = get_device_map()
    if device_map is not None:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map=device_map,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16,
        ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


# ---------------------------------------------------------------------------
# Chat template
# ---------------------------------------------------------------------------

def _format_prompt(tokenizer, model_id: str, instruction: str) -> str:
    """All Exp 11 models support system prompts (none are gemma-2)."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": instruction},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ---------------------------------------------------------------------------
# Batched generation
# ---------------------------------------------------------------------------

def _generate_batch(
    model, tokenizer, input_texts: list[str], config: Experiment11Config,
) -> list[str]:
    device = next(model.parameters()).device
    inputs = tokenizer(
        input_texts, return_tensors="pt", padding=True, truncation=False,
    ).to(device)
    torch.manual_seed(config.seed)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            do_sample=True,
        )
    n_input = inputs["input_ids"].shape[1]
    return [
        tokenizer.decode(outputs[i][n_input:], skip_special_tokens=True)
        for i in range(len(input_texts))
    ]


def _count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


# ---------------------------------------------------------------------------
# Per-model generation with per-(model, dataset) output files
# ---------------------------------------------------------------------------

def _generate_for_model(
    model_id: str,
    model_key: str,
    label: str,
    all_prompts: dict[str, list[dict]],
    config: Experiment11Config,
    force: bool = False,
) -> dict[str, dict[int, dict]]:
    """
    Generate responses for one model across all datasets.

    Returns {dataset: {prompt_id: record}} where each record contains
    response_{model_key} and response_{model_key}_normalized.
    """
    raw_key  = f"response_{model_key}"
    norm_key = f"response_{model_key}_normalized"

    # Check if all outputs are already complete
    if not force:
        all_done = True
        for ds in config.datasets:
            path = config.generations_dir / f"{model_key}_{ds}_responses.json"
            if not path.exists():
                all_done = False
                break
            with open(path) as f:
                data = json.load(f)
            if not data or not all(raw_key in r and norm_key in r for r in data):
                all_done = False
                break
        if all_done:
            print(f"  {label}: all outputs found, loading from disk.")
            result = {}
            for ds in config.datasets:
                path = config.generations_dir / f"{model_key}_{ds}_responses.json"
                with open(path) as f:
                    records = json.load(f)
                result[ds] = {r["prompt_id"]: r for r in records}
            return result

    # Load partial checkpoints
    results_per_dataset: dict[str, dict[int, dict]] = {}
    for ds in config.datasets:
        results_per_dataset[ds] = {}
        partial_path = config.generations_dir / f"{model_key}_{ds}_partial.json"
        if partial_path.exists() and not force:
            with open(partial_path) as f:
                partial = json.load(f)
            for pid_str, entry in partial.items():
                results_per_dataset[ds][int(pid_str)] = entry

    total_needed = sum(
        sum(1 for p in prompts if p["prompt_id"] not in results_per_dataset[ds])
        for ds, prompts in all_prompts.items()
    )
    if total_needed == 0:
        print(f"  {label}: all responses in partials, skipping model load.")
    else:
        print(f"  Loading {model_id} ({total_needed} responses needed)...")
        model, tokenizer = _load_model_and_tokenizer(model_id)
        model.eval()

        for ds_name, prompts in all_prompts.items():
            partial_path = config.generations_dir / f"{model_key}_{ds_name}_partial.json"
            done: set[int] = set(results_per_dataset[ds_name].keys())
            remaining = [p for p in prompts if p["prompt_id"] not in done]

            if not remaining:
                print(f"    [{ds_name}] {label}: already complete.")
                continue
            if done:
                print(f"    [{ds_name}] Resuming {label} from {len(done)}/{len(prompts)}")

            bs = config.batch_size
            for batch_start in tqdm(range(0, len(remaining), bs), desc=f"    {label}/{ds_name}"):
                batch = remaining[batch_start : batch_start + bs]
                input_texts = [_format_prompt(tokenizer, model_id, p["instruction"]) for p in batch]
                raw_responses = _generate_batch(model, tokenizer, input_texts, config)

                for prompt, raw in zip(batch, raw_responses):
                    pid = prompt["prompt_id"]
                    norm = normalize_response(raw)
                    results_per_dataset[ds_name][pid] = {
                        "prompt_id":  pid,
                        "dataset":    ds_name,
                        "instruction": prompt["instruction"],
                        "split":      prompt["split"],
                        raw_key:  raw,
                        norm_key: norm,
                    }
                    done.add(pid)

                if len(done) % config.checkpoint_interval == 0:
                    with open(partial_path, "w") as f:
                        json.dump(
                            {str(pid): results_per_dataset[ds_name][pid] for pid in done},
                            f,
                        )

            # Final partial save
            with open(partial_path, "w") as f:
                json.dump(
                    {str(pid): results_per_dataset[ds_name][pid] for pid in done},
                    f,
                )

        del model
        clear_device_cache()
        gc.collect()

    # Filter by min_response_tokens and write final files
    # Use a lightweight tokenizer for token counting (load once)
    print(f"  {label}: filtering and saving...")
    ref_tokenizer = AutoTokenizer.from_pretrained(model_id)

    for ds_name, prompts in all_prompts.items():
        partial_path = config.generations_dir / f"{model_key}_{ds_name}_partial.json"
        out_path = config.generations_dir / f"{model_key}_{ds_name}_responses.json"

        valid, discarded = [], []
        for p in prompts:
            pid = p["prompt_id"]
            r = results_per_dataset[ds_name].get(pid)
            if r is None or raw_key not in r:
                discarded.append(pid)
                continue
            n_toks = _count_tokens(ref_tokenizer, r[norm_key])
            if n_toks >= config.min_response_tokens:
                valid.append(r)
            else:
                discarded.append(pid)

        valid.sort(key=lambda r: r["prompt_id"])
        with open(out_path, "w") as f:
            json.dump(valid, f, indent=2)
        print(
            f"  [{ds_name}] {label}: {len(valid)} retained "
            f"({len(discarded)} discarded) → {out_path}"
        )

        if partial_path.exists():
            partial_path.unlink()

    del ref_tokenizer
    gc.collect()
    return results_per_dataset


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_all_responses(
    config: Experiment11Config,
    force: bool = False,
    target_model: str = "all",
) -> dict[str, dict[str, list[dict]]]:
    """
    Generate responses from all 6 models for all 3 datasets.

    Returns {model_key: {dataset: [records]}}.

    target_model controls which model(s) to generate for:
      "all"     — all 6 models
      "small"   — 3 small source models only
      "llama70b" / "gemma31b" / "qwen32b"   — one large target model + small sources
      "llama8b" / "gemma4b" / "qwen7b"       — one small source model only
    """
    print("  Loading prompt datasets...")
    all_prompts = load_all_prompts(config)

    # Determine which models to run
    all_model_keys = {k for k, _ in config.target_models + config.source_models}
    if target_model == "all":
        keys_to_run = all_model_keys
    elif target_model == "small":
        keys_to_run = {k for k, _ in config.source_models}
    elif target_model in all_model_keys:
        # Run the requested model; always include source models (needed for cross conditions)
        source_keys = {k for k, _ in config.source_models}
        keys_to_run = source_keys | {target_model}
    else:
        raise ValueError(f"Unknown target_model: {target_model!r}")

    all_model_ids = dict(config.target_models + config.source_models)

    labels = {
        "llama70b": "Llama 3.3 70B",
        "gemma31b": "Gemma 4 31B",
        "qwen32b":  "Qwen 32B",
        "llama8b":  "Llama 3.1 8B",
        "gemma4b":  "Gemma 4 4B",
        "qwen7b":   "Qwen 7B",
    }

    # Run small models first (they free VRAM before loading large models)
    ordered_keys = (
        [k for k, _ in config.source_models if k in keys_to_run]
        + [k for k, _ in config.target_models if k in keys_to_run]
    )

    all_results: dict[str, dict[str, list[dict]]] = {}
    for model_key in ordered_keys:
        model_id = all_model_ids[model_key]
        print(f"\n  --- {labels[model_key]} ({model_key}) ---")
        per_ds = _generate_for_model(
            model_id=model_id,
            model_key=model_key,
            label=labels[model_key],
            all_prompts=all_prompts,
            config=config,
            force=force,
        )
        all_results[model_key] = {
            ds: sorted(per_ds[ds].values(), key=lambda r: r["prompt_id"])
            for ds in config.datasets
        }

    return all_results


def load_all_responses(
    config: Experiment11Config,
) -> dict[str, dict[str, list[dict]]]:
    """Load all pre-generated response files. Raises if any file is missing."""
    all_model_ids = dict(config.target_models + config.source_models)
    result: dict[str, dict[str, list[dict]]] = {}
    for model_key in all_model_ids:
        result[model_key] = {}
        for ds in config.datasets:
            path = config.generations_dir / f"{model_key}_{ds}_responses.json"
            if not path.exists():
                raise FileNotFoundError(
                    f"Responses missing: {path}. Run the generate step first."
                )
            with open(path) as f:
                result[model_key][ds] = json.load(f)
    return result
