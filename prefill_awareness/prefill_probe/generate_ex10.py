"""
Response generation for Experiment 10 (Probe Generalisation).

Generates responses from four source models across three prompt datasets:
  - Self:  Llama 3.3 70B (the target model generates its own responses)
  - Cross: Llama 3.1 8B, Gemma 4 31B, Mistral Small 24B

Output per dataset — outputs/experiment10/generations/{dataset}_responses.json —
has one record per prompt with keys:
  prompt_id, dataset, instruction, split,
  response_llama70b, response_llama8b, response_gemma31b, response_mistral24b

IMPORTANT: max_new_tokens=512 — no response truncation.
"""

import gc
import json

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import Experiment10Config
from .data_ex10 import load_all_prompts
from .utils import clear_device_cache, get_device, get_device_map


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


def _format_prompt(tokenizer, model_id: str, instruction: str) -> str:
    supports_system = "gemma-2" not in model_id.lower()
    if supports_system:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": instruction},
        ]
    else:
        messages = [{"role": "user", "content": instruction}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _generate_response(
    model, tokenizer, input_text: str, config: Experiment10Config,
) -> str:
    device = next(model.parameters()).device
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    torch.manual_seed(config.seed)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            do_sample=True,
        )
    n_input = inputs["input_ids"].shape[1]
    return tokenizer.decode(output[0][n_input:], skip_special_tokens=True)


def _count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def _generate_for_model(
    model_id: str,
    response_key: str,
    label: str,
    all_prompts: dict[str, list[dict]],
    results_per_dataset: dict[str, dict[int, dict]],
    config: Experiment10Config,
) -> None:
    """Load one source model, generate responses for all datasets, unload."""
    total_needed = sum(
        sum(
            1 for p in prompts
            if response_key not in results_per_dataset[ds].get(p["prompt_id"], {})
        )
        for ds, prompts in all_prompts.items()
    )
    if total_needed == 0:
        print(f"  {label}: all responses already present, skipping.")
        return

    print(f"  Loading {model_id}...")
    model, tokenizer = _load_model_and_tokenizer(model_id)
    model.eval()

    for ds_name, prompts in all_prompts.items():
        partial_path = config.generations_dir / f"{ds_name}_{response_key}_partial.json"

        done: set[int] = {
            pid for pid, r in results_per_dataset[ds_name].items()
            if response_key in r
        }

        if partial_path.exists():
            with open(partial_path) as f:
                partial = json.load(f)
            for pid_str, text in partial.items():
                pid = int(pid_str)
                if pid not in results_per_dataset[ds_name]:
                    results_per_dataset[ds_name][pid] = {}
                results_per_dataset[ds_name][pid][response_key] = text
                done.add(pid)
            print(f"    [{ds_name}] Resuming {label} from {len(done)}/{len(prompts)}")

        remaining = [p for p in prompts if p["prompt_id"] not in done]
        if not remaining:
            print(f"    [{ds_name}] {label}: already complete.")
            continue

        for prompt in tqdm(remaining, desc=f"    {label} / {ds_name}"):
            pid = prompt["prompt_id"]
            input_text = _format_prompt(tokenizer, model_id, prompt["instruction"])
            response_text = _generate_response(model, tokenizer, input_text, config)

            if pid not in results_per_dataset[ds_name]:
                results_per_dataset[ds_name][pid] = {
                    "prompt_id": pid,
                    "dataset": ds_name,
                    "instruction": prompt["instruction"],
                    "split": prompt["split"],
                }
            results_per_dataset[ds_name][pid][response_key] = response_text
            done.add(pid)

            if len(done) % config.checkpoint_interval == 0:
                partial_data = {
                    str(p): results_per_dataset[ds_name][p][response_key]
                    for p in done
                    if response_key in results_per_dataset[ds_name].get(p, {})
                }
                with open(partial_path, "w") as f:
                    json.dump(partial_data, f)

        partial_data = {
            str(p): results_per_dataset[ds_name][p][response_key]
            for p in done
            if response_key in results_per_dataset[ds_name].get(p, {})
        }
        with open(partial_path, "w") as f:
            json.dump(partial_data, f)

    del model
    clear_device_cache()
    print(f"  {label}: generation complete.")


def generate_all_responses(
    config: Experiment10Config,
    force: bool = False,
) -> dict[str, list[dict]]:
    """
    Generate responses from all 4 source models for all 3 datasets.
    Returns {dataset_name: [response_dicts]}.
    """
    required_keys = [
        "response_llama70b", "response_llama8b",
        "response_gemma31b", "response_mistral24b",
    ]

    # Check if all outputs are already complete
    if not force:
        all_complete = True
        for ds in config.datasets:
            path = config.generations_dir / f"{ds}_responses.json"
            if not path.exists():
                all_complete = False
                break
            with open(path) as f:
                data = json.load(f)
            if not data or not all(all(k in r for k in required_keys) for r in data):
                all_complete = False
                break

        if all_complete:
            print("  All generation outputs found, loading...")
            return {
                ds: json.load(open(config.generations_dir / f"{ds}_responses.json"))
                for ds in config.datasets
            }

    print("  Loading prompt datasets...")
    all_prompts = load_all_prompts(config)

    # Working dict: {dataset: {pid: record}}
    results_per_dataset: dict[str, dict[int, dict]] = {ds: {} for ds in config.datasets}

    for ds in config.datasets:
        path = config.generations_dir / f"{ds}_responses.json"
        if path.exists() and not force:
            with open(path) as f:
                existing = json.load(f)
            for r in existing:
                results_per_dataset[ds][r["prompt_id"]] = r
            print(f"  Loaded {len(existing)} existing records for {ds}.")
        else:
            for p in all_prompts[ds]:
                pid = p["prompt_id"]
                results_per_dataset[ds][pid] = {
                    "prompt_id": pid,
                    "dataset": ds,
                    "instruction": p["instruction"],
                    "split": p["split"],
                }

    source_models = [
        (config.target_model_id,    "response_llama70b",    "Llama 3.3 70B (self)"),
        (config.llama8b_model_id,   "response_llama8b",     "Llama 3.1 8B"),
        (config.gemma31b_model_id,  "response_gemma31b",    "Gemma 4 31B"),
        (config.mistral24b_model_id, "response_mistral24b", "Mistral Small 24B"),
    ]

    for model_id, response_key, label in source_models:
        print(f"\n  --- {label} ---")
        _generate_for_model(
            model_id=model_id,
            response_key=response_key,
            label=label,
            all_prompts=all_prompts,
            results_per_dataset=results_per_dataset,
            config=config,
        )

    print("\n  Loading Llama tokenizer for token-count filtering...")
    ref_tokenizer = AutoTokenizer.from_pretrained(config.target_model_id)

    outputs: dict[str, list[dict]] = {}
    for ds in config.datasets:
        valid, discarded = [], []
        for pid, r in results_per_dataset[ds].items():
            if not all(k in r for k in required_keys):
                discarded.append(pid)
                continue
            min_toks = min(_count_tokens(ref_tokenizer, r[k]) for k in required_keys)
            if min_toks >= config.min_response_tokens:
                valid.append(r)
            else:
                discarded.append(pid)

        valid.sort(key=lambda r: r["prompt_id"])
        out_path = config.generations_dir / f"{ds}_responses.json"
        with open(out_path, "w") as f:
            json.dump(valid, f, indent=2)
        print(f"  [{ds}] Retained {len(valid)} / {len(results_per_dataset[ds])} "
              f"(discarded {len(discarded)}) → {out_path}")

        for _, response_key, _ in source_models:
            partial = config.generations_dir / f"{ds}_{response_key}_partial.json"
            if partial.exists():
                partial.unlink()

        outputs[ds] = valid

    del ref_tokenizer
    gc.collect()
    return outputs


def load_all_dataset_responses(
    config: Experiment10Config,
) -> dict[str, list[dict]]:
    """Load pre-generated responses from disk. Raises if any file is missing."""
    outputs = {}
    for ds in config.datasets:
        path = config.generations_dir / f"{ds}_responses.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Responses for dataset '{ds}' not found at {path}. "
                "Run the generate step first."
            )
        with open(path) as f:
            outputs[ds] = json.load(f)
    return outputs
