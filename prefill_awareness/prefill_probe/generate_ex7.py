"""
Data preparation for Experiment 7 (Implicit Behavioral Measures).

Assembles the three response conditions for Llama 70B self-rating:
  - llama70b:  Llama 70B's own response (reused from Ex4 if available)
  - llama8b:   Llama 8B response (from Ex1 response_target)
  - gemma9b:   Gemma 9B response (from Ex1 response_source)

If Ex4 responses are not available, generates Llama 70B responses on-the-fly.

Saves:
  generations_dir_ex7/responses_ex7.json

Each record:
  {prompt_id, instruction, split,
   response_llama70b, response_llama8b, response_gemma9b}
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from tqdm import tqdm

from .config import Experiment7Config


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_ex1(config: Experiment7Config) -> dict[int, dict]:
    """Load Ex1 records keyed by prompt_id."""
    path = config.ex1_generations_dir / "responses.json"
    if not path.exists():
        raise FileNotFoundError(f"Experiment 1 responses not found at {path}.")
    with open(path) as f:
        records = json.load(f)
    return {r["prompt_id"]: r for r in records}


def _load_ex4(config: Experiment7Config) -> dict[int, dict] | None:
    """Load Ex4 records keyed by prompt_id, or None if Ex4 was not run."""
    path = config.ex4_generations_dir / "responses.json"
    if not path.exists():
        return None
    with open(path) as f:
        records = json.load(f)
    return {r["prompt_id"]: r for r in records}


# ---------------------------------------------------------------------------
# Llama 70B generation (fallback if Ex4 not available)
# ---------------------------------------------------------------------------

def _generate_llama70b_responses(
    instructions: list[tuple[int, str, str]],  # (prompt_id, instruction, split)
    config: Experiment7Config,
) -> dict[int, str]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .utils import get_device, get_device_map, clear_device_cache
    import gc

    device_str = get_device()
    device_map = get_device_map()
    print(f"  Loading {config.llama70b_model_id} for generation on {device_str}...")

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

    responses: dict[int, str] = {}
    for pid, instruction, _ in tqdm(instructions, desc="Generating Llama 70B"):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": instruction},
        ]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = tokenizer(input_text, return_tensors="pt")["input_ids"].to(device)

        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                do_sample=True,
            )

        new_tokens = out[0, input_ids.shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        responses[pid] = response

    del model
    gc.collect()
    clear_device_cache()
    return responses


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def prepare_ex7_data(config: Experiment7Config, force: bool = False) -> list[dict]:
    """
    Assemble three-condition rating dataset and save to generations_dir_ex7.
    Returns the assembled list of records.
    """
    out_path = config.generations_dir_ex7 / "responses_ex7.json"
    if out_path.exists() and not force:
        print(f"  Found existing Ex7 responses at {out_path}, loading.")
        with open(out_path) as f:
            return json.load(f)

    ex1 = _load_ex1(config)
    ex4 = _load_ex4(config)

    if ex4 is None:
        print("  Experiment 4 responses not found — will generate Llama 70B responses.")

    # Use the intersection of prompt IDs from Ex1 (and Ex4 if available)
    pids = list(ex1.keys())
    if ex4 is not None:
        pids = [p for p in pids if p in ex4]

    # Limit to n_prompts
    pids = pids[: config.n_prompts]

    # If we need to generate 70B responses
    if ex4 is None:
        instructions = [(pid, ex1[pid]["instruction"], ex1[pid]["split"]) for pid in pids]
        llama70b_by_pid = _generate_llama70b_responses(instructions, config)
    else:
        llama70b_by_pid = {pid: ex4[pid]["response_llama70b"] for pid in pids if pid in ex4}

    records: list[dict] = []
    for pid in pids:
        r1 = ex1[pid]
        r70 = llama70b_by_pid.get(pid)
        if r70 is None:
            continue

        records.append({
            "prompt_id": pid,
            "instruction": r1["instruction"],
            "split": r1["split"],
            "response_llama70b": r70,
            "response_llama8b": r1.get("response_target", ""),
            "response_gemma9b": r1.get("response_source", ""),
        })

    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"  Saved {len(records)} records → {out_path}")
    return records
