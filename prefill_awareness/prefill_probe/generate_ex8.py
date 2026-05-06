"""
Response generation for Experiment 8 (Cross-Architecture Probing).

Generates responses from Mistral Small 3.2 24B and Gemma 4 31B for the same
300 Alpaca instructions used in Experiment 1.  Llama 8B responses from Ex1
serve as the cross-model source for both new models.

Saves:
  generations_dir_ex8/responses_mistral.json
  generations_dir_ex8/responses_gemma31b.json

Each file is a list of dicts:
  {prompt_id, instruction, split, response_self, response_cross_llama8b,
   self_response_tokens, cross_response_tokens}
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from tqdm import tqdm

from .config import Experiment8Config


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ex1_data(config: Experiment8Config) -> list[dict]:
    """Load Ex1 responses — instruction, split, and Llama 8B response."""
    resp_path = config.ex1_generations_dir / "responses.json"
    if not resp_path.exists():
        raise FileNotFoundError(
            f"Experiment 1 responses not found at {resp_path}. "
            "Run Experiment 1 first."
        )
    with open(resp_path) as f:
        records = json.load(f)
    return records[: config.n_prompts]


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def _load_model_and_tokenizer(model_id: str, seed: int):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .utils import get_device, get_device_map
    import gc

    device_str = get_device()
    device_map = get_device_map()
    print(f"  Loading {model_id} (fp16) on {device_str}...")

    if device_map is not None:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map=device_map
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16
        ).to(device_str)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    torch.manual_seed(seed)
    return model, tokenizer


def _unload(model) -> None:
    import gc
    from .utils import clear_device_cache
    del model
    gc.collect()
    clear_device_cache()


# ---------------------------------------------------------------------------
# Response generation
# ---------------------------------------------------------------------------

def _generate_response(
    model,
    tokenizer,
    instruction: str,
    config: Experiment8Config,
    is_gemma: bool = False,
) -> str | None:
    """
    Generate one response for the given instruction.
    Returns None if generation fails or the response is too short.
    """
    messages = [
        {"role": "user", "content": instruction},
    ]

    # Gemma 4 thinking mode must be disabled to get a deterministic, non-thinking response.
    # Pass enable_thinking=False via extra kwargs if the tokenizer supports it.
    template_kwargs: dict = {}
    if is_gemma:
        template_kwargs["enable_thinking"] = False

    try:
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
    except TypeError:
        # Older tokenizer without enable_thinking — try without it
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    input_ids = tokenizer(input_text, return_tensors="pt")["input_ids"]
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            do_sample=True,
        )

    new_tokens = output_ids[0, input_ids.shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return response if response else None


# ---------------------------------------------------------------------------
# Main generation functions
# ---------------------------------------------------------------------------

def generate_responses_for_model(
    model_id: str,
    out_path: Path,
    ex1_data: list[dict],
    config: Experiment8Config,
    force: bool = False,
    is_gemma: bool = False,
) -> list[dict]:
    """Generate responses for one model, save to out_path."""
    if out_path.exists() and not force:
        print(f"  Found existing responses at {out_path}, skipping.")
        with open(out_path) as f:
            return json.load(f)

    model, tokenizer = _load_model_and_tokenizer(model_id, config.seed)
    records: list[dict] = []
    skipped = 0

    print(f"  Generating {len(ex1_data)} responses with {model_id}...")
    for r in tqdm(ex1_data, desc=f"Generating ({model_id.split('/')[-1]})"):
        response = _generate_response(model, tokenizer, r["instruction"], config, is_gemma=is_gemma)
        if response is None:
            skipped += 1
            continue

        cross_resp = r.get("response_target", "")  # Llama 8B response from Ex1
        cross_tokens = r.get("target_response_tokens", 0)
        self_tokens = len(tokenizer.encode(response))

        if self_tokens < config.min_response_tokens or cross_tokens < config.min_response_tokens:
            skipped += 1
            continue

        records.append({
            "prompt_id": r["prompt_id"],
            "instruction": r["instruction"],
            "split": r["split"],
            "response_self": response,
            "response_cross_llama8b": cross_resp,
            "self_response_tokens": self_tokens,
            "cross_response_tokens": cross_tokens,
        })

    _unload(model)
    print(f"  Generated {len(records)} records ({skipped} skipped).")

    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"  Saved → {out_path}")
    return records


def run_generation_ex8(config: Experiment8Config, model: str = "both", force: bool = False):
    """
    Generate responses for Mistral and/or Gemma 31B.
    model: "mistral" | "gemma" | "both"
    """
    ex1_data = load_ex1_data(config)
    print(f"  Loaded {len(ex1_data)} Alpaca prompts from Experiment 1.")

    if model in ("mistral", "both"):
        out = config.generations_dir_ex8 / "responses_mistral.json"
        generate_responses_for_model(
            config.mistral_model_id, out, ex1_data, config, force=force, is_gemma=False
        )

    if model in ("gemma", "both"):
        out = config.generations_dir_ex8 / "responses_gemma31b.json"
        generate_responses_for_model(
            config.gemma31b_model_id, out, ex1_data, config, force=force, is_gemma=True
        )
