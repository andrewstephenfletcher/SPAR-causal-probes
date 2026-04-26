"""
Response generation for Experiment 4 (Scaling Analysis).

Loads existing Experiment 1 responses (Llama 3.1 8B + Gemma 2 9B) and
generates new responses from Llama 3.3 70B and Gemma 4 31B for the same
300 prompts.

Output — outputs/experiment4/generations/responses.json — has one record
per prompt with fields:
    prompt_id, instruction, split,
    response_llama8b, response_gemma9b,   (loaded from Exp 1)
    response_llama70b, response_gemma31b  (generated here)
"""

import gc
import json

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import Experiment4Config
from .utils import clear_device_cache, get_device, get_device_map


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def _load_model_and_tokenizer(model_id: str) -> tuple:
    """Load a causal LM and tokenizer, preferring fp16 on CUDA, else fp16 on MPS/CPU."""
    device = get_device()
    device_map = get_device_map()

    if device_map is not None:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map=device_map,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
        ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def _supports_system_prompt(model_id: str) -> bool:
    """
    Gemma 2 (gemma-2-*) does not support the system role in its chat template.
    All other models used here (Llama 3.x, Gemma 4) support system prompts.
    """
    return "gemma-2" not in model_id.lower()


def _format_generation_prompt(tokenizer, instruction: str, model_id: str) -> str:
    """Apply the model's chat template to produce a generation prompt string."""
    if _supports_system_prompt(model_id):
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
    model,
    tokenizer,
    input_text: str,
    config: Experiment4Config,
) -> str:
    """Generate a single response and return the response text."""
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
    response_ids = output[0][n_input:]
    return tokenizer.decode(response_ids, skip_special_tokens=True)


def _count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_all_responses(
    config: Experiment4Config,
    force: bool = False,
) -> list[dict]:
    """
    Produce experiment4 responses.json by:
      1. Loading Exp 1 responses for llama8b + gemma9b responses and prompts.
      2. Generating Llama 3.3 70B responses for the same prompts.
      3. Generating Gemma 4 31B responses for the same prompts.

    Returns the merged list of response dicts (one per prompt).
    """
    output_path = config.generations_dir_ex4 / "responses.json"

    if output_path.exists() and not force:
        print(f"  Found existing responses at {output_path}, loading...")
        with open(output_path) as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Load Exp 1 responses (provides prompt text, splits, llama8b/gemma9b)
    # ------------------------------------------------------------------
    exp1_path = config.ex1_generations_dir / "responses.json"
    if not exp1_path.exists():
        raise FileNotFoundError(
            f"Experiment 1 responses not found at {exp1_path}. "
            "Run Experiment 1 first (python -m prefill_probe.run_experiment1)."
        )

    with open(exp1_path) as f:
        exp1_responses = json.load(f)

    print(f"  Loaded {len(exp1_responses)} prompts from Experiment 1 responses.")

    # Build working dict keyed by prompt_id
    results: dict[int, dict] = {}
    for r in exp1_responses:
        pid = r["prompt_id"]
        results[pid] = {
            "prompt_id": pid,
            "instruction": r["instruction"],
            "split": r["split"],
            "response_llama8b": r["response_target"],
            "response_gemma9b": r["response_source"],
        }

    prompts = list(results.values())

    # ------------------------------------------------------------------
    # Generate Llama 3.3 70B responses
    # ------------------------------------------------------------------
    llama70b_path = config.generations_dir_ex4 / "responses_llama70b_partial.json"

    done_llama70b: set[int] = set()
    if llama70b_path.exists():
        with open(llama70b_path) as f:
            partial = json.load(f)
        for pid, text in partial.items():
            results[int(pid)]["response_llama70b"] = text
            done_llama70b.add(int(pid))
        print(f"  Resuming Llama 70B generation from {len(done_llama70b)} / {len(prompts)}")

    remaining_llama70b = [p for p in prompts if p["prompt_id"] not in done_llama70b]
    if remaining_llama70b:
        device_str = get_device()
        print(f"  Loading {config.llama70b_model_id} on {device_str}...")
        model, tokenizer = _load_model_and_tokenizer(config.llama70b_model_id)
        model.eval()

        for prompt in tqdm(remaining_llama70b, desc="Llama 70B"):
            pid = prompt["prompt_id"]
            input_text = _format_generation_prompt(
                tokenizer, prompt["instruction"], config.llama70b_model_id
            )
            response_text = _generate_response(model, tokenizer, input_text, config)
            results[pid]["response_llama70b"] = response_text
            done_llama70b.add(pid)

            # Checkpoint every 20 prompts
            if len(done_llama70b) % 20 == 0:
                partial_data = {str(p): results[p]["response_llama70b"]
                                for p in done_llama70b}
                with open(llama70b_path, "w") as f:
                    json.dump(partial_data, f)

        # Final checkpoint save
        partial_data = {str(p): results[p]["response_llama70b"] for p in done_llama70b}
        with open(llama70b_path, "w") as f:
            json.dump(partial_data, f)

        del model
        clear_device_cache()
        print(f"  Llama 70B generation done ({len(done_llama70b)} prompts).")

    # ------------------------------------------------------------------
    # Generate Gemma 4 31B responses
    # ------------------------------------------------------------------
    gemma31b_path = config.generations_dir_ex4 / "responses_gemma31b_partial.json"

    done_gemma31b: set[int] = set()
    if gemma31b_path.exists():
        with open(gemma31b_path) as f:
            partial = json.load(f)
        for pid, text in partial.items():
            results[int(pid)]["response_gemma31b"] = text
            done_gemma31b.add(int(pid))
        print(f"  Resuming Gemma 31B generation from {len(done_gemma31b)} / {len(prompts)}")

    remaining_gemma31b = [p for p in prompts if p["prompt_id"] not in done_gemma31b]
    if remaining_gemma31b:
        device_str = get_device()
        print(f"  Loading {config.gemma31b_model_id} on {device_str}...")
        model, tokenizer = _load_model_and_tokenizer(config.gemma31b_model_id)
        model.eval()

        for prompt in tqdm(remaining_gemma31b, desc="Gemma 31B"):
            pid = prompt["prompt_id"]
            input_text = _format_generation_prompt(
                tokenizer, prompt["instruction"], config.gemma31b_model_id
            )
            response_text = _generate_response(model, tokenizer, input_text, config)
            results[pid]["response_gemma31b"] = response_text
            done_gemma31b.add(pid)

            if len(done_gemma31b) % 20 == 0:
                partial_data = {str(p): results[p]["response_gemma31b"]
                                for p in done_gemma31b}
                with open(gemma31b_path, "w") as f:
                    json.dump(partial_data, f)

        partial_data = {str(p): results[p]["response_gemma31b"] for p in done_gemma31b}
        with open(gemma31b_path, "w") as f:
            json.dump(partial_data, f)

        del model
        clear_device_cache()
        print(f"  Gemma 31B generation done ({len(done_gemma31b)} prompts).")

    # ------------------------------------------------------------------
    # Filter: require all four response fields and min token count
    # For token counting we use the llama8b tokenizer as a common reference
    # (same tokenizer as Exp 1 filtering).
    # ------------------------------------------------------------------
    required_fields = ["response_llama8b", "response_gemma9b",
                       "response_llama70b", "response_gemma31b"]

    print(f"  Filtering responses (min {config.min_response_tokens} tokens per response)...")
    print(f"  Loading Llama 8B tokenizer for token counting...")
    ref_tokenizer = AutoTokenizer.from_pretrained(config.llama8b_model_id)

    valid = []
    discarded = []
    for pid, r in results.items():
        if not all(f in r for f in required_fields):
            discarded.append(pid)
            continue
        min_n = min(
            _count_tokens(ref_tokenizer, r[f])
            for f in required_fields
        )
        if min_n >= config.min_response_tokens:
            valid.append(r)
        else:
            discarded.append(pid)

    del ref_tokenizer
    gc.collect()

    valid.sort(key=lambda r: r["prompt_id"])
    print(f"  Retained {len(valid)} / {len(results)} prompts "
          f"(discarded {len(discarded)})")

    with open(output_path, "w") as f:
        json.dump(valid, f, indent=2)

    # Clean up partial checkpoints on success
    for partial_file in [llama70b_path, gemma31b_path]:
        if partial_file.exists():
            partial_file.unlink()

    print(f"  Saved {len(valid)} merged responses → {output_path}")
    return valid
