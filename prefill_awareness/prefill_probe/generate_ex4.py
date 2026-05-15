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
      2. Generating Llama 3.3 70B responses (skipped if already present).
      3. Generating Gemma 4 31B responses (skipped if already present).
      4. Generating Gemma 4 4B responses (skipped if already present).

    Returns the merged list of response dicts (one per prompt).
    """
    output_path = config.generations_dir_ex4 / "responses.json"
    required_fields = ["response_llama8b", "response_gemma9b",
                       "response_llama70b", "response_gemma31b", "response_gemma4b",
                       "response_mistral7b", "response_mistral24b"]

    if output_path.exists() and not force:
        with open(output_path) as f:
            existing = json.load(f)
        if existing and all(all(f in r for f in required_fields) for r in existing):
            print(f"  Found complete responses at {output_path}, loading...")
            return existing
        print(f"  Found partial responses at {output_path}; extending with missing fields...")

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

    # Build working dict keyed by prompt_id, seeding from existing file if present
    results: dict[int, dict] = {}
    if output_path.exists() and not force:
        with open(output_path) as f:
            for r in json.load(f):
                results[r["prompt_id"]] = r

    for r in exp1_responses:
        pid = r["prompt_id"]
        if pid not in results:
            results[pid] = {
                "prompt_id": pid,
                "instruction": r["instruction"],
                "split": r["split"],
                "response_llama8b": r["response_target"],
                "response_gemma9b": r["response_source"],
            }

    prompts = list(results.values())

    # ------------------------------------------------------------------
    # Generate Llama 3.3 70B responses (skip if already in results)
    # ------------------------------------------------------------------
    llama70b_path = config.generations_dir_ex4 / "responses_llama70b_partial.json"

    done_llama70b: set[int] = {
        pid for pid, r in results.items() if "response_llama70b" in r
    }
    if done_llama70b:
        print(f"  Llama 70B: {len(done_llama70b)} responses already present, skipping generation.")

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
    # Generate Gemma 4 31B responses (skip if already in results)
    # ------------------------------------------------------------------
    gemma31b_path = config.generations_dir_ex4 / "responses_gemma31b_partial.json"

    done_gemma31b: set[int] = {
        pid for pid, r in results.items() if "response_gemma31b" in r
    }
    if done_gemma31b:
        print(f"  Gemma 31B: {len(done_gemma31b)} responses already present, skipping generation.")

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
    # Generate Gemma 4 4B responses (skip if already in results)
    # ------------------------------------------------------------------
    gemma4b_path = config.generations_dir_ex4 / "responses_gemma4b_partial.json"

    done_gemma4b: set[int] = {
        pid for pid, r in results.items() if "response_gemma4b" in r
    }
    if done_gemma4b:
        print(f"  Gemma 4B: {len(done_gemma4b)} responses already present, skipping generation.")

    if gemma4b_path.exists():
        with open(gemma4b_path) as f:
            partial = json.load(f)
        for pid, text in partial.items():
            results[int(pid)]["response_gemma4b"] = text
            done_gemma4b.add(int(pid))
        print(f"  Resuming Gemma 4B generation from {len(done_gemma4b)} / {len(prompts)}")

    remaining_gemma4b = [p for p in prompts if p["prompt_id"] not in done_gemma4b]
    if remaining_gemma4b:
        device_str = get_device()
        print(f"  Loading {config.gemma4b_model_id} on {device_str}...")
        model, tokenizer = _load_model_and_tokenizer(config.gemma4b_model_id)
        model.eval()

        for prompt in tqdm(remaining_gemma4b, desc="Gemma 4B"):
            pid = prompt["prompt_id"]
            input_text = _format_generation_prompt(
                tokenizer, prompt["instruction"], config.gemma4b_model_id
            )
            response_text = _generate_response(model, tokenizer, input_text, config)
            results[pid]["response_gemma4b"] = response_text
            done_gemma4b.add(pid)

            if len(done_gemma4b) % 20 == 0:
                partial_data = {str(p): results[p]["response_gemma4b"]
                                for p in done_gemma4b}
                with open(gemma4b_path, "w") as f:
                    json.dump(partial_data, f)

        partial_data = {str(p): results[p]["response_gemma4b"] for p in done_gemma4b}
        with open(gemma4b_path, "w") as f:
            json.dump(partial_data, f)

        del model
        clear_device_cache()
        print(f"  Gemma 4B generation done ({len(done_gemma4b)} prompts).")

    # ------------------------------------------------------------------
    # Filter: require all seven response fields and min token count
    # ------------------------------------------------------------------
    required_fields = ["response_llama8b", "response_gemma9b",
                       "response_llama70b", "response_gemma31b", "response_gemma4b",
                       "response_mistral7b", "response_mistral24b"]

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

    # ------------------------------------------------------------------
    # Generate Mistral 7B responses (skip if already in results)
    # ------------------------------------------------------------------
    mistral7b_path = config.generations_dir_ex4 / "responses_mistral7b_partial.json"

    done_mistral7b: set[int] = {
        pid for pid, r in results.items() if "response_mistral7b" in r
    }
    if done_mistral7b:
        print(f"  Mistral 7B: {len(done_mistral7b)} responses already present, skipping generation.")

    if mistral7b_path.exists():
        with open(mistral7b_path) as f:
            partial = json.load(f)
        for pid, text in partial.items():
            results[int(pid)]["response_mistral7b"] = text
            done_mistral7b.add(int(pid))
        print(f"  Resuming Mistral 7B generation from {len(done_mistral7b)} / {len(prompts)}")

    remaining_mistral7b = [p for p in prompts if p["prompt_id"] not in done_mistral7b]
    if remaining_mistral7b:
        device_str = get_device()
        print(f"  Loading {config.mistral7b_model_id} on {device_str}...")
        model, tokenizer = _load_model_and_tokenizer(config.mistral7b_model_id)
        model.eval()

        for prompt in tqdm(remaining_mistral7b, desc="Mistral 7B"):
            pid = prompt["prompt_id"]
            input_text = _format_generation_prompt(
                tokenizer, prompt["instruction"], config.mistral7b_model_id
            )
            response_text = _generate_response(model, tokenizer, input_text, config)
            results[pid]["response_mistral7b"] = response_text
            done_mistral7b.add(pid)

            if len(done_mistral7b) % 20 == 0:
                partial_data = {str(p): results[p]["response_mistral7b"]
                                for p in done_mistral7b}
                with open(mistral7b_path, "w") as f:
                    json.dump(partial_data, f)

        partial_data = {str(p): results[p]["response_mistral7b"] for p in done_mistral7b}
        with open(mistral7b_path, "w") as f:
            json.dump(partial_data, f)

        del model
        clear_device_cache()
        print(f"  Mistral 7B generation done ({len(done_mistral7b)} prompts).")

    # ------------------------------------------------------------------
    # Generate Mistral Small 24B responses (skip if already in results)
    # ------------------------------------------------------------------
    mistral24b_path = config.generations_dir_ex4 / "responses_mistral24b_partial.json"

    done_mistral24b: set[int] = {
        pid for pid, r in results.items() if "response_mistral24b" in r
    }
    if done_mistral24b:
        print(f"  Mistral 24B: {len(done_mistral24b)} responses already present, skipping generation.")

    if mistral24b_path.exists():
        with open(mistral24b_path) as f:
            partial = json.load(f)
        for pid, text in partial.items():
            results[int(pid)]["response_mistral24b"] = text
            done_mistral24b.add(int(pid))
        print(f"  Resuming Mistral 24B generation from {len(done_mistral24b)} / {len(prompts)}")

    remaining_mistral24b = [p for p in prompts if p["prompt_id"] not in done_mistral24b]
    if remaining_mistral24b:
        device_str = get_device()
        print(f"  Loading {config.mistral24b_model_id} on {device_str}...")
        model, tokenizer = _load_model_and_tokenizer(config.mistral24b_model_id)
        model.eval()

        for prompt in tqdm(remaining_mistral24b, desc="Mistral 24B"):
            pid = prompt["prompt_id"]
            input_text = _format_generation_prompt(
                tokenizer, prompt["instruction"], config.mistral24b_model_id
            )
            response_text = _generate_response(model, tokenizer, input_text, config)
            results[pid]["response_mistral24b"] = response_text
            done_mistral24b.add(pid)

            if len(done_mistral24b) % 20 == 0:
                partial_data = {str(p): results[p]["response_mistral24b"]
                                for p in done_mistral24b}
                with open(mistral24b_path, "w") as f:
                    json.dump(partial_data, f)

        partial_data = {str(p): results[p]["response_mistral24b"] for p in done_mistral24b}
        with open(mistral24b_path, "w") as f:
            json.dump(partial_data, f)

        del model
        clear_device_cache()
        print(f"  Mistral 24B generation done ({len(done_mistral24b)} prompts).")

    # Clean up partial checkpoints on success
    for partial_file in [llama70b_path, gemma31b_path, gemma4b_path,
                         mistral7b_path, mistral24b_path]:
        if partial_file.exists():
            partial_file.unlink()

    print(f"  Saved {len(valid)} merged responses → {output_path}")
    return valid
