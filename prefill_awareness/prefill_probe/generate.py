"""
Response generation for Experiment 1.

Generates responses from both the target (Llama 3.1 8B Instruct) and source
(Gemma 2 9B IT) models.  The two models are loaded and unloaded sequentially
to avoid holding both in accelerator memory simultaneously.
"""

import gc
import json

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import Config
from .utils import clear_device_cache, get_device, get_device_map, gpu_memory_gb


# ---------------------------------------------------------------------------
# Model loading / unloading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(
    model_id: str,
    load_in_4bit: bool = False,
) -> tuple:
    """
    Load a causal LM and its tokenizer.

    Device priority: CUDA (device_map="auto") > MPS > CPU.
    4-bit quantisation is only supported on CUDA (bitsandbytes).
    """
    device = get_device()
    device_map = get_device_map()

    if load_in_4bit and device == "cuda":
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(load_in_4bit=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
            )
        except ImportError:
            print("  WARNING: bitsandbytes not installed — loading in float16 instead.")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
            )
    elif device_map is not None:
        # CUDA with multi-GPU support
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map=device_map,
        )
    else:
        # MPS or CPU: load then move
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
        ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def unload_model(model) -> None:
    """Delete model object and free accelerator memory."""
    del model
    clear_device_cache()


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def format_prompt_for_model(
    tokenizer,
    instruction: str,
    is_target: bool,
) -> str:
    """
    Apply the model's chat template to a single instruction.

    Target model (Llama 3.1): system + user messages.
    Source model (Gemma 2):   user message only — Gemma's template does not
                               support the system role in the same way.
    """
    if is_target:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": instruction},
        ]
    else:
        messages = [
            {"role": "user", "content": instruction},
        ]

    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_response(
    model,
    tokenizer,
    input_text: str,
    config: Config,
) -> tuple[str, int]:
    """
    Generate a single response.  Sets torch seed before the call for
    reproducibility.  Returns (response_text, n_new_tokens).
    """
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
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
    return response_text, len(response_ids)


def _use_4bit_for_source() -> bool:
    """
    Decide whether to quantise the source model (Gemma 9B) to 4-bit.

    4-bit is only available on CUDA.  On MPS/CPU we rely on unified memory
    (Apple Silicon has enough headroom for 9B at float16 ≈ 18 GB).
    On CUDA, use 4-bit if free VRAM < 20 GB.
    """
    import torch as _torch
    if not _torch.cuda.is_available():
        return False  # MPS / CPU: no bitsandbytes support
    free_gb = gpu_memory_gb()
    if free_gb < 20.0:
        print(f"  Free CUDA memory: {free_gb:.1f} GB < 20 GB — "
              "using 4-bit quantisation for source model.")
        return True
    return False


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_all_responses(
    prompts: list[dict],
    config: Config,
    force: bool = False,
) -> list[dict]:
    """
    Generate responses from both target and source models.

    Models are loaded sequentially and unloaded between steps to keep peak
    memory usage below the capacity of a single accelerator.

    Saves results to generations_dir/responses.json.
    Returns the filtered list of response dicts.
    """
    output_path = config.generations_dir / "responses.json"

    if output_path.exists() and not force:
        print(f"  Found existing responses at {output_path}, loading...")
        with open(output_path) as f:
            return json.load(f)

    results: dict[int, dict] = {p["prompt_id"]: dict(p) for p in prompts}

    # ------------------------------------------------------------------ #
    # Step A: Target model (Llama 3.1 8B Instruct)
    # ------------------------------------------------------------------ #
    device_str = get_device()
    print(f"  Loading target model ({config.target_model_id}) on {device_str}...")
    target_model, target_tokenizer = load_model_and_tokenizer(config.target_model_id)
    target_model.eval()

    print("  Generating target model responses...")
    for prompt in tqdm(prompts, desc="Target (Llama)"):
        input_text = format_prompt_for_model(
            target_tokenizer, prompt["instruction"], is_target=True
        )
        response_text, _ = generate_response(
            target_model, target_tokenizer, input_text, config
        )
        # Token count under the target tokeniser
        n_tokens = len(
            target_tokenizer(response_text, add_special_tokens=False)["input_ids"]
        )
        results[prompt["prompt_id"]]["response_target"] = response_text
        results[prompt["prompt_id"]]["target_response_tokens"] = n_tokens

    print("  Unloading target model...")
    unload_model(target_model)
    # Keep target_tokenizer alive for counting source response lengths

    # ------------------------------------------------------------------ #
    # Step B: Source model (Gemma 2 9B IT)
    # ------------------------------------------------------------------ #
    use_4bit = _use_4bit_for_source()
    print(f"  Loading source model ({config.source_model_id}) on {device_str}...")
    source_model, source_tokenizer = load_model_and_tokenizer(
        config.source_model_id, load_in_4bit=use_4bit
    )
    source_model.eval()

    print("  Generating source model responses...")
    for prompt in tqdm(prompts, desc="Source (Gemma)"):
        input_text = format_prompt_for_model(
            source_tokenizer, prompt["instruction"], is_target=False
        )
        response_text, _ = generate_response(
            source_model, source_tokenizer, input_text, config
        )
        # Token count under the TARGET tokeniser (per spec)
        n_tokens = len(
            target_tokenizer(response_text, add_special_tokens=False)["input_ids"]
        )
        results[prompt["prompt_id"]]["response_source"] = response_text
        results[prompt["prompt_id"]]["source_response_tokens"] = n_tokens

    print("  Unloading source model...")
    unload_model(source_model)

    # ------------------------------------------------------------------ #
    # Filter: discard prompts where either response is too short
    # ------------------------------------------------------------------ #
    print("  Filtering by minimum response token count "
          f"(≥ {config.min_response_tokens} tokens under target tokeniser)...")
    valid: list[dict] = []
    discarded: list[int] = []

    for pid, r in results.items():
        t_ok = r.get("target_response_tokens", 0) >= config.min_response_tokens
        s_ok = r.get("source_response_tokens", 0) >= config.min_response_tokens
        if t_ok and s_ok:
            valid.append(r)
        else:
            discarded.append(pid)

    print(f"  Retained {len(valid)} / {len(prompts)} prompts "
          f"(discarded {len(discarded)})")
    if discarded:
        print(f"  Discarded prompt_ids: {discarded}")
    if len(valid) < 250:
        print(f"  WARNING: Only {len(valid)} prompts retained. "
              "Consider raising config.n_prompts and re-running with --force.")

    valid.sort(key=lambda r: r["prompt_id"])

    with open(output_path, "w") as f:
        json.dump(valid, f, indent=2)

    print(f"  Saved {len(valid)} responses → {output_path}")
    return valid
