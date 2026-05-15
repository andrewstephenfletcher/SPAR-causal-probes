"""
Activation extraction for Experiment 4 (Scaling Analysis).

For each target model, runs forward passes for every (prompt, condition) pair
and extracts the residual stream at the last content token of the assistant
response at every transformer layer.

Llama 3.3 70B conditions (activations_dir_llama70b/):
  self_prefill.pt   — Llama 70B's own response
  cross_gemma9b.pt  — Gemma 9B response in Llama 70B's template
  cross_llama8b.pt  — Llama 8B response in Llama 70B's template

Gemma 4 31B conditions (activations_dir_gemma31b/):
  self_prefill.pt   — Gemma 31B's own response
  cross_llama8b.pt  — Llama 8B response in Gemma 31B's template
  cross_gemma9b.pt  — Gemma 9B response in Gemma 31B's template

Each .pt file is a list[dict] with keys:
  prompt_id, condition, layer_activations (dict[int, np.ndarray float16]),
  last_response_pos, n_response_tokens

Activations are saved incrementally every checkpoint_interval prompts.
"""

from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import Experiment4Config
from .utils import clear_device_cache, get_device, get_device_map


# ---------------------------------------------------------------------------
# Chat template helpers
# ---------------------------------------------------------------------------

def _supports_system_prompt(model_id: str) -> bool:
    return "gemma-2" not in model_id.lower()


def build_prefill_input(
    tokenizer,
    model_id: str,
    instruction: str,
    response_text: str,
) -> tuple[torch.Tensor, int, int]:
    """
    Tokenize a complete conversation using the target model's chat template.

    Returns (prefill_ids, last_response_pos, n_response_tokens) where
    last_response_pos is the index of the last non-special token of the
    assistant response content.
    """
    if _supports_system_prompt(model_id):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response_text},
        ]
    else:
        messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response_text},
        ]

    prefill_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    prefill_ids = tokenizer(prefill_text, return_tensors="pt")["input_ids"]

    response_only_ids = tokenizer(
        response_text, add_special_tokens=False
    )["input_ids"]
    n_response_tokens = len(response_only_ids)

    n_trailing = _count_trailing_special_tokens(tokenizer, prefill_ids)
    last_response_pos = prefill_ids.shape[1] - 1 - n_trailing

    return prefill_ids, last_response_pos, n_response_tokens


def _count_trailing_special_tokens(tokenizer, prefill_ids: torch.Tensor) -> int:
    """Count consecutive trailing special tokens (up to 4) from the end."""
    all_special_ids = set(tokenizer.all_special_ids)
    n_trailing = 0
    seq_len = prefill_ids.shape[1]
    for offset in range(1, 5):
        pos = seq_len - offset
        if pos < 0:
            break
        tok_id = prefill_ids[0, pos].item()
        if tok_id in all_special_ids:
            n_trailing = offset
        else:
            break
    return n_trailing


# ---------------------------------------------------------------------------
# Position verification
# ---------------------------------------------------------------------------

def verify_extraction_positions(
    tokenizer,
    model_id: str,
    responses: list[dict],
    response_key: str,
    n_examples: int = 5,
) -> None:
    """
    Confirm that last_response_pos points to the expected last content token.
    Raises RuntimeError on any mismatch.
    """
    print(f"\n=== Position Verification: {response_key} (first {n_examples}) ===")
    all_ok = True

    for r in responses[:n_examples]:
        prefill_ids, last_pos, _ = build_prefill_input(
            tokenizer, model_id, r["instruction"], r[response_key]
        )
        extracted_id = prefill_ids[0, last_pos].item()
        extracted_tok = tokenizer.decode([extracted_id])

        expected_id = tokenizer(
            r[response_key], add_special_tokens=False
        )["input_ids"][-1]
        expected_tok = tokenizer.decode([expected_id])

        match = extracted_id == expected_id
        if not match:
            all_ok = False

        print(
            f"  pid={r['prompt_id']:3d}  pos={last_pos:5d}  "
            f"extracted='{extracted_tok}'  expected='{expected_tok}'  "
            f"[{'OK' if match else 'MISMATCH'}]"
        )
        if not match:
            tail = prefill_ids[0, -6:].tolist()
            print(f"    tail tokens: {[tokenizer.decode([t]) for t in tail]}")

    if not all_ok:
        raise RuntimeError(
            f"Position verification FAILED for '{response_key}'. "
            "Check _count_trailing_special_tokens for this model's chat template."
        )
    print("  All OK.\n")


# ---------------------------------------------------------------------------
# Layer access
# ---------------------------------------------------------------------------

def get_transformer_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise RuntimeError(
        f"Cannot find model.model.layers for {type(model).__name__}. "
        "Inspect the model architecture and update get_transformer_layers()."
    )


# ---------------------------------------------------------------------------
# Single forward pass
# ---------------------------------------------------------------------------

def _run_forward_pass(
    model,
    tokenizer,
    model_id: str,
    instruction: str,
    response_text: str,
    n_layers: int,
    device,
) -> tuple[dict, int, int]:
    """
    One forward pass with hooks at all n_layers layers.

    Returns (layer_activations, last_response_pos, n_response_tokens) where
    layer_activations maps layer_idx -> np.ndarray float16 (hidden_dim,).
    """
    prefill_ids, last_response_pos, n_response_tokens = build_prefill_input(
        tokenizer, model_id, instruction, response_text
    )
    prefill_ids = prefill_ids.to(device)

    activations: dict[int, np.ndarray] = {}
    hooks = []
    transformer_layers = get_transformer_layers(model)

    def make_hook(layer_idx: int):
        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            activations[layer_idx] = (
                hidden[0, last_response_pos, :]
                .detach().float().cpu().numpy().astype(np.float16)
            )
        return hook_fn

    for i in range(n_layers):
        hooks.append(transformer_layers[i].register_forward_hook(make_hook(i)))

    with torch.no_grad():
        model(prefill_ids)

    for h in hooks:
        h.remove()

    return activations, last_response_pos, n_response_tokens


# ---------------------------------------------------------------------------
# Per-condition extraction with checkpointing
# ---------------------------------------------------------------------------

def _extract_one_condition(
    condition_name: str,
    response_key: str,
    target_model_id: str,
    model,
    tokenizer,
    responses: list[dict],
    activations_dir: Path,
    n_layers: int,
    checkpoint_interval: int,
) -> None:
    """Extract activations for one (model, condition) pair, with resume support."""
    final_path = activations_dir / f"{condition_name}.pt"

    if final_path.exists():
        print(f"  [{condition_name}] Already complete, skipping.")
        return

    verify_extraction_positions(
        tokenizer, target_model_id, responses, response_key, n_examples=5
    )

    partial_path = activations_dir / f"{condition_name}_partial.pt"
    data_list: list[dict] = []
    completed_pids: set[int] = set()

    if partial_path.exists():
        data_list = torch.load(partial_path, weights_only=False)
        completed_pids = {d["prompt_id"] for d in data_list}
        print(f"  [{condition_name}] Resuming from "
              f"{len(completed_pids)} / {len(responses)} prompts")

    device = next(model.parameters()).device
    remaining = [r for r in responses if r["prompt_id"] not in completed_pids]

    for i, r in enumerate(tqdm(remaining, desc=f"  {condition_name}")):
        acts, pos, n = _run_forward_pass(
            model, tokenizer, target_model_id,
            r["instruction"], r[response_key],
            n_layers, device,
        )
        data_list.append({
            "prompt_id": r["prompt_id"],
            "condition": condition_name,
            "layer_activations": acts,
            "last_response_pos": pos,
            "n_response_tokens": n,
        })

        if (i + 1) % checkpoint_interval == 0:
            torch.save(data_list, partial_path)

    torch.save(data_list, final_path)
    if partial_path.exists():
        partial_path.unlink()
    print(f"  [{condition_name}] Saved {len(data_list)} records → {final_path}")


# ---------------------------------------------------------------------------
# Per-model entry points
# ---------------------------------------------------------------------------

def extract_all_activations_llama70b(
    responses: list[dict],
    config: Experiment4Config,
) -> None:
    """Extract activations for Llama 3.3 70B across all three conditions."""
    # {output filename stem: response field}
    conditions = {
        "self_prefill":  "response_llama70b",
        "cross_gemma9b": "response_gemma9b",
        "cross_llama8b": "response_llama8b",
    }

    if all((config.activations_dir_llama70b / f"{c}.pt").exists() for c in conditions):
        print("  Llama 70B: all activation files present, skipping extraction.")
        return

    device_str = get_device()
    device_map = get_device_map()

    print(f"  Loading {config.llama70b_model_id} (fp16) on {device_str}...")
    if device_map is not None:
        model = AutoModelForCausalLM.from_pretrained(
            config.llama70b_model_id,
            torch_dtype=torch.float16,
            device_map=device_map,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.llama70b_model_id,
            torch_dtype=torch.float16,
        ).to(device_str)

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(config.llama70b_model_id)
    torch.use_deterministic_algorithms(True, warn_only=True)

    n_layers = len(get_transformer_layers(model))
    print(f"  Detected {n_layers} transformer layers.")

    for condition_name, response_key in conditions.items():
        _extract_one_condition(
            condition_name=condition_name,
            response_key=response_key,
            target_model_id=config.llama70b_model_id,
            model=model,
            tokenizer=tokenizer,
            responses=responses,
            activations_dir=config.activations_dir_llama70b,
            n_layers=n_layers,
            checkpoint_interval=config.checkpoint_interval,
        )

    del model
    clear_device_cache()


def extract_all_activations_gemma31b(
    responses: list[dict],
    config: Experiment4Config,
) -> None:
    """Extract activations for Gemma 4 31B across all three conditions."""
    conditions = {
        "self_prefill":  "response_gemma31b",
        "cross_llama8b": "response_llama8b",
        "cross_gemma9b": "response_gemma9b",
    }

    if all((config.activations_dir_gemma31b / f"{c}.pt").exists() for c in conditions):
        print("  Gemma 31B: all activation files present, skipping extraction.")
        return

    device_str = get_device()
    device_map = get_device_map()

    print(f"  Loading {config.gemma31b_model_id} (fp16) on {device_str}...")
    if device_map is not None:
        model = AutoModelForCausalLM.from_pretrained(
            config.gemma31b_model_id,
            torch_dtype=torch.float16,
            device_map=device_map,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.gemma31b_model_id,
            torch_dtype=torch.float16,
        ).to(device_str)

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(config.gemma31b_model_id)
    torch.use_deterministic_algorithms(True, warn_only=True)

    n_layers = len(get_transformer_layers(model))
    print(f"  Detected {n_layers} transformer layers.")

    for condition_name, response_key in conditions.items():
        _extract_one_condition(
            condition_name=condition_name,
            response_key=response_key,
            target_model_id=config.gemma31b_model_id,
            model=model,
            tokenizer=tokenizer,
            responses=responses,
            activations_dir=config.activations_dir_gemma31b,
            n_layers=n_layers,
            checkpoint_interval=config.checkpoint_interval,
        )

    del model
    clear_device_cache()


def extract_all_activations_gemma4b(
    responses: list[dict],
    config: Experiment4Config,
) -> None:
    _extract_all_activations_for(
        model_id=config.gemma4b_model_id,
        conditions={
            "self_prefill":  "response_gemma4b",
            "cross_llama8b": "response_llama8b",
            "cross_gemma9b": "response_gemma9b",
        },
        activations_dir=config.activations_dir_gemma4b,
        responses=responses,
        config=config,
        label="Gemma 4B",
    )


def _extract_all_activations_for(
    model_id: str,
    conditions: dict[str, str],
    activations_dir: Path,
    responses: list[dict],
    config: Experiment4Config,
    label: str,
) -> None:
    """Generic extraction helper — load model, extract all conditions, unload."""
    if all((activations_dir / f"{c}.pt").exists() for c in conditions):
        print(f"  {label}: all activation files present, skipping extraction.")
        return

    device_str = get_device()
    device_map = get_device_map()

    print(f"  Loading {model_id} (fp16) on {device_str}...")
    if device_map is not None:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map=device_map,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16,
        ).to(device_str)

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    torch.use_deterministic_algorithms(True, warn_only=True)

    n_layers = len(get_transformer_layers(model))
    print(f"  Detected {n_layers} transformer layers.")

    for condition_name, response_key in conditions.items():
        _extract_one_condition(
            condition_name=condition_name,
            response_key=response_key,
            target_model_id=model_id,
            model=model,
            tokenizer=tokenizer,
            responses=responses,
            activations_dir=activations_dir,
            n_layers=n_layers,
            checkpoint_interval=config.checkpoint_interval,
        )

    del model
    clear_device_cache()


def extract_all_activations_mistral7b(
    responses: list[dict],
    config: Experiment4Config,
) -> None:
    _extract_all_activations_for(
        model_id=config.mistral7b_model_id,
        conditions={
            "self_prefill":  "response_mistral7b",
            "cross_llama8b": "response_llama8b",
            "cross_gemma9b": "response_gemma9b",
        },
        activations_dir=config.activations_dir_mistral7b,
        responses=responses,
        config=config,
        label="Mistral 7B",
    )


def extract_all_activations_mistral24b(
    responses: list[dict],
    config: Experiment4Config,
) -> None:
    _extract_all_activations_for(
        model_id=config.mistral24b_model_id,
        conditions={
            "self_prefill":  "response_mistral24b",
            "cross_llama8b": "response_llama8b",
            "cross_gemma9b": "response_gemma9b",
        },
        activations_dir=config.activations_dir_mistral24b,
        responses=responses,
        config=config,
        label="Mistral 24B",
    )


def run_all_extractions(responses: list[dict], config: Experiment4Config) -> None:
    print("\n--- Extracting activations: Llama 3.3 70B ---")
    extract_all_activations_llama70b(responses, config)

    print("\n--- Extracting activations: Gemma 4 31B ---")
    extract_all_activations_gemma31b(responses, config)

    print("\n--- Extracting activations: Gemma 4 4B ---")
    extract_all_activations_gemma4b(responses, config)

    print("\n--- Extracting activations: Mistral 7B ---")
    extract_all_activations_mistral7b(responses, config)

    print("\n--- Extracting activations: Mistral Small 24B ---")
    extract_all_activations_mistral24b(responses, config)
