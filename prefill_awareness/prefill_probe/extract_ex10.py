"""
Activation extraction for Experiment 10 (Probe Generalisation).

Target model is always Llama 3.3 70B.  For every (dataset, condition) pair we
run a forward pass over the Llama 70B template with the relevant response text
and extract the residual stream at the last content token of the assistant
response at a single fixed layer (probe_layer=60, the best layer from Exp 4).

Files written per (dataset, condition):
  activations/{dataset}_self.pt
  activations/{dataset}_cross_llama8b.pt
  activations/{dataset}_cross_gemma31b.pt
  activations/{dataset}_cross_mistral24b.pt

Each .pt file is a list[dict] with keys:
  prompt_id, split, activation (np.ndarray float16, shape (hidden_dim,))
"""

from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import Experiment10Config
from .extract_ex4 import (
    build_prefill_input,
    get_transformer_layers,
    verify_extraction_positions,
)
from .utils import clear_device_cache, get_device, get_device_map


# ---------------------------------------------------------------------------
# Single-layer forward pass
# ---------------------------------------------------------------------------

def _run_forward_pass_single_layer(
    model,
    tokenizer,
    model_id: str,
    instruction: str,
    response_text: str,
    probe_layer: int,
    device,
) -> np.ndarray:
    """Forward pass with a hook at probe_layer; return last-response-pos activation."""
    prefill_ids, last_response_pos, _ = build_prefill_input(
        tokenizer, model_id, instruction, response_text
    )
    prefill_ids = prefill_ids.to(device)

    activation_holder: list[np.ndarray] = []
    transformer_layers = get_transformer_layers(model)

    def hook_fn(module, input, output):
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        activation_holder.append(
            hidden[0, last_response_pos, :]
            .detach().float().cpu().numpy().astype(np.float16)
        )

    hook = transformer_layers[probe_layer].register_forward_hook(hook_fn)
    with torch.no_grad():
        model(prefill_ids)
    hook.remove()

    return activation_holder[0]


# ---------------------------------------------------------------------------
# Per-condition extraction with checkpointing
# ---------------------------------------------------------------------------

def _extract_one_condition(
    condition_name: str,
    response_key: str,
    dataset_name: str,
    responses: list[dict],
    model,
    tokenizer,
    model_id: str,
    probe_layer: int,
    activations_dir: Path,
    checkpoint_interval: int,
    device,
) -> None:
    """Extract single-layer activations for one (dataset, condition), with resume."""
    final_path = activations_dir / f"{dataset_name}_{condition_name}.pt"
    if final_path.exists():
        print(f"  [{dataset_name}/{condition_name}] Already complete, skipping.")
        return

    verify_extraction_positions(
        tokenizer, model_id, responses, response_key, n_examples=3
    )

    partial_path = activations_dir / f"{dataset_name}_{condition_name}_partial.pt"
    data_list: list[dict] = []
    completed_pids: set[int] = set()

    if partial_path.exists():
        data_list = torch.load(partial_path, weights_only=False)
        completed_pids = {d["prompt_id"] for d in data_list}
        print(f"  [{dataset_name}/{condition_name}] Resuming from "
              f"{len(completed_pids)} / {len(responses)}")

    remaining = [r for r in responses if r["prompt_id"] not in completed_pids]

    for i, r in enumerate(tqdm(remaining, desc=f"  {dataset_name}/{condition_name}")):
        act = _run_forward_pass_single_layer(
            model, tokenizer, model_id,
            r["instruction"], r[response_key],
            probe_layer, device,
        )
        data_list.append({
            "prompt_id": r["prompt_id"],
            "split": r["split"],
            "activation": act,
        })

        if (i + 1) % checkpoint_interval == 0:
            torch.save(data_list, partial_path)

    torch.save(data_list, final_path)
    if partial_path.exists():
        partial_path.unlink()
    print(f"  [{dataset_name}/{condition_name}] Saved {len(data_list)} records → {final_path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def extract_all_activations(
    all_responses: dict[str, list[dict]],
    config: Experiment10Config,
) -> None:
    """
    Extract Llama 70B activations at probe_layer for all (dataset, condition) pairs.

    Conditions per dataset:
      self              — response_llama70b
      cross_llama8b     — response_llama8b
      cross_gemma31b    — response_gemma31b
      cross_mistral24b  — response_mistral24b
    """
    conditions = {
        "self":             "response_llama70b",
        "cross_llama8b":    "response_llama8b",
        "cross_gemma31b":   "response_gemma31b",
        "cross_mistral24b": "response_mistral24b",
    }

    # Skip if everything already exists
    all_done = all(
        (config.activations_dir / f"{ds}_{cond}.pt").exists()
        for ds in config.datasets
        for cond in conditions
    )
    if all_done:
        print("  All activation files already present, skipping extraction.")
        return

    device_str = get_device()
    device_map = get_device_map()

    print(f"  Loading {config.target_model_id} (fp16) on {device_str}...")
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
    torch.use_deterministic_algorithms(True, warn_only=True)

    n_layers = len(get_transformer_layers(model))
    print(f"  Model has {n_layers} layers; extracting at layer {config.probe_layer}.")

    device = next(model.parameters()).device

    for ds_name, responses in all_responses.items():
        print(f"\n  --- Dataset: {ds_name} ({len(responses)} prompts) ---")
        for condition_name, response_key in conditions.items():
            _extract_one_condition(
                condition_name=condition_name,
                response_key=response_key,
                dataset_name=ds_name,
                responses=responses,
                model=model,
                tokenizer=tokenizer,
                model_id=config.target_model_id,
                probe_layer=config.probe_layer,
                activations_dir=config.activations_dir,
                checkpoint_interval=config.checkpoint_interval,
                device=device,
            )

    del model
    clear_device_cache()


def load_all_activations(
    config: Experiment10Config,
) -> dict[str, dict[str, list[dict]]]:
    """
    Load all activation files from disk.
    Returns {dataset: {condition: [records]}}.
    """
    conditions = ["self", "cross_llama8b", "cross_gemma31b", "cross_mistral24b"]
    result = {}
    for ds in config.datasets:
        result[ds] = {}
        for cond in conditions:
            path = config.activations_dir / f"{ds}_{cond}.pt"
            if not path.exists():
                raise FileNotFoundError(
                    f"Activations not found at {path}. Run the extract step first."
                )
            result[ds][cond] = torch.load(path, weights_only=False)
    return result
