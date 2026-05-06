"""
Activation extraction for Experiment 9 (Probe Transfer and Representational Geometry).

Two extraction tasks:

  1. Truth activations (9.1)
     Load Geometry of Truth datasets (cities, larger_than), format each statement as
     a chat prompt, and extract residual-stream activations at the last input token
     at layers [16, 24, 30] of Llama 8B.

  2. Eval/deploy CAA direction (9.2)
     Compute a contrastive activation addition (CAA) vector at layer 30 by
     differencing mean activations for eval-context vs. deployed-context prompts.

Both tasks require Llama 8B loaded in memory and are run from run_experiment9.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .config import Experiment9Config
from .utils import clear_device_cache, get_device, get_device_map


# ---------------------------------------------------------------------------
# Geometry of Truth dataset loading
# ---------------------------------------------------------------------------

def load_got_dataset(config: Experiment9Config, dataset_name: str) -> list[dict]:
    """
    Load a Geometry of Truth CSV and return records with a train/test split.

    Each record: {"statement": str, "label": int (1=true, 0=false),
                  "dataset": str, "split": "train" | "test"}
    """
    csv_path = config.got_datasets_dir / f"{dataset_name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Geometry of Truth dataset not found: {csv_path}\n"
            "Expected path relative to project root."
        )

    df = pd.read_csv(csv_path)
    if "statement" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"Expected columns 'statement' and 'label' in {csv_path}. "
            f"Got: {list(df.columns)}"
        )

    records = df[["statement", "label"]].to_dict(orient="records")

    rng = np.random.default_rng(42)
    n = len(records)
    perm = rng.permutation(n)
    n_train = int(n * config.got_train_frac)

    for i, record in enumerate(records):
        record["label"] = int(record["label"])
        record["dataset"] = dataset_name

    for idx in perm[:n_train]:
        records[idx]["split"] = "train"
    for idx in perm[n_train:]:
        records[idx]["split"] = "test"

    print(f"  Loaded {dataset_name}: {n} statements "
          f"({n_train} train, {n - n_train} test)")
    return records


# ---------------------------------------------------------------------------
# Truth activation extraction
# ---------------------------------------------------------------------------

def _format_truth_input(tokenizer, statement: str) -> torch.Tensor:
    """Format a true/false statement as a Llama 8B chat prompt for classification."""
    messages = [
        {"role": "user", "content": f"Is the following statement true or false? {statement}"}
    ]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return tokenizer(input_text, return_tensors="pt")["input_ids"]


def _extract_at_layers(
    model,
    input_ids: torch.Tensor,
    layers: list[int],
    device,
    position: int = -1,
) -> dict[int, np.ndarray]:
    """
    Run one forward pass and return hidden states at `position` for each layer.
    Returns {layer_idx: np.ndarray (float16, shape (hidden_dim,))}.
    """
    activations: dict[int, np.ndarray] = {}
    hooks = []

    def make_hook(layer_idx: int):
        def hook_fn(module, inp, output):
            hidden = output[0]
            # position=-1 gives the last token of the input
            activations[layer_idx] = (
                hidden[0, position, :]
                .detach()
                .float()
                .cpu()
                .numpy()
                .astype(np.float16)
            )
        return hook_fn

    for layer_idx in layers:
        h = model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
        hooks.append(h)

    with torch.no_grad():
        model(input_ids.to(device))

    for h in hooks:
        h.remove()

    return activations


def run_truth_extraction(
    model,
    tokenizer,
    config: Experiment9Config,
    device,
    force: bool = False,
) -> dict[str, list[dict]]:
    """
    Extract truth activations for all GoT datasets.

    Saves:  activations_dir_ex9 / "truth_activations_{dataset}.pt"

    Returns: {"cities": [...], "larger_than": [...]}
    Each record: {statement, label, dataset, split,
                  layer_activations: {layer_idx: np.ndarray}}
    """
    results: dict[str, list[dict]] = {}

    for dataset_name in config.got_datasets:
        out_path = config.activations_dir_ex9 / f"truth_activations_{dataset_name}.pt"

        if out_path.exists() and not force:
            print(f"  Found existing truth activations for '{dataset_name}', skipping.")
            results[dataset_name] = torch.load(out_path, weights_only=False)
            continue

        records = load_got_dataset(config, dataset_name)
        extracted: list[dict] = []

        print(f"  Extracting truth activations: {dataset_name} ({len(records)} statements)...")
        for rec in tqdm(records, desc=f"Truth/{dataset_name}"):
            input_ids = _format_truth_input(tokenizer, rec["statement"])
            layer_acts = _extract_at_layers(
                model, input_ids, config.truth_layers, device, position=-1
            )
            extracted.append({
                "statement": rec["statement"],
                "label": rec["label"],
                "dataset": dataset_name,
                "split": rec["split"],
                "layer_activations": layer_acts,
            })

        torch.save(extracted, out_path)
        print(f"  Saved truth activations → {out_path}")
        results[dataset_name] = extracted

    return results


# ---------------------------------------------------------------------------
# Eval / deploy CAA direction extraction
# ---------------------------------------------------------------------------

def _format_eval_deploy_input(tokenizer, system_prompt: str, user_message: str) -> torch.Tensor:
    """Format an eval/deploy prompt as a system+user chat with generation prompt."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return tokenizer(input_text, return_tensors="pt")["input_ids"]


def run_eval_deploy_extraction(
    model,
    tokenizer,
    config: Experiment9Config,
    device,
    force: bool = False,
) -> np.ndarray:
    """
    Compute the eval/deploy CAA direction at layer 30.

    The direction is: mean(eval activations) - mean(deploy activations), unit-normalised.

    Saves: results_dir_ex9 / "eval_deploy_direction.npy"
           results_dir_ex9 / "eval_deploy_direction.json"

    Returns: unit-norm np.ndarray of shape (hidden_dim,)
    """
    out_npy  = config.results_dir_ex9 / "eval_deploy_direction.npy"
    out_json = config.results_dir_ex9 / "eval_deploy_direction.json"

    if out_npy.exists() and not force:
        print("  Found existing eval/deploy direction, skipping.")
        return np.load(out_npy).astype(np.float32)

    layer_idx = config.primary_layer
    user_msg  = config.eval_deploy_neutral_message

    eval_acts:   list[np.ndarray] = []
    deploy_acts: list[np.ndarray] = []

    print(f"  Extracting eval/deploy activations at layer {layer_idx}...")
    for prompt in tqdm(config.eval_prompts, desc="Eval prompts"):
        input_ids = _format_eval_deploy_input(tokenizer, prompt, user_msg)
        acts = _extract_at_layers(model, input_ids, [layer_idx], device, position=-1)
        eval_acts.append(acts[layer_idx].astype(np.float32))

    for prompt in tqdm(config.deploy_prompts, desc="Deploy prompts"):
        input_ids = _format_eval_deploy_input(tokenizer, prompt, user_msg)
        acts = _extract_at_layers(model, input_ids, [layer_idx], device, position=-1)
        deploy_acts.append(acts[layer_idx].astype(np.float32))

    eval_mean   = np.mean(eval_acts, axis=0)
    deploy_mean = np.mean(deploy_acts, axis=0)
    direction   = eval_mean - deploy_mean
    norm = np.linalg.norm(direction)
    direction = direction / norm

    np.save(out_npy, direction.astype(np.float32))
    meta = {
        "layer": layer_idx,
        "n_eval_prompts":   len(config.eval_prompts),
        "n_deploy_prompts": len(config.deploy_prompts),
        "raw_norm": float(norm),
    }
    with open(out_json, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Eval/deploy direction saved → {out_npy}  (raw norm={norm:.2f})")
    return direction.astype(np.float32)


# ---------------------------------------------------------------------------
# Model loading / unloading helpers (re-exported for run_experiment9.py)
# ---------------------------------------------------------------------------

def load_model(config: Experiment9Config):
    """Load Llama 8B in fp16 on the best available device."""
    device_str = get_device()
    device_map = get_device_map()
    print(f"  Loading {config.model_id} (fp16) on {device_str}...")

    from transformers import AutoModelForCausalLM
    if device_map is not None:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_id, torch_dtype=torch.float16, device_map=device_map
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_id, torch_dtype=torch.float16
        ).to(device_str)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    device = next(model.parameters()).device
    return model, tokenizer, device


def unload_model(model) -> None:
    import gc
    del model
    gc.collect()
    clear_device_cache()
