"""
Steering vector extraction and hook construction for Experiment 5.

The steering direction is the weight vector of a logistic-regression probe
(LinearProbe from probe.py) retrained at a specific layer on Experiment 4
activations.  The direction is converted to the original (un-normalised)
activation space so it can be added directly to the residual stream.

alpha > 0 → steer toward "not-self" (cross-model direction)
alpha < 0 → steer toward "self"
alpha = 0 → no effect (baseline)
"""

import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .config import Experiment5Config
from .extract_ex4 import build_prefill_input, get_transformer_layers
from .probe import (
    apply_normaliser,
    build_dataset_for_layer,
    evaluate_probe,
    fit_normaliser,
    train_probe,
)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_pt(path: Path) -> list[dict]:
    return torch.load(path, weights_only=False)


def _build_split_map(responses: list[dict]) -> dict[int, str]:
    return {r["prompt_id"]: r["split"] for r in responses}


# ---------------------------------------------------------------------------
# Probe direction extraction
# ---------------------------------------------------------------------------

def extract_probe_direction(
    responses: list[dict],
    config: Experiment5Config,
    layer_idx: int,
    condition: str = "cross_gemma9b",
    force: bool = False,
) -> np.ndarray:
    """
    Retrain the probe at `layer_idx` for the given condition and return the
    steering direction in the original activation space (unit norm, float32).

    The positive direction points toward class 1 ("not-self" / cross-model).
    Saved as .npy alongside a .json metadata file.
    """
    out_path = config.results_dir_ex5 / f"probe_direction_layer{layer_idx}_{condition}.npy"
    if out_path.exists() and not force:
        vec = np.load(out_path).astype(np.float32)
        print(f"  Loaded steering direction from {out_path}  norm={np.linalg.norm(vec):.4f}")
        return vec

    acts_dir = config.ex4_activations_dir_llama70b
    self_data = _load_pt(acts_dir / "self_prefill.pt")
    cross_data = _load_pt(acts_dir / f"{condition}.pt")
    split_map = _build_split_map(responses)

    dataset = build_dataset_for_layer(self_data, cross_data, split_map, layer_idx)

    X_train, y_train = dataset["train"]["X"], dataset["train"]["y"]
    X_val,   y_val   = dataset["val"]["X"],   dataset["val"]["y"]
    X_test,  y_test  = dataset["test"]["X"],  dataset["test"]["y"]

    if len(X_train) < 2 or len(np.unique(y_train)) < 2:
        raise ValueError(f"Insufficient training data at layer {layer_idx}")

    print(f"  Training probe at layer {layer_idx} "
          f"(n_train={len(X_train)}, n_test={len(X_test)})...")
    probe, mean, std, best_wd, _ = train_probe(
        X_train, y_train, X_val, y_val, config.probe_regularisation_grid
    )
    test_acc, test_auroc, _ = evaluate_probe(probe, mean, std, X_test, y_test)
    print(f"  Layer {layer_idx}: AUROC={test_auroc:.4f}, acc={test_acc:.4f}, wd={best_wd}")

    # Convert from normalised feature space to original activation space.
    # Probe decision: w^T * (x - mean) / std = 0  ⟹ direction = w / std (unnorm).
    weight_norm = probe.linear.weight.data[0].numpy().astype(np.float32)
    weight_orig = weight_norm / std.astype(np.float32)
    weight_orig = weight_orig / np.linalg.norm(weight_orig)

    np.save(out_path, weight_orig)

    meta = {
        "layer_idx": layer_idx,
        "condition": condition,
        "test_auroc": float(test_auroc),
        "test_acc": float(test_acc),
        "best_wd": float(best_wd),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    with open(out_path.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved steering direction → {out_path}")
    return weight_orig


def random_steering_direction(hidden_dim: int, seed: int = 99) -> np.ndarray:
    """Unit-norm random vector for the random-vector control."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(hidden_dim).astype(np.float32)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# Steering hook
# ---------------------------------------------------------------------------

def make_steering_hook(steering_vec: torch.Tensor, alpha: float):
    """
    Returns a forward hook that adds alpha * steering_vec to the layer output.
    Works with tuple outputs (LlamaDecoderLayer returns a tuple).
    """
    def hook_fn(module, input, output):
        if isinstance(output, (tuple, list)):
            hidden = output[0]
        else:
            hidden = output

        sv = steering_vec.to(hidden.device).to(hidden.dtype)
        hidden = hidden + alpha * sv.unsqueeze(0).unsqueeze(0)

        if isinstance(output, (tuple, list)):
            return (hidden,) + tuple(output[1:])
        return hidden

    return hook_fn


@contextmanager
def steering_context(
    model,
    layer_idx: int,
    steering_vec: torch.Tensor,
    alpha: float,
):
    """Context manager: registers hook, yields, removes hook."""
    layers = get_transformer_layers(model)
    hook = layers[layer_idx].register_forward_hook(
        make_steering_hook(steering_vec, alpha)
    )
    try:
        yield
    finally:
        hook.remove()


# ---------------------------------------------------------------------------
# Alpha calibration
# ---------------------------------------------------------------------------

def _measure_residual_norm(
    model,
    tokenizer,
    model_id: str,
    responses: list[dict],
    layer_idx: int,
    n_samples: int,
) -> float:
    """Mean L2 norm of the residual stream at layer_idx over n_samples prompts."""
    layers = get_transformer_layers(model)
    device = next(model.parameters()).device
    norms: list[float] = []

    for r in responses[:n_samples]:
        prefill_ids, last_pos, _ = build_prefill_input(
            tokenizer, model_id, r["instruction"], r["response_llama70b"]
        )
        prefill_ids = prefill_ids.to(device)

        captured: dict = {}

        def _hook(module, input, output):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            captured["norm"] = hidden[0, last_pos, :].detach().float().norm().item()

        h = layers[layer_idx].register_forward_hook(_hook)
        with torch.no_grad():
            model(prefill_ids)
        h.remove()

        norms.append(captured["norm"])

    return float(np.mean(norms))


def calibrate_alpha(
    model,
    tokenizer,
    model_id: str,
    responses: list[dict],
    config: Experiment5Config,
    force: bool = False,
) -> dict:
    """
    Compute mean residual norm at the steering layer and propose alpha values
    as fractions of that norm.  Saves calibration.json.

    Conservative ≈ 0.5%, moderate ≈ 2%, aggressive ≈ 8% of mean residual norm.
    """
    out_path = config.results_dir_ex5 / "alpha_calibration.json"
    if out_path.exists() and not force:
        with open(out_path) as f:
            result = json.load(f)
        print(f"  Loaded calibration: mean_norm={result['mean_norm']:.1f}  "
              f"moderate={result['alpha_moderate']:.4f}")
        return result

    print(f"  Measuring residual norm at layer {config.steering_layer} "
          f"over {config.n_prompts_calibrate} prompts...")
    mean_norm = _measure_residual_norm(
        model, tokenizer, model_id, responses,
        config.steering_layer, config.n_prompts_calibrate,
    )
    print(f"  Mean residual norm: {mean_norm:.2f}")

    # Alpha = fraction × (mean_norm / 100)
    scale = mean_norm / 100.0
    fractions = [0.5, 1.0, 2.0, 4.0, 8.0]
    candidates = [round(f * scale, 4) for f in fractions]

    result = {
        "mean_norm": float(mean_norm),
        "scale": float(scale),
        "layer": config.steering_layer,
        "alpha_candidates": candidates,
        "alpha_conservative": candidates[0],  # 0.5% of mean_norm
        "alpha_moderate": candidates[2],       # 2% of mean_norm
        "alpha_aggressive": candidates[4],     # 8% of mean_norm
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  Alpha candidates (×scale={scale:.4f}): {candidates}")
    print(f"    conservative={result['alpha_conservative']:.4f}")
    print(f"    moderate    ={result['alpha_moderate']:.4f}")
    print(f"    aggressive  ={result['alpha_aggressive']:.4f}")
    print(f"  Calibration saved → {out_path}")
    return result
