"""
Steering utilities for Experiment 6 (Steering Control Analysis).

Provides:
  - generate_random_steering_vectors: n random unit-norm vectors
  - compute_layer_norms: per-layer mean residual stream norm over a prompt sample
  - extract_probe_directions: probe direction at each steering layer, cached to Ex6 results dir

Reuses make_steering_hook and steering_context from steer_ex5.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .config import Experiment6Config
from .extract_ex4 import build_prefill_input, get_transformer_layers
from .probe import build_dataset_for_layer, evaluate_probe, train_probe
from .steer_ex5 import make_steering_hook, steering_context  # noqa: F401 (re-exported)


# ---------------------------------------------------------------------------
# Random vector generation
# ---------------------------------------------------------------------------

def generate_random_steering_vectors(
    hidden_dim: int,
    n_vectors: int,
    seed: int = 42,
) -> list[torch.Tensor]:
    """
    Return n_vectors random unit-norm vectors in the residual stream space.
    Each is drawn from a standard normal distribution and L2-normalised.
    Stored as float16 to match model activations.
    """
    rng = np.random.RandomState(seed)
    vecs: list[torch.Tensor] = []
    for _ in range(n_vectors):
        v = rng.randn(hidden_dim).astype(np.float32)
        v /= np.linalg.norm(v)
        vecs.append(torch.tensor(v, dtype=torch.float16))
    return vecs


# ---------------------------------------------------------------------------
# Per-layer residual norm estimation
# ---------------------------------------------------------------------------

def compute_layer_norms(
    model,
    tokenizer,
    model_id: str,
    responses: list[dict],
    layers: list[int],
    n_samples: int = 20,
) -> dict[int, float]:
    """
    Compute the mean L2 norm of the residual stream at each layer in `layers`
    over the first n_samples prompts, using the self-prefill forward pass.

    Returns {layer_idx: mean_norm}.
    """
    tf_layers = get_transformer_layers(model)
    device = next(model.parameters()).device
    layer_norms: dict[int, float] = {}

    for layer_idx in tqdm(layers, desc="Layer norms"):
        norms: list[float] = []
        captured: dict[str, float] = {}

        def _hook(module, input, output, _cap=captured):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            # Mean norm across all token positions in the sequence
            _cap["norm"] = hidden[0].detach().float().norm(dim=-1).mean().item()

        handle = tf_layers[layer_idx].register_forward_hook(_hook)

        for r in responses[:n_samples]:
            prefill_ids, _, _ = build_prefill_input(
                tokenizer, model_id, r["instruction"], r["response_llama70b"]
            )
            with torch.no_grad():
                model(prefill_ids.to(device))
            norms.append(captured["norm"])

        handle.remove()
        mean_norm = float(np.mean(norms))
        layer_norms[layer_idx] = mean_norm
        print(f"  Layer {layer_idx:2d}: mean residual norm = {mean_norm:.1f}")

    return layer_norms


# ---------------------------------------------------------------------------
# Probe direction extraction at multiple layers
# ---------------------------------------------------------------------------

def _load_pt(path: Path) -> list[dict]:
    return torch.load(path, weights_only=False)


def _build_split_map(responses: list[dict]) -> dict[int, str]:
    return {r["prompt_id"]: r["split"] for r in responses}


def extract_probe_directions(
    responses: list[dict],
    config: Experiment6Config,
    force: bool = False,
) -> dict[int, np.ndarray]:
    """
    Train (or load cached) probe direction at each layer in config.steering_layers.

    Uses Experiment 4 activations (self_prefill vs probe_condition).
    Saves each direction as probe_direction_layer{N}_{condition}.npy in results_dir_ex6.

    Returns {layer_idx: unit_norm_float32_vector}.
    """
    acts_dir = config.ex4_activations_dir_llama70b
    self_data = _load_pt(acts_dir / "self_prefill.pt")
    cross_data = _load_pt(acts_dir / f"{config.probe_condition}.pt")
    split_map = _build_split_map(responses)

    directions: dict[int, np.ndarray] = {}

    for layer_idx in config.steering_layers:
        out_npy = (
            config.results_dir_ex6
            / f"probe_direction_layer{layer_idx}_{config.probe_condition}.npy"
        )
        out_json = out_npy.with_suffix(".json")

        if out_npy.exists() and not force:
            vec = np.load(out_npy).astype(np.float32)
            print(f"  Layer {layer_idx:2d}: loaded from cache  norm={np.linalg.norm(vec):.4f}")
            directions[layer_idx] = vec
            continue

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
        print(f"  Layer {layer_idx:2d}: AUROC={test_auroc:.4f}  acc={test_acc:.4f}  wd={best_wd}")

        # Convert normalised-space weight to original activation space, then unit-normalise.
        # Probe decision boundary: w^T (x - mean) / std = 0  =>  direction = w / std (unnorm).
        weight_norm = probe.linear.weight.data[0].numpy().astype(np.float32)
        weight_orig = weight_norm / std.astype(np.float32)
        weight_orig = weight_orig / np.linalg.norm(weight_orig)

        np.save(out_npy, weight_orig)
        with open(out_json, "w") as f:
            json.dump({
                "layer_idx": layer_idx,
                "condition": config.probe_condition,
                "test_auroc": float(test_auroc),
                "test_acc": float(test_acc),
                "best_wd": float(best_wd),
                "n_train": int(len(X_train)),
                "n_test": int(len(X_test)),
            }, f, indent=2)

        directions[layer_idx] = weight_orig

    return directions


# ---------------------------------------------------------------------------
# Norm computation persistence
# ---------------------------------------------------------------------------

def save_layer_norms(layer_norms: dict[int, float], config: Experiment6Config) -> None:
    out = config.results_dir_ex6 / "layer_norms.json"
    with open(out, "w") as f:
        json.dump({str(k): v for k, v in layer_norms.items()}, f, indent=2)
    print(f"  Layer norms saved → {out}")


def load_layer_norms(config: Experiment6Config) -> dict[int, float] | None:
    out = config.results_dir_ex6 / "layer_norms.json"
    if not out.exists():
        return None
    with open(out) as f:
        raw = json.load(f)
    return {int(k): float(v) for k, v in raw.items()}
