"""
Probe utilities for Experiment 11b.

Provides:
  - DimProbe: Difference-in-Means probe (no gradient training)
  - Data loading from Experiment 11 .pt activation files
  - Split map loading from Experiment 11 generation JSONs
  - Dataset building: pooled (self vs. all 3 cross-sources) and per-source modes
  - Prompt-ID alignment across conditions (handles 1-2 missing records per file)
  - Probe saving helpers

Reuses from probe.py:
  LinearProbe, train_probe, evaluate_probe, fit_normaliser, apply_normaliser
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from .config import Experiment11bConfig
from .probe import (
    LinearProbe,
    apply_normaliser,
    evaluate_probe,
    fit_normaliser,
    train_probe,
)


# ---------------------------------------------------------------------------
# DimProbe
# ---------------------------------------------------------------------------

@dataclass
class DimProbe:
    """Difference-in-Means probe. No gradient training needed."""
    direction: np.ndarray  # unit vector, shape (hidden_dim,)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return raw dot-product scores (higher = more cross-model)."""
        return X.astype(np.float32) @ self.direction


def train_dim_probe(X_train: np.ndarray, y_train: np.ndarray) -> DimProbe:
    pos = X_train[y_train == 1]
    neg = X_train[y_train == 0]
    direction = pos.mean(0) - neg.mean(0)
    norm = np.linalg.norm(direction)
    return DimProbe(direction / norm if norm > 1e-12 else direction)


def evaluate_dim_probe(
    probe: DimProbe,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[float, float]:
    """Returns (auroc, balanced_accuracy)."""
    scores = probe.score(X_test)
    auroc = float(roc_auc_score(y_test, scores))
    preds = (scores > np.median(scores)).astype(int)
    bacc = float(balanced_accuracy_score(y_test, preds))
    return auroc, bacc


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_activations_ex11(
    target_key: str,
    condition: str,
    dataset: str,
    config: Experiment11bConfig,
) -> list[dict]:
    path = config.exp11_activations_dir / target_key / condition / f"{dataset}.pt"
    return torch.load(path, weights_only=False)


def load_token_position_activations_ex11(
    target_key: str,
    condition: str,
    dataset: str,
    config: Experiment11bConfig,
) -> list[dict]:
    path = (
        config.exp11_activations_dir / target_key / condition
        / f"{dataset}_token_positions.pt"
    )
    return torch.load(path, weights_only=False)


def get_split_map(
    target_key: str,
    dataset: str,
    config: Experiment11bConfig,
) -> dict[int, str]:
    """Load prompt_id → split from the target model's self generation file."""
    path = config.exp11_generations_dir / f"{target_key}_{dataset}_responses.json"
    with open(path) as f:
        records = json.load(f)
    return {r["prompt_id"]: r["split"] for r in records}


def get_raw_responses(
    model_key: str,
    dataset: str,
    config: Experiment11bConfig,
) -> list[dict]:
    """Load full response records for a model+dataset."""
    path = config.exp11_generations_dir / f"{model_key}_{dataset}_responses.json"
    with open(path) as f:
        return json.load(f)


def detect_n_layers(
    target_key: str,
    config: Experiment11bConfig,
) -> int:
    data = load_activations_ex11(target_key, "self", config.datasets[0], config)
    return len(data[0]["layer_activations"])


def get_common_prompt_ids(data_lists: list[list[dict]]) -> set[int]:
    """Intersection of prompt_ids across all provided activation lists."""
    sets = [set(r["prompt_id"] for r in d) for d in data_lists]
    return set.intersection(*sets)


# ---------------------------------------------------------------------------
# Dataset building
# ---------------------------------------------------------------------------

def build_dataset_pooled(
    self_data: list[dict],
    cross_data_list: list[list[dict]],  # one per cross condition
    split_map: dict[int, str],
    layer_idx: int,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Build X/y arrays for one layer in pooled mode (self vs. all cross-sources).

    Uses only prompt_ids present in self AND every cross condition.
    Label 0 = self, 1 = cross-model.
    Returns {"train": {"X": ..., "y": ...}, "val": {...}, "test": {...}}.
    """
    common_pids = get_common_prompt_ids([self_data] + cross_data_list)
    self_by_pid = {r["prompt_id"]: r for r in self_data}
    cross_by_pid_list = [{r["prompt_id"]: r for r in cd} for cd in cross_data_list]

    splits: dict[str, dict] = {s: {"X": [], "y": []} for s in ("train", "val", "test")}

    for pid in common_pids:
        split = split_map.get(pid)
        if split not in splits:
            continue
        self_act = self_by_pid[pid]["layer_activations"][layer_idx].astype(np.float32)
        splits[split]["X"].append(self_act)
        splits[split]["y"].append(0)
        for cross_by_pid in cross_by_pid_list:
            if pid not in cross_by_pid:
                continue
            cross_act = cross_by_pid[pid]["layer_activations"][layer_idx].astype(np.float32)
            splits[split]["X"].append(cross_act)
            splits[split]["y"].append(1)

    return _finalise_splits(splits)


def build_dataset_per_source(
    self_data: list[dict],
    cross_data: list[dict],
    split_map: dict[int, str],
    layer_idx: int,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Build X/y arrays for one layer in per-source mode (self vs. one source).
    Same interface as build_dataset_pooled.
    """
    common_pids = get_common_prompt_ids([self_data, cross_data])
    self_by_pid  = {r["prompt_id"]: r for r in self_data}
    cross_by_pid = {r["prompt_id"]: r for r in cross_data}

    splits: dict[str, dict] = {s: {"X": [], "y": []} for s in ("train", "val", "test")}

    for pid in common_pids:
        split = split_map.get(pid)
        if split not in splits:
            continue
        self_act  = self_by_pid[pid]["layer_activations"][layer_idx].astype(np.float32)
        cross_act = cross_by_pid[pid]["layer_activations"][layer_idx].astype(np.float32)
        splits[split]["X"].extend([self_act, cross_act])
        splits[split]["y"].extend([0, 1])

    return _finalise_splits(splits)


def _finalise_splits(
    splits: dict[str, dict],
) -> dict[str, dict[str, np.ndarray]]:
    result = {}
    d_model = None
    for split_name, d in splits.items():
        if d["X"]:
            X = np.stack(d["X"])
            d_model = X.shape[1]
        else:
            X = np.empty((0, d_model or 4096))
        result[split_name] = {"X": X, "y": np.array(d["y"], dtype=int)}
    return result


# ---------------------------------------------------------------------------
# One-layer probe training convenience wrapper
# ---------------------------------------------------------------------------

def train_and_evaluate_layer(
    dataset: dict[str, dict[str, np.ndarray]],
    wd_grid: list[float],
) -> dict:
    """
    Train LinearProbe + DimProbe on a pre-built split dict.
    Returns per-layer metric dict.
    """
    X_train, y_train = dataset["train"]["X"], dataset["train"]["y"]
    X_val,   y_val   = dataset["val"]["X"],   dataset["val"]["y"]
    X_test,  y_test  = dataset["test"]["X"],  dataset["test"]["y"]

    result: dict = {}

    # Skip under-populated splits
    if (len(X_train) < 4 or len(X_val) < 2 or len(X_test) < 2
            or len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2):
        result["skipped"] = True
        result["lr_auroc"] = float("nan")
        result["dim_auroc"] = float("nan")
        result["lr_balanced_acc"] = float("nan")
        result["dim_balanced_acc"] = float("nan")
        return result

    # LinearProbe
    probe, mean, std, best_wd, _ = train_probe(
        X_train, y_train, X_val, y_val, wd_grid
    )
    lr_bacc, lr_auroc, weight_norm = evaluate_probe(probe, mean, std, X_test, y_test)

    # DimProbe (trained on train, no val selection needed)
    dim_probe = train_dim_probe(X_train, y_train)
    dim_auroc, dim_bacc = evaluate_dim_probe(dim_probe, X_test, y_test)

    result.update({
        "skipped": False,
        "lr_auroc":        lr_auroc,
        "lr_balanced_acc": lr_bacc,
        "lr_best_wd":      best_wd,
        "lr_weight_norm":  weight_norm,
        "dim_auroc":       dim_auroc,
        "dim_balanced_acc": dim_bacc,
        "n_train": len(X_train),
        "n_val":   len(X_val),
        "n_test":  len(X_test),
    })
    return result


# ---------------------------------------------------------------------------
# Probe saving / loading
# ---------------------------------------------------------------------------

def save_lr_probe(
    probe: LinearProbe,
    mean: np.ndarray,
    std: np.ndarray,
    wd: float,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": probe.state_dict(), "mean": mean, "std": std, "wd": wd}, path)


def save_dim_probe(probe: DimProbe, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), probe.direction)


def load_lr_probe(path: Path, d_model: int) -> tuple[LinearProbe, np.ndarray, np.ndarray]:
    ckpt = torch.load(path, weights_only=False)
    probe = LinearProbe(d_model)
    probe.load_state_dict(ckpt["state_dict"])
    probe.eval()
    return probe, ckpt["mean"], ckpt["std"]


def load_dim_probe(path: Path) -> DimProbe:
    return DimProbe(np.load(str(path)))
