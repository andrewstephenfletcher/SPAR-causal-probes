"""
Probe training for Experiment 8 (Cross-Architecture Probing).

Trains one LinearProbe per (layer, position) cell for Qwen 32B and
Gemma 31B, using the same training procedure as Experiments 1 and 2.
Also computes the cumulative perplexity baseline AUROC at each position.

Results saved as:
  results_dir_ex8/probe_results_mistral.csv
  results_dir_ex8/probe_results_gemma31b.csv

CSV columns: model, layer, position, balanced_accuracy, auroc, n_test,
             perplexity_baseline_auroc, best_wd
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from .config import Experiment8Config
from .probe import evaluate_probe, train_probe
from .probe_positions import compute_cumulative_perplexity_auroc


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_activations(activations_dir: Path) -> tuple[list[dict], list[dict]]:
    self_data  = torch.load(activations_dir / "self_prefill_positions.pt", weights_only=False)
    cross_data = torch.load(activations_dir / "cross_prefill_positions.pt", weights_only=False)
    return self_data, cross_data


# ---------------------------------------------------------------------------
# Cell dataset assembly
# ---------------------------------------------------------------------------

def _build_cell_dataset(
    self_data: list[dict],
    cross_data: list[dict],
    split_map: dict[int, str],
    layer: int,
    position: int,
) -> dict[str, dict]:
    """Build {train, val, test} dataset for one (layer, position) cell."""
    cross_by_pid = {d["prompt_id"]: d for d in cross_data}
    data: dict[str, dict] = {s: {"X": [], "y": []} for s in ("train", "val", "test")}
    key = (layer, position)

    for self_item in self_data:
        pid = self_item["prompt_id"]
        split = split_map.get(pid)
        if split is None:
            continue
        cross_item = cross_by_pid.get(pid)
        if cross_item is None:
            continue

        self_act  = self_item["activations"].get(key)
        cross_act = cross_item["activations"].get(key)
        if self_act is None or cross_act is None:
            continue

        # For train, carve last 20% as val (matching probe_ex9.py convention)
        data[split]["X"].extend([self_act.astype(np.float32), cross_act.astype(np.float32)])
        data[split]["y"].extend([0, 1])

    # Carve val from train (last 20%)
    if data["train"]["X"] and not data["val"]["X"]:
        n = len(data["train"]["X"])
        cut = max(1, int(n * 0.8))
        data["val"]["X"] = data["train"]["X"][cut:]
        data["val"]["y"] = data["train"]["y"][cut:]
        data["train"]["X"] = data["train"]["X"][:cut]
        data["train"]["y"] = data["train"]["y"][:cut]

    result = {}
    d_model = None
    for split, d in data.items():
        if d["X"]:
            X = np.stack(d["X"])
            d_model = X.shape[1]
        else:
            X = np.empty((0, d_model or 4096))
        result[split] = {"X": X, "y": np.array(d["y"], dtype=int)}

    return result


# ---------------------------------------------------------------------------
# Main probe training
# ---------------------------------------------------------------------------

def train_probes_for_model(
    model_name: str,
    activations_dir: Path,
    layers: list[int],
    config: Experiment8Config,
    force: bool = False,
) -> pd.DataFrame:
    """Train all (layer, position) probes for one model."""
    csv_path = config.results_dir_ex8 / f"probe_results_{model_name}.csv"
    if csv_path.exists() and not force:
        print(f"  Found existing probe results at {csv_path}, loading...")
        return pd.read_csv(csv_path)

    self_data, cross_data = _load_activations(activations_dir)
    split_map = {r["prompt_id"]: r["split"] for r in self_data}

    print(f"  Computing cumulative perplexity baselines ({model_name})...")
    ppl_auroc_by_pos: dict[int, float] = {}
    for pos in config.positions:
        ppl_auroc_by_pos[pos] = compute_cumulative_perplexity_auroc(
            self_data, cross_data, split_map, pos
        )

    n_cells = len(layers) * len(config.positions)
    print(f"  Training {n_cells} probes ({len(layers)} layers × "
          f"{len(config.positions)} positions) for {model_name}...")

    rows: list[dict] = []
    with tqdm(total=n_cells, desc=f"Probes ({model_name})") as pbar:
        for layer in layers:
            for pos in config.positions:
                dataset = _build_cell_dataset(self_data, cross_data, split_map, layer, pos)
                X_tr, y_tr = dataset["train"]["X"], dataset["train"]["y"]
                X_v,  y_v  = dataset["val"]["X"],   dataset["val"]["y"]
                X_te, y_te = dataset["test"]["X"],  dataset["test"]["y"]

                n_test = len(y_te)
                if len(X_tr) < 4 or len(X_v) < 2 or len(X_te) < 2:
                    rows.append(_empty_row(model_name, layer, pos, n_test, ppl_auroc_by_pos[pos]))
                    pbar.update(1)
                    continue
                if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                    rows.append(_empty_row(model_name, layer, pos, n_test, ppl_auroc_by_pos[pos]))
                    pbar.update(1)
                    continue

                probe, mean, std, best_wd, _ = train_probe(
                    X_tr, y_tr, X_v, y_v,
                    wd_grid=[1e-3, 1e-2, 1e-1, 1.0],
                )
                test_acc, test_auroc, _ = evaluate_probe(probe, mean, std, X_te, y_te)

                rows.append({
                    "model": model_name,
                    "layer": layer,
                    "position": pos,
                    "balanced_accuracy": round(test_acc, 5),
                    "auroc": round(test_auroc, 5),
                    "n_test": n_test,
                    "perplexity_baseline_auroc": round(ppl_auroc_by_pos[pos], 5),
                    "best_wd": best_wd,
                })
                pbar.update(1)

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"  Saved probe results → {csv_path}")
    return df


def _empty_row(model: str, layer: int, pos: int, n_test: int, ppl_auroc: float) -> dict:
    return {
        "model": model, "layer": layer, "position": pos,
        "balanced_accuracy": float("nan"), "auroc": float("nan"),
        "n_test": n_test, "perplexity_baseline_auroc": ppl_auroc,
        "best_wd": float("nan"),
    }


def run_probing_ex8(
    config: Experiment8Config,
    model: str = "both",
    force: bool = False,
) -> dict[str, pd.DataFrame]:
    """Train probes for Mistral and/or Gemma 31B. Returns {model_name: DataFrame}."""
    results = {}

    if model in ("mistral", "both"):
        df = train_probes_for_model(
            "mistral", config.activations_dir_mistral, config.mistral_layers, config, force=force
        )
        results["mistral"] = df

    if model in ("gemma", "both"):
        df = train_probes_for_model(
            "gemma31b", config.activations_dir_gemma31b, config.gemma_layers, config, force=force
        )
        results["gemma31b"] = df

    return results
