"""
Probe training and evaluation for Experiment 2 (position-wise analysis).

Trains one LinearProbe per (layer, position) cell using the same architecture
and training procedure as Experiment 1.  Also computes the cumulative
perplexity baseline AUROC at each position.

Results are saved as probe_results.csv with one row per (layer, position) cell.
"""

import csv
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from .config import Experiment2Config
from .probe import evaluate_probe, fit_normaliser, train_probe


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_position_activations(
    ex2_config: Experiment2Config,
) -> tuple[list[dict], list[dict]]:
    """Load self- and cross-model activation records from Experiment 2."""
    self_data = torch.load(
        ex2_config.activations_dir_ex2 / "self_prefill_positions.pt",
        weights_only=False,
    )
    cross_data = torch.load(
        ex2_config.activations_dir_ex2 / "cross_gemma_prefill_positions.pt",
        weights_only=False,
    )
    return self_data, cross_data


# ---------------------------------------------------------------------------
# Dataset assembly for a single (layer, position) cell
# ---------------------------------------------------------------------------

def build_cell_dataset(
    self_data: list[dict],
    cross_data: list[dict],
    split_map: dict[int, str],
    layer: int,
    position: int,
) -> dict[str, dict]:
    """
    Return {'train': {'X': ..., 'y': ...}, 'val': ..., 'test': ...}
    for a single (layer, position) cell.

    Label 0 = self-prefill, 1 = cross-model prefill.
    """
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

        self_act = self_item["activations"].get(key)
        cross_act = cross_item["activations"].get(key)
        if self_act is None or cross_act is None:
            continue

        data[split]["X"].extend([
            self_act.astype(np.float32),
            cross_act.astype(np.float32),
        ])
        data[split]["y"].extend([0, 1])

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
# Cumulative perplexity baseline
# ---------------------------------------------------------------------------

def compute_cumulative_perplexity_auroc(
    self_data: list[dict],
    cross_data: list[dict],
    split_map: dict[int, str],
    position: int,
) -> float:
    """
    AUROC of the single-feature classifier: -cumulative_mean_log_prob up to
    `position` (inclusive), evaluated on the TEST split.

    Higher -mean_log_prob → more surprising → more likely cross-model.
    """
    cross_by_pid = {d["prompt_id"]: d for d in cross_data}

    scores: list[float] = []
    labels: list[int] = []

    for self_item in self_data:
        pid = self_item["prompt_id"]
        if split_map.get(pid) != "test":
            continue
        cross_item = cross_by_pid.get(pid)
        if cross_item is None:
            continue

        # Cumulative mean log-prob up to position (inclusive)
        self_lps = [
            lp for lp in self_item["log_probs_per_response_token"][: position + 1]
            if not np.isnan(lp)
        ]
        cross_lps = [
            lp for lp in cross_item["log_probs_per_response_token"][: position + 1]
            if not np.isnan(lp)
        ]
        if not self_lps or not cross_lps:
            continue

        scores.append(-float(np.mean(self_lps)))   # self (label 0)
        labels.append(0)
        scores.append(-float(np.mean(cross_lps)))  # cross (label 1)
        labels.append(1)

    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


# ---------------------------------------------------------------------------
# Main probe training entry point
# ---------------------------------------------------------------------------

def train_all_position_probes(
    responses: list[dict],
    ex2_config: Experiment2Config,
    force: bool = False,
) -> pd.DataFrame:
    """
    Train one LinearProbe per (layer, position) cell and return results as a
    DataFrame.  Saves probe_results.csv to results_dir_ex2.
    """
    csv_path = ex2_config.results_dir_ex2 / "probe_results.csv"

    if csv_path.exists() and not force:
        print(f"  Found existing probe results at {csv_path}, loading...")
        return pd.read_csv(csv_path)

    self_data, cross_data = load_position_activations(ex2_config)
    split_map = {r["prompt_id"]: r["split"] for r in responses}

    # Pre-compute perplexity baseline for every position (independent of layer)
    print("  Computing cumulative perplexity baselines per position...")
    ppl_auroc_by_pos: dict[int, float] = {}
    for pos in ex2_config.positions:
        ppl_auroc_by_pos[pos] = compute_cumulative_perplexity_auroc(
            self_data, cross_data, split_map, pos
        )

    rows: list[dict] = []
    n_cells = len(ex2_config.layers) * len(ex2_config.positions)

    print(f"  Training {n_cells} probes ({len(ex2_config.layers)} layers × "
          f"{len(ex2_config.positions)} positions)...")

    with tqdm(total=n_cells, desc="Probing (layer, pos)") as pbar:
        for layer in ex2_config.layers:
            for pos in ex2_config.positions:
                dataset = build_cell_dataset(
                    self_data, cross_data, split_map, layer, pos
                )
                X_tr, y_tr = dataset["train"]["X"], dataset["train"]["y"]
                X_v,  y_v  = dataset["val"]["X"],   dataset["val"]["y"]
                X_te, y_te = dataset["test"]["X"],  dataset["test"]["y"]

                n_test = len(y_te)

                if len(X_tr) < 2 or len(X_v) < 2 or len(X_te) < 2:
                    rows.append(_empty_row(layer, pos, n_test, ppl_auroc_by_pos[pos]))
                    pbar.update(1)
                    continue
                if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                    rows.append(_empty_row(layer, pos, n_test, ppl_auroc_by_pos[pos]))
                    pbar.update(1)
                    continue

                probe, mean, std, best_wd, _val_acc = train_probe(
                    X_tr, y_tr, X_v, y_v,
                    # Use a tighter grid for speed — the best layer (30) is already
                    # identified; here we scan many cells.
                    wd_grid=[1e-3, 1e-2, 1e-1, 1.0],
                )
                test_acc, test_auroc, _ = evaluate_probe(probe, mean, std, X_te, y_te)

                rows.append({
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


def _empty_row(layer: int, pos: int, n_test: int, ppl_auroc: float) -> dict:
    return {
        "layer": layer, "position": pos,
        "balanced_accuracy": float("nan"), "auroc": float("nan"),
        "n_test": n_test, "perplexity_baseline_auroc": ppl_auroc,
        "best_wd": float("nan"),
    }
