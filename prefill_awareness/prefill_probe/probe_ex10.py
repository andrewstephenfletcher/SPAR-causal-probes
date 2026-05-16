"""
Probe training and transfer-matrix evaluation for Experiment 10.

For each of the 9 training conditions (cross_src × dataset) we train a
LinearProbe to separate Llama 70B self-prefill activations from cross-prefill
activations.  Every trained probe is then evaluated on all 9 test conditions,
producing a 9×9 AUROC transfer matrix.

Output — results_dir/transfer_matrix.json:
{
  "conditions": ["llama8b_bigcodebench", ...],   # 9 labels, row/col order
  "matrix": {
    "llama8b_bigcodebench": {                    # trained on this condition
      "llama8b_bigcodebench": {"auroc": 0.97, "n_test": 22},
      "llama8b_oasst1":       {"auroc": 0.88, "n_test": 20},
      ...
    },
    ...
  }
}
"""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .config import Experiment10Config
from .probe import apply_normaliser, evaluate_probe, train_probe


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _build_xy(
    self_records: list[dict],
    cross_records: list[dict],
    target_split: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pair self/cross records by prompt_id; return (X, y) for target_split.
    Label 0 = self, 1 = cross.
    """
    cross_by_pid = {r["prompt_id"]: r for r in cross_records}
    X_list, y_list = [], []
    for self_r in self_records:
        pid = self_r["prompt_id"]
        if self_r["split"] != target_split:
            continue
        cross_r = cross_by_pid.get(pid)
        if cross_r is None:
            continue
        X_list.append(self_r["activation"].astype(np.float32))
        y_list.append(0)
        X_list.append(cross_r["activation"].astype(np.float32))
        y_list.append(1)

    if not X_list:
        return np.empty((0, 1), dtype=np.float32), np.empty(0, dtype=int)
    return np.stack(X_list), np.array(y_list, dtype=int)


# ---------------------------------------------------------------------------
# Main training + evaluation
# ---------------------------------------------------------------------------

def train_and_evaluate(
    all_activations: dict[str, dict[str, list[dict]]],
    config: Experiment10Config,
    force: bool = False,
) -> dict:
    """
    Train 9 probes and evaluate in a 9×9 transfer matrix.
    Returns the result dict (also saved to results_dir/transfer_matrix.json).
    """
    output_path = config.results_dir / "transfer_matrix.json"
    if output_path.exists() and not force:
        print(f"  Loading existing transfer matrix from {output_path}")
        with open(output_path) as f:
            return json.load(f)

    wd_grid = config.probe_regularisation_grid
    cross_sources = config.cross_sources  # ["llama8b", "gemma31b", "qwen32b"]
    datasets = config.datasets            # ["bigcodebench", "oasst1", "gpqa"]

    # Condition label: "{cross_src}_{dataset}"
    condition_labels = [
        f"{src}_{ds}" for src in cross_sources for ds in datasets
    ]

    matrix: dict[str, dict] = {}

    for train_src in cross_sources:
        for train_ds in datasets:
            train_label = f"{train_src}_{train_ds}"
            print(f"\n  Training probe: {train_label}")

            self_records  = all_activations[train_ds]["self"]
            cross_records = all_activations[train_ds][f"cross_{train_src}"]

            X_train, y_train = _build_xy(self_records, cross_records, "train")
            X_val,   y_val   = _build_xy(self_records, cross_records, "val")

            if len(X_train) < 4 or len(np.unique(y_train)) < 2:
                print(f"    Insufficient training data for {train_label}, skipping.")
                matrix[train_label] = {}
                continue

            probe, mean, std, best_wd, val_acc = train_probe(
                X_train, y_train, X_val, y_val, wd_grid
            )
            print(f"    val_acc={val_acc:.4f}  best_wd={best_wd}")

            # Evaluate on all 9 test conditions
            test_results: dict[str, dict] = {}
            for test_src in cross_sources:
                for test_ds in datasets:
                    test_label = f"{test_src}_{test_ds}"
                    self_test  = all_activations[test_ds]["self"]
                    cross_test = all_activations[test_ds][f"cross_{test_src}"]

                    X_test, y_test = _build_xy(self_test, cross_test, "test")

                    if len(X_test) < 2 or len(np.unique(y_test)) < 2:
                        test_results[test_label] = {"auroc": None, "n_test": len(X_test)}
                        continue

                    _, test_auroc, _ = evaluate_probe(probe, mean, std, X_test, y_test)
                    test_results[test_label] = {
                        "auroc": float(test_auroc),
                        "n_test": int(len(X_test)),
                    }
                    print(f"    → {test_label}: AUROC={test_auroc:.4f}")

            matrix[train_label] = test_results

    result = {
        "conditions": condition_labels,
        "matrix": matrix,
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Transfer matrix saved → {output_path}")
    return result
