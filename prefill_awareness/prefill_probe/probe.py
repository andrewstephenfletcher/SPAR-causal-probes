"""
Linear probe training and evaluation for Experiment 1.

Architecture: LinearProbe — a single nn.Linear(d_model, 1) layer trained
with BCEWithLogitsLoss and Adam.  L2 regularisation is applied via weight
decay; the value is chosen by grid search on the validation set.

Probe training is done on CPU with full-batch gradient descent (probes are
small — the bottleneck is activation loading, not gradient computation).
"""

import json

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from tqdm import tqdm

from .config import Config


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

class Probe(nn.Module):
    pass


class LinearProbe(Probe):
    """Single linear layer: d_model → 1 logit."""

    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)  # (batch,)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def fit_normaliser(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-feature mean and std from training data."""
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    return mean, std


def apply_normaliser(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


# ---------------------------------------------------------------------------
# Probe training
# ---------------------------------------------------------------------------

def train_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    wd_grid: list[float],
    n_epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    clip_grad_norm: float = 1.0,
) -> tuple:
    """
    Train LinearProbe with AdamW (mini-batch) for each weight_decay in wd_grid.
    Select best model by validation balanced accuracy.

    Mini-batch training with gradient clipping generalises better than full-batch
    Adam in the high-dimensional, low-sample regime of LLM activations.

    Returns (best_probe, mean, std, best_wd, best_val_acc).
    """
    d_model = X_train.shape[1]
    n_train = X_train.shape[0]
    device = torch.device("cpu")  # probes are small; CPU is fine

    # Normalise features using training-set statistics
    mean, std = fit_normaliser(X_train)
    X_tr_s = apply_normaliser(X_train, mean, std)
    X_v_s = apply_normaliser(X_val, mean, std)

    X_tr_t = torch.tensor(X_tr_s, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_v_t = torch.tensor(X_v_s, dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss()

    best_probe = None
    best_val_acc = -1.0
    best_wd = wd_grid[0]

    rng = torch.Generator()

    for wd in wd_grid:
        torch.manual_seed(0)
        rng.manual_seed(0)
        probe = LinearProbe(d_model).to(device)
        # AdamW applies weight-decay as decoupled L2 regularisation, which
        # behaves more predictably than coupled L2 in Adam.
        optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)

        for epoch in range(n_epochs):
            probe.train()
            perm = torch.randperm(n_train, generator=rng)
            for i in range(0, n_train, batch_size):
                batch_idx = perm[i: i + batch_size]
                if len(batch_idx) == 0:
                    continue
                optimizer.zero_grad()
                logits = probe(X_tr_t[batch_idx])
                loss = criterion(logits, y_tr_t[batch_idx])
                loss.backward()
                if clip_grad_norm > 0:
                    nn.utils.clip_grad_norm_(probe.parameters(), clip_grad_norm)
                optimizer.step()

        probe.eval()
        with torch.no_grad():
            val_logits = probe(X_v_t)
            val_preds = (val_logits > 0).long().cpu().numpy()

        val_acc = balanced_accuracy_score(y_val, val_preds)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_probe = probe
            best_wd = wd

    return best_probe, mean, std, best_wd, best_val_acc


def evaluate_probe(
    probe: LinearProbe,
    mean: np.ndarray,
    std: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[float, float, float]:
    """
    Evaluate probe on test data.
    Returns (balanced_accuracy, auroc, weight_norm).
    """
    device = torch.device("cpu")
    X_s = apply_normaliser(X_test, mean, std)
    X_t = torch.tensor(X_s, dtype=torch.float32, device=device)

    probe.eval()
    with torch.no_grad():
        logits = probe(X_t).cpu().numpy()

    preds = (logits > 0).astype(int)
    test_acc = balanced_accuracy_score(y_test, preds)
    test_auroc = roc_auc_score(y_test, logits)
    weight_norm = float(probe.linear.weight.norm().item())

    return test_acc, test_auroc, weight_norm


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_activations(config: Config) -> tuple[list[dict], list[dict]]:
    """Load self-prefill and cross-model activation files."""
    self_data = torch.load(
        config.activations_dir / "self_prefill.pt", weights_only=False
    )
    cross_data = torch.load(
        config.activations_dir / "cross_gemma_prefill.pt", weights_only=False
    )
    return self_data, cross_data


def build_dataset_for_layer(
    self_data: list[dict],
    cross_data: list[dict],
    split_map: dict[int, str],
    layer_idx: int,
) -> dict[str, dict]:
    """
    Build X / y arrays for one layer, split into train / val / test.
    Label 0 = self-prefill, 1 = cross-model.

    Returns dict with keys 'train', 'val', 'test', each containing
    {'X': np.ndarray (n, d_model, float32), 'y': np.ndarray (n,)}.
    """
    cross_by_pid = {d["prompt_id"]: d for d in cross_data}

    data: dict[str, dict] = {
        s: {"X": [], "y": []} for s in ("train", "val", "test")
    }

    for self_item in self_data:
        pid = self_item["prompt_id"]
        split = split_map.get(pid)
        if split is None:
            continue
        cross_item = cross_by_pid.get(pid)
        if cross_item is None:
            continue

        self_act = self_item["layer_activations"][layer_idx].astype(np.float32)
        cross_act = cross_item["layer_activations"][layer_idx].astype(np.float32)

        data[split]["X"].extend([self_act, cross_act])
        data[split]["y"].extend([0, 1])

    result = {}
    d_model = None
    for split, d in data.items():
        if d["X"]:
            X = np.stack(d["X"])
            d_model = X.shape[1]
        else:
            X = np.empty((0, d_model or 4096))
        y = np.array(d["y"], dtype=int)
        result[split] = {"X": X, "y": y}

    return result


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def run_sanity_check_self_vs_self(
    self_data: list[dict],
    split_map: dict[int, str],
    config: Config,
    layer_idx: int = 15,
) -> dict:
    """
    Randomly assign labels within the self-prefill activations and train a
    probe.  Balanced accuracy should be ~50% on every split (chance level).

    Construction:
      - Gather one activation per prompt (self-prefill only).
      - Assign a random binary label to each activation, independent of
        prompt content, using a fixed seed.  Approximately half get label 0,
        half get label 1.
      - Use the same train/val/test prompt split as the main experiment.
      - Train LinearProbe on the random labels and evaluate.

    Since labels are random w.r.t. activation content, a well-behaved probe
    should not generalise: both train and test accuracy should be ~50%.

    Returns a dict with balanced accuracy on train / val / test and label
    counts per split, to make the failure mode transparent if the check fails.
    """
    acts: list[np.ndarray] = []
    splits: list[str] = []
    pids: list[int] = []

    for item in self_data:
        pid = item["prompt_id"]
        split = split_map.get(pid)
        if split is not None:
            acts.append(item["layer_activations"][layer_idx].astype(np.float32))
            splits.append(split)
            pids.append(pid)

    X = np.stack(acts)
    n = len(X)
    splits_arr = np.array(splits)

    # Assign random binary labels, independent of prompt id or split
    rng = np.random.default_rng(seed=0)
    perm = rng.permutation(n)
    y = np.zeros(n, dtype=int)
    y[perm[: n // 2]] = 1

    # Report label balance per split to verify there is no systematic skew
    for sp in ("train", "val", "test"):
        mask = splits_arr == sp
        n_sp = mask.sum()
        n1 = y[mask].sum()
        print(f"    Sanity split '{sp}': n={n_sp}, label-1={n1} ({100*n1/max(n_sp,1):.0f}%)")

    probe, mean, std, best_wd, val_acc = train_probe(
        X[splits_arr == "train"], y[splits_arr == "train"],
        X[splits_arr == "val"],   y[splits_arr == "val"],
        config.probe_regularisation_grid,
    )

    # Evaluate on all three splits so we can distinguish overfit from no-signal
    train_acc, _, _ = evaluate_probe(
        probe, mean, std,
        X[splits_arr == "train"], y[splits_arr == "train"],
    )
    val_acc_eval, _, _ = evaluate_probe(
        probe, mean, std,
        X[splits_arr == "val"], y[splits_arr == "val"],
    )
    test_acc, _, _ = evaluate_probe(
        probe, mean, std,
        X[splits_arr == "test"], y[splits_arr == "test"],
    )

    print(f"    Sanity probe  train={train_acc:.4f}  val={val_acc_eval:.4f}  "
          f"test={test_acc:.4f}  best_wd={best_wd}")

    n_test = (splits_arr == "test").sum()
    print(f"    (n_test={n_test}; 1-sigma band ≈ ±{100*0.5/n_test**0.5:.1f}pp "
          f"around 50%, so [{50 - 100*0.5/n_test**0.5:.1f}%, {50 + 100*0.5/n_test**0.5:.1f}%] expected)")

    return {
        "train_acc": train_acc,
        "val_acc": val_acc_eval,
        "test_acc": test_acc,
        "best_wd": best_wd,
        "n_train": int((splits_arr == "train").sum()),
        "n_val":   int((splits_arr == "val").sum()),
        "n_test":  n_test,
    }


def load_l2_check_result(config: Config) -> dict:
    """Load reproducibility check result written by extract.py."""
    repro_path = config.activations_dir / "reproducibility_check.json"
    if repro_path.exists():
        with open(repro_path) as f:
            return json.load(f)
    return {"max_l2_distance": None, "pass": None}


# ---------------------------------------------------------------------------
# Perplexity baseline
# ---------------------------------------------------------------------------

def compute_perplexity_baseline_auroc(
    perplexity_data: list[dict],
    split_map: dict[int, str],
) -> float:
    """
    Single-feature AUROC using -mean_log_prob as the classifier score on
    the TEST split.  Higher score → more surprising → more likely cross-model.
    """
    pid_cond: dict[int, dict] = {}
    for item in perplexity_data:
        pid = item["prompt_id"]
        if pid not in pid_cond:
            pid_cond[pid] = {}
        pid_cond[pid][item["condition"]] = item

    scores: list[float] = []
    labels: list[int] = []

    for pid, cond_map in pid_cond.items():
        if split_map.get(pid) != "test":
            continue
        if "self" not in cond_map or "cross_gemma" not in cond_map:
            continue
        scores.append(-cond_map["self"]["mean_log_prob"])
        labels.append(0)
        scores.append(-cond_map["cross_gemma"]["mean_log_prob"])
        labels.append(1)

    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def train_and_evaluate_all_probes(
    responses: list[dict],
    config: Config,
) -> dict:
    """
    Train probes for all layers and return a results dict.
    """
    self_data, cross_data = load_activations(config)
    split_map = {r["prompt_id"]: r["split"] for r in responses}

    results: dict = {
        "layer_results": {},
        "sanity_check_acc": None,
        "l2_check": None,
        "perplexity_baseline_auroc": None,
    }

    # --- Sanity check: self vs. self (should be ~50% on all splits) ---
    print("  Running sanity check (self vs. self probe at layer 15)...")
    sanity = run_sanity_check_self_vs_self(
        self_data, split_map, config, layer_idx=15
    )
    results["sanity_check"] = sanity
    # Back-compat alias used by analysis / summary table
    results["sanity_check_acc"] = sanity["test_acc"]

    # Interpret the result:
    #   train ≈ 50% AND test ≈ 50%  → probe learned nothing  (GOOD)
    #   train >> 50%, test ≈ 50%   → minor overfit, still OK
    #   train >> 50%, test << 50%  → overfit reversed on test (small n effect, likely OK)
    #   train << 50%               → something is wrong with the pipeline
    train_ok = sanity["train_acc"] >= 0.45
    test_ok  = 0.40 <= sanity["test_acc"] <= 0.60
    if not train_ok:
        status = "PIPELINE ERROR — train acc below 45%, labels may be leaking real signal"
    elif not test_ok:
        status = "WARNING — test acc outside [40%, 60%], but check n_test for sampling noise"
    else:
        status = "OK (noise-level; expected given small n_test)"
    print(f"  Sanity check: train={sanity['train_acc']:.4f}  "
          f"test={sanity['test_acc']:.4f}  [{status}]")

    # --- L2 reproducibility result from extraction step ---
    results["l2_check"] = load_l2_check_result(config)
    print(f"  L2 reproducibility check: {results['l2_check']}")

    # --- Perplexity baseline AUROC ---
    ppl_path = config.activations_dir / "perplexity.json"
    if ppl_path.exists():
        with open(ppl_path) as f:
            ppl_data = json.load(f)
        ppl_auroc = compute_perplexity_baseline_auroc(ppl_data, split_map)
        results["perplexity_baseline_auroc"] = ppl_auroc
        print(f"  Perplexity baseline AUROC (test): {ppl_auroc:.4f}")
    else:
        print("  Perplexity data not found; skipping baseline.")

    # --- Layer-wise probe training ---
    print("\n  Training LinearProbe for each layer...")
    for layer_idx in tqdm(config.extract_layers, desc="Probing layers"):
        dataset = build_dataset_for_layer(
            self_data, cross_data, split_map, layer_idx
        )

        X_train, y_train = dataset["train"]["X"], dataset["train"]["y"]
        X_val,   y_val   = dataset["val"]["X"],   dataset["val"]["y"]
        X_test,  y_test  = dataset["test"]["X"],  dataset["test"]["y"]

        if len(X_train) < 2 or len(X_val) < 2 or len(X_test) < 2:
            print(f"  Layer {layer_idx}: insufficient data, skipping.")
            continue
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            print(f"  Layer {layer_idx}: only one class present, skipping.")
            continue

        probe, mean, std, best_wd, val_acc = train_probe(
            X_train, y_train, X_val, y_val,
            config.probe_regularisation_grid,
        )
        test_acc, test_auroc, weight_norm = evaluate_probe(
            probe, mean, std, X_test, y_test
        )

        results["layer_results"][layer_idx] = {
            "test_balanced_accuracy": test_acc,
            "test_auroc": test_auroc,
            "val_balanced_accuracy": val_acc,
            "best_weight_decay": best_wd,
            "weight_norm": weight_norm,
        }

    # Brief console summary
    if results["layer_results"]:
        best_layer = max(
            results["layer_results"],
            key=lambda l: results["layer_results"][l]["test_balanced_accuracy"],
        )
        best = results["layer_results"][best_layer]
        print(f"\n  Best layer: {best_layer}  "
              f"acc={best['test_balanced_accuracy']:.4f}  "
              f"AUROC={best['test_auroc']:.4f}")

    return results
