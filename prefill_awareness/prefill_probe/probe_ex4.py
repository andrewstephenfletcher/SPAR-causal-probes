"""
Probe training and evaluation for Experiment 4 (Scaling Analysis).

Trains per-layer LinearProbes for each (target model, cross condition) pair.
Also re-runs probes on Experiment 1 activations so all three models' results
are available as JSON for the scaling analysis.

Output files:
  results_dir_ex4/probe_results_llama70b.json
  results_dir_ex4/probe_results_gemma31b.json
  ex1_results_dir/probe_results_llama8b.json   (from existing Exp 1 activations)

JSON schema per file:
{
  "model_id": "...",
  "model_name": "llama70b",
  "n_layers": 80,
  "sanity_check": {...},
  "cross_conditions": {
    "cross_gemma9b": {
      "layer_results": {"0": {...}, ..., "79": {...}},
      "perplexity_baseline_auroc": 0.95,
      "best_layer": 60,
      "best_auroc": 0.98
    },
    "cross_llama8b": { ... }
  }
}
"""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from .config import Config, Experiment4Config
from .probe import build_dataset_for_layer, evaluate_probe, train_probe


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_pt(path: Path) -> list[dict]:
    return torch.load(path, weights_only=False)


def _build_split_map(responses: list[dict]) -> dict[int, str]:
    return {r["prompt_id"]: r["split"] for r in responses}


# ---------------------------------------------------------------------------
# Perplexity baseline AUROC
# ---------------------------------------------------------------------------

def _perplexity_baseline_auroc(
    ppl_path: Path,
    self_condition: str,
    cross_condition: str,
    split_map: dict[int, str],
) -> float:
    if not ppl_path.exists():
        return float("nan")

    with open(ppl_path) as f:
        ppl_data = json.load(f)

    pid_cond: dict[int, dict] = {}
    for item in ppl_data:
        pid = item["prompt_id"]
        if pid not in pid_cond:
            pid_cond[pid] = {}
        pid_cond[pid][item["condition"]] = item

    scores, labels = [], []
    for pid, cond_map in pid_cond.items():
        if split_map.get(pid) != "test":
            continue
        if self_condition not in cond_map or cross_condition not in cond_map:
            continue
        scores.append(-cond_map[self_condition]["mean_log_prob"])
        labels.append(0)
        scores.append(-cond_map[cross_condition]["mean_log_prob"])
        labels.append(1)

    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


# ---------------------------------------------------------------------------
# Sanity check: random labels on self-activations → should be ~50%
# ---------------------------------------------------------------------------

def _sanity_check(
    self_data: list[dict],
    split_map: dict[int, str],
    wd_grid: list[float],
    layer_idx: int = 0,
) -> dict:
    """Probe trained on random labels within self-activations; test acc should be ~50%."""
    acts, splits = [], []
    for item in self_data:
        pid = item["prompt_id"]
        sp = split_map.get(pid)
        if sp is not None:
            acts.append(item["layer_activations"][layer_idx].astype(np.float32))
            splits.append(sp)

    X = np.stack(acts)
    n = len(X)
    splits_arr = np.array(splits)

    rng = np.random.default_rng(seed=0)
    y = np.zeros(n, dtype=int)
    y[rng.permutation(n)[: n // 2]] = 1

    probe, mean, std, best_wd, _ = train_probe(
        X[splits_arr == "train"], y[splits_arr == "train"],
        X[splits_arr == "val"],   y[splits_arr == "val"],
        wd_grid,
    )
    test_acc, _, _ = evaluate_probe(
        probe, mean, std,
        X[splits_arr == "test"], y[splits_arr == "test"],
    )
    return {
        "test_acc": float(test_acc),
        "n_train": int((splits_arr == "train").sum()),
        "n_val":   int((splits_arr == "val").sum()),
        "n_test":  int((splits_arr == "test").sum()),
    }


# ---------------------------------------------------------------------------
# Core probe loop for one (model, cross_condition) pair
# ---------------------------------------------------------------------------

def _train_probes_for_condition(
    self_data: list[dict],
    cross_data: list[dict],
    split_map: dict[int, str],
    wd_grid: list[float],
    ppl_baseline_auroc: float,
    layer_ids: list[int],
    desc: str = "",
) -> dict:
    """Train LinearProbe at every layer; return condition-level results dict."""
    layer_results: dict[str, dict] = {}

    for layer_idx in tqdm(layer_ids, desc=f"  Probing {desc}"):
        dataset = build_dataset_for_layer(self_data, cross_data, split_map, layer_idx)

        X_train, y_train = dataset["train"]["X"], dataset["train"]["y"]
        X_val,   y_val   = dataset["val"]["X"],   dataset["val"]["y"]
        X_test,  y_test  = dataset["test"]["X"],  dataset["test"]["y"]

        if any(len(a) < 2 for a in [X_train, X_val, X_test]):
            continue
        if any(len(np.unique(y)) < 2 for y in [y_train, y_test]):
            continue

        probe, mean, std, best_wd, val_acc = train_probe(
            X_train, y_train, X_val, y_val, wd_grid
        )
        test_acc, test_auroc, weight_norm = evaluate_probe(
            probe, mean, std, X_test, y_test
        )

        layer_results[str(layer_idx)] = {
            "test_balanced_accuracy": float(test_acc),
            "test_auroc": float(test_auroc),
            "val_balanced_accuracy": float(val_acc),
            "best_weight_decay": float(best_wd),
            "weight_norm": float(weight_norm),
        }

    if not layer_results:
        return {
            "layer_results": {},
            "perplexity_baseline_auroc": None,
            "best_layer": None,
            "best_auroc": None,
        }

    best_layer_key = max(
        layer_results, key=lambda k: layer_results[k]["test_auroc"]
    )
    return {
        "layer_results": layer_results,
        "perplexity_baseline_auroc": (
            float(ppl_baseline_auroc) if not np.isnan(ppl_baseline_auroc) else None
        ),
        "best_layer": int(best_layer_key),
        "best_auroc": layer_results[best_layer_key]["test_auroc"],
    }


# ---------------------------------------------------------------------------
# Per-model probe runners
# ---------------------------------------------------------------------------

def train_probes_llama70b(
    responses: list[dict],
    config: Experiment4Config,
    force: bool = False,
) -> dict:
    output_path = config.results_dir_ex4 / "probe_results_llama70b.json"
    if output_path.exists() and not force:
        print(f"  Loading existing Llama 70B probe results from {output_path}")
        with open(output_path) as f:
            return json.load(f)

    split_map = _build_split_map(responses)
    wd_grid = config.probe_regularisation_grid
    ppl_path = config.activations_dir_llama70b / "perplexity.json"

    self_data = _load_pt(config.activations_dir_llama70b / "self_prefill.pt")
    layer_ids = sorted(self_data[0]["layer_activations"].keys())

    print("  Sanity check (Llama 70B)...")
    sanity = _sanity_check(self_data, split_map, wd_grid, layer_idx=layer_ids[0])
    print(f"    test_acc={sanity['test_acc']:.4f} (expected ~0.50)")

    results: dict = {
        "model_id": config.llama70b_model_id,
        "model_name": "llama70b",
        "n_layers": max(layer_ids) + 1,
        "sanity_check": sanity,
        "cross_conditions": {},
    }

    for condition_name, cross_file in [
        ("cross_gemma9b", "cross_gemma9b.pt"),
        ("cross_llama8b", "cross_llama8b.pt"),
    ]:
        print(f"\n  Llama 70B vs. {condition_name}...")
        cross_data = _load_pt(config.activations_dir_llama70b / cross_file)
        ppl_auroc = _perplexity_baseline_auroc(
            ppl_path, "self_prefill", condition_name, split_map
        )
        cond_results = _train_probes_for_condition(
            self_data, cross_data, split_map, wd_grid, ppl_auroc, layer_ids,
            desc=f"llama70b/{condition_name}",
        )
        results["cross_conditions"][condition_name] = cond_results
        if cond_results["best_layer"] is not None:
            print(f"    Best layer {cond_results['best_layer']}: "
                  f"AUROC={cond_results['best_auroc']:.4f}  "
                  f"perp={ppl_auroc:.4f}")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved → {output_path}")
    return results


def train_probes_gemma31b(
    responses: list[dict],
    config: Experiment4Config,
    force: bool = False,
) -> dict:
    output_path = config.results_dir_ex4 / "probe_results_gemma31b.json"
    if output_path.exists() and not force:
        print(f"  Loading existing Gemma 31B probe results from {output_path}")
        with open(output_path) as f:
            return json.load(f)

    split_map = _build_split_map(responses)
    wd_grid = config.probe_regularisation_grid
    ppl_path = config.activations_dir_gemma31b / "perplexity.json"

    self_data = _load_pt(config.activations_dir_gemma31b / "self_prefill.pt")
    layer_ids = sorted(self_data[0]["layer_activations"].keys())

    print("  Sanity check (Gemma 31B)...")
    sanity = _sanity_check(self_data, split_map, wd_grid, layer_idx=layer_ids[0])
    print(f"    test_acc={sanity['test_acc']:.4f} (expected ~0.50)")

    results: dict = {
        "model_id": config.gemma31b_model_id,
        "model_name": "gemma31b",
        "n_layers": max(layer_ids) + 1,
        "sanity_check": sanity,
        "cross_conditions": {},
    }

    for condition_name, cross_file in [
        ("cross_llama8b", "cross_llama8b.pt"),
        ("cross_gemma9b", "cross_gemma9b.pt"),
    ]:
        print(f"\n  Gemma 31B vs. {condition_name}...")
        cross_data = _load_pt(config.activations_dir_gemma31b / cross_file)
        ppl_auroc = _perplexity_baseline_auroc(
            ppl_path, "self_prefill", condition_name, split_map
        )
        cond_results = _train_probes_for_condition(
            self_data, cross_data, split_map, wd_grid, ppl_auroc, layer_ids,
            desc=f"gemma31b/{condition_name}",
        )
        results["cross_conditions"][condition_name] = cond_results
        if cond_results["best_layer"] is not None:
            print(f"    Best layer {cond_results['best_layer']}: "
                  f"AUROC={cond_results['best_auroc']:.4f}  "
                  f"perp={ppl_auroc:.4f}")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved → {output_path}")
    return results


# ---------------------------------------------------------------------------
# Exp 1 (Llama 8B) re-run
# ---------------------------------------------------------------------------

def ensure_llama8b_probe_results(
    responses: list[dict],
    config: Experiment4Config,
    force: bool = False,
) -> dict:
    """
    Re-run probes on Experiment 1 activations and save as JSON.
    Provides the Llama 8B data point for the scaling comparison.
    """
    output_path = config.ex1_results_dir / "probe_results_llama8b.json"
    if output_path.exists() and not force:
        print(f"  Loading existing Llama 8B probe results from {output_path}")
        with open(output_path) as f:
            return json.load(f)

    self_path = config.ex1_activations_dir / "self_prefill.pt"
    cross_path = config.ex1_activations_dir / "cross_gemma_prefill.pt"

    if not self_path.exists() or not cross_path.exists():
        raise FileNotFoundError(
            f"Experiment 1 activations not found at {config.ex1_activations_dir}. "
            "Run Experiment 1 first."
        )

    print("\n  Computing Llama 8B probe results from Exp 1 activations...")
    exp1_config = Config()
    split_map = _build_split_map(responses)
    wd_grid = config.probe_regularisation_grid
    ppl_path = config.ex1_activations_dir / "perplexity.json"

    self_data = _load_pt(self_path)
    cross_data = _load_pt(cross_path)

    layer_ids = sorted(exp1_config.extract_layers)

    print("  Sanity check (Llama 8B)...")
    sanity = _sanity_check(self_data, split_map, wd_grid, layer_idx=layer_ids[0])
    print(f"    test_acc={sanity['test_acc']:.4f} (expected ~0.50)")

    # Exp 1 uses "cross_gemma" condition name in the perplexity JSON
    ppl_auroc = _perplexity_baseline_auroc(
        ppl_path, "self", "cross_gemma", split_map
    )

    print(f"  Training per-layer probes for Llama 8B ({len(layer_ids)} layers)...")
    cond_results = _train_probes_for_condition(
        self_data, cross_data, split_map, wd_grid, ppl_auroc, layer_ids,
        desc="llama8b/cross_gemma9b",
    )

    results = {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "model_name": "llama8b",
        "n_layers": 32,
        "sanity_check": sanity,
        "cross_conditions": {"cross_gemma9b": cond_results},
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved → {output_path}")
    return results


def _train_probes_for_model(
    model_name: str,
    model_id: str,
    activations_dir: Path,
    cross_conditions: list[tuple[str, str]],
    responses: list[dict],
    config: Experiment4Config,
    output_path: Path,
    force: bool = False,
) -> dict:
    """Generic probe trainer — loads activations, runs per-layer probes, saves JSON."""
    if output_path.exists() and not force:
        print(f"  Loading existing {model_name} probe results from {output_path}")
        with open(output_path) as f:
            return json.load(f)

    split_map = _build_split_map(responses)
    wd_grid = config.probe_regularisation_grid
    ppl_path = activations_dir / "perplexity.json"

    self_data = _load_pt(activations_dir / "self_prefill.pt")
    layer_ids = sorted(self_data[0]["layer_activations"].keys())

    print(f"  Sanity check ({model_name})...")
    sanity = _sanity_check(self_data, split_map, wd_grid, layer_idx=layer_ids[0])
    print(f"    test_acc={sanity['test_acc']:.4f} (expected ~0.50)")

    results: dict = {
        "model_id": model_id,
        "model_name": model_name,
        "n_layers": max(layer_ids) + 1,
        "sanity_check": sanity,
        "cross_conditions": {},
    }

    for condition_name, cross_file in cross_conditions:
        print(f"\n  {model_name} vs. {condition_name}...")
        cross_data = _load_pt(activations_dir / cross_file)
        ppl_auroc = _perplexity_baseline_auroc(
            ppl_path, "self_prefill", condition_name, split_map
        )
        cond_results = _train_probes_for_condition(
            self_data, cross_data, split_map, wd_grid, ppl_auroc, layer_ids,
            desc=f"{model_name}/{condition_name}",
        )
        results["cross_conditions"][condition_name] = cond_results
        if cond_results["best_layer"] is not None:
            print(f"    Best layer {cond_results['best_layer']}: "
                  f"AUROC={cond_results['best_auroc']:.4f}  "
                  f"perp={ppl_auroc:.4f}")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved → {output_path}")
    return results


def train_probes_gemma4b(
    responses: list[dict],
    config: Experiment4Config,
    force: bool = False,
) -> dict:
    return _train_probes_for_model(
        model_name="gemma4b",
        model_id=config.gemma4b_model_id,
        activations_dir=config.activations_dir_gemma4b,
        cross_conditions=[("cross_llama8b", "cross_llama8b.pt"),
                          ("cross_gemma9b",  "cross_gemma9b.pt")],
        responses=responses,
        config=config,
        output_path=config.results_dir_ex4 / "probe_results_gemma4b.json",
        force=force,
    )


def train_probes_mistral7b(
    responses: list[dict],
    config: Experiment4Config,
    force: bool = False,
) -> dict:
    return _train_probes_for_model(
        model_name="mistral7b",
        model_id=config.mistral7b_model_id,
        activations_dir=config.activations_dir_mistral7b,
        cross_conditions=[("cross_llama8b", "cross_llama8b.pt"),
                          ("cross_gemma9b",  "cross_gemma9b.pt")],
        responses=responses,
        config=config,
        output_path=config.results_dir_ex4 / "probe_results_mistral7b.json",
        force=force,
    )


def train_probes_mistral24b(
    responses: list[dict],
    config: Experiment4Config,
    force: bool = False,
) -> dict:
    return _train_probes_for_model(
        model_name="mistral24b",
        model_id=config.mistral24b_model_id,
        activations_dir=config.activations_dir_mistral24b,
        cross_conditions=[("cross_llama8b", "cross_llama8b.pt"),
                          ("cross_gemma9b",  "cross_gemma9b.pt")],
        responses=responses,
        config=config,
        output_path=config.results_dir_ex4 / "probe_results_mistral24b.json",
        force=force,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def train_all_probes(
    responses: list[dict],
    config: Experiment4Config,
    force: bool = False,
) -> dict[str, dict]:
    print("\n--- Probe training: Llama 8B (Exp 1 re-run) ---")
    llama8b = ensure_llama8b_probe_results(responses, config, force=force)

    print("\n--- Probe training: Llama 3.3 70B ---")
    llama70b = train_probes_llama70b(responses, config, force=force)

    print("\n--- Probe training: Gemma 4 31B ---")
    gemma31b = train_probes_gemma31b(responses, config, force=force)

    print("\n--- Probe training: Gemma 4 4B ---")
    gemma4b = train_probes_gemma4b(responses, config, force=force)

    print("\n--- Probe training: Mistral 7B ---")
    mistral7b = train_probes_mistral7b(responses, config, force=force)

    print("\n--- Probe training: Mistral Small 24B ---")
    mistral24b = train_probes_mistral24b(responses, config, force=force)

    return {
        "llama8b": llama8b, "llama70b": llama70b,
        "gemma4b": gemma4b, "gemma31b": gemma31b,
        "mistral7b": mistral7b, "mistral24b": mistral24b,
    }
