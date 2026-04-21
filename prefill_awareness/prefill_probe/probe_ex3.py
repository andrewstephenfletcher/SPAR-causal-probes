"""
Probe training and analysis for Experiment 3.

Trains linear probes at layer 30 (last token) for every self-vs-X pairing,
then runs four analysis types:

  1. Base probes     — one per cross-model condition, all datasets pooled
  2. Unified probe   — self vs. all cross-model conditions pooled; also
                       evaluated on held-out llama-70b-equivalent (here,
                       on each condition excluded from training)
  3. Cross-source transfer matrix (4 × 4)
  4. Cross-topic transfer matrices (3 × 3, one per cross-model condition)
  5. Perplexity-matched analysis
  6. Outlier-excluded analysis
  7. Altered-self vs. cross-model comparison (Section 9.5)

Results are saved to results_dir_ex3 / "probe_results_ex3.json".
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from .config import Experiment3Config
from .probe import apply_normaliser, evaluate_probe, fit_normaliser, train_probe

# Condition names (matches generate_ex3.py)
CONDITIONS = ["self", "altered_self", "gemma", "mistral", "style_imitated"]
CROSS_CONDITIONS = ["altered_self", "gemma", "mistral", "style_imitated"]
DATASETS = ["alpaca", "oasst1", "mmlu"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_activations(ex3_config: Experiment3Config) -> dict[str, list[dict]]:
    """Load activation files for all 5 conditions."""
    all_acts: dict[str, list[dict]] = {}
    for cond in CONDITIONS:
        path = ex3_config.activations_dir_ex3 / f"activations_{cond}.pt"
        if path.exists():
            all_acts[cond] = torch.load(path, weights_only=False)
        else:
            print(f"  WARNING: activation file not found for '{cond}': {path}")
            all_acts[cond] = []
    return all_acts


def load_perplexity(ex3_config: Experiment3Config) -> dict[str, dict[str, dict]]:
    """
    Load perplexity_all.json and return as nested dict:
    {prompt_id: {condition: {"mean_log_prob": float, "perplexity": float, ...}}}
    """
    ppl_path = ex3_config.activations_dir_ex3 / "perplexity_all.json"
    if not ppl_path.exists():
        print(f"  WARNING: perplexity file not found at {ppl_path}.")
        return {}
    with open(ppl_path) as f:
        records = json.load(f)
    nested: dict[str, dict[str, dict]] = {}
    for rec in records:
        pid = rec["prompt_id"]
        if pid not in nested:
            nested[pid] = {}
        nested[pid][rec["condition"]] = rec
    return nested


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def build_dataset(
    self_acts: list[dict],
    other_acts: list[dict],
    split: str | None = None,
    dataset: str | None = None,
    pid_whitelist: set[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) for self (label=0) vs. other (label=1).

    Filters:
      split         — include only records with this split label
      dataset       — include only records from this dataset
      pid_whitelist — include only these prompt IDs
    """
    self_by_pid = {d["prompt_id"]: d for d in self_acts}
    other_by_pid = {d["prompt_id"]: d for d in other_acts}

    X_list: list[np.ndarray] = []
    y_list: list[int] = []

    for pid, self_item in self_by_pid.items():
        other_item = other_by_pid.get(pid)
        if other_item is None:
            continue
        if split is not None and self_item.get("split") != split:
            continue
        if dataset is not None and self_item.get("dataset") != dataset:
            continue
        if pid_whitelist is not None and pid not in pid_whitelist:
            continue

        X_list.extend([
            self_item["activation"].astype(np.float32),
            other_item["activation"].astype(np.float32),
        ])
        y_list.extend([0, 1])

    if not X_list:
        d_model = 4096
        return np.empty((0, d_model), dtype=np.float32), np.array([], dtype=int)

    return np.stack(X_list), np.array(y_list, dtype=int)


def _train_eval(
    self_acts: list[dict],
    other_acts: list[dict],
    wd_grid: list[float],
    train_split: str = "train",
    val_split: str = "val",
    test_split: str = "test",
    train_dataset: str | None = None,
    eval_dataset: str | None = None,
    pid_whitelist: set[str] | None = None,
) -> dict:
    """
    Train a probe and evaluate on test set.  Returns a metrics dict.
    """
    X_tr, y_tr = build_dataset(self_acts, other_acts, split=train_split,
                                dataset=train_dataset)
    X_v,  y_v  = build_dataset(self_acts, other_acts, split=val_split,
                                dataset=train_dataset)
    X_te, y_te = build_dataset(self_acts, other_acts, split=test_split,
                                dataset=eval_dataset, pid_whitelist=pid_whitelist)

    if (len(X_tr) < 4 or len(X_v) < 2 or len(X_te) < 2
            or len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2):
        return {"auroc": float("nan"), "balanced_accuracy": float("nan"),
                "n_test": len(y_te), "best_wd": float("nan")}

    probe, mean, std, best_wd, _ = train_probe(X_tr, y_tr, X_v, y_v, wd_grid)
    acc, auroc, _ = evaluate_probe(probe, mean, std, X_te, y_te)

    return {
        "auroc": round(float(auroc), 5),
        "balanced_accuracy": round(float(acc), 5),
        "n_test": int(len(y_te) // 2),
        "best_wd": best_wd,
    }


def _eval_probe_on(probe, mean, std, self_acts, other_acts,
                   split=None, dataset=None, pid_whitelist=None) -> dict:
    """Evaluate an already-trained probe on a custom subset."""
    X_te, y_te = build_dataset(self_acts, other_acts,
                                split=split, dataset=dataset,
                                pid_whitelist=pid_whitelist)
    if len(X_te) < 2 or len(np.unique(y_te)) < 2:
        return {"auroc": float("nan"), "balanced_accuracy": float("nan"),
                "n_test": 0}
    acc, auroc, _ = evaluate_probe(probe, mean, std, X_te, y_te)
    return {"auroc": round(float(auroc), 5),
            "balanced_accuracy": round(float(acc), 5),
            "n_test": int(len(y_te) // 2)}


# ---------------------------------------------------------------------------
# Perplexity baseline AUROC on an arbitrary subset
# ---------------------------------------------------------------------------

def _ppl_auroc_subset(
    ppl_nested: dict,
    condition: str,
    pid_whitelist: set[str],
    split: str | None = "test",
) -> float:
    scores: list[float] = []
    labels: list[int] = []
    for pid, cond_map in ppl_nested.items():
        if pid not in pid_whitelist:
            continue
        if split is not None:
            rec = cond_map.get("self", {})
            if rec.get("split") != split:
                continue
        self_lp  = cond_map.get("self", {}).get("mean_log_prob")
        other_lp = cond_map.get(condition, {}).get("mean_log_prob")
        if self_lp is None or other_lp is None:
            continue
        scores.extend([-self_lp, -other_lp])
        labels.extend([0, 1])
    if len(set(labels)) < 2:
        return float("nan")
    return round(float(roc_auc_score(labels, scores)), 5)


# ---------------------------------------------------------------------------
# Perplexity-matched and outlier-excluded subsets
# ---------------------------------------------------------------------------

def _matched_pids(
    ppl_nested: dict,
    condition: str,
    threshold: float,
    split: str = "test",
) -> set[str]:
    """Prompt IDs where |self_mean_lp - other_mean_lp| <= threshold (test split)."""
    result = set()
    for pid, cond_map in ppl_nested.items():
        rec_self  = cond_map.get("self", {})
        rec_other = cond_map.get(condition, {})
        if rec_self.get("split") != split:
            continue
        lp_s = rec_self.get("mean_log_prob")
        lp_o = rec_other.get("mean_log_prob")
        if lp_s is not None and lp_o is not None:
            if abs(lp_s - lp_o) <= threshold:
                result.add(pid)
    return result


def _outlier_free_pids(
    ppl_nested: dict,
    condition: str,
    threshold: float,
    split: str = "test",
) -> set[str]:
    """Prompt IDs where neither self nor other has perplexity > threshold."""
    result = set()
    for pid, cond_map in ppl_nested.items():
        rec_self  = cond_map.get("self", {})
        rec_other = cond_map.get(condition, {})
        if rec_self.get("split") != split:
            continue
        ppl_s = rec_self.get("perplexity", float("inf"))
        ppl_o = rec_other.get("perplexity", float("inf"))
        if ppl_s <= threshold and ppl_o <= threshold:
            result.add(pid)
    return result


# ---------------------------------------------------------------------------
# Analysis: base probes
# ---------------------------------------------------------------------------

def _run_base_probes(
    all_acts: dict,
    ppl_nested: dict,
    ex3_config: Experiment3Config,
) -> dict:
    wd_grid = ex3_config.probe_regularisation_grid
    results = {}

    for cond in CROSS_CONDITIONS:
        print(f"    self vs. {cond} ...")

        full = _train_eval(all_acts["self"], all_acts[cond], wd_grid)

        # Outlier-excluded test subset
        outlier_free = _outlier_free_pids(
            ppl_nested, cond, ex3_config.outlier_perplexity_threshold
        )
        # Re-train on full train, evaluate on outlier-free test subset
        X_tr, y_tr = build_dataset(all_acts["self"], all_acts[cond], split="train")
        X_v,  y_v  = build_dataset(all_acts["self"], all_acts[cond], split="val")
        if len(X_tr) >= 4 and len(X_v) >= 2 and len(np.unique(y_tr)) >= 2:
            probe, mean, std, _, _ = train_probe(X_tr, y_tr, X_v, y_v, wd_grid)
            oe = _eval_probe_on(probe, mean, std,
                                all_acts["self"], all_acts[cond],
                                split="test", pid_whitelist=outlier_free)
        else:
            probe = mean = std = None
            oe = {"auroc": float("nan"), "balanced_accuracy": float("nan"), "n_test": 0}

        # Perplexity-matched test subset
        matched = _matched_pids(
            ppl_nested, cond, ex3_config.perplexity_match_threshold
        )
        if len(matched) < 10 and ex3_config.perplexity_match_threshold < 1.0:
            # Relax threshold if too few matched pairs
            matched = _matched_pids(ppl_nested, cond, 1.0)
            threshold_used = 1.0
            print(f"      (relaxed perp threshold to 1.0 nat; n_matched={len(matched)})")
        else:
            threshold_used = ex3_config.perplexity_match_threshold

        if probe is not None:
            pm = _eval_probe_on(probe, mean, std,
                                all_acts["self"], all_acts[cond],
                                split="test", pid_whitelist=matched)
        else:
            pm = {"auroc": float("nan"), "balanced_accuracy": float("nan"), "n_test": 0}

        # Perplexity baselines
        all_test_pids = {
            pid for pid, cond_map in ppl_nested.items()
            if cond_map.get("self", {}).get("split") == "test"
        }
        ppl_full     = _ppl_auroc_subset(ppl_nested, cond, all_test_pids)
        ppl_oe       = _ppl_auroc_subset(ppl_nested, cond, outlier_free)
        ppl_matched  = _ppl_auroc_subset(ppl_nested, cond, matched)

        results[f"self_vs_{cond}"] = {
            "full":             {**full,  "ppl_baseline_auroc": ppl_full},
            "outlier_excluded": {**oe,    "ppl_baseline_auroc": ppl_oe,
                                 "n_excluded": len(all_test_pids) - len(outlier_free)},
            "perp_matched":     {**pm,    "ppl_baseline_auroc": ppl_matched,
                                 "n_matched": len(matched),
                                 "threshold_used": threshold_used},
        }

    return results


# ---------------------------------------------------------------------------
# Analysis: unified "not-self" probe
# ---------------------------------------------------------------------------

def _run_unified_probe(
    all_acts: dict,
    ppl_nested: dict,
    ex3_config: Experiment3Config,
) -> dict:
    """
    Pool all cross-model conditions as label=1.
    Also evaluate with each condition held out from training.
    """
    wd_grid = ex3_config.probe_regularisation_grid

    def _pool_acts(conds: list[str], split: str, dataset: str | None = None):
        X_list, y_list = [], []
        for cond in conds:
            X, y = build_dataset(all_acts["self"], all_acts[cond],
                                 split=split, dataset=dataset)
            X_list.append(X)
            y_list.append(y)
        if any(len(x) > 0 for x in X_list):
            return np.concatenate(X_list), np.concatenate(y_list)
        return np.empty((0, 4096), dtype=np.float32), np.array([], dtype=int)

    X_tr, y_tr = _pool_acts(CROSS_CONDITIONS, "train")
    X_v,  y_v  = _pool_acts(CROSS_CONDITIONS, "val")
    X_te, y_te = _pool_acts(CROSS_CONDITIONS, "test")

    if len(X_tr) < 4 or len(X_v) < 2 or len(np.unique(y_tr)) < 2:
        return {"full": {"auroc": float("nan")}, "holdout_eval": {}}

    probe, mean, std, best_wd, _ = train_probe(X_tr, y_tr, X_v, y_v, wd_grid)
    acc, auroc, _ = evaluate_probe(probe, mean, std, X_te, y_te)

    # Holdout eval: remove one condition from training, eval only on that condition
    holdout_evals = {}
    for held_out in CROSS_CONDITIONS:
        train_conds = [c for c in CROSS_CONDITIONS if c != held_out]
        X_tr_ho, y_tr_ho = _pool_acts(train_conds, "train")
        X_v_ho,  y_v_ho  = _pool_acts(train_conds, "val")
        if len(X_tr_ho) < 4 or len(X_v_ho) < 2:
            holdout_evals[held_out] = {"auroc": float("nan")}
            continue
        p_ho, m_ho, s_ho, _, _ = train_probe(X_tr_ho, y_tr_ho, X_v_ho, y_v_ho, wd_grid)
        X_ho_te, y_ho_te = build_dataset(all_acts["self"], all_acts[held_out], split="test")
        if len(X_ho_te) < 2 or len(np.unique(y_ho_te)) < 2:
            holdout_evals[held_out] = {"auroc": float("nan")}
            continue
        _, ho_auroc, _ = evaluate_probe(p_ho, m_ho, s_ho, X_ho_te, y_ho_te)
        holdout_evals[held_out] = {"auroc": round(float(ho_auroc), 5)}

    return {
        "full": {
            "auroc": round(float(auroc), 5),
            "balanced_accuracy": round(float(acc), 5),
            "n_test": int(len(y_te) // 2),
            "best_wd": best_wd,
        },
        "holdout_eval": holdout_evals,
    }


# ---------------------------------------------------------------------------
# Analysis: cross-source transfer matrix
# ---------------------------------------------------------------------------

def _run_cross_source_matrix(
    all_acts: dict,
    ex3_config: Experiment3Config,
) -> list[list]:
    """
    4 × 4 matrix: rows = train condition, cols = eval condition.
    Values are AUROC on the test split of the eval condition.
    Returns as nested list (row-major) with row/col labels.
    """
    wd_grid = ex3_config.probe_regularisation_grid
    rows = []
    for train_cond in CROSS_CONDITIONS:
        row = []
        X_tr, y_tr = build_dataset(all_acts["self"], all_acts[train_cond], split="train")
        X_v,  y_v  = build_dataset(all_acts["self"], all_acts[train_cond], split="val")
        if len(X_tr) < 4 or len(X_v) < 2 or len(np.unique(y_tr)) < 2:
            rows.append([float("nan")] * len(CROSS_CONDITIONS))
            continue
        probe, mean, std, _, _ = train_probe(X_tr, y_tr, X_v, y_v, wd_grid)
        for eval_cond in CROSS_CONDITIONS:
            X_te, y_te = build_dataset(all_acts["self"], all_acts[eval_cond], split="test")
            if len(X_te) < 2 or len(np.unique(y_te)) < 2:
                row.append(float("nan"))
                continue
            _, auroc, _ = evaluate_probe(probe, mean, std, X_te, y_te)
            row.append(round(float(auroc), 5))
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Analysis: cross-topic transfer matrices
# ---------------------------------------------------------------------------

def _run_cross_topic_matrix(
    all_acts: dict,
    condition: str,
    ex3_config: Experiment3Config,
) -> list[list]:
    """
    3 × 3 matrix: rows = train dataset, cols = eval dataset.
    """
    wd_grid = ex3_config.probe_regularisation_grid
    rows = []
    for train_ds in DATASETS:
        row = []
        X_tr, y_tr = build_dataset(all_acts["self"], all_acts[condition],
                                   split="train", dataset=train_ds)
        X_v,  y_v  = build_dataset(all_acts["self"], all_acts[condition],
                                   split="val", dataset=train_ds)
        if len(X_tr) < 4 or len(X_v) < 2 or len(np.unique(y_tr)) < 2:
            rows.append([float("nan")] * len(DATASETS))
            continue
        probe, mean, std, _, _ = train_probe(X_tr, y_tr, X_v, y_v, wd_grid)
        for eval_ds in DATASETS:
            X_te, y_te = build_dataset(all_acts["self"], all_acts[condition],
                                       split="test", dataset=eval_ds)
            if len(X_te) < 2 or len(np.unique(y_te)) < 2:
                row.append(float("nan"))
                continue
            _, auroc, _ = evaluate_probe(probe, mean, std, X_te, y_te)
            row.append(round(float(auroc), 5))
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Analysis 9.5: altered-self vs. cross-model
# ---------------------------------------------------------------------------

def _run_altered_self_comparison(
    all_acts: dict,
    ex3_config: Experiment3Config,
) -> dict:
    """
    Three probes: self vs. altered_self, self vs. gemma, altered_self vs. gemma.
    """
    wd_grid = ex3_config.probe_regularisation_grid

    def _probe(self_acts, other_acts):
        return _train_eval(self_acts, other_acts, wd_grid)

    return {
        "self_vs_altered_self": _probe(all_acts["self"], all_acts["altered_self"]),
        "self_vs_gemma":        _probe(all_acts["self"], all_acts["gemma"]),
        "altered_self_vs_gemma": _probe(all_acts["altered_self"], all_acts["gemma"]),
    }


# ---------------------------------------------------------------------------
# Per-dataset evaluation helper
# ---------------------------------------------------------------------------

def _per_dataset_eval(
    all_acts: dict,
    ex3_config: Experiment3Config,
) -> dict:
    """
    For each cross-model condition, report AUROC per dataset (test split).
    Probe trained on all datasets pooled (full train split).
    """
    wd_grid = ex3_config.probe_regularisation_grid
    results = {}
    for cond in CROSS_CONDITIONS:
        X_tr, y_tr = build_dataset(all_acts["self"], all_acts[cond], split="train")
        X_v,  y_v  = build_dataset(all_acts["self"], all_acts[cond], split="val")
        if len(X_tr) < 4 or len(X_v) < 2 or len(np.unique(y_tr)) < 2:
            results[cond] = {ds: float("nan") for ds in DATASETS + ["pooled"]}
            continue
        probe, mean, std, _, _ = train_probe(X_tr, y_tr, X_v, y_v, wd_grid)
        per_ds = {}
        for ds in DATASETS:
            X_te, y_te = build_dataset(all_acts["self"], all_acts[cond],
                                       split="test", dataset=ds)
            if len(X_te) < 2 or len(np.unique(y_te)) < 2:
                per_ds[ds] = float("nan")
                continue
            _, auroc, _ = evaluate_probe(probe, mean, std, X_te, y_te)
            per_ds[ds] = round(float(auroc), 5)
        # pooled
        X_te_all, y_te_all = build_dataset(all_acts["self"], all_acts[cond], split="test")
        if len(X_te_all) >= 2 and len(np.unique(y_te_all)) >= 2:
            _, auroc_all, _ = evaluate_probe(probe, mean, std, X_te_all, y_te_all)
            per_ds["pooled"] = round(float(auroc_all), 5)
        else:
            per_ds["pooled"] = float("nan")
        results[cond] = per_ds
    return results


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _apply_split_overrides(
    all_acts: dict[str, list[dict]],
    responses: list[dict],
) -> None:
    """
    Update the split and dataset fields in activation records to match
    the (post-filter, post-reassignment) values in responses.

    Activation files are written at extraction time and may carry stale
    split labels.  This override is applied in-memory before any probe
    training so that cross-dataset and cross-split analyses are correct.
    """
    override_map = {r["prompt_id"]: r for r in responses}
    for cond, acts in all_acts.items():
        for d in acts:
            resp = override_map.get(d["prompt_id"])
            if resp is not None:
                d["split"]   = resp["split"]
                d["dataset"] = resp["dataset"]


def run_all_probe_analyses(
    responses: list[dict],
    ex3_config: Experiment3Config,
    force: bool = False,
) -> dict:
    """
    Run all probe analyses and save results to probe_results_ex3.json.
    Returns the results dict.
    """
    out_path = ex3_config.results_dir_ex3 / "probe_results_ex3.json"

    if out_path.exists() and not force:
        print(f"  Found existing probe results at {out_path}, loading...")
        with open(out_path) as f:
            return json.load(f)

    all_acts = load_all_activations(ex3_config)
    _apply_split_overrides(all_acts, responses)  # use reassigned splits from responses
    ppl_nested = load_perplexity(ex3_config)

    results: dict = {}

    # 1. Base probes
    print("  Training base probes (self vs. each condition)...")
    results["base_probes"] = _run_base_probes(all_acts, ppl_nested, ex3_config)

    # 2. Unified "not-self" probe
    print("  Training unified not-self probe...")
    results["unified_probe"] = _run_unified_probe(all_acts, ppl_nested, ex3_config)

    # 3. Cross-source transfer matrix
    print("  Building cross-source transfer matrix (4 × 4)...")
    results["cross_source_matrix"] = {
        "labels": CROSS_CONDITIONS,
        "values": _run_cross_source_matrix(all_acts, ex3_config),
    }

    # 4. Cross-topic transfer matrices
    print("  Building cross-topic transfer matrices (3 × 3 per condition)...")
    results["cross_topic_matrices"] = {}
    for cond in CROSS_CONDITIONS:
        print(f"    condition: {cond}")
        results["cross_topic_matrices"][cond] = {
            "labels": DATASETS,
            "values": _run_cross_topic_matrix(all_acts, cond, ex3_config),
        }

    # 5. Per-dataset evaluation
    print("  Per-dataset evaluation...")
    results["per_dataset"] = _per_dataset_eval(all_acts, ex3_config)

    # 6. Altered-self comparison
    print("  Altered-self vs. cross-model comparison...")
    results["altered_self_comparison"] = _run_altered_self_comparison(all_acts, ex3_config)

    # 7. Perplexity summary stats (mean perplexity per condition)
    ppl_summary: dict[str, dict] = {}
    for cond in CONDITIONS:
        ppls = [
            rec[cond]["perplexity"]
            for rec in ppl_nested.values()
            if cond in rec and np.isfinite(rec[cond].get("perplexity", float("nan")))
        ]
        if ppls:
            ppl_summary[cond] = {
                "mean": round(float(np.mean(ppls)), 4),
                "median": round(float(np.median(ppls)), 4),
                "p95": round(float(np.percentile(ppls, 95)), 4),
                "n": len(ppls),
            }
        else:
            ppl_summary[cond] = {"mean": float("nan"), "median": float("nan"),
                                  "p95": float("nan"), "n": 0}
    results["perplexity_summary"] = ppl_summary

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved probe results → {out_path}")

    return results
