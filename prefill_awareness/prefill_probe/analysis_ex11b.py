"""
Analysis and figures for Experiment 11b (Probe Training and Analysis).

Eight analyses in execution order:
  0  validate            — confirm all activation files present, log counts/shapes
  3  token_distributions — first/last token distributions (pre-normalisation)
  1  depth_curves        — per-layer AUROC vs. relative depth, all targets × datasets
  4  position0_diagnostic— layer-0/1 AUROC vs. token-identity baseline
  2  last_token_confound — pre-normalisation last-token overlap analysis
  5  cross_dataset       — 3×3 generalisation heatmaps (train on A, test on B)
  6  cross_model         — 3×3 source-transfer heatmaps (train on src A, test on src B)
  7  token_position      — accumulation curves from token-position activations
  8  summary             — headline numbers from all prior results

Each function is independently runnable: it skips if its output JSON already exists
(unless force=True is passed).

All activations were extracted with response_normalized (ending in "."), so the
last-token confound is controlled in the activations. Analyses 2 and 3 characterise
the pre-normalisation distributions to motivate the design choice.
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from .analysis_ex4 import MODEL_COLORS, MODEL_DISPLAY
from .config import Experiment11bConfig
from .probe import train_probe, evaluate_probe, apply_normaliser
from .probe_ex11b import (
    DimProbe,
    build_dataset_per_source,
    build_dataset_pooled,
    detect_n_layers,
    evaluate_dim_probe,
    get_common_prompt_ids,
    get_raw_responses,
    get_split_map,
    load_activations_ex11,
    load_token_position_activations_ex11,
    save_dim_probe,
    save_lr_probe,
    train_and_evaluate_layer,
    train_dim_probe,
)
from .sanity_last_token_ex4 import _last_token_label


# ---------------------------------------------------------------------------
# Extended display / color maps for Exp 11b models
# ---------------------------------------------------------------------------

_DISPLAY = {
    **MODEL_DISPLAY,
    "llama70b": "Llama 3.3 70B",
    "gemma31b": "Gemma 4 31B",
    "qwen32b":  "Qwen 2.5 32B",
    "llama8b":  "Llama 3.1 8B",
    "gemma4b":  "Gemma 4 4B",
    "qwen7b":   "Qwen 2.5 7B",
}

_COLORS = {**MODEL_COLORS}

_DS_COLORS = {
    "bigcodebench": "#2980B9",
    "oasst1":       "#E74C3C",
    "gpqa":         "#8E44AD",
}

_DS_MARKERS = {
    "bigcodebench": "o",
    "oasst1":       "s",
    "gpqa":         "^",
}

# ---------------------------------------------------------------------------
# Plot style constants
# ---------------------------------------------------------------------------
_FS_TITLE       = 14   # ax.set_title (single-panel figure titles)
_FS_SUPTITLE    = 14   # fig.suptitle
_FS_FAMILY      = 11   # panel labels in multi-panel figures (bold)
_FS_LABEL       = 12   # axis labels (xlabel / ylabel)
_FS_TICK        = 10   # tick labels
_FS_LEGEND      = 10   # legend
_FS_CELL        = 11   # bar/value annotations
_FS_CBAR        = 11   # colorbar label
_FS_SMALL_ANNOT = 9    # small inline annotations ("chance", "self")


def _save_fig(fig: plt.Figure, path_stem: Path) -> None:
    fig.savefig(path_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _json_exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 2


# ---------------------------------------------------------------------------
# Analysis 0: Validate
# ---------------------------------------------------------------------------

def validate_data(config: Experiment11bConfig, force: bool = False) -> dict:
    out_path = config.results_dir / "validation.json"
    if not force and _json_exists(out_path):
        print("[validate] Already complete, loading.")
        with open(out_path) as f:
            return json.load(f)

    print("\n=== Analysis 0: Validation ===")
    report = {"files": {}, "issues": []}

    for target in config.target_models:
        n_layers = detect_n_layers(target, config)
        data = load_activations_ex11(target, "self", config.datasets[0], config)
        hidden_dim = data[0]["layer_activations"][0].shape[0]
        print(f"  {target}: {n_layers} layers, hidden_dim={hidden_dim}")
        report[target] = {"n_layers": n_layers, "hidden_dim": hidden_dim}

        for cond in ["self"] + config.cross_conditions_for(target):
            for ds in config.datasets:
                for suffix in ["", "_token_positions"]:
                    path = (config.exp11_activations_dir / target / cond
                            / f"{ds}{suffix}.pt")
                    if not path.exists():
                        msg = f"MISSING: {path}"
                        print(f"  WARNING: {msg}")
                        report["issues"].append(msg)
                        report["files"][str(path)] = {"exists": False}
                        continue
                    d = torch.load(path, weights_only=False)
                    n = len(d)
                    report["files"][str(path)] = {"exists": True, "n_records": n}
                    if n < 90:
                        msg = f"LOW RECORD COUNT ({n}): {path}"
                        print(f"  WARNING: {msg}")
                        report["issues"].append(msg)
                    else:
                        print(f"  {target}/{cond}/{ds}{suffix}: {n} records  OK")

    if not report["issues"]:
        print("  Validation passed with no issues.")

    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    return report


# ---------------------------------------------------------------------------
# Analysis 3: Token Distributions
# ---------------------------------------------------------------------------

def _first_token_label(text: str) -> str:
    text = text.strip()
    if not text:
        return "<empty>"
    m = re.match(r"[A-Za-z0-9À-ÿ']+", text)
    if m:
        word = m.group(0)
        return word[:5] if len(word) > 5 else word
    return text[0]


def analysis_token_distributions(config: Experiment11bConfig, force: bool = False) -> None:
    out_dir = config.results_dir / "token_distributions"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = config.figures_dir / "token_distributions"
    fig_dir.mkdir(parents=True, exist_ok=True)

    sentinel = out_dir / "done.json"
    if not force and sentinel.exists():
        print("[token_distributions] Already complete.")
        return

    print("\n=== Analysis 3: Token Distributions ===")

    all_model_keys = config.target_models + config.source_models

    for ds in config.datasets:
        for kind, label_fn in [("last", _last_token_label), ("first", _first_token_label)]:
            token_counts: dict[str, dict[str, int]] = {}

            for model_key in all_model_keys:
                records = get_raw_responses(model_key, ds, config)
                raw_field = f"response_{model_key}"
                counts: dict[str, int] = {}
                for r in records:
                    raw = r.get(raw_field, "")
                    tok = label_fn(raw)
                    counts[tok] = counts.get(tok, 0) + 1
                token_counts[model_key] = counts

            # Save JSON
            out_path = out_dir / f"{kind}_token_{ds}.json"
            with open(out_path, "w") as f:
                json.dump(token_counts, f, indent=2)

            # Figure: grouped bar chart (top-10 tokens by total frequency)
            total: dict[str, int] = {}
            for counts in token_counts.values():
                for tok, cnt in counts.items():
                    total[tok] = total.get(tok, 0) + cnt
            top_tokens = [t for t, _ in sorted(total.items(), key=lambda x: -x[1])[:10]]

            n_models = len(all_model_keys)
            n_tokens = len(top_tokens)
            width = 0.8 / n_models
            x = np.arange(n_tokens)

            fig, ax = plt.subplots(figsize=(13, 5))
            for i, model_key in enumerate(all_model_keys):
                total_model = sum(token_counts[model_key].values())
                heights = [
                    100 * token_counts[model_key].get(t, 0) / max(total_model, 1)
                    for t in top_tokens
                ]
                offset = (i - n_models / 2 + 0.5) * width
                ax.bar(x + offset, heights, width=width * 0.9,
                       label=_DISPLAY.get(model_key, model_key),
                       color=_COLORS.get(model_key, "gray"), alpha=0.85)

            ax.set_xticks(x)
            ax.set_xticklabels([repr(t) for t in top_tokens], fontsize=_FS_TICK, rotation=30, ha="right")
            ax.set_xlabel("Token", fontsize=_FS_LABEL)
            ax.set_ylabel("% of responses", fontsize=_FS_LABEL)
            ax.set_title(f"{kind.capitalize()} token distribution — {ds}", fontsize=_FS_TITLE, fontweight="bold")
            ax.legend(fontsize=_FS_LEGEND, ncol=3, framealpha=0.9)
            ax.tick_params(axis="x", length=0)
            ax.grid(axis="y", alpha=0.3)
            _save_fig(fig, fig_dir / f"{kind}_token_{ds}")

    sentinel.write_text("{}")
    print("  Token distribution analysis complete.")


# ---------------------------------------------------------------------------
# Analysis 1: Depth Curves
# ---------------------------------------------------------------------------

def analysis_depth_curves(config: Experiment11bConfig, force: bool = False) -> dict:
    """
    Per-layer AUROC (pooled: self vs. all 3 cross-sources) for every target × dataset.
    Returns {target: {dataset: results_dict}} for use by later analyses.
    """
    out_dir = config.results_dir / "depth_curves"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = config.figures_dir
    fig_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict[str, dict]] = {}

    for target in config.target_models:
        n_layers = detect_n_layers(target, config)
        all_results[target] = {}

        for ds in config.datasets:
            out_path = out_dir / f"{target}_{ds}.json"
            if not force and _json_exists(out_path):
                print(f"[depth_curves] {target}/{ds}: loading cached results.")
                with open(out_path) as f:
                    all_results[target][ds] = json.load(f)
                continue

            print(f"\n  {target}/{ds}: training probes for {n_layers} layers...")
            split_map = get_split_map(target, ds, config)

            # Load all condition data once
            self_data = load_activations_ex11(target, "self", ds, config)
            cross_data_list = [
                load_activations_ex11(target, cond, ds, config)
                for cond in config.cross_conditions_for(target)
            ]

            lr_auroc, dim_auroc, lr_bacc, dim_bacc = [], [], [], []

            for layer_idx in tqdm(range(n_layers), desc=f"    {target}/{ds}"):
                dataset = build_dataset_pooled(
                    self_data, cross_data_list, split_map, layer_idx
                )
                metrics = train_and_evaluate_layer(dataset, config.probe_regularisation_grid)
                lr_auroc.append(metrics["lr_auroc"])
                dim_auroc.append(metrics["dim_auroc"])
                lr_bacc.append(metrics["lr_balanced_acc"])
                dim_bacc.append(metrics["dim_balanced_acc"])

                # Save best probe (for cross-dataset/cross-model analyses)
                if not metrics.get("skipped"):
                    # Retrain to get the probe object for saving
                    _save_best_probe_if_needed(
                        target, ds, "pooled", layer_idx,
                        self_data, cross_data_list, split_map, config,
                    )

            best_lr_layer = int(np.nanargmax(lr_auroc))
            best_dim_layer = int(np.nanargmax(dim_auroc))

            result = {
                "target": target, "dataset": ds, "n_layers": n_layers,
                "layers": list(range(n_layers)),
                "lr_auroc": lr_auroc, "dim_auroc": dim_auroc,
                "lr_balanced_acc": lr_bacc, "dim_balanced_acc": dim_bacc,
                "best_lr_layer": best_lr_layer,
                "best_lr_auroc": lr_auroc[best_lr_layer] if not np.isnan(lr_auroc[best_lr_layer]) else None,
                "best_dim_layer": best_dim_layer,
                "best_dim_auroc": dim_auroc[best_dim_layer] if not np.isnan(dim_auroc[best_dim_layer]) else None,
            }

            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"  {target}/{ds}: best LR layer={best_lr_layer} "
                  f"AUROC={result['best_lr_auroc']:.4f}  "
                  f"best DIM layer={best_dim_layer} "
                  f"AUROC={result['best_dim_auroc']:.4f}")
            all_results[target][ds] = result

    # Figures — one panel per target, lines per dataset
    for target in config.target_models:
        n_layers = all_results[target][config.datasets[0]]["n_layers"]
        fig, ax = plt.subplots(figsize=(11, 5))

        for ds in config.datasets:
            res = all_results[target][ds]
            layers = res["layers"]
            rel = [l / (n_layers - 1) for l in layers]
            ax.plot(rel, res["lr_auroc"], color=_DS_COLORS[ds],
                    marker=_DS_MARKERS[ds], markersize=3,
                    label=f"{ds} LR", linewidth=1.5)
            ax.plot(rel, res["dim_auroc"], color=_DS_COLORS[ds],
                    linestyle="--", linewidth=1.2, label=f"{ds} DIM")

        ax.axhline(0.5, linestyle=":", color="gray", linewidth=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0.35, 1.05)
        ax.set_xlabel("Relative layer depth", fontsize=_FS_LABEL)
        ax.set_ylabel("AUROC (test)", fontsize=_FS_LABEL)
        ax.set_title(f"Depth curves — {_DISPLAY.get(target, target)}", fontsize=_FS_TITLE, fontweight="bold")
        ax.legend(fontsize=_FS_LEGEND, ncol=2, framealpha=0.9)
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="y", alpha=0.3)
        _save_fig(fig, fig_dir / f"depth_curves_{target}")

    return all_results


def _save_best_probe_if_needed(
    target, dataset, source_label, layer_idx,
    self_data, cross_data_list, split_map, config,
) -> None:
    """Save LR and DIM probes at the given layer (only if not already saved)."""
    lr_path  = config.probes_dir / target / "lr"  / f"{dataset}_{source_label}_layer{layer_idx}.pt"
    dim_path = config.probes_dir / target / "dim" / f"{dataset}_{source_label}_layer{layer_idx}.npy"
    if lr_path.exists() and dim_path.exists():
        return

    dataset_splits = build_dataset_pooled(
        self_data, cross_data_list, split_map, layer_idx
    )
    X_train, y_train = dataset_splits["train"]["X"], dataset_splits["train"]["y"]
    X_val,   y_val   = dataset_splits["val"]["X"],   dataset_splits["val"]["y"]
    if len(X_train) < 4 or len(np.unique(y_train)) < 2:
        return

    probe, mean, std, best_wd, _ = train_probe(
        X_train, y_train, X_val, y_val, config.probe_regularisation_grid
    )
    save_lr_probe(probe, mean, std, best_wd, lr_path)
    dim_probe = train_dim_probe(X_train, y_train)
    save_dim_probe(dim_probe, dim_path)


# ---------------------------------------------------------------------------
# Helper: optimal layer per target
# ---------------------------------------------------------------------------

def _optimal_layer(target: str, depth_results: dict, config: Experiment11bConfig) -> int:
    """
    Median best layer across datasets for this target.
    Falls back to 60% depth if depth_results not available.
    """
    n_layers = detect_n_layers(target, config)
    fallback = round(0.60 * (n_layers - 1))
    if target not in depth_results:
        return fallback
    best_layers = [
        depth_results[target][ds]["best_lr_layer"]
        for ds in config.datasets
        if ds in depth_results[target] and depth_results[target][ds].get("best_lr_layer") is not None
    ]
    return int(np.median(best_layers)) if best_layers else fallback


# ---------------------------------------------------------------------------
# Analysis 4: Position-0 Diagnostic
# ---------------------------------------------------------------------------

def analysis_position0_diagnostic(
    config: Experiment11bConfig, force: bool = False,
) -> None:
    out_dir = config.results_dir / "position0_diagnostic"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Analysis 4: Position-0 Diagnostic ===")

    for target in config.target_models:
        out_path = out_dir / f"{target}.json"
        if not force and _json_exists(out_path):
            print(f"[position0_diagnostic] {target}: already complete.")
            continue

        result = {"target": target, "layers": {}}

        for ds in config.datasets:
            split_map = get_split_map(target, ds, config)
            self_data = load_activations_ex11(target, "self", ds, config)
            cross_data_list = [
                load_activations_ex11(target, cond, ds, config)
                for cond in config.cross_conditions_for(target)
            ]

            ds_result = {}
            for layer_idx in (0, 1):
                dataset = build_dataset_pooled(
                    self_data, cross_data_list, split_map, layer_idx
                )
                metrics = train_and_evaluate_layer(dataset, config.probe_regularisation_grid)

                # Token-identity baseline: all normalized last tokens are "."
                # so a token-identity classifier has zero discriminative power.
                # We report this explicitly rather than building a one-hot encoder.
                ds_result[f"layer{layer_idx}"] = {
                    "lr_auroc":  metrics["lr_auroc"],
                    "dim_auroc": metrics["dim_auroc"],
                    "token_identity_auroc": 0.5,  # all normalized responses end with "."
                    "n_test": metrics.get("n_test", 0),
                    "note": "token_identity_auroc=0.5 by construction: all activations "
                            "extracted on normalised responses ending in '.'",
                }
                print(f"  {target}/{ds}/layer{layer_idx}: "
                      f"LR={metrics['lr_auroc']:.4f}  "
                      f"DIM={metrics['dim_auroc']:.4f}  "
                      f"token-identity=0.50 (constant)")
            result["layers"][ds] = ds_result

        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)


# ---------------------------------------------------------------------------
# Analysis 2: Last-Token Confound
# ---------------------------------------------------------------------------

def analysis_last_token_confound(
    config: Experiment11bConfig, force: bool = False,
) -> None:
    out_dir = config.results_dir / "last_token_confound"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = config.figures_dir
    fig_dir.mkdir(parents=True, exist_ok=True)

    overlap_path = out_dir / "token_overlap.json"
    if not force and _json_exists(overlap_path):
        print("[last_token_confound] Already complete.")
        return

    print("\n=== Analysis 2: Last-Token Confound (pre-normalisation) ===")

    overlap: dict = {}
    source_model_keys = config.source_models  # llama8b, gemma4b, qwen7b

    for ds in config.datasets:
        overlap[ds] = {}
        for target in config.target_models:
            overlap[ds][target] = {}
            target_records = get_raw_responses(target, ds, config)
            target_by_pid = {r["prompt_id"]: r for r in target_records}
            self_raw_field = f"response_{target}"

            for source_key in source_model_keys:
                source_records = get_raw_responses(source_key, ds, config)
                source_by_pid = {r["prompt_id"]: r for r in source_records}
                cross_raw_field = f"response_{source_key}"

                common = set(target_by_pid) & set(source_by_pid)
                n_match = 0
                for pid in common:
                    self_tok = _last_token_label(target_by_pid[pid].get(self_raw_field, ""))
                    cross_tok = _last_token_label(source_by_pid[pid].get(cross_raw_field, ""))
                    if self_tok == cross_tok:
                        n_match += 1

                frac = n_match / len(common) if common else float("nan")
                overlap[ds][target][source_key] = {
                    "n_common": len(common),
                    "n_match": n_match,
                    "overlap_fraction": frac,
                }
                print(f"  {target} vs {source_key} [{ds}]: overlap={frac:.3f} "
                      f"({n_match}/{len(common)})")

    with open(overlap_path, "w") as f:
        json.dump(overlap, f, indent=2)

    # Figure: grouped bar chart of pre-normalisation last-token distributions
    # (one figure per dataset showing top tokens per source model)
    for ds in config.datasets:
        all_counts: dict[str, dict[str, int]] = {}
        for model_key in config.target_models + config.source_models:
            records = get_raw_responses(model_key, ds, config)
            raw_field = f"response_{model_key}"
            counts: dict[str, int] = {}
            for r in records:
                tok = _last_token_label(r.get(raw_field, ""))
                counts[tok] = counts.get(tok, 0) + 1
            all_counts[model_key] = counts

        total: dict[str, int] = {}
        for counts in all_counts.values():
            for tok, cnt in counts.items():
                total[tok] = total.get(tok, 0) + cnt
        top_tokens = [t for t, _ in sorted(total.items(), key=lambda x: -x[1])[:10]]

        n_models = len(all_counts)
        width = 0.8 / n_models
        x = np.arange(len(top_tokens))
        fig, ax = plt.subplots(figsize=(13, 5))
        for i, (model_key, counts) in enumerate(all_counts.items()):
            total_model = sum(counts.values())
            heights = [100 * counts.get(t, 0) / max(total_model, 1) for t in top_tokens]
            offset = (i - n_models / 2 + 0.5) * width
            ax.bar(x + offset, heights, width=width * 0.9,
                   label=_DISPLAY.get(model_key, model_key),
                   color=_COLORS.get(model_key, "gray"), alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([repr(t) for t in top_tokens], fontsize=_FS_TICK, rotation=30, ha="right")
        ax.set_xlabel("Last token (pre-normalisation)", fontsize=_FS_LABEL)
        ax.set_ylabel("% of responses", fontsize=_FS_LABEL)
        ax.set_title(f"Last-token distribution (pre-normalisation) — {ds}", fontsize=_FS_TITLE, fontweight="bold")
        ax.legend(fontsize=_FS_LEGEND, ncol=3, framealpha=0.9)
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="y", alpha=0.3)
        _save_fig(fig, fig_dir / f"last_token_confound_{ds}")

    print("  Note: post-normalisation all responses end with '.'"
          " — no matched-subset analysis needed (confound fully controlled).")


# ---------------------------------------------------------------------------
# Analysis 5: Cross-Dataset Generalization
# ---------------------------------------------------------------------------

def analysis_cross_dataset(
    config: Experiment11bConfig,
    depth_results: dict | None = None,
    force: bool = False,
) -> None:
    out_dir = config.results_dir / "generalization"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = config.figures_dir

    print("\n=== Analysis 5: Cross-Dataset Generalization ===")

    for target in config.target_models:
        out_path = out_dir / f"cross_dataset_{target}.json"
        if not force and _json_exists(out_path):
            print(f"[cross_dataset] {target}: already complete.")
            continue

        opt_layer = _optimal_layer(target, depth_results or {}, config)
        n_layers  = detect_n_layers(target, config)
        print(f"  {target}: using layer {opt_layer} / {n_layers-1} "
              f"({100*opt_layer/(n_layers-1):.0f}% depth)")

        # Load activations for all datasets once
        act: dict[str, dict] = {}
        for ds in config.datasets:
            act[ds] = {
                "self":  load_activations_ex11(target, "self", ds, config),
                "cross": [load_activations_ex11(target, cond, ds, config)
                          for cond in config.cross_conditions_for(target)],
                "split_map": get_split_map(target, ds, config),
            }

        matrix: dict[str, dict[str, float]] = {ds: {} for ds in config.datasets}

        for train_ds in config.datasets:
            split_map_train = act[train_ds]["split_map"]
            train_dataset = build_dataset_pooled(
                act[train_ds]["self"],
                act[train_ds]["cross"],
                split_map_train,
                opt_layer,
            )
            X_train, y_train = train_dataset["train"]["X"], train_dataset["train"]["y"]
            X_val,   y_val   = train_dataset["val"]["X"],   train_dataset["val"]["y"]
            if len(X_train) < 4 or len(np.unique(y_train)) < 2:
                for test_ds in config.datasets:
                    matrix[train_ds][test_ds] = float("nan")
                continue

            probe, mean, std, _, _ = train_probe(
                X_train, y_train, X_val, y_val, config.probe_regularisation_grid
            )

            for test_ds in config.datasets:
                test_dataset = build_dataset_pooled(
                    act[test_ds]["self"],
                    act[test_ds]["cross"],
                    act[test_ds]["split_map"],
                    opt_layer,
                )
                X_test = test_dataset["test"]["X"]
                y_test = test_dataset["test"]["y"]
                if len(X_test) < 2 or len(np.unique(y_test)) < 2:
                    matrix[train_ds][test_ds] = float("nan")
                    continue
                _, auroc, _ = evaluate_probe(probe, mean, std, X_test, y_test)
                matrix[train_ds][test_ds] = auroc
                print(f"    train={train_ds} test={test_ds}: AUROC={auroc:.4f}")

        result = {
            "target": target, "opt_layer": opt_layer,
            "matrix": matrix,
            "mean_off_diagonal": _mean_off_diagonal(matrix, config.datasets),
        }
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        _plot_heatmap(matrix, config.datasets,
                      title=f"Cross-dataset generalisation — {_DISPLAY.get(target, target)}",
                      path_stem=fig_dir / f"cross_dataset_{target}")


# ---------------------------------------------------------------------------
# Analysis 6: Cross-Model Generalization
# ---------------------------------------------------------------------------

def analysis_cross_model(
    config: Experiment11bConfig,
    depth_results: dict | None = None,
    force: bool = False,
) -> None:
    out_dir = config.results_dir / "generalization"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = config.figures_dir

    print("\n=== Analysis 6: Cross-Model Generalization ===")

    # Map condition name → source model display key (all possible sources)
    _all_cond_to_source = {
        "cross_llama8b":  "llama8b",
        "cross_gemma4b":  "gemma4b",
        "cross_qwen7b":   "qwen7b",
        "cross_llama70b": "llama70b",
        "cross_gemma31b": "gemma31b",
        "cross_qwen32b":  "qwen32b",
    }

    for target in config.target_models:
        out_path = out_dir / f"cross_model_{target}.json"
        if not force and _json_exists(out_path):
            print(f"[cross_model] {target}: already complete.")
            continue

        # Build per-target condition→source mapping (excludes self-referential condition)
        cond_to_source = {
            c: _all_cond_to_source[c]
            for c in config.cross_conditions_for(target)
            if c in _all_cond_to_source
        }
        source_keys = sorted(
            cond_to_source.values(),
            key=lambda k: _FAMILY_ORDER.index(k) if k in _FAMILY_ORDER else 99,
        )

        opt_layer = _optimal_layer(target, depth_results or {}, config)
        print(f"  {target}: using layer {opt_layer}")

        # Pool across datasets for more data
        self_all:  list[list[dict]] = []
        cross_all: dict[str, list[list[dict]]] = {sk: [] for sk in source_keys}

        for ds in config.datasets:
            self_all.append(load_activations_ex11(target, "self", ds, config))
            for cond, sk in cond_to_source.items():
                cross_all[sk].append(load_activations_ex11(target, cond, ds, config))

        # Flatten across datasets (merge prompt_ids won't collide since they're per-dataset)
        def _merge(lists: list[list[dict]]) -> list[dict]:
            out = []
            for i, lst in enumerate(lists):
                for r in lst:
                    out.append({**r, "prompt_id": r["prompt_id"] + i * 10000})
            return out

        self_merged = _merge(self_all)
        cross_merged = {sk: _merge(cross_all[sk]) for sk in source_keys}

        # Combined split map (offset prompt_ids same way)
        split_map_merged: dict[int, str] = {}
        for i, ds in enumerate(config.datasets):
            sm = get_split_map(target, ds, config)
            for pid, sp in sm.items():
                split_map_merged[pid + i * 10000] = sp

        matrix: dict[str, dict[str, float]] = {sk: {} for sk in source_keys}

        for train_src in source_keys:
            train_dataset = build_dataset_per_source(
                self_merged, cross_merged[train_src], split_map_merged, opt_layer
            )
            X_train, y_train = train_dataset["train"]["X"], train_dataset["train"]["y"]
            X_val,   y_val   = train_dataset["val"]["X"],   train_dataset["val"]["y"]
            if len(X_train) < 4 or len(np.unique(y_train)) < 2:
                for test_src in source_keys:
                    matrix[train_src][test_src] = float("nan")
                continue

            probe, mean, std, _, _ = train_probe(
                X_train, y_train, X_val, y_val, config.probe_regularisation_grid
            )

            for test_src in source_keys:
                test_dataset = build_dataset_per_source(
                    self_merged, cross_merged[test_src], split_map_merged, opt_layer
                )
                X_test = test_dataset["test"]["X"]
                y_test = test_dataset["test"]["y"]
                if len(X_test) < 2 or len(np.unique(y_test)) < 2:
                    matrix[train_src][test_src] = float("nan")
                    continue
                _, auroc, _ = evaluate_probe(probe, mean, std, X_test, y_test)
                matrix[train_src][test_src] = auroc
                print(f"    train_src={train_src} test_src={test_src}: AUROC={auroc:.4f}")

        result = {
            "target": target, "opt_layer": opt_layer,
            "matrix": matrix,
            "mean_off_diagonal": _mean_off_diagonal(matrix, source_keys),
        }
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        _plot_heatmap(matrix, source_keys,
                      title=f"Cross-model generalisation — {_DISPLAY.get(target, target)}",
                      path_stem=fig_dir / f"cross_model_{target}",
                      labels=[_DISPLAY.get(sk, sk) for sk in source_keys])


# ---------------------------------------------------------------------------
# Analysis 7: Token-Position Accumulation
# ---------------------------------------------------------------------------

def analysis_token_position(config: Experiment11bConfig, force: bool = False) -> None:
    out_dir = config.results_dir / "token_position"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = config.figures_dir

    print("\n=== Analysis 7: Token-Position Accumulation ===")

    for target in config.target_models:
        for ds in config.datasets:
            out_path = out_dir / f"{target}_{ds}.json"
            if not force and _json_exists(out_path):
                print(f"[token_position] {target}/{ds}: already complete.")
                continue

            print(f"  {target}/{ds}: loading token-position activations...")
            split_map = get_split_map(target, ds, config)

            # Load self + all cross conditions
            self_tp  = load_token_position_activations_ex11(target, "self", ds, config)
            cross_tp = [
                load_token_position_activations_ex11(target, cond, ds, config)
                for cond in config.cross_conditions_for(target)
            ]

            if not self_tp:
                print(f"  {target}/{ds}: empty token-position file, skipping.")
                continue

            layers = self_tp[0]["layers"]  # e.g. [24, 35, 47]
            print(f"    Layers: {layers}")

            # Determine available percentage positions
            # Use n_response_tokens to compute which absolute positions correspond to
            # each percentage; find prompts that have each position available.
            pct_targets = list(range(10, 101, 10))  # 10%, 20%, ..., 100%

            position_results: dict[int, dict[int, dict]] = {l: {} for l in layers}

            # Build lookup: pid → record for each condition
            self_by_pid  = {r["prompt_id"]: r for r in self_tp}
            cross_by_pid_list = [{r["prompt_id"]: r for r in cd} for cd in cross_tp]

            for pct in pct_targets:
                for layer_idx in layers:
                    X_train, y_train = [], []
                    X_val,   y_val   = [], []
                    X_test,  y_test  = [], []

                    for self_r in self_tp:
                        pid = self_r["prompt_id"]
                        split = split_map.get(pid)
                        if split not in ("train", "val", "test"):
                            continue
                        n_resp = self_r["n_response_tokens"]
                        target_offset = round((pct / 100.0) * (n_resp - 1))

                        # Find the actual stored position closest to target_offset from start
                        # token_positions are absolute; response_start ≈ positions[0] - 0 offset
                        # Use the position in the list that's closest to target_offset
                        tp_list = self_r["token_positions"]
                        if not tp_list:
                            continue
                        # Infer response_start as tp_list[0] (first absolute position, offset 0)
                        response_start = tp_list[0]
                        target_abs = response_start + target_offset
                        # Find closest stored position
                        closest = min(tp_list, key=lambda p: abs(p - target_abs))
                        if abs(closest - target_abs) > max(5, 0.1 * n_resp):
                            continue  # no close enough position available

                        # Get self activation at this position + layer
                        self_act = self_r["activations"].get(layer_idx, {}).get(closest)
                        if self_act is None:
                            continue
                        self_act = self_act.astype(np.float32)

                        cross_acts = []
                        for cross_by_pid in cross_by_pid_list:
                            cross_r = cross_by_pid.get(pid)
                            if cross_r is None:
                                continue
                            cross_n = cross_r["n_response_tokens"]
                            c_offset = round((pct / 100.0) * (cross_n - 1))
                            c_tp_list = cross_r["token_positions"]
                            if not c_tp_list:
                                continue
                            c_response_start = c_tp_list[0]
                            c_target_abs = c_response_start + c_offset
                            c_closest = min(c_tp_list, key=lambda p: abs(p - c_target_abs))
                            if abs(c_closest - c_target_abs) > max(5, 0.1 * cross_n):
                                continue
                            c_act = cross_r["activations"].get(layer_idx, {}).get(c_closest)
                            if c_act is None:
                                continue
                            cross_acts.append(c_act.astype(np.float32))

                        if not cross_acts:
                            continue

                        if split == "train":
                            X_train.append(self_act); y_train.append(0)
                            for ca in cross_acts:
                                X_train.append(ca); y_train.append(1)
                        elif split == "val":
                            X_val.append(self_act); y_val.append(0)
                            for ca in cross_acts:
                                X_val.append(ca); y_val.append(1)
                        else:
                            X_test.append(self_act); y_test.append(0)
                            for ca in cross_acts:
                                X_test.append(ca); y_test.append(1)

                    splits = {
                        "train": {"X": np.stack(X_train) if X_train else np.empty((0, 1)),
                                  "y": np.array(y_train, dtype=int)},
                        "val":   {"X": np.stack(X_val)   if X_val   else np.empty((0, 1)),
                                  "y": np.array(y_val,   dtype=int)},
                        "test":  {"X": np.stack(X_test)  if X_test  else np.empty((0, 1)),
                                  "y": np.array(y_test,  dtype=int)},
                    }
                    metrics = train_and_evaluate_layer(splits, config.probe_regularisation_grid)
                    n_test = len(X_test)
                    position_results[layer_idx][pct] = {
                        **metrics,
                        "n_test": n_test,
                        "low_confidence": n_test < 30,
                    }

            result = {
                "target": target, "dataset": ds, "layers": layers,
                "pct_targets": pct_targets,
                "results": {str(l): {str(p): position_results[l][p]
                                     for p in pct_targets}
                            for l in layers},
            }
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)

            # Figure
            fig, ax = plt.subplots(figsize=(10, 5))
            for layer_idx in layers:
                n_layers = detect_n_layers(target, config)
                rel_depth = layer_idx / (n_layers - 1)
                aurocs = [position_results[layer_idx].get(p, {}).get("lr_auroc", float("nan"))
                          for p in pct_targets]
                ax.plot(pct_targets, aurocs, marker="o", markersize=4,
                        label=f"Layer {layer_idx} ({100*rel_depth:.0f}% depth)")
            ax.axhline(0.5, linestyle=":", color="gray")
            ax.set_xlabel("Response position (% of length)", fontsize=_FS_LABEL)
            ax.set_ylabel("AUROC (test)", fontsize=_FS_LABEL)
            ax.set_title(f"Token-position accumulation — {_DISPLAY.get(target, target)} / {ds}",
                         fontsize=_FS_TITLE, fontweight="bold")
            ax.legend(fontsize=_FS_LEGEND, framealpha=0.9)
            ax.tick_params(axis="x", length=0)
            ax.set_xlim(0, 105)
            ax.set_ylim(0.35, 1.05)
            ax.grid(alpha=0.3)
            _save_fig(fig, fig_dir / f"token_position_{target}_{ds}")
            print(f"  {target}/{ds}: saved.")


# ---------------------------------------------------------------------------
# Analysis 8: Summary
# ---------------------------------------------------------------------------

def analysis_summary(config: Experiment11bConfig, force: bool = False) -> dict:
    out_path = config.results_dir / "summary.json"
    if not force and _json_exists(out_path):
        print("[summary] Already complete.")
        with open(out_path) as f:
            return json.load(f)

    print("\n=== Analysis 8: Summary Statistics ===")
    summary: dict = {}

    for target in config.target_models:
        t_sum: dict = {}
        n_layers = detect_n_layers(target, config)

        # 1. Peak AUROC per dataset + relative depth at first > 0.90
        for ds in config.datasets:
            dc_path = config.results_dir / "depth_curves" / f"{target}_{ds}.json"
            if not dc_path.exists():
                continue
            with open(dc_path) as f:
                dc = json.load(f)
            best_layer = dc.get("best_lr_layer")
            best_auroc = dc.get("best_lr_auroc")
            lr_auroc_curve = dc.get("lr_auroc", [])
            # First layer where AUROC > 0.90
            first_90_layer = next(
                (l for l, a in enumerate(lr_auroc_curve) if isinstance(a, float) and a > 0.90),
                None,
            )
            t_sum[ds] = {
                "best_lr_layer":       best_layer,
                "best_lr_auroc":       best_auroc,
                "rel_peak":            best_layer / (n_layers - 1) if best_layer is not None else None,
                "first_layer_auroc_gt_0.90": first_90_layer,
                "rel_depth_first_0.90": (first_90_layer / (n_layers - 1)
                                         if first_90_layer is not None else None),
            }
            if best_auroc is not None:
                print(f"  {target}/{ds}: peak AUROC={best_auroc:.4f} "
                      f"at layer {best_layer} ({100*t_sum[ds]['rel_peak']:.0f}% depth)")

        # 2. Cross-dataset off-diagonal
        cd_path = config.results_dir / "generalization" / f"cross_dataset_{target}.json"
        if cd_path.exists():
            with open(cd_path) as f:
                cd = json.load(f)
            t_sum["cross_dataset_mean_off_diagonal"] = cd.get("mean_off_diagonal")

        # 3. Cross-model off-diagonal
        cm_path = config.results_dir / "generalization" / f"cross_model_{target}.json"
        if cm_path.exists():
            with open(cm_path) as f:
                cm = json.load(f)
            t_sum["cross_model_mean_off_diagonal"] = cm.get("mean_off_diagonal")

        # 4. Token-position detection latency (first pct where AUROC > 0.90)
        for ds in config.datasets:
            tp_path = config.results_dir / "token_position" / f"{target}_{ds}.json"
            if not tp_path.exists():
                continue
            with open(tp_path) as f:
                tp = json.load(f)
            for layer_str, pct_dict in tp.get("results", {}).items():
                first_90_pct = next(
                    (int(p) for p, m in sorted(pct_dict.items(), key=lambda x: int(x[0]))
                     if isinstance(m.get("lr_auroc"), float) and m["lr_auroc"] > 0.90),
                    None,
                )
                t_sum.setdefault("token_position", {})[f"{ds}_layer{layer_str}"] = {
                    "first_pct_auroc_gt_0.90": first_90_pct
                }

        summary[target] = t_sum

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Markdown table
    table_path = config.results_dir / "summary_table.md"
    with open(table_path, "w") as f:
        f.write("# Experiment 11b Summary\n\n")
        f.write("| Target | Dataset | Peak AUROC | Peak Layer (rel.) | First >0.90 (rel.) |\n")
        f.write("|--------|---------|-----------|-------------------|--------------------|\n")
        for target in config.target_models:
            for ds in config.datasets:
                ts = summary.get(target, {}).get(ds, {})
                auroc = ts.get("best_lr_auroc")
                rel   = ts.get("rel_peak")
                first = ts.get("rel_depth_first_0.90")
                auroc_s = f"{auroc:.4f}" if auroc is not None else "N/A"
                rel_s   = f"{rel:.2f}"   if rel   is not None else "N/A"
                first_s = f"{first:.2f}" if first is not None else "N/A"
                f.write(f"| {_DISPLAY.get(target, target)} | {ds} "
                        f"| {auroc_s} | {rel_s} | {first_s} |\n")

    print(f"  Summary saved → {out_path}")
    print(f"  Table saved   → {table_path}")
    return summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Analysis 9: Per-Source Depth Curves
# ---------------------------------------------------------------------------

_COND_SOURCE_DISPLAY = {
    "cross_llama8b":  "Llama 3.1 8B",
    "cross_gemma4b":  "Gemma 4 4B",
    "cross_qwen7b":   "Qwen 2.5 7B",
    "cross_llama70b": "Llama 3.3 70B",
    "cross_gemma31b": "Gemma 4 31B",
    "cross_qwen32b":  "Qwen 2.5 32B",
    "pooled":         "Pooled (all sources)",
}

# One colour per family; solid = large model, dashed = small model
_COND_COLORS = {
    "cross_gemma4b":  MODEL_COLORS["gemma31b"],  # Google blue
    "cross_gemma31b": MODEL_COLORS["gemma31b"],  # Google blue
    "cross_llama8b":  MODEL_COLORS["llama70b"],  # purple
    "cross_llama70b": MODEL_COLORS["llama70b"],  # purple
    "cross_qwen7b":   MODEL_COLORS["qwen32b"],   # amber
    "cross_qwen32b":  MODEL_COLORS["qwen32b"],   # amber
    "pooled":         "dimgray",
}

_COND_LINESTYLES = {
    "cross_gemma4b":  "--",               # small → dashed
    "cross_gemma31b": "-",                # large → solid
    "cross_llama8b":  "--",               # small → dashed
    "cross_llama70b": "-",                # large → solid
    "cross_qwen7b":   "--",               # small → dashed
    "cross_qwen32b":  "-",                # large → solid
    "pooled":         (0, (3, 1, 1, 1)),
}

_COND_LINEWIDTHS = {
    "cross_gemma4b":  1.4,
    "cross_gemma31b": 2.0,
    "cross_llama8b":  1.4,
    "cross_llama70b": 2.0,
    "cross_qwen7b":   1.4,
    "cross_qwen32b":  2.0,
    "pooled":         1.4,
}

# Family-grouped legend order
_COND_LEGEND_ORDER = [
    "cross_gemma4b", "cross_gemma31b",
    "cross_llama8b", "cross_llama70b",
    "cross_qwen7b",  "cross_qwen32b",
    "pooled",
]


def analysis_per_source_depth_curves(
    config: Experiment11bConfig,
    targets: list[str] | None = None,
    force: bool = False,
) -> None:
    """
    Per-source depth curves: train self vs. each cross-source separately,
    so we can see which source model drives (or suppresses) probe AUROC.

    By default runs on all target models; pass targets=["llama70b"] to restrict.
    Also overlays the pooled curve (loaded from Analysis 1 results) for comparison.

    Output:
      results/per_source_depth_curves/{target}_{dataset}.json
      figures/per_source_depth_curves_{target}.pdf/png
        One panel per dataset, 4 lines (3 per-source + 1 pooled dashed).
    """
    out_dir = config.results_dir / "per_source_depth_curves"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = config.figures_dir
    fig_dir.mkdir(parents=True, exist_ok=True)

    if targets is None:
        targets = config.target_models

    cond_to_source = {
        "cross_llama8b":  "cross_llama8b",
        "cross_gemma4b":  "cross_gemma4b",
        "cross_qwen7b":   "cross_qwen7b",
        "cross_llama70b": "cross_llama70b",
        "cross_gemma31b": "cross_gemma31b",
        "cross_qwen32b":  "cross_qwen32b",
    }

    for target in targets:
        n_layers = detect_n_layers(target, config)
        n_ds = len(config.datasets)

        # One figure per target, one subplot per dataset
        fig, axes = plt.subplots(1, n_ds, figsize=(5 * n_ds, 5), sharey=True)
        if n_ds == 1:
            axes = [axes]

        for ax, ds in zip(axes, config.datasets):
            out_path = out_dir / f"{target}_{ds}.json"

            if not force and _json_exists(out_path):
                print(f"[per_source_depth_curves] {target}/{ds}: loading cached.")
                with open(out_path) as f:
                    ds_result = json.load(f)
            else:
                print(f"\n  {target}/{ds}: per-source depth curves ({n_layers} layers)...")
                split_map = get_split_map(target, ds, config)
                self_data = load_activations_ex11(target, "self", ds, config)

                ds_result: dict[str, dict] = {}
                for cond in config.cross_conditions_for(target):
                    cross_data = load_activations_ex11(target, cond, ds, config)
                    lr_auroc, dim_auroc = [], []
                    for layer_idx in tqdm(range(n_layers), desc=f"    {cond}"):
                        dataset = build_dataset_per_source(
                            self_data, cross_data, split_map, layer_idx
                        )
                        metrics = train_and_evaluate_layer(
                            dataset, config.probe_regularisation_grid
                        )
                        lr_auroc.append(metrics["lr_auroc"])
                        dim_auroc.append(metrics["dim_auroc"])
                    best_lr_layer = int(np.nanargmax(lr_auroc))
                    ds_result[cond] = {
                        "lr_auroc":  lr_auroc,
                        "dim_auroc": dim_auroc,
                        "best_lr_layer":  best_lr_layer,
                        "best_lr_auroc":  lr_auroc[best_lr_layer],
                    }
                    print(f"    {cond}: best AUROC={lr_auroc[best_lr_layer]:.4f} "
                          f"at layer {best_lr_layer}")

                with open(out_path, "w") as f:
                    json.dump(ds_result, f, indent=2)

            # ---- Plot ----
            rel = [l / (n_layers - 1) for l in range(n_layers)]

            # Load pooled curve first so it can be included in the legend dict
            pooled_path = config.results_dir / "depth_curves" / f"{target}_{ds}.json"
            pooled_auroc = None
            if pooled_path.exists():
                with open(pooled_path) as f:
                    pooled_auroc = json.load(f)["lr_auroc"]

            # Build {cond: line_handle} in legend order, skipping absent conds
            legend_handles, legend_labels = [], []
            for cond in _COND_LEGEND_ORDER:
                if cond == "pooled":
                    if pooled_auroc is None:
                        continue
                    (line,) = ax.plot(
                        rel, pooled_auroc,
                        color=_COND_COLORS["pooled"],
                        linestyle=_COND_LINESTYLES["pooled"],
                        linewidth=_COND_LINEWIDTHS["pooled"],
                    )
                else:
                    if cond not in ds_result:
                        continue
                    (line,) = ax.plot(
                        rel, ds_result[cond]["lr_auroc"],
                        color=_COND_COLORS[cond],
                        linestyle=_COND_LINESTYLES[cond],
                        linewidth=_COND_LINEWIDTHS[cond],
                    )
                legend_handles.append(line)
                legend_labels.append(_COND_SOURCE_DISPLAY.get(cond, cond))

            ax.axhline(0.5, linestyle=":", color="lightgray", linewidth=1)
            ax.set_xlim(0, 1)
            ax.set_ylim(0.35, 1.05)
            ax.set_xlabel("Relative layer depth", fontsize=_FS_LABEL)
            if ax is axes[0]:
                ax.set_ylabel("AUROC (test)", fontsize=_FS_LABEL)
            ax.set_title(ds, fontsize=_FS_FAMILY, fontweight="bold")
            ax.legend(legend_handles, legend_labels, fontsize=_FS_LEGEND, loc="lower right", framealpha=0.9)
            ax.tick_params(axis="x", length=0)
            ax.grid(axis="y", alpha=0.3)

        fig.suptitle(
            f"Per-source depth curves — {_DISPLAY.get(target, target)}",
            fontsize=_FS_SUPTITLE, fontweight="bold", y=1.02,
        )
        fig.tight_layout()
        _save_fig(fig, fig_dir / f"per_source_depth_curves_{target}")
        print(f"  Figure saved → per_source_depth_curves_{target}.pdf/png")


# ---------------------------------------------------------------------------
# Analysis 10: Joint Cross-Model × Cross-Dataset Generalization
# ---------------------------------------------------------------------------

def analysis_joint_generalization(
    config: Experiment11bConfig,
    depth_results: dict | None = None,
    force: bool = False,
) -> None:
    """
    Joint cross-model × cross-dataset generalization (Analysis 10).

    For each target model, runs 3-fold CV over datasets:
      - Hold out one dataset as the test domain
      - Train probe: self vs. cross_{train_src} on the 2 non-heldout datasets
        (using their built-in train/val splits for regularisation selection)
      - Test probe: self vs. cross_{test_src} on ALL records from the held-out dataset

    The diagonal (train_src == test_src) is a pure cross-dataset transfer control;
    the off-diagonal cells are the joint cross-model × cross-dataset test.

    Produces per-target:
      - results/joint_generalization/{target}.json
          {fold_results, mean_matrix, std_matrix, mean_off_diagonal}
      - figures/joint_gen_mean_{target}.png/pdf   — mean heatmap (primary)
      - figures/joint_gen_folds_{target}.png/pdf  — 3-panel per-fold breakdown (secondary)
    """
    out_dir = config.results_dir / "joint_generalization"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = config.figures_dir

    print("\n=== Analysis 10: Joint Cross-Model × Cross-Dataset Generalization ===")

    _all_cond_to_source = {
        "cross_llama8b":  "llama8b",
        "cross_gemma4b":  "gemma4b",
        "cross_qwen7b":   "qwen7b",
        "cross_llama70b": "llama70b",
        "cross_gemma31b": "gemma31b",
        "cross_qwen32b":  "qwen32b",
    }

    datasets = config.datasets

    for target in config.target_models:
        out_path = out_dir / f"{target}.json"
        if not force and _json_exists(out_path):
            print(f"[joint_generalization] {target}: loading cached results.")
            with open(out_path) as f:
                result = json.load(f)
            source_keys  = result["source_keys"]
            fold_results = result["fold_results"]
            mean_matrix  = result["mean_matrix"]
        else:
            cond_to_source = {
                c: _all_cond_to_source[c]
                for c in config.cross_conditions_for(target)
                if c in _all_cond_to_source
            }
            source_keys = sorted(
                cond_to_source.values(),
                key=lambda k: _FAMILY_ORDER.index(k) if k in _FAMILY_ORDER else 99,
            )
            source_to_cond = {v: k for k, v in cond_to_source.items()}

            opt_layer = _optimal_layer(target, depth_results or {}, config)
            print(f"  {target}: layer {opt_layer}, {len(source_keys)} cross-sources")

            # Pre-load all activations: self[ds] and cross[src][ds]
            self_acts: dict[str, list[dict]] = {
                ds: load_activations_ex11(target, "self", ds, config)
                for ds in datasets
            }
            cross_acts: dict[str, dict[str, list[dict]]] = {
                sk: {
                    ds: load_activations_ex11(target, source_to_cond[sk], ds, config)
                    for ds in datasets
                }
                for sk in source_keys
            }
            split_maps: dict[str, dict[int, str]] = {
                ds: get_split_map(target, ds, config) for ds in datasets
            }

            fold_results: dict[str, dict[str, dict[str, float]]] = {}

            for held_out_ds in datasets:
                train_dsets = [ds for ds in datasets if ds != held_out_ds]
                print(f"  {target} | held-out: {held_out_ds} | train: {train_dsets}")

                fold_matrix: dict[str, dict[str, float]] = {sk: {} for sk in source_keys}

                for train_src in source_keys:
                    # Merge 2 training datasets (offset prompt_ids to avoid collision)
                    def _merge(lists: list[list[dict]]) -> list[dict]:
                        out = []
                        for i, lst in enumerate(lists):
                            for r in lst:
                                out.append({**r, "prompt_id": r["prompt_id"] + i * 10000})
                        return out

                    self_train   = _merge([self_acts[ds]            for ds in train_dsets])
                    cross_train  = _merge([cross_acts[train_src][ds] for ds in train_dsets])
                    split_merged = {
                        pid + i * 10000: sp
                        for i, ds in enumerate(train_dsets)
                        for pid, sp in split_maps[ds].items()
                    }

                    train_ds_splits = build_dataset_per_source(
                        self_train, cross_train, split_merged, opt_layer
                    )
                    X_train, y_train = train_ds_splits["train"]["X"], train_ds_splits["train"]["y"]
                    X_val,   y_val   = train_ds_splits["val"]["X"],   train_ds_splits["val"]["y"]

                    if len(X_train) < 4 or len(X_val) < 2 or len(np.unique(y_train)) < 2:
                        for test_src in source_keys:
                            fold_matrix[train_src][test_src] = float("nan")
                        continue

                    probe, mean, std, _, _ = train_probe(
                        X_train, y_train, X_val, y_val, config.probe_regularisation_grid
                    )

                    for test_src in source_keys:
                        # Test on ALL records from held-out dataset (ignore split assignment)
                        self_test  = self_acts[held_out_ds]
                        cross_test = cross_acts[test_src][held_out_ds]
                        common_pids = get_common_prompt_ids([self_test, cross_test])
                        self_by_pid  = {r["prompt_id"]: r for r in self_test}
                        cross_by_pid = {r["prompt_id"]: r for r in cross_test}

                        X_test_list, y_test_list = [], []
                        for pid in common_pids:
                            X_test_list.append(
                                self_by_pid[pid]["layer_activations"][opt_layer].astype(np.float32)
                            )
                            y_test_list.append(0)
                            X_test_list.append(
                                cross_by_pid[pid]["layer_activations"][opt_layer].astype(np.float32)
                            )
                            y_test_list.append(1)

                        if len(X_test_list) < 2:
                            fold_matrix[train_src][test_src] = float("nan")
                            continue

                        X_test = np.stack(X_test_list)
                        y_test = np.array(y_test_list, dtype=int)

                        if len(np.unique(y_test)) < 2:
                            fold_matrix[train_src][test_src] = float("nan")
                            continue

                        _, auroc, _ = evaluate_probe(probe, mean, std, X_test, y_test)
                        fold_matrix[train_src][test_src] = float(auroc)
                        print(
                            f"    heldout={held_out_ds} train={train_src} test={test_src}: "
                            f"AUROC={auroc:.4f}"
                        )

                fold_results[held_out_ds] = fold_matrix

            # Aggregate mean ± std over the 3 folds
            mean_matrix: dict[str, dict[str, float]] = {sk: {} for sk in source_keys}
            std_matrix:  dict[str, dict[str, float]] = {sk: {} for sk in source_keys}
            for train_src in source_keys:
                for test_src in source_keys:
                    vals = [
                        fold_results[ds][train_src].get(test_src, float("nan"))
                        for ds in datasets
                    ]
                    valid = [v for v in vals if not np.isnan(v)]
                    mean_matrix[train_src][test_src] = float(np.mean(valid)) if valid else float("nan")
                    std_matrix[train_src][test_src]  = (
                        float(np.std(valid)) if len(valid) > 1 else float("nan")
                    )

            result = {
                "target":            target,
                "opt_layer":         opt_layer,
                "source_keys":       source_keys,
                "datasets":          datasets,
                "fold_results":      fold_results,
                "mean_matrix":       mean_matrix,
                "std_matrix":        std_matrix,
                "mean_off_diagonal": _mean_off_diagonal(mean_matrix, source_keys),
            }
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)

        src_labels = [_DISPLAY.get(sk, sk) for sk in source_keys]

        # Primary figure: mean AUROC heatmap (same style as Analysis 6)
        _plot_heatmap(
            mean_matrix, source_keys,
            title=f"Joint generalization (mean ± std over dataset folds) — "
                  f"{_DISPLAY.get(target, target)}",
            path_stem=fig_dir / f"joint_gen_mean_{target}",
            labels=src_labels,
        )

        # Secondary figure: one heatmap per held-out dataset
        n_ds = len(datasets)
        fig, axes = plt.subplots(
            1, n_ds, figsize=(5 * n_ds + 1, 5), constrained_layout=True
        )
        axes_list = list(axes) if n_ds > 1 else [axes]
        last_im = None
        for ax, held_out_ds in zip(axes_list, datasets):
            matrix = fold_results[held_out_ds]
            n = len(source_keys)
            data = np.full((n, n), np.nan)
            for i, k1 in enumerate(source_keys):
                for j, k2 in enumerate(source_keys):
                    v = matrix.get(k1, {}).get(k2)
                    if v is not None and not np.isnan(v):
                        data[i, j] = v
            cmap = plt.get_cmap("RdYlGn")
            last_im = ax.imshow(data, vmin=0.5, vmax=1.0, cmap="RdYlGn", aspect="auto")
            for i in range(n):
                for j in range(n):
                    val = data[i, j]
                    if not np.isnan(val):
                        norm_val = (val - 0.5) / 0.5
                        r, g, b, _ = cmap(norm_val)
                        luminance = 0.299 * r + 0.587 * g + 0.114 * b
                        text_color = "white" if luminance < 0.45 else "black"
                        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                                fontsize=_FS_CELL, fontweight="bold", color=text_color)
            ax.set_xticks(range(n))
            ax.set_xticklabels(src_labels, rotation=30, ha="right", fontsize=_FS_TICK)
            ax.set_yticks(range(n))
            ax.set_yticklabels(src_labels, fontsize=_FS_TICK)
            ax.set_title(f"Held-out: {_DS_SHORT.get(held_out_ds, held_out_ds)}", fontsize=_FS_FAMILY, fontweight="bold")
            ax.set_xlabel("Test source", fontsize=_FS_LABEL)
            if ax is axes_list[0]:
                ax.set_ylabel("Train source", fontsize=_FS_LABEL)

        if last_im is not None:
            cbar = fig.colorbar(last_im, ax=axes_list, shrink=0.75, pad=0.03)
            cbar.set_label("AUROC", fontsize=_FS_CBAR)

        fig.suptitle(
            f"Joint generalization (per fold) — {_DISPLAY.get(target, target)}",
            fontsize=_FS_SUPTITLE, fontweight="bold",
        )
        _save_fig(fig, fig_dir / f"joint_gen_folds_{target}")
        print(
            f"  {target}: mean off-diagonal AUROC = "
            f"{result['mean_off_diagonal']:.4f}"
        )

    print("  Analysis 10 complete.")


def _mean_off_diagonal(matrix: dict, keys: list[str]) -> float:
    vals = [
        v for i, k1 in enumerate(keys)
        for j, k2 in enumerate(keys)
        if i != j
        for v in [matrix.get(k1, {}).get(k2)]
        if v is not None and not (isinstance(v, float) and np.isnan(v))
    ]
    return float(np.mean(vals)) if vals else float("nan")


def _plot_heatmap(
    matrix: dict,
    keys: list[str],
    title: str,
    path_stem: Path,
    labels: list[str] | None = None,
) -> None:
    if labels is None:
        labels = keys
    n = len(keys)
    data = np.full((n, n), np.nan)
    for i, k1 in enumerate(keys):
        for j, k2 in enumerate(keys):
            v = matrix.get(k1, {}).get(k2)
            if v is not None:
                data[i, j] = v

    fig, ax = plt.subplots(figsize=(6, 5))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        im = ax.imshow(data, vmin=0.5, vmax=1.0, cmap="RdYlGn", aspect="auto")
    plt.colorbar(im, ax=ax, label="AUROC")

    for i in range(n):
        for j in range(n):
            val = data[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=_FS_CELL, color="black")

    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=_FS_TICK)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=_FS_TICK)
    ax.set_xlabel("Test", fontsize=_FS_LABEL)
    ax.set_ylabel("Train", fontsize=_FS_LABEL)
    ax.set_title(title, fontsize=_FS_TITLE, fontweight="bold")
    _save_fig(fig, path_stem)


# ---------------------------------------------------------------------------
# Blog figures: combined 1×3 heatmaps
# ---------------------------------------------------------------------------

_DS_SHORT = {
    "bigcodebench": "BCB",
    "oasst1":       "OASST1",
    "gpqa":         "GPQA",
}

_SRC_SHORT = {
    "gemma4b":  "Gemma\n4 4B",
    "gemma31b": "Gemma\n4 31B",
    "llama8b":  "Llama\n3.1 8B",
    "llama70b": "Llama\n3.3 70B",
    "qwen7b":   "Qwen\n2.5 7B",
    "qwen32b":  "Qwen\n2.5 32B",
}

# Canonical family-sorted order for cross-model axes
_FAMILY_ORDER = ["gemma4b", "gemma31b", "llama8b", "llama70b", "qwen7b", "qwen32b"]


def _plot_combined_heatmaps(
    matrices: dict[str, dict],
    targets: list[str],
    keys: list[str],
    tick_labels: list[str],
    title: str,
    path_stem: Path,
    per_panel_keys: dict[str, tuple[list[str], list[str]]] | None = None,
) -> None:
    n_targets = len(targets)
    fig, axes = plt.subplots(
        1, n_targets, figsize=(5 * n_targets + 1, 5), constrained_layout=True
    )
    axes_list = list(axes) if n_targets > 1 else [axes]

    all_vals = [
        v
        for target in targets
        for row in matrices.get(target, {}).values()
        for v in row.values()
        if v is not None and not np.isnan(v)
    ]
    vmin = min(0.5, min(all_vals)) if all_vals else 0.5
    vmax = 1.0
    cmap = plt.get_cmap("RdYlGn")

    last_im = None
    for ax, target in zip(axes_list, targets):
        panel_keys, panel_labels = (
            per_panel_keys[target] if per_panel_keys and target in per_panel_keys
            else (keys, tick_labels)
        )
        n = len(panel_keys)
        matrix = matrices.get(target, {})
        data = np.full((n, n), np.nan)
        for i, k1 in enumerate(panel_keys):
            for j, k2 in enumerate(panel_keys):
                v = matrix.get(k1, {}).get(k2)
                if v is not None:
                    data[i, j] = float(v)

        last_im = ax.imshow(data, vmin=vmin, vmax=vmax, cmap="RdYlGn", aspect="auto")

        for i in range(n):
            for j in range(n):
                val = data[i, j]
                if not np.isnan(val):
                    norm_val = (val - vmin) / max(vmax - vmin, 1e-9)
                    r, g, b, _ = cmap(norm_val)
                    luminance = 0.299 * r + 0.587 * g + 0.114 * b
                    text_color = "white" if luminance < 0.45 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=_FS_CELL, fontweight="bold", color=text_color)

        ax.set_xticks(range(n))
        ax.set_xticklabels(panel_labels, fontsize=_FS_TICK)
        ax.set_yticks(range(n))
        ax.set_yticklabels(panel_labels, fontsize=_FS_TICK)
        ax.set_title(_DISPLAY.get(target, target), fontsize=_FS_FAMILY, fontweight="bold", pad=8)
        ax.set_xlabel("Test", fontsize=_FS_LABEL)
        if ax is axes_list[0]:
            ax.set_ylabel("Train", fontsize=_FS_LABEL)

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes_list, shrink=0.75, pad=0.03)
        cbar.set_label("AUROC", fontsize=_FS_CBAR)
        cbar.ax.tick_params(labelsize=_FS_TICK)

    fig.suptitle(title, fontsize=_FS_SUPTITLE, fontweight="bold")
    _save_fig(fig, path_stem)


def analysis_blog_depth_curves(config: Experiment11bConfig, force: bool = False) -> None:
    """
    Combined 1×3 blog depth-curve figure: one panel per dataset, all 3 target models
    overlaid, LR AUROC only.  Family order: Gemma → Llama → Qwen.
    Output: figures/blog_depth_curves.png/pdf
    """
    fig_dir = config.figures_dir
    out_png = fig_dir / "blog_depth_curves.png"
    if not force and out_png.exists():
        print("[blog_depth_curves] Already exists.")
        return

    print("\n=== Blog Depth Curves (combined 1×3) ===")

    _TARGET_COLORS = {
        "gemma31b": MODEL_COLORS["gemma31b"],
        "llama70b": MODEL_COLORS["llama70b"],
        "qwen32b":  MODEL_COLORS["qwen32b"],
    }
    ordered_targets = ["gemma31b", "llama70b", "qwen32b"]

    # Load cached depth-curve results
    dc_dir = config.results_dir / "depth_curves"
    results: dict[str, dict[str, dict]] = {}
    for target in ordered_targets:
        results[target] = {}
        for ds in config.datasets:
            path = dc_dir / f"{target}_{ds}.json"
            if not path.exists():
                print(f"  WARNING: {path} not found — run analysis 1 first.")
                return
            with open(path) as f:
                results[target][ds] = json.load(f)

    n_ds = len(config.datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(5 * n_ds + 1, 4.5),
                             sharey=True, constrained_layout=True)
    axes_list = list(axes)

    for ax, ds in zip(axes_list, config.datasets):
        for target in ordered_targets:
            res = results[target][ds]
            n_layers = res["n_layers"]
            rel = [l / (n_layers - 1) for l in res["layers"]]
            ax.plot(rel, res["lr_auroc"],
                    color=_TARGET_COLORS[target],
                    linewidth=2.0,
                    label=_DISPLAY.get(target, target))

        ax.axhline(0.5, linestyle=":", color="lightgray", linewidth=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0.35, 1.05)
        ax.set_xlabel("Relative layer depth", fontsize=_FS_LABEL)
        ax.set_title(_DS_SHORT.get(ds, ds), fontsize=_FS_FAMILY, fontweight="bold")
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="y", alpha=0.3)
        if ax is axes_list[0]:
            ax.set_ylabel("AUROC (test)", fontsize=_FS_LABEL)
            ax.legend(fontsize=_FS_LEGEND, loc="lower right", framealpha=0.9)

    fig.suptitle("Depth curves — LR probe AUROC by dataset",
                 fontsize=_FS_SUPTITLE, fontweight="bold")
    _save_fig(fig, fig_dir / "blog_depth_curves")
    print("  Saved → blog_depth_curves.png/pdf")


def analysis_blog_heatmaps(config: Experiment11bConfig, force: bool = False) -> None:
    """
    Generate combined 1×3 heatmap figures (one column per target model) for blog use.
    Requires analyses 5 (cross_dataset) and 6 (cross_model) to have been run first.
    Output: figures/blog_cross_dataset.png/pdf, figures/blog_cross_model.png/pdf
    """
    gen_dir = config.results_dir / "generalization"
    fig_dir = config.figures_dir

    print("\n=== Blog Heatmaps (combined 1×3) ===")

    for kind, filename_tmpl, keys, tick_labels, title, stem in [
        (
            "cross_dataset",
            "cross_dataset_{target}.json",
            config.datasets,
            [_DS_SHORT[d] for d in config.datasets],
            "Cross-dataset generalisation",
            "blog_cross_dataset",
        ),
        (
            "cross_model",
            "cross_model_{target}.json",
            _FAMILY_ORDER,
            [_SRC_SHORT[s] for s in _FAMILY_ORDER],
            "Cross-source generalisation",
            "blog_cross_model",
        ),
    ]:
        out_png = fig_dir / f"{stem}.png"
        if not force and out_png.exists():
            print(f"[blog_heatmaps] {stem}: already exists.")
            continue

        matrices: dict[str, dict] = {}
        missing = False
        for target in config.target_models:
            path = gen_dir / filename_tmpl.format(target=target)
            if not path.exists():
                print(f"  WARNING: {path} not found — run analysis 5/6 first.")
                missing = True
                break
            with open(path) as f:
                matrices[target] = json.load(f)["matrix"]

        if missing:
            continue

        # For cross_model each target excludes itself, so derive per-panel keys
        # from the actual matrix keys (sorted by family order).
        per_panel_keys = None
        if kind == "cross_model":
            per_panel_keys = {}
            for target in config.target_models:
                t_keys = sorted(
                    matrices[target].keys(),
                    key=lambda k: _FAMILY_ORDER.index(k) if k in _FAMILY_ORDER else 99,
                )
                per_panel_keys[target] = (t_keys, [_SRC_SHORT[k] for k in t_keys])

        ordered_targets = sorted(
            config.target_models,
            key=lambda t: _FAMILY_ORDER.index(t) if t in _FAMILY_ORDER else 99,
        )
        _plot_combined_heatmaps(
            matrices, ordered_targets, keys, tick_labels, title,
            fig_dir / stem,
            per_panel_keys=per_panel_keys,
        )
        print(f"  Saved → {stem}.png/pdf")
