"""
Analysis and figures for Experiment 11c (Norm-Controlled Probes).

Four analyses in execution order:
  0  norm_profiles           — mean L2 norm per layer for self and each cross condition
  1  depth_curves_normalized — per-source depth curves with L2-normalised activations
  2  comparison_and_step     — unnorm vs. norm AUROC overlay + step retention ratios
  3  cosine_similarity       — DIM direction cosine similarity between adjacent layers

The central question: is Gemma 31B's AUROC step at ~60–70% depth (on within-family
Gemma E4B conditions) genuine new computation, or explained by residual stream norm
growth? If the step survives L2 normalisation (retention ratio ≈ 1), it reflects
directional signal. If it collapses (retention ratio ≈ 0), it was norm-driven.

All activations read from Experiment 11 .pt files. CPU-only; no model inference.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from .analysis_ex11b import (
    _COND_COLORS,
    _COND_LINESTYLES,
    _COND_SOURCE_DISPLAY,
    _DISPLAY,
    _json_exists,
    _save_fig,
)
from .config import Experiment11cConfig
from .probe_ex11b import (
    detect_n_layers,
    get_common_prompt_ids,
    get_split_map,
    load_activations_ex11,
    train_and_evaluate_layer,
    train_dim_probe,
)


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x)
    return x / (norm + 1e-12)


def _build_dataset_per_source_normalized(
    self_data: list[dict],
    cross_data: list[dict],
    split_map: dict[int, str],
    layer_idx: int,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Same as build_dataset_per_source but L2-normalises each activation vector.
    Label 0 = self, 1 = cross.
    """
    common_pids = get_common_prompt_ids([self_data, cross_data])
    self_by_pid  = {r["prompt_id"]: r for r in self_data}
    cross_by_pid = {r["prompt_id"]: r for r in cross_data}

    splits: dict[str, dict] = {s: {"X": [], "y": []} for s in ("train", "val", "test")}

    for pid in common_pids:
        split = split_map.get(pid)
        if split not in splits:
            continue
        self_act  = _l2_normalize(
            self_by_pid[pid]["layer_activations"][layer_idx].astype(np.float32)
        )
        cross_act = _l2_normalize(
            cross_by_pid[pid]["layer_activations"][layer_idx].astype(np.float32)
        )
        splits[split]["X"].extend([self_act, cross_act])
        splits[split]["y"].extend([0, 1])

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


def _mean_norm_per_layer(data: list[dict], n_layers: int) -> list[float]:
    norms: list[list[float]] = [[] for _ in range(n_layers)]
    for r in data:
        for l in range(n_layers):
            act = r["layer_activations"][l].astype(np.float32)
            norms[l].append(float(np.linalg.norm(act)))
    return [float(np.mean(norms[l])) if norms[l] else float("nan") for l in range(n_layers)]


# ---------------------------------------------------------------------------
# Analysis 0: Norm Profiles
# ---------------------------------------------------------------------------

def analysis_norm_profiles(config: Experiment11cConfig, force: bool = False) -> None:
    out_dir = config.results_dir / "norm_profiles"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = config.figures_dir
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Analysis 0: Norm Profiles ===")

    cond_display = {
        "self":           "Self",
        "cross_llama8b":  "Llama 3.1 8B",
        "cross_gemma4b":  "Gemma 4 4B",
        "cross_qwen7b":   "Qwen 2.5 7B",
        "cross_llama70b": "Llama 3.3 70B",
        "cross_gemma31b": "Gemma 4 31B",
        "cross_qwen32b":  "Qwen 2.5 32B",
    }
    cond_colors  = {"self": "black", **_COND_COLORS}

    for target in config.target_models:
        all_conds = ["self"] + config.cross_conditions_for(target)
        out_path = out_dir / f"{target}.json"
        if not force and _json_exists(out_path):
            print(f"[norm_profiles] {target}: already complete.")
            continue

        n_layers = detect_n_layers(target, config)
        rel = [l / (n_layers - 1) for l in range(n_layers)]
        n_ds = len(config.datasets)

        all_norms: dict[str, dict[str, list[float]]] = {ds: {} for ds in config.datasets}

        for ds in config.datasets:
            print(f"  {target}/{ds}: computing norm profiles...")
            for cond in all_conds:
                data = load_activations_ex11(target, cond, ds, config)
                all_norms[ds][cond] = _mean_norm_per_layer(data, n_layers)

        with open(out_path, "w") as f:
            json.dump({"target": target, "norms": all_norms}, f, indent=2)

        fig, axes = plt.subplots(1, n_ds, figsize=(5 * n_ds, 4.5), sharey=True)
        if n_ds == 1:
            axes = [axes]

        for ax, ds in zip(axes, config.datasets):
            for cond in all_conds:
                norms = all_norms[ds].get(cond, [])
                if not norms:
                    continue
                ax.plot(
                    rel, norms,
                    color=cond_colors.get(cond, "gray"),
                    linestyle="-" if cond == "self" else "--",
                    linewidth=1.6,
                    label=cond_display.get(cond, cond),
                )
            ax.set_xlabel("Relative layer depth", fontsize=11)
            if ax is axes[0]:
                ax.set_ylabel("Mean L2 norm", fontsize=11)
            ax.set_title(ds, fontsize=12)
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.3)

        fig.suptitle(
            f"Residual stream L2 norm — {_DISPLAY.get(target, target)}",
            fontsize=14, y=1.02,
        )
        fig.tight_layout()
        _save_fig(fig, fig_dir / f"norm_profiles_{target}")
        print(f"  {target}: norm profiles saved.")


# ---------------------------------------------------------------------------
# Analysis 1: Depth Curves (L2-Normalised)
# ---------------------------------------------------------------------------

def analysis_depth_curves_normalized(
    config: Experiment11cConfig, force: bool = False
) -> dict:
    """
    Per-source depth curves with L2-normalised activation vectors.
    Mirrors Analysis 9 (per_source_depth_curves) from Exp 11b, but each activation
    vector is L2-normalised before probe training.

    Returns {target: {dataset: {cond: {"lr_auroc": [...], ...}}}}
    """
    out_dir = config.results_dir / "depth_curves_normalized"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = config.figures_dir

    print("\n=== Analysis 1: Depth Curves (L2-Normalised) ===")

    all_results: dict = {}

    for target in config.target_models:
        n_layers = detect_n_layers(target, config)
        n_ds = len(config.datasets)
        all_results[target] = {}

        fig, axes = plt.subplots(1, n_ds, figsize=(5 * n_ds, 5), sharey=True)
        if n_ds == 1:
            axes = [axes]

        for ax, ds in zip(axes, config.datasets):
            out_path = out_dir / f"{target}_{ds}.json"

            if not force and _json_exists(out_path):
                print(f"[depth_curves_normalized] {target}/{ds}: loading cached.")
                with open(out_path) as f:
                    ds_result = json.load(f)
            else:
                print(f"\n  {target}/{ds}: normalised depth curves ({n_layers} layers)...")
                split_map = get_split_map(target, ds, config)
                self_data = load_activations_ex11(target, "self", ds, config)

                ds_result: dict[str, dict] = {}
                for cond in config.cross_conditions_for(target):
                    cross_data = load_activations_ex11(target, cond, ds, config)
                    lr_auroc, dim_auroc = [], []
                    for layer_idx in tqdm(range(n_layers), desc=f"    {cond} (norm)"):
                        dataset = _build_dataset_per_source_normalized(
                            self_data, cross_data, split_map, layer_idx
                        )
                        metrics = train_and_evaluate_layer(
                            dataset, config.probe_regularisation_grid
                        )
                        lr_auroc.append(metrics["lr_auroc"])
                        dim_auroc.append(metrics["dim_auroc"])
                    best_lr_layer = int(np.nanargmax(lr_auroc))
                    ds_result[cond] = {
                        "lr_auroc":      lr_auroc,
                        "dim_auroc":     dim_auroc,
                        "best_lr_layer": best_lr_layer,
                        "best_lr_auroc": lr_auroc[best_lr_layer],
                    }
                    print(f"    {cond} (norm): best AUROC={lr_auroc[best_lr_layer]:.4f} "
                          f"at layer {best_lr_layer}")

                with open(out_path, "w") as f:
                    json.dump(ds_result, f, indent=2)

            all_results[target][ds] = ds_result

            rel = [l / (n_layers - 1) for l in range(n_layers)]
            for cond in config.cross_conditions_for(target):
                if cond not in ds_result:
                    continue
                ax.plot(
                    rel, ds_result[cond]["lr_auroc"],
                    color=_COND_COLORS[cond],
                    linestyle=_COND_LINESTYLES[cond],
                    linewidth=1.8,
                    label=_COND_SOURCE_DISPLAY.get(cond, cond),
                )

            ax.axhline(0.5, linestyle=":", color="lightgray", linewidth=1)
            ax.set_xlim(0, 1)
            ax.set_ylim(0.35, 1.05)
            ax.set_xlabel("Relative layer depth", fontsize=11)
            if ax is axes[0]:
                ax.set_ylabel("AUROC (test, L2-normalised)", fontsize=11)
            ax.set_title(ds, fontsize=12)
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.3)

        fig.suptitle(
            f"Depth curves (L2-normalised) — {_DISPLAY.get(target, target)}",
            fontsize=14, y=1.02,
        )
        fig.tight_layout()
        _save_fig(fig, fig_dir / f"depth_curves_normalized_{target}")
        print(f"  Figure saved → depth_curves_normalized_{target}.pdf/png")

    return all_results


# ---------------------------------------------------------------------------
# Analysis 2: Comparison and Step Retention
# ---------------------------------------------------------------------------

def _find_step_layers(auroc_curve: list[float], min_delta: float = 0.03) -> list[int]:
    """
    Find layer indices l where auroc_curve[l+1] - auroc_curve[l] >= min_delta.
    """
    steps = []
    for l in range(len(auroc_curve) - 1):
        a_prev = auroc_curve[l]
        a_next = auroc_curve[l + 1]
        if not (isinstance(a_prev, (int, float)) and isinstance(a_next, (int, float))):
            continue
        if np.isnan(a_prev) or np.isnan(a_next):
            continue
        if a_next - a_prev >= min_delta:
            steps.append(l)
    return steps


def analysis_comparison_and_step(
    config: Experiment11cConfig, force: bool = False
) -> None:
    """
    Overlay unnormalized (11b) vs. L2-normalised (11c) per-source depth curves.
    Identify step layers and compute step retention ratios.

    Prerequisite: Experiment 11b analysis 9 must have been run.
    """
    out_dir = config.results_dir / "comparison_and_step"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = config.figures_dir

    print("\n=== Analysis 2: Comparison and Step Retention ===")

    for target in config.target_models:
        n_layers = detect_n_layers(target, config)
        rel = [l / (n_layers - 1) for l in range(n_layers)]
        n_ds = len(config.datasets)

        target_result: dict = {}
        missing_prereq = False

        for ds in config.datasets:
            out_path = out_dir / f"{target}_{ds}.json"
            if not force and _json_exists(out_path):
                print(f"[comparison_and_step] {target}/{ds}: already complete.")
                with open(out_path) as f:
                    target_result[ds] = json.load(f)
                continue

            unnorm_path = (
                config.exp11b_results_dir / "per_source_depth_curves"
                / f"{target}_{ds}.json"
            )
            norm_path = (
                config.results_dir / "depth_curves_normalized" / f"{target}_{ds}.json"
            )

            if not unnorm_path.exists():
                print(f"  WARNING: {unnorm_path} missing — run Exp 11b analysis 9 first.")
                missing_prereq = True
                continue
            if not norm_path.exists():
                print(f"  WARNING: {norm_path} missing — run analysis 1 first.")
                missing_prereq = True
                continue

            with open(unnorm_path) as f:
                unnorm = json.load(f)
            with open(norm_path) as f:
                norm = json.load(f)

            ds_result: dict[str, dict] = {}

            for cond in config.cross_conditions_for(target):
                if cond not in unnorm or cond not in norm:
                    continue
                ua = unnorm[cond]["lr_auroc"]
                na = norm[cond]["lr_auroc"]

                # Per-layer deltas (shift by 1)
                deltas_unnorm = [
                    float(ua[l + 1] - ua[l]) if l + 1 < len(ua) else float("nan")
                    for l in range(len(ua))
                ]
                deltas_norm = [
                    float(na[l + 1] - na[l]) if l + 1 < len(na) else float("nan")
                    for l in range(len(na))
                ]

                step_layers = _find_step_layers(ua)

                retention: dict[int, float] = {}
                for l in step_layers:
                    if l < len(deltas_unnorm) and l < len(deltas_norm):
                        du = deltas_unnorm[l]
                        dn = deltas_norm[l]
                        if not np.isnan(du) and not np.isnan(dn) and du > 0:
                            retention[l] = float(dn / du)

                mean_ret = float(np.mean(list(retention.values()))) if retention else None
                ds_result[cond] = {
                    "lr_auroc_unnorm":  ua,
                    "lr_auroc_norm":    na,
                    "deltas_unnorm":    deltas_unnorm,
                    "deltas_norm":      deltas_norm,
                    "step_layers":      step_layers,
                    "retention_ratios": {str(k): v for k, v in retention.items()},
                    "mean_retention":   mean_ret,
                }
                if mean_ret is not None:
                    print(f"  {target}/{ds}/{cond}: {len(step_layers)} step(s), "
                          f"mean retention={mean_ret:.3f}")
                else:
                    print(f"  {target}/{ds}/{cond}: no significant steps found.")

            target_result[ds] = ds_result
            with open(out_path, "w") as f:
                json.dump(ds_result, f, indent=2)

        if missing_prereq:
            continue

        # Figure: 1×n_ds grid, unnorm (solid) vs. norm (dashed) per cross-condition
        fig, axes = plt.subplots(1, n_ds, figsize=(5 * n_ds, 5), sharey=True)
        if n_ds == 1:
            axes = [axes]

        for ax, ds in zip(axes, config.datasets):
            ds_r = target_result.get(ds, {})
            for cond in config.cross_conditions_for(target):
                if cond not in ds_r:
                    continue
                cr = ds_r[cond]
                color = _COND_COLORS.get(cond, "gray")
                label_base = _COND_SOURCE_DISPLAY.get(cond, cond)
                ax.plot(rel, cr["lr_auroc_unnorm"], color=color, linewidth=1.8,
                        linestyle="-",  label=f"{label_base} (raw)")
                ax.plot(rel, cr["lr_auroc_norm"],   color=color, linewidth=1.4,
                        linestyle="--", label=f"{label_base} (L2-norm)")
                for l in cr.get("step_layers", []):
                    if l < len(rel):
                        ax.axvline(rel[l], color=color, alpha=0.25, linewidth=0.8)

            ax.axhline(0.5, linestyle=":", color="lightgray", linewidth=1)
            ax.set_xlim(0, 1)
            ax.set_ylim(0.35, 1.05)
            ax.set_xlabel("Relative layer depth", fontsize=11)
            if ax is axes[0]:
                ax.set_ylabel("AUROC (test)", fontsize=11)
            ax.set_title(ds, fontsize=12)
            ax.legend(fontsize=7, ncol=2)
            ax.grid(axis="y", alpha=0.3)

        fig.suptitle(
            f"Unnorm vs. L2-norm — {_DISPLAY.get(target, target)}",
            fontsize=14, y=1.02,
        )
        fig.tight_layout()
        _save_fig(fig, fig_dir / f"comparison_and_step_{target}")
        print(f"  Figure saved → comparison_and_step_{target}.pdf/png")


# ---------------------------------------------------------------------------
# Analysis 3: Cosine Similarity of DIM Directions
# ---------------------------------------------------------------------------

def analysis_cosine_similarity(
    config: Experiment11cConfig, force: bool = False
) -> None:
    """
    For each layer, fit a DimProbe (unnormalised activations) and compute the
    cosine similarity between adjacent-layer DIM directions.

    A sharp drop toward 0 or negative at a particular layer indicates the DIM
    direction is rotating sharply — new information being written. This is the
    expected signature of genuine computation at the step layers.

    Prioritises Gemma 31B (clearest step in within-family condition) by running
    it first, but processes all target models.
    """
    out_dir = config.results_dir / "cosine_similarity"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = config.figures_dir

    print("\n=== Analysis 3: DIM Direction Cosine Similarity ===")

    targets_ordered = (
        ["gemma31b"] + [t for t in config.target_models if t != "gemma31b"]
    )

    for target in targets_ordered:
        out_path = out_dir / f"{target}.json"
        if not force and _json_exists(out_path):
            print(f"[cosine_similarity] {target}: already complete.")
            continue

        n_layers = detect_n_layers(target, config)
        rel = [l / (n_layers - 1) for l in range(n_layers)]
        n_ds = len(config.datasets)

        fig, axes = plt.subplots(1, n_ds, figsize=(5 * n_ds, 4.5), sharey=True)
        if n_ds == 1:
            axes = [axes]

        target_result: dict = {}

        for ax, ds in zip(axes, config.datasets):
            split_map = get_split_map(target, ds, config)
            self_data = load_activations_ex11(target, "self", ds, config)

            ds_result: dict[str, list[float]] = {}
            print(f"  {target}/{ds}: computing DIM directions ({n_layers} layers)...")

            for cond in config.cross_conditions_for(target):
                cross_data = load_activations_ex11(target, cond, ds, config)
                common_pids = get_common_prompt_ids([self_data, cross_data])
                self_by_pid  = {r["prompt_id"]: r for r in self_data}
                cross_by_pid = {r["prompt_id"]: r for r in cross_data}

                directions: list[np.ndarray] = []
                for layer_idx in tqdm(range(n_layers), desc=f"    {cond} dirs"):
                    X_all, y_all = [], []
                    for pid in common_pids:
                        self_act  = self_by_pid[pid]["layer_activations"][layer_idx].astype(np.float32)
                        cross_act = cross_by_pid[pid]["layer_activations"][layer_idx].astype(np.float32)
                        X_all.extend([self_act, cross_act])
                        y_all.extend([0, 1])
                    X = np.stack(X_all)
                    y = np.array(y_all, dtype=int)
                    probe = train_dim_probe(X, y)
                    directions.append(probe.direction)

                cosine_sims: list[float] = []
                for l in range(len(directions) - 1):
                    cosine_sims.append(float(np.dot(directions[l], directions[l + 1])))
                cosine_sims.append(float("nan"))

                ds_result[cond] = cosine_sims
                ax.plot(
                    rel, cosine_sims,
                    color=_COND_COLORS.get(cond, "gray"),
                    linewidth=1.6,
                    label=_COND_SOURCE_DISPLAY.get(cond, cond),
                )

            target_result[ds] = ds_result

            ax.axhline(0.0, linestyle=":", color="lightgray", linewidth=1)
            ax.set_xlim(0, 1)
            ax.set_ylim(-1.05, 1.05)
            ax.set_xlabel("Relative layer depth", fontsize=11)
            if ax is axes[0]:
                ax.set_ylabel("Cosine similarity (adjacent layers)", fontsize=11)
            ax.set_title(ds, fontsize=12)
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.3)

        with open(out_path, "w") as f:
            json.dump({"target": target, "cosine_sims": target_result}, f, indent=2)

        fig.suptitle(
            f"DIM direction similarity — {_DISPLAY.get(target, target)}",
            fontsize=14, y=1.02,
        )
        fig.tight_layout()
        _save_fig(fig, fig_dir / f"cosine_similarity_{target}")
        print(f"  {target}: cosine similarity saved.")
