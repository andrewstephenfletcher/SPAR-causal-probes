"""
Analysis and figures for Experiment 10 (Probe Generalisation).

Three outputs:
  fig1_transfer_matrix.png     — full 9×9 AUROC heatmap
  fig2_by_dataset.png          — 3×3 heatmap averaged over cross-source models
  fig3_by_cross_source.png     — 3×3 heatmap averaged over datasets

Rows = trained-on condition, columns = tested-on condition.
Values are AUROC; white cells indicate insufficient data (None).
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from .config import Experiment10Config

DATASET_DISPLAY = {
    "bigcodebench": "BigCodeBench",
    "oasst1":       "OASST1",
    "gpqa":         "GPQA",
}

SOURCE_DISPLAY = {
    "llama8b":    "Llama 8B",
    "gemma31b":   "Gemma 31B",
    "mistral24b": "Mistral 24B",
}


def _load_matrix(config: Experiment10Config) -> dict:
    path = config.results_dir / "transfer_matrix.json"
    if not path.exists():
        raise FileNotFoundError(f"Transfer matrix not found at {path}. Run probe step first.")
    with open(path) as f:
        return json.load(f)


def _extract_auroc_grid(
    matrix_data: dict,
    row_labels: list[str],
    col_labels: list[str],
) -> np.ndarray:
    """Build a (n_row, n_col) float array; NaN where auroc is None."""
    grid = np.full((len(row_labels), len(col_labels)), np.nan)
    for i, row in enumerate(row_labels):
        for j, col in enumerate(col_labels):
            val = matrix_data["matrix"].get(row, {}).get(col, {}).get("auroc")
            if val is not None:
                grid[i, j] = val
    return grid


def _heatmap(
    ax,
    grid: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    vmin: float = 0.5,
    vmax: float = 1.0,
) -> None:
    im = ax.imshow(grid, vmin=vmin, vmax=vmax, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xlabel("Tested on", fontsize=9)
    ax.set_ylabel("Trained on", fontsize=9)
    ax.set_title(title, fontsize=10)

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = grid[i, j]
            if not np.isnan(val):
                text_color = "white" if val < (vmin + vmax) / 2 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=text_color)
    return im


def make_fig1_full_matrix(matrix_data: dict, config: Experiment10Config) -> None:
    cross_sources = config.cross_sources
    datasets = config.datasets
    conditions = [f"{src}_{ds}" for src in cross_sources for ds in datasets]

    tick_labels = [
        f"{SOURCE_DISPLAY[src]}\n{DATASET_DISPLAY[ds]}"
        for src in cross_sources for ds in datasets
    ]

    grid = _extract_auroc_grid(matrix_data, conditions, conditions)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = _heatmap(ax, grid, tick_labels, tick_labels,
                  title="Probe Transfer: 9×9 AUROC Matrix\n(Llama 70B activations at layer 60)")
    plt.colorbar(im, ax=ax, label="AUROC")
    plt.tight_layout()

    out = config.figures_dir / "fig1_transfer_matrix.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


def make_fig2_by_dataset(matrix_data: dict, config: Experiment10Config) -> None:
    """3×3 heatmap: rows = trained-on dataset, cols = tested-on dataset, averaged over source."""
    cross_sources = config.cross_sources
    datasets = config.datasets

    grid = np.full((len(datasets), len(datasets)), np.nan)
    for i, train_ds in enumerate(datasets):
        for j, test_ds in enumerate(datasets):
            vals = []
            for src in cross_sources:
                train_label = f"{src}_{train_ds}"
                test_label  = f"{src}_{test_ds}"
                val = matrix_data["matrix"].get(train_label, {}).get(test_label, {}).get("auroc")
                if val is not None:
                    vals.append(val)
            if vals:
                grid[i, j] = np.mean(vals)

    tick_labels = [DATASET_DISPLAY[ds] for ds in datasets]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = _heatmap(ax, grid, tick_labels, tick_labels,
                  title="Transfer by Dataset\n(averaged over cross-source models)")
    plt.colorbar(im, ax=ax, label="AUROC")
    plt.tight_layout()

    out = config.figures_dir / "fig2_by_dataset.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


def make_fig3_by_cross_source(matrix_data: dict, config: Experiment10Config) -> None:
    """3×3 heatmap: rows = trained-on cross-source, cols = tested-on cross-source, averaged over dataset."""
    cross_sources = config.cross_sources
    datasets = config.datasets

    grid = np.full((len(cross_sources), len(cross_sources)), np.nan)
    for i, train_src in enumerate(cross_sources):
        for j, test_src in enumerate(cross_sources):
            vals = []
            for train_ds in datasets:
                for test_ds in datasets:
                    train_label = f"{train_src}_{train_ds}"
                    test_label  = f"{test_src}_{test_ds}"
                    val = matrix_data["matrix"].get(train_label, {}).get(test_label, {}).get("auroc")
                    if val is not None:
                        vals.append(val)
            if vals:
                grid[i, j] = np.mean(vals)

    tick_labels = [SOURCE_DISPLAY[src] for src in cross_sources]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = _heatmap(ax, grid, tick_labels, tick_labels,
                  title="Transfer by Cross-Source Model\n(averaged over datasets)")
    plt.colorbar(im, ax=ax, label="AUROC")
    plt.tight_layout()

    out = config.figures_dir / "fig3_by_cross_source.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


def print_summary_table(matrix_data: dict, config: Experiment10Config) -> None:
    cross_sources = config.cross_sources
    datasets = config.datasets

    print("\n=== Transfer Matrix Summary (AUROC) ===")
    conditions = [f"{src}_{ds}" for src in cross_sources for ds in datasets]
    header = "Train \\ Test".ljust(28) + "  ".join(f"{c[:18]:>18}" for c in conditions)
    print(header)
    print("-" * len(header))

    for train_cond in conditions:
        row_vals = []
        for test_cond in conditions:
            val = matrix_data["matrix"].get(train_cond, {}).get(test_cond, {}).get("auroc")
            row_vals.append(f"{val:.3f}" if val is not None else "  N/A")
        print(f"{train_cond[:28]:28s}" + "  ".join(f"{v:>18}" for v in row_vals))

    print()
    all_vals = [
        v["auroc"]
        for row in matrix_data["matrix"].values()
        for v in row.values()
        if v.get("auroc") is not None
    ]
    diag_vals = [
        matrix_data["matrix"].get(c, {}).get(c, {}).get("auroc")
        for c in conditions
        if matrix_data["matrix"].get(c, {}).get(c, {}).get("auroc") is not None
    ]
    off_diag_vals = [
        v["auroc"]
        for i, row_c in enumerate(conditions)
        for j, col_c in enumerate(conditions)
        if i != j
        for v in [matrix_data["matrix"].get(row_c, {}).get(col_c, {})]
        if v.get("auroc") is not None
    ]

    if all_vals:
        print(f"  Mean AUROC (all):      {np.mean(all_vals):.4f}")
    if diag_vals:
        print(f"  Mean AUROC (diagonal): {np.mean(diag_vals):.4f}")
    if off_diag_vals:
        print(f"  Mean AUROC (off-diag): {np.mean(off_diag_vals):.4f}")


def generate_all_figures(matrix_data: dict, config: Experiment10Config) -> None:
    print("\n  Fig 1: full 9×9 transfer matrix")
    make_fig1_full_matrix(matrix_data, config)
    print("\n  Fig 2: by-dataset 3×3 marginal")
    make_fig2_by_dataset(matrix_data, config)
    print("\n  Fig 3: by-cross-source 3×3 marginal")
    make_fig3_by_cross_source(matrix_data, config)
    print_summary_table(matrix_data, config)
