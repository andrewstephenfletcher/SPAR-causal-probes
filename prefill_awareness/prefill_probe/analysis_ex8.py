"""
Figure generation for Experiment 8 (Cross-Architecture Probing).

Figures:
  1. 2D heatmaps — AUROC at each (layer, position) cell for Mistral 24B and
     Gemma 31B, shown side-by-side.
  2. Scaling curve — peak probe AUROC vs. model size (parameters), combining
     Ex2 (Llama 8B) + Ex8 (Mistral 24B, Gemma 31B).
  3. Accumulation curves — AUROC at the peak layer as a function of position,
     comparing all available models.
  4. Functional threshold scatter — first position where AUROC > 0.75 vs.
     model size.
  5. Probe vs. perplexity — probe AUROC vs. perplexity baseline AUROC per
     (layer, position) cell, for each model.

Saves figures to results_dir_ex8/figures/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import Experiment8Config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_df(config: Experiment8Config, model_name: str) -> pd.DataFrame | None:
    csv_path = config.results_dir_ex8 / f"probe_results_{model_name}.csv"
    if not csv_path.exists():
        print(f"  WARNING: probe results not found: {csv_path}")
        return None
    return pd.read_csv(csv_path)


def _load_ex2_df(config: Experiment8Config) -> pd.DataFrame | None:
    csv_path = config.ex2_results_dir / "probe_results.csv"
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


def _figures_dir(config: Experiment8Config) -> Path:
    d = config.results_dir_ex8 / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Figure 1: 2D heatmaps
# ---------------------------------------------------------------------------

def figure1_heatmaps(dfs: dict[str, pd.DataFrame], config: Experiment8Config) -> None:
    figs_dir = _figures_dir(config)
    available = {k: v for k, v in dfs.items() if v is not None}
    if not available:
        print("  Skipping Figure 1: no probe results available.")
        return

    n_models = len(available)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 6))
    if n_models == 1:
        axes = [axes]

    model_labels = {"mistral": "Mistral 24B", "gemma31b": "Gemma 31B"}

    for ax, (model_name, df) in zip(axes, available.items()):
        layers_sorted = sorted(df["layer"].unique())
        positions_sorted = sorted(df["position"].unique())

        grid = np.full((len(layers_sorted), len(positions_sorted)), np.nan)
        layer_idx = {l: i for i, l in enumerate(layers_sorted)}
        pos_idx   = {p: j for j, p in enumerate(positions_sorted)}

        for _, row in df.iterrows():
            i = layer_idx.get(int(row["layer"]))
            j = pos_idx.get(int(row["position"]))
            if i is not None and j is not None:
                grid[i, j] = row["auroc"]

        im = ax.imshow(
            grid, aspect="auto", cmap="RdYlGn", vmin=0.5, vmax=1.0,
            origin="lower",
        )
        plt.colorbar(im, ax=ax, label="AUROC")

        ax.set_xticks(range(len(positions_sorted)))
        ax.set_xticklabels(positions_sorted, fontsize=8)
        ax.set_xlabel("Response Position")

        # Show every 5th layer label to avoid crowding
        ytick_every = max(1, len(layers_sorted) // 10)
        ax.set_yticks(range(0, len(layers_sorted), ytick_every))
        ax.set_yticklabels(layers_sorted[::ytick_every], fontsize=8)
        ax.set_ylabel("Layer")

        ax.set_title(model_labels.get(model_name, model_name))

    fig.suptitle("Experiment 8: 2D Probe AUROC (Layer × Position)", fontsize=12)
    plt.tight_layout()
    out = figs_dir / "figure1_heatmaps.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure 1 saved → {out}")


# ---------------------------------------------------------------------------
# Figure 2: Scaling curve
# ---------------------------------------------------------------------------

_MODEL_SIZES = {
    "llama8b":   8,
    "gemma9b":   9,
    "mistral24b": 24,
    "gemma31b":  31,
    "llama70b":  70,
}


def figure2_scaling_curve(
    dfs: dict[str, pd.DataFrame],
    ex2_df: pd.DataFrame | None,
    config: Experiment8Config,
) -> None:
    figs_dir = _figures_dir(config)
    points: list[tuple[float, float, str]] = []  # (size_B, peak_auroc, label)

    if ex2_df is not None:
        peak = ex2_df["auroc"].max()
        points.append((8, peak, "Llama 8B"))

    labels = {"mistral": ("mistral24b", "Mistral 24B"), "gemma31b": ("gemma31b", "Gemma 31B")}
    for model_name, df in dfs.items():
        if df is None:
            continue
        size_key, label = labels.get(model_name, (model_name, model_name))
        size = _MODEL_SIZES.get(size_key, 0)
        if size == 0:
            continue
        peak = df["auroc"].max()
        points.append((size, peak, label))

    if len(points) < 2:
        print("  Skipping Figure 2: fewer than 2 data points.")
        return

    sizes, aurocs, labels_str = zip(*sorted(points, key=lambda x: x[0]))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sizes, aurocs, "o-", color="steelblue", markersize=8, linewidth=2)
    for s, a, l in zip(sizes, aurocs, labels_str):
        ax.annotate(l, (s, a), textcoords="offset points", xytext=(6, 4), fontsize=9)

    ax.set_xlabel("Model Size (B parameters)", fontsize=11)
    ax.set_ylabel("Peak Probe AUROC", fontsize=11)
    ax.set_title("Experiment 8: Probe AUROC vs. Model Scale", fontsize=12)
    ax.set_ylim(0.5, 1.02)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = figs_dir / "figure2_scaling_curve.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure 2 saved → {out}")


# ---------------------------------------------------------------------------
# Figure 3: Accumulation curves
# ---------------------------------------------------------------------------

def figure3_accumulation_curves(
    dfs: dict[str, pd.DataFrame],
    ex2_df: pd.DataFrame | None,
    config: Experiment8Config,
) -> None:
    figs_dir = _figures_dir(config)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]
    color_idx = 0

    def _plot_model(df: pd.DataFrame, label: str, color) -> None:
        peak_layer = int(df.loc[df["auroc"].idxmax(), "layer"])
        layer_df = df[df["layer"] == peak_layer].sort_values("position")
        ax.plot(layer_df["position"], layer_df["auroc"], "o-",
                label=f"{label} (layer {peak_layer})", color=color, linewidth=2, markersize=5)

    if ex2_df is not None:
        _plot_model(ex2_df, "Llama 8B", colors[color_idx])
        color_idx += 1

    model_labels = {"mistral": "Mistral 24B", "gemma31b": "Gemma 31B"}
    for model_name, df in dfs.items():
        if df is None:
            continue
        _plot_model(df, model_labels.get(model_name, model_name), colors[color_idx])
        color_idx += 1

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Chance")
    ax.set_xlabel("Response Position (tokens)", fontsize=11)
    ax.set_ylabel("Probe AUROC", fontsize=11)
    ax.set_title("Experiment 8: Probe AUROC Accumulation by Position", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_ylim(0.4, 1.02)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = figs_dir / "figure3_accumulation_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure 3 saved → {out}")


# ---------------------------------------------------------------------------
# Figure 4: Functional threshold scatter
# ---------------------------------------------------------------------------

def figure4_threshold_scatter(
    dfs: dict[str, pd.DataFrame],
    ex2_df: pd.DataFrame | None,
    config: Experiment8Config,
    threshold: float = 0.75,
) -> None:
    figs_dir = _figures_dir(config)

    def _first_threshold_pos(df: pd.DataFrame) -> float | None:
        peak_layer = int(df.loc[df["auroc"].idxmax(), "layer"])
        layer_df = df[df["layer"] == peak_layer].sort_values("position")
        above = layer_df[layer_df["auroc"] >= threshold]
        if above.empty:
            return None
        return float(above["position"].iloc[0])

    points: list[tuple[float, float, str]] = []
    if ex2_df is not None:
        pos = _first_threshold_pos(ex2_df)
        if pos is not None:
            points.append((8, pos, "Llama 8B"))

    model_labels = {"mistral": ("Mistral 24B", 24), "gemma31b": ("Gemma 31B", 31)}
    for model_name, df in dfs.items():
        if df is None:
            continue
        label, size = model_labels.get(model_name, (model_name, 0))
        if size == 0:
            continue
        pos = _first_threshold_pos(df)
        if pos is not None:
            points.append((size, pos, label))

    if len(points) < 2:
        print("  Skipping Figure 4: insufficient data.")
        return

    sizes, positions, labels_str = zip(*sorted(points, key=lambda x: x[0]))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(sizes, positions, s=100, color="darkorange", zorder=3)
    for s, p, l in zip(sizes, positions, labels_str):
        ax.annotate(l, (s, p), textcoords="offset points", xytext=(6, 4), fontsize=9)

    ax.set_xlabel("Model Size (B parameters)", fontsize=11)
    ax.set_ylabel(f"First Position with AUROC ≥ {threshold}", fontsize=11)
    ax.set_title("Experiment 8: Self-Awareness Functional Threshold by Scale", fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = figs_dir / "figure4_threshold_scatter.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure 4 saved → {out}")


# ---------------------------------------------------------------------------
# Figure 5: Probe AUROC vs. perplexity baseline
# ---------------------------------------------------------------------------

def figure5_probe_vs_perplexity(dfs: dict[str, pd.DataFrame], config: Experiment8Config) -> None:
    figs_dir = _figures_dir(config)
    available = {k: v for k, v in dfs.items() if v is not None}
    if not available:
        return

    n_models = len(available)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), sharey=True)
    if n_models == 1:
        axes = [axes]

    model_labels = {"mistral": "Mistral 24B", "gemma31b": "Gemma 31B"}
    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]

    for ax, (model_name, df), color in zip(axes, available.items(), colors):
        clean = df.dropna(subset=["auroc", "perplexity_baseline_auroc"])
        ax.scatter(
            clean["perplexity_baseline_auroc"], clean["auroc"],
            alpha=0.4, s=20, color=color,
        )
        ax.plot([0.5, 1.0], [0.5, 1.0], "k--", linewidth=0.8, label="Equal")
        ax.set_xlabel("Perplexity Baseline AUROC", fontsize=10)
        ax.set_ylabel("Probe AUROC", fontsize=10)
        ax.set_title(model_labels.get(model_name, model_name))
        ax.set_xlim(0.45, 1.0)
        ax.set_ylim(0.45, 1.0)
        ax.legend(fontsize=8)

    fig.suptitle("Experiment 8: Probe vs. Perplexity Baseline", fontsize=12)
    plt.tight_layout()
    out = figs_dir / "figure5_probe_vs_perplexity.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure 5 saved → {out}")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def generate_all_figures_ex8(
    results: dict[str, pd.DataFrame | None],
    config: Experiment8Config,
) -> None:
    """
    Generate all 5 figures for Experiment 8.

    results: {"mistral": DataFrame | None, "gemma31b": DataFrame | None}
    """
    ex2_df = _load_ex2_df(config)
    if ex2_df is None:
        print("  NOTE: Experiment 2 probe results not found — scaling figures will be partial.")

    figure1_heatmaps(results, config)
    figure2_scaling_curve(results, ex2_df, config)
    figure3_accumulation_curves(results, ex2_df, config)
    figure4_threshold_scatter(results, ex2_df, config)
    figure5_probe_vs_perplexity(results, config)
