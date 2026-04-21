"""
Plotting and summary statistics for Experiment 2 (position-wise analysis).

Produces:
  Figure 1 — accumulation_curves.png   (AUROC per layer vs. position)
  Figure 2 — layer30_vs_baseline.png   (layer 30 probe AUROC vs. perplexity baseline)
  Figure 3 — heatmap.png               (layers × positions, coloured by AUROC)
  Figure 4 — derivatives.png           (delta AUROC between consecutive positions, per layer)
  results/summary_ex2.csv
  stdout: interpretive checklist
"""

import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from .config import Experiment2Config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pivot(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Return (layers × positions) pivot table."""
    return df.pivot(index="layer", columns="position", values=value_col)


# ---------------------------------------------------------------------------
# Figure 1: Accumulation curves per layer
# ---------------------------------------------------------------------------

def _figure_accumulation_curves(df: pd.DataFrame, ex2_config: Experiment2Config) -> None:
    pivot = _pivot(df, "auroc")
    positions = sorted(df["position"].unique())

    fig, ax = plt.subplots(figsize=(11, 5))
    cmap = plt.get_cmap("tab10")

    for i, layer in enumerate(sorted(pivot.index)):
        aurocs = [pivot.loc[layer, p] if p in pivot.columns else np.nan for p in positions]
        ax.plot(positions, aurocs, marker="o", markersize=4, linewidth=1.5,
                label=f"Layer {layer}", color=cmap(i))

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="Chance (0.5)")
    ax.set_xlabel("Relative token position (0 = first response token)", fontsize=12)
    ax.set_ylabel("Probe AUROC (test set)", fontsize=12)
    ax.set_title(
        "Prefill signal accumulation: probe AUROC vs. position by layer",
        fontsize=13,
    )
    ax.set_xlim(-0.5, max(positions) + 0.5)
    ax.set_ylim(0.3, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = ex2_config.results_dir_ex2 / "accumulation_curves.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


# ---------------------------------------------------------------------------
# Figure 2: Layer 30 probe AUROC vs. cumulative perplexity baseline
# ---------------------------------------------------------------------------

def _figure_layer30_vs_baseline(df: pd.DataFrame, ex2_config: Experiment2Config) -> None:
    best_layer = int(df.groupby("layer")["auroc"].mean().idxmax())
    layer_df = df[df["layer"] == best_layer].sort_values("position")
    positions = layer_df["position"].tolist()

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(positions, layer_df["auroc"].tolist(), marker="s", markersize=4,
            linewidth=1.5, color="steelblue", label=f"Probe AUROC (layer {best_layer})")
    ax.plot(positions, layer_df["perplexity_baseline_auroc"].tolist(),
            marker="^", markersize=4, linewidth=1.5, linestyle="--",
            color="darkorange", label="Cumulative perplexity baseline AUROC")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="Chance (0.5)")

    ax.set_xlabel("Relative token position (0 = first response token)", fontsize=12)
    ax.set_ylabel("AUROC (test set)", fontsize=12)
    ax.set_title(
        f"Layer {best_layer} probe vs. perplexity baseline: AUROC vs. position",
        fontsize=13,
    )
    ax.set_xlim(-0.5, max(positions) + 0.5)
    ax.set_ylim(0.3, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = ex2_config.results_dir_ex2 / "layer30_vs_baseline.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


# ---------------------------------------------------------------------------
# Figure 3: Heatmap (layers × positions)
# ---------------------------------------------------------------------------

def _figure_heatmap(df: pd.DataFrame, ex2_config: Experiment2Config) -> None:
    pivot = _pivot(df, "auroc")
    layers = sorted(pivot.index)
    positions = sorted(pivot.columns)

    matrix = pivot.loc[layers, positions].values.astype(float)

    fig, ax = plt.subplots(figsize=(13, 4))
    im = ax.imshow(matrix, aspect="auto", vmin=0.4, vmax=1.0,
                   cmap="RdYlGn", origin="upper")
    plt.colorbar(im, ax=ax, label="AUROC")

    ax.set_xticks(range(len(positions)))
    ax.set_xticklabels([str(p) for p in positions], fontsize=9)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f"L{l}" for l in layers], fontsize=10)
    ax.set_xlabel("Relative token position", fontsize=12)
    ax.set_ylabel("Layer", fontsize=12)
    ax.set_title("Probe AUROC heatmap (layers × positions)", fontsize=13)

    # Annotate cells
    for r, layer in enumerate(layers):
        for c, pos in enumerate(positions):
            val = matrix[r, c]
            if np.isfinite(val):
                color = "white" if val < 0.65 or val > 0.92 else "black"
                ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=color)

    plt.tight_layout()
    out = ex2_config.results_dir_ex2 / "heatmap.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


# ---------------------------------------------------------------------------
# Figure 4: Derivatives (delta AUROC between consecutive positions)
# ---------------------------------------------------------------------------

def _figure_derivatives(df: pd.DataFrame, ex2_config: Experiment2Config) -> None:
    pivot = _pivot(df, "auroc")
    positions = sorted(df["position"].unique())
    layers = sorted(pivot.index)

    fig, ax = plt.subplots(figsize=(11, 5))
    cmap = plt.get_cmap("tab10")

    for i, layer in enumerate(layers):
        aurocs = np.array([
            pivot.loc[layer, p] if p in pivot.columns else np.nan
            for p in positions
        ], dtype=float)
        deltas = np.diff(aurocs)
        midpoints = [(positions[j] + positions[j + 1]) / 2 for j in range(len(positions) - 1)]
        ax.plot(midpoints, deltas, marker="o", markersize=3, linewidth=1.2,
                label=f"Layer {layer}", color=cmap(i))

    ax.axhline(0.0, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Position (midpoint between sampled positions)", fontsize=12)
    ax.set_ylabel("ΔAUROC between consecutive positions", fontsize=12)
    ax.set_title(
        "Rate of prefill signal accumulation (derivative of AUROC vs. position)",
        fontsize=13,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = ex2_config.results_dir_ex2 / "derivatives.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


# ---------------------------------------------------------------------------
# Interpretive checklist
# ---------------------------------------------------------------------------

def _classify_accumulation_pattern(
    positions: list[int],
    aurocs: list[float],
) -> str:
    """
    Classify the accumulation pattern for one layer's AUROC curve.

    Rules (applied in order):
      SMOOTH       — AUROC increases monotonically (no drop > 0.02), spread > 0.10
      STEP         — single position where delta > 0.15
      EARLY PLATEAU — reaches ≥ 0.85 AUROC within first 25% of positions
      LATE EMERGENCE — first position with AUROC ≥ 0.70 is in the last 33% of positions
      MIXED        — none of the above, or multiple patterns co-present
    """
    valid = [(p, a) for p, a in zip(positions, aurocs) if np.isfinite(a)]
    if len(valid) < 3:
        return "INSUFFICIENT DATA"

    pos_arr = np.array([v[0] for v in valid])
    aur_arr = np.array([v[1] for v in valid])
    deltas = np.diff(aur_arr)
    spread = float(aur_arr.max() - aur_arr.min())
    n = len(pos_arr)

    # STEP: single large jump
    if np.any(deltas > 0.15):
        return "STEP"

    # EARLY PLATEAU: near-peak AUROC reached quickly
    first_25_pct_cutoff = pos_arr[max(1, n // 4)]
    early_aur = aur_arr[pos_arr <= first_25_pct_cutoff]
    if len(early_aur) > 0 and float(early_aur.max()) >= 0.85:
        return "EARLY PLATEAU"

    # SMOOTH: mostly monotone with meaningful range
    drops = deltas[deltas < -0.02]
    if len(drops) == 0 and spread > 0.10:
        return "SMOOTH"

    # LATE EMERGENCE
    above_threshold = pos_arr[aur_arr >= 0.70]
    if len(above_threshold) > 0:
        first_high = float(above_threshold[0])
        total_range = float(pos_arr[-1] - pos_arr[0])
        if total_range > 0 and (first_high - pos_arr[0]) / total_range >= 0.67:
            return "LATE EMERGENCE"

    return "MIXED"


def _print_interpretive_checklist(
    df: pd.DataFrame,
    ex2_config: Experiment2Config,
) -> None:
    pivot = _pivot(df, "auroc")
    positions = sorted(df["position"].unique())
    layers = sorted(pivot.index)

    # Overall stats
    best_layer_mean = df.groupby("layer")["auroc"].mean().idxmax()
    best_pos_mean = df.groupby("position")["auroc"].mean().idxmax()

    ppl_df = df[df["layer"] == layers[0]][["position", "perplexity_baseline_auroc"]].drop_duplicates()
    ppl_at_end = float(ppl_df[ppl_df["position"] == max(positions)]["perplexity_baseline_auroc"].iloc[0])
    probe_at_end = float(pivot.loc[best_layer_mean, max(positions)]) if max(positions) in pivot.columns else float("nan")

    print("\n" + "=" * 65)
    print("  EXPERIMENT 2 — INTERPRETIVE CHECKLIST")
    print("=" * 65)

    print(f"\n  Best layer (mean AUROC): Layer {best_layer_mean}")
    print(f"  Best position (mean AUROC): Position {best_pos_mean}")
    print(f"  Probe AUROC at max position (layer {best_layer_mean}): {probe_at_end:.4f}")
    print(f"  Perplexity baseline AUROC at max position:             {ppl_at_end:.4f}")

    if np.isfinite(probe_at_end) and np.isfinite(ppl_at_end):
        if probe_at_end > ppl_at_end + 0.03:
            print("  [CHECK] Probe exceeds perplexity baseline — structural signal detected.")
        elif probe_at_end > ppl_at_end - 0.03:
            print("  [NOTE] Probe ≈ perplexity baseline — signal may be largely perplexity-driven.")
        else:
            print("  [NOTE] Perplexity baseline exceeds probe — consider re-checking pipeline.")

    print("\n  Accumulation patterns per layer:")
    for layer in layers:
        aurocs = [
            float(pivot.loc[layer, p]) if p in pivot.columns and np.isfinite(pivot.loc[layer, p])
            else float("nan")
            for p in positions
        ]
        pattern = _classify_accumulation_pattern(positions, aurocs)
        mean_aur = np.nanmean(aurocs)
        print(f"    Layer {layer:2d}: {pattern:<18s}  mean AUROC={mean_aur:.4f}")

    print("=" * 65)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _save_summary_csv(df: pd.DataFrame, ex2_config: Experiment2Config) -> None:
    pivot_auroc = _pivot(df, "auroc")
    pivot_acc = _pivot(df, "balanced_accuracy")
    positions = sorted(df["position"].unique())
    layers = sorted(df["layer"].unique())
    ppl_by_pos = (
        df[["position", "perplexity_baseline_auroc"]]
        .drop_duplicates()
        .set_index("position")["perplexity_baseline_auroc"]
        .to_dict()
    )

    rows = []

    # Per-layer summary
    for layer in layers:
        for pos in positions:
            auroc = pivot_auroc.loc[layer, pos] if pos in pivot_auroc.columns else float("nan")
            acc = pivot_acc.loc[layer, pos] if pos in pivot_acc.columns else float("nan")
            rows.append({
                "layer": layer,
                "position": pos,
                "auroc": round(float(auroc), 5) if np.isfinite(auroc) else "",
                "balanced_accuracy": round(float(acc), 5) if np.isfinite(acc) else "",
                "perplexity_baseline_auroc": round(float(ppl_by_pos.get(pos, float("nan"))), 5),
            })

    out = ex2_config.results_dir_ex2 / "summary_ex2.csv"
    fieldnames = ["layer", "position", "auroc", "balanced_accuracy", "perplexity_baseline_auroc"]
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Summary saved → {out}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_all_figures_ex2(
    probe_results_df: pd.DataFrame,
    ex2_config: Experiment2Config,
) -> None:
    """Generate all four figures, print interpretive checklist, save summary CSV."""
    print("  Generating Figure 1: accumulation curves...")
    _figure_accumulation_curves(probe_results_df, ex2_config)

    print("  Generating Figure 2: best-layer vs. perplexity baseline...")
    _figure_layer30_vs_baseline(probe_results_df, ex2_config)

    print("  Generating Figure 3: AUROC heatmap...")
    _figure_heatmap(probe_results_df, ex2_config)

    print("  Generating Figure 4: derivatives...")
    _figure_derivatives(probe_results_df, ex2_config)

    _print_interpretive_checklist(probe_results_df, ex2_config)
    _save_summary_csv(probe_results_df, ex2_config)
