"""
Scaling analysis for Experiment 4.

Produces four figures and a summary table comparing probe performance across
Llama 3.1 8B (32 layers), Gemma 4 31B (60 layers), and Llama 3.3 70B (80 layers).

Figure 1 — normalized_auroc.png
    Layer-wise AUROC vs. relative layer depth (layer / n_layers) for all three
    models, using the primary cross-family condition.  Horizontal dashed lines
    show each model's perplexity baseline.

Figure 2 — peak_layer_analysis.png
    Grouped bar chart: absolute peak layer, relative peak position, and
    remaining-depth fraction for each model.

Figure 3 — probe_vs_perplexity_gap.png
    Bar chart of (best-layer probe AUROC − perplexity baseline AUROC) per model.
    A gap > 0 means the probe captures information beyond per-token surprisal.

Figure 4 — within_vs_cross_family.png
    Llama 70B only: layer-wise AUROC for cross_gemma9b (cross-family) and
    cross_llama8b (within-family), using raw layer indices on the x-axis.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import Experiment4Config


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

MODEL_DISPLAY = {
    "llama8b":   "Llama 3.1 8B",
    "llama70b":  "Llama 3.3 70B",
    "gemma4b":   "Gemma 4 4B",
    "gemma31b":  "Gemma 4 31B",
    "qwen7b":  "Qwen 7B",
    "qwen32b": "Qwen 32B",
}

MODEL_COLORS = {
    "llama8b":   "#A569BD",  # purple light
    "llama70b":  "#6C3483",  # purple
    "gemma4b":   "#7BAAF7",  # Google light
    "gemma31b":  "#4285F4",  # Google
    "gemma9b":   "#4285F4",  # Google (Gemma 2 9B, legacy)
    "qwen7b":    "#F0B27A",  # amber light
    "qwen32b":   "#E67E22",  # amber
}

# Primary cross-family condition used for the main scaling comparison
PRIMARY_CROSS = {
    "llama8b":   "cross_gemma9b",
    "llama70b":  "cross_gemma9b",
    "gemma4b":   "cross_llama8b",
    "gemma31b":  "cross_llama8b",
    "qwen7b":  "cross_llama8b",
    "qwen32b": "cross_llama8b",
}


def _layer_auroc_curve(
    model_results: dict,
    condition_name: str,
) -> tuple[list[int], list[float]]:
    """Extract (layer_indices, auroc_values) for one model / condition."""
    layer_results = model_results["cross_conditions"][condition_name]["layer_results"]
    layers = sorted(int(k) for k in layer_results)
    aurocs = [layer_results[str(l)]["test_auroc"] for l in layers]
    return layers, aurocs


def _best_layer_stats(
    model_results: dict,
    condition_name: str,
) -> dict:
    """Return peak-layer metrics for one model / condition."""
    cond = model_results["cross_conditions"][condition_name]
    n_layers = model_results["n_layers"]

    best_layer = cond["best_layer"]
    best_auroc = cond["best_auroc"]
    ppl_auroc = cond.get("perplexity_baseline_auroc") or float("nan")

    rel_peak = best_layer / (n_layers - 1) if best_layer is not None else float("nan")
    remaining = 1.0 - rel_peak

    return {
        "n_layers": n_layers,
        "best_layer": best_layer,
        "relative_peak": rel_peak,
        "remaining_depth": remaining,
        "best_auroc": best_auroc,
        "ppl_auroc": ppl_auroc,
        "gap": (best_auroc - ppl_auroc)
        if (best_auroc is not None and not np.isnan(ppl_auroc))
        else float("nan"),
    }


# ---------------------------------------------------------------------------
# Figure 1: normalised layer-wise AUROC across models
# ---------------------------------------------------------------------------

def _figure_normalized_auroc(
    all_results: dict[str, dict],
    config: Experiment4Config,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))

    for model_name in ("llama8b", "llama70b", "gemma4b", "gemma31b", "qwen7b", "qwen32b"):
        if model_name not in all_results:
            continue
        model_r = all_results[model_name]
        cond_name = PRIMARY_CROSS[model_name]
        if cond_name not in model_r.get("cross_conditions", {}):
            continue

        n_layers = model_r["n_layers"]
        layers, aurocs = _layer_auroc_curve(model_r, cond_name)
        rel_layers = [l / (n_layers - 1) for l in layers]

        color = MODEL_COLORS[model_name]
        label = f"{MODEL_DISPLAY[model_name]} ({n_layers} layers)"
        ax.plot(rel_layers, aurocs, color=color, linewidth=1.8, label=label)

        ppl_auroc = model_r["cross_conditions"][cond_name].get("perplexity_baseline_auroc")
        if ppl_auroc is not None and not np.isnan(ppl_auroc):
            ax.axhline(
                ppl_auroc,
                color=color,
                linestyle="--",
                linewidth=1.0,
                alpha=0.7,
                label=f"{MODEL_DISPLAY[model_name]} perplexity baseline ({ppl_auroc:.3f})",
            )

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.0, label="Chance (0.5)")
    ax.set_xlabel("Relative layer depth (layer / n_layers)", fontsize=12)
    ax.set_ylabel("AUROC (test set)", fontsize=12)
    ax.set_title(
        "Prefill detection: layer-wise AUROC vs. relative depth across model scales",
        fontsize=13,
    )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.35, 1.05)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = config.results_dir_ex4 / "fig1_normalized_auroc.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


# ---------------------------------------------------------------------------
# Figure 2: peak layer analysis
# ---------------------------------------------------------------------------

def _figure_peak_layer_analysis(
    all_results: dict[str, dict],
    config: Experiment4Config,
) -> None:
    model_order = ["llama8b", "llama70b", "gemma4b", "gemma31b", "qwen7b", "qwen32b"]
    stats = {
        m: _best_layer_stats(all_results[m], PRIMARY_CROSS[m])
        for m in model_order
        if m in all_results and PRIMARY_CROSS[m] in all_results[m].get("cross_conditions", {})
    }

    labels = [MODEL_DISPLAY[m] for m in model_order if m in stats]
    rel_peaks = [stats[m]["relative_peak"] for m in model_order if m in stats]
    remaining = [stats[m]["remaining_depth"] for m in model_order if m in stats]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width / 2, rel_peaks, width, label="Relative peak position",
                   color="steelblue", alpha=0.85)
    bars2 = ax.bar(x + width / 2, remaining, width, label="Remaining depth fraction",
                   color="tomato", alpha=0.85)

    def _label_bars(bars, values):
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.2f}",
                    ha="center", va="bottom", fontsize=9,
                )

    _label_bars(bars1, rel_peaks)
    _label_bars(bars2, remaining)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Fraction of total depth", fontsize=12)
    ax.set_title(
        "Peak layer position and remaining computational headroom by model",
        fontsize=13,
    )
    ax.set_ylim(0, 1.15)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.5)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # Annotate absolute layer numbers below x-axis
    present = [m for m in model_order if m in stats]
    for i, m in enumerate(present):
        n_layers = stats[m]["n_layers"]
        best_layer = stats[m]["best_layer"]
        ax.text(
            x[i], -0.08,
            f"peak={best_layer}/{n_layers - 1}",
            ha="center", va="top", fontsize=8, color="dimgray",
            transform=ax.get_xaxis_transform(),
        )

    plt.tight_layout()
    out = config.results_dir_ex4 / "fig2_peak_layer_analysis.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


# ---------------------------------------------------------------------------
# Figure 3: probe vs. perplexity gap
# ---------------------------------------------------------------------------

def _figure_probe_vs_perplexity_gap(
    all_results: dict[str, dict],
    config: Experiment4Config,
) -> None:
    model_order = ["llama8b", "llama70b", "gemma4b", "gemma31b", "qwen7b", "qwen32b"]
    stats = {
        m: _best_layer_stats(all_results[m], PRIMARY_CROSS[m])
        for m in model_order
        if m in all_results and PRIMARY_CROSS[m] in all_results[m].get("cross_conditions", {})
    }

    labels = [MODEL_DISPLAY[m] for m in model_order if m in stats]
    gaps = [stats[m]["gap"] for m in model_order if m in stats]
    colors = [
        "steelblue" if g >= 0 else "tomato"
        for g in gaps
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, gaps, color=colors, alpha=0.85, width=0.5)

    for bar, val in zip(bars, gaps):
        if not np.isnan(val):
            ypos = bar.get_height() + 0.002 if val >= 0 else bar.get_height() - 0.004
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                ypos,
                f"{val:+.3f}",
                ha="center", va="bottom" if val >= 0 else "top",
                fontsize=10,
            )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Probe AUROC − Perplexity baseline AUROC", fontsize=11)
    ax.set_title(
        "Probe advantage over perplexity baseline by model scale",
        fontsize=13,
    )
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = config.results_dir_ex4 / "fig3_probe_vs_perplexity_gap.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


# ---------------------------------------------------------------------------
# Figure 4: within-family vs. cross-family (Llama 70B only)
# ---------------------------------------------------------------------------

def _figure_within_vs_cross_family(
    all_results: dict[str, dict],
    config: Experiment4Config,
) -> None:
    llama70b = all_results.get("llama70b", {})
    cross_conds = llama70b.get("cross_conditions", {})

    if "cross_gemma9b" not in cross_conds or "cross_llama8b" not in cross_conds:
        print("  Skipping Figure 4: Llama 70B cross-condition data incomplete.")
        return

    n_layers = llama70b["n_layers"]

    layers_cf, aurocs_cf = _layer_auroc_curve(llama70b, "cross_gemma9b")
    layers_sf, aurocs_sf = _layer_auroc_curve(llama70b, "cross_llama8b")

    ppl_cf = cross_conds["cross_gemma9b"].get("perplexity_baseline_auroc")
    ppl_sf = cross_conds["cross_llama8b"].get("perplexity_baseline_auroc")

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(layers_cf, aurocs_cf, color=MODEL_COLORS["gemma9b"], linewidth=1.8, marker="s",
            markersize=3, label="Cross-family: vs. Gemma 9B")
    ax.plot(layers_sf, aurocs_sf, color=MODEL_COLORS["llama8b"], linewidth=1.8, marker="o",
            markersize=3, label="Within-family: vs. Llama 8B")

    if ppl_cf is not None and not np.isnan(ppl_cf):
        ax.axhline(ppl_cf, color=MODEL_COLORS["gemma9b"], linestyle="--", linewidth=1.0,
                   alpha=0.7, label=f"Perp. baseline cross-family ({ppl_cf:.3f})")
    if ppl_sf is not None and not np.isnan(ppl_sf):
        ax.axhline(ppl_sf, color=MODEL_COLORS["llama8b"], linestyle="--", linewidth=1.0,
                   alpha=0.7, label=f"Perp. baseline within-family ({ppl_sf:.3f})")

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.0, label="Chance (0.5)")
    ax.set_xlabel("Layer index", fontsize=12)
    ax.set_ylabel("AUROC (test set)", fontsize=12)
    ax.set_title(
        "Llama 3.3 70B: within-family vs. cross-family prefill detection",
        fontsize=13,
    )
    ax.set_xlim(-1, n_layers)
    ax.set_ylim(0.35, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = config.results_dir_ex4 / "fig4_within_vs_cross_family.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(
    all_results: dict[str, dict],
    config: Experiment4Config,
) -> None:
    model_order = ["llama8b", "llama70b", "gemma4b", "gemma31b", "qwen7b", "qwen32b"]
    primary = PRIMARY_CROSS

    header = (
        f"{'Model':<16} {'Layers':>6} {'Best Layer':>10} "
        f"{'Rel. Peak':>10} {'Remaining':>10} "
        f"{'Probe AUROC':>12} {'Perp AUROC':>11} {'Gap':>8}"
    )
    sep = "-" * len(header)

    rows = []
    for m in model_order:
        if m not in all_results:
            continue
        cond = primary[m]
        if cond not in all_results[m].get("cross_conditions", {}):
            continue
        s = _best_layer_stats(all_results[m], cond)
        rows.append((
            MODEL_DISPLAY[m],
            s["n_layers"],
            s["best_layer"],
            s["relative_peak"],
            s["remaining_depth"],
            s["best_auroc"],
            s["ppl_auroc"],
            s["gap"],
        ))

    print("\n" + "=" * len(header))
    print("  EXPERIMENT 4 — SCALING ANALYSIS SUMMARY")
    print("=" * len(header))
    print(header)
    print(sep)
    for model_name, n_layers, best_layer, rel_peak, remaining, probe_auroc, ppl_auroc, gap in rows:
        def _fmt(v):
            return f"{v:.4f}" if isinstance(v, float) and not np.isnan(v) else "N/A"

        print(
            f"  {model_name:<14} {n_layers:>6} {best_layer:>10} "
            f"{_fmt(rel_peak):>10} {_fmt(remaining):>10} "
            f"{_fmt(probe_auroc):>12} {_fmt(ppl_auroc):>11} {_fmt(gap):>8}"
        )
    print("=" * len(header))

    # Interpretive checklist
    present_models = [m for m in model_order
                      if m in all_results
                      and PRIMARY_CROSS[m] in all_results[m].get("cross_conditions", {})]

    if len(present_models) >= 2:
        stats_present = {m: _best_layer_stats(all_results[m], PRIMARY_CROSS[m])
                         for m in present_models}
        rel_peaks = [stats_present[m]["relative_peak"] for m in present_models]
        gaps_vals = [stats_present[m]["gap"] for m in present_models]

        print("\nInterpretive Checklist:")
        print("  Relative peak positions: "
              + ", ".join(f"{MODEL_DISPLAY[m]}={v:.3f}"
                          for m, v in zip(present_models, rel_peaks)))

        if rel_peaks[-1] < rel_peaks[0]:
            print("  [x] EARLIER PEAK WITH SCALE — supports larger models having more headroom")
        elif abs(rel_peaks[-1] - rel_peaks[0]) < 0.05:
            print("  [ ] NO PEAK SHIFT — signal is a last-layer phenomenon at all scales")
        else:
            print("  [?] LATER PEAK — unexpected, warrants investigation")

        if (not np.isnan(gaps_vals[-1]) and not np.isnan(gaps_vals[0])
                and gaps_vals[-1] > gaps_vals[0]):
            print("  [x] GAP NARROWS/FLIPS WITH SCALE — larger models encode richer info")
        elif not any(np.isnan(g) for g in gaps_vals):
            print("  [ ] GAP STAYS NEGATIVE — signal dominated by surprisal at all scales")

    # Save CSV
    csv_path = config.results_dir_ex4 / "summary_ex4.csv"
    import csv
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Model", "Layers", "Best Layer", "Relative Peak", "Remaining Depth",
            "Probe AUROC", "Perplexity AUROC", "Gap"
        ])
        for row in rows:
            writer.writerow(row)
    print(f"\n  Summary saved → {csv_path}")


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def load_probe_results_if_present(config: Experiment4Config) -> dict[str, dict]:
    """
    Load all three probe result JSONs if they exist.
    Returns a (possibly partial) dict keyed by model name.
    """
    paths = {
        "llama8b":   config.ex1_results_dir / "probe_results_llama8b.json",
        "llama70b":  config.results_dir_ex4  / "probe_results_llama70b.json",
        "gemma31b":  config.results_dir_ex4  / "probe_results_gemma31b.json",
        "gemma4b":   config.results_dir_ex4  / "probe_results_gemma4b.json",
        "qwen7b":  config.results_dir_ex4  / "probe_results_qwen7b.json",
        "qwen32b": config.results_dir_ex4  / "probe_results_qwen32b.json",
    }
    results = {}
    for name, path in paths.items():
        if path.exists():
            with open(path) as f:
                results[name] = json.load(f)
            print(f"  Loaded {name} probe results from {path}")
        else:
            print(f"  WARNING: probe results for {name} not found at {path}")
    return results


def generate_all_figures(
    all_results: dict[str, dict],
    config: Experiment4Config,
) -> None:
    """Produce all four scaling-analysis figures and the summary table."""
    print("\n--- Generating figures ---")
    _figure_normalized_auroc(all_results, config)
    _figure_peak_layer_analysis(all_results, config)
    _figure_probe_vs_perplexity_gap(all_results, config)
    _figure_within_vs_cross_family(all_results, config)
    print_summary_table(all_results, config)
