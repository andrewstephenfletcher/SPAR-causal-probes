"""
Analysis and figure generation for Experiment 6 (Steering Control Analysis).

Figure 1 — fig1_random_histogram.png
    Histogram of 10 random vectors' "not me" rates (Analysis A).
    Red vertical line = probe direction rate. z-score annotated.

Figure 2 — fig2_alpha_sweep.png
    Dose-response: "not me" rate vs signed alpha fraction for probe (red)
    and random vector (grey) at layer 24 (Analysis B).

Figure 3 — fig3_layer_sweep.png
    Two stacked subplots:
      Top:    "not me" rate by layer for probe, random, and baseline (Analysis C).
      Bottom: Specificity = probe_effect − random_effect by layer.

Figure 4 — fig4_magnitude_direction.png
    Bar chart: "not me" rate for probe α=1.0 vs random at escalating magnitudes
    (Analysis D). Bars annotated with the absolute perturbation norm (= alpha).

Figure 5 — fig5_summary.png
    Summary bar chart comparing four key numbers across analyses.

Summary table written to summary_ex6.csv.
Interpretive checklist printed to stdout and summary_ex6_checklist.txt.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .config import Experiment6Config


# ---------------------------------------------------------------------------
# Binomial CI helper
# ---------------------------------------------------------------------------

def _binom_ci(p: float, n: int, z: float = 1.96) -> float:
    """Half-width Wald 95% CI for a proportion."""
    return z * np.sqrt(p * (1 - p) / n) if n > 0 else 0.0


# ---------------------------------------------------------------------------
# Per-analysis aggregation
# ---------------------------------------------------------------------------

def compute_analysis_a(results: list[dict]) -> dict:
    """
    Returns probe rate, per-random-vector rates, and z-score
    (how many std-devs the probe rate is above the random distribution).
    """
    df = pd.DataFrame(results)
    df["not_me"] = df["parsed"] == "not_me"

    baseline_rate = df[df["label"] == "baseline"]["not_me"].mean()
    probe_rate    = df[df["label"] == "probe"]["not_me"].mean()

    random_rates: list[float] = []
    i = 0
    while True:
        sub = df[df["label"] == f"random_{i}"]
        if len(sub) == 0:
            break
        random_rates.append(float(sub["not_me"].mean()))
        i += 1

    random_mean = float(np.mean(random_rates)) if random_rates else float("nan")
    random_std  = float(np.std(random_rates, ddof=1)) if len(random_rates) > 1 else float("nan")
    z_score     = (
        (probe_rate - random_mean) / random_std
        if random_std and not np.isnan(random_std) and random_std > 0
        else float("nan")
    )

    return {
        "baseline_not_me_rate": float(baseline_rate),
        "probe_not_me_rate":    float(probe_rate),
        "random_not_me_rates":  random_rates,
        "random_mean":          random_mean,
        "random_std":           random_std,
        "probe_z_score":        float(z_score),
        "n_random":             len(random_rates),
    }


def compute_analysis_b(results: list[dict]) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      vector_type (probe/random), alpha_fraction, not_me_rate, n
    Includes the baseline row (alpha_fraction=0, vector_type=baseline).
    """
    df = pd.DataFrame(results)
    df["not_me"] = df["parsed"] == "not_me"

    rows = []
    for label, grp in df.groupby("label"):
        if label == "baseline":
            vec_type = "baseline"
            frac = 0.0
        elif label.startswith("probe"):
            vec_type = "probe"
            frac = float(grp["alpha_fraction"].iloc[0])
        elif label.startswith("random"):
            vec_type = "random"
            frac = float(grp["alpha_fraction"].iloc[0])
        else:
            continue

        rows.append({
            "vector_type":    vec_type,
            "alpha_fraction": frac,
            "not_me_rate":    float(grp["not_me"].mean()),
            "n":              len(grp),
        })

    return pd.DataFrame(rows).sort_values(["vector_type", "alpha_fraction"]).reset_index(drop=True)


def compute_analysis_c(results: list[dict], steering_layers: list[int]) -> pd.DataFrame:
    """
    Returns a DataFrame with per-layer probe/random rates and specificity.
    Columns: layer, baseline_rate, probe_rate, random_rate,
             probe_effect, random_effect, specificity
    """
    df = pd.DataFrame(results)
    df["not_me"] = df["parsed"] == "not_me"

    baseline_rate = float(df[df["label"] == "baseline"]["not_me"].mean())

    rows = []
    for layer in steering_layers:
        probe_sub  = df[df["condition_id"] == f"probe_l{layer}"]
        random_sub = df[df["condition_id"] == f"random_l{layer}"]

        probe_rate  = float(probe_sub["not_me"].mean())  if len(probe_sub)  > 0 else float("nan")
        random_rate = float(random_sub["not_me"].mean()) if len(random_sub) > 0 else float("nan")

        probe_effect  = probe_rate  - baseline_rate
        random_effect = random_rate - baseline_rate
        specificity   = probe_effect - random_effect

        rows.append({
            "layer":          layer,
            "baseline_rate":  baseline_rate,
            "probe_rate":     probe_rate,
            "random_rate":    random_rate,
            "probe_effect":   probe_effect,
            "random_effect":  random_effect,
            "specificity":    specificity,
        })

    return pd.DataFrame(rows)


def compute_analysis_d(results: list[dict], alpha_fracs_random: list[float]) -> pd.DataFrame:
    """
    Returns a DataFrame with "not me" rate per condition for Analysis D.
    Columns: condition_label, alpha_fraction, not_me_rate, perturbation_norm, n
    The perturbation norm = alpha (since the steering vector is unit-norm).
    """
    df = pd.DataFrame(results)
    df["not_me"] = df["parsed"] == "not_me"

    rows = []
    for label, grp in df.groupby("label"):
        rows.append({
            "condition_label":  label,
            "alpha_fraction":   float(grp["alpha_fraction"].iloc[0]),
            "alpha":            float(grp["alpha"].iloc[0]),
            "not_me_rate":      float(grp["not_me"].mean()),
            "n":                len(grp),
        })

    return pd.DataFrame(rows).sort_values("alpha_fraction").reset_index(drop=True)


def find_best_layer(c_df: pd.DataFrame) -> int:
    """Return the steering layer with maximum probe specificity."""
    idx = c_df["specificity"].idxmax()
    return int(c_df.loc[idx, "layer"])


# ---------------------------------------------------------------------------
# Figure 1: Random histogram (Analysis A)
# ---------------------------------------------------------------------------

def figure1_random_histogram(a_stats: dict, config: Experiment6Config) -> None:
    random_rates  = a_stats["random_not_me_rates"]
    probe_rate    = a_stats["probe_not_me_rate"]
    baseline_rate = a_stats["baseline_not_me_rate"]
    z_score       = a_stats["probe_z_score"]
    n_random      = a_stats["n_random"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(
        random_rates, bins=max(5, n_random // 2),
        color="#95a5a6", edgecolor="black", alpha=0.85,
        label=f"Random vectors (n={n_random})",
    )
    ax.axvline(probe_rate, color="#e74c3c", linewidth=2.0, linestyle="-",
               label=f"Probe direction ({probe_rate:.2f})")
    ax.axvline(baseline_rate, color="black", linewidth=1.5, linestyle="--",
               label=f"Baseline ({baseline_rate:.2f})")

    z_text = f"z = {z_score:.2f}" if not np.isnan(z_score) else "z = n/a"
    ax.text(0.97, 0.95, z_text, transform=ax.transAxes,
            ha="right", va="top", fontsize=12, color="#e74c3c",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

    ax.set_xlabel('"Not me" rate', fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_xlim(0, 1)
    ax.legend(fontsize=9)
    ax.set_title(
        f"Analysis A: Probe vs. {n_random} random vectors at layer 24 (α={config.alpha_fractions[-1]}×)",
        fontsize=11,
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out = config.results_dir_ex6 / "fig1_random_histogram.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Figure 1 saved → {out}")


# ---------------------------------------------------------------------------
# Figure 2: Alpha sweep dose-response (Analysis B)
# ---------------------------------------------------------------------------

def figure2_alpha_sweep(b_df: pd.DataFrame, config: Experiment6Config) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    colors  = {"probe": "#e74c3c", "random": "#7f8c8d"}
    markers = {"probe": "o",       "random": "s"}

    baseline_rate = float(
        b_df[b_df["vector_type"] == "baseline"]["not_me_rate"].iloc[0]
        if len(b_df[b_df["vector_type"] == "baseline"]) > 0
        else 0.5
    )

    for vec_type in ("probe", "random"):
        sub = b_df[b_df["vector_type"] == vec_type].copy()
        # Include baseline as alpha_fraction=0
        bl_row = pd.DataFrame([{
            "vector_type": vec_type, "alpha_fraction": 0.0,
            "not_me_rate": baseline_rate, "n": 0,
        }])
        sub = pd.concat([bl_row, sub], ignore_index=True)
        sub = sub.sort_values("alpha_fraction")

        ax.plot(
            sub["alpha_fraction"], sub["not_me_rate"],
            marker=markers[vec_type], color=colors[vec_type],
            label=vec_type.capitalize(), linewidth=1.8, markersize=7,
        )

    ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axhline(baseline_rate, color="black", linestyle=":", linewidth=0.8,
               alpha=0.5, label=f"Baseline ({baseline_rate:.2f})")

    # Mark the alpha used in Experiment 5 (alpha_fraction ≈ 1.5)
    ax.axvline(config.alpha_fractions[-1], color="#e74c3c",
               linestyle=":", linewidth=0.8, alpha=0.6, label="Exp 5 alpha")

    ax.set_xlabel("Alpha fraction  (negative → self, positive → not-self)", fontsize=10)
    ax.set_ylabel('"Not me" rate', fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.set_title("Analysis B: Dose-response — probe vs. random (layer 24)", fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = config.results_dir_ex6 / "fig2_alpha_sweep.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Figure 2 saved → {out}")


# ---------------------------------------------------------------------------
# Figure 3: Layer sweep (Analysis C)
# ---------------------------------------------------------------------------

def figure3_layer_sweep(c_df: pd.DataFrame, config: Experiment6Config) -> None:
    layers = c_df["layer"].tolist()
    x      = np.arange(len(layers))

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    # Top: not_me_rate by layer
    ax_top.plot(x, c_df["probe_rate"],   marker="o", color="#e74c3c", label="Probe",   linewidth=1.8)
    ax_top.plot(x, c_df["random_rate"],  marker="s", color="#7f8c8d", label="Random",  linewidth=1.8)
    ax_top.axhline(c_df["baseline_rate"].iloc[0], color="black", linestyle="--",
                   linewidth=1.0, alpha=0.7, label=f"Baseline ({c_df['baseline_rate'].iloc[0]:.2f})")

    # Shade 60-70% depth range (layers 48-56 for 80-layer Llama)
    depth_60 = layers.index(48) if 48 in layers else None
    depth_70 = layers.index(56) if 56 in layers else None
    if depth_60 is not None and depth_70 is not None:
        ax_top.axvspan(depth_60 - 0.4, depth_70 + 0.4, alpha=0.08, color="#3498db",
                       label="60-70% depth (Macar et al.)")

    ax_top.set_ylabel('"Not me" rate', fontsize=11)
    ax_top.set_ylim(0, 1.05)
    ax_top.legend(fontsize=9)
    ax_top.set_title("Analysis C: Layer sweep — probe vs. random", fontsize=11)
    ax_top.grid(alpha=0.3)

    # Bottom: specificity
    colors_spec = ["#27ae60" if v > 0.05 else "#e74c3c" if v < -0.05 else "#95a5a6"
                   for v in c_df["specificity"]]
    ax_bot.bar(x, c_df["specificity"], color=colors_spec, edgecolor="black", linewidth=0.5)
    ax_bot.axhline(0, color="black", linewidth=0.8)
    ax_bot.axhline(0.15, color="#27ae60", linestyle="--", linewidth=0.8, alpha=0.7,
                   label="Specificity threshold (0.15)")
    ax_bot.set_ylabel("Specificity\n(probe − random effect)", fontsize=10)
    ax_bot.legend(fontsize=8)
    ax_bot.grid(axis="y", alpha=0.3)

    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels([str(l) for l in layers], fontsize=10)
    ax_bot.set_xlabel("Transformer layer", fontsize=11)

    fig.tight_layout()
    out = config.results_dir_ex6 / "fig3_layer_sweep.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Figure 3 saved → {out}")


# ---------------------------------------------------------------------------
# Figure 4: Magnitude vs. direction (Analysis D)
# ---------------------------------------------------------------------------

def figure4_magnitude_direction(
    d_df: pd.DataFrame,
    best_layer: int,
    config: Experiment6Config,
) -> None:
    # Build display order: baseline, probe, random_a1.0, random_a1.5, ...
    order_map = {
        "baseline": 0,
        "probe":    1,
    }
    for row in d_df.itertuples():
        if row.condition_label.startswith("random_a"):
            order_map[row.condition_label] = 2 + float(row.condition_label.split("_a")[1])

    d_df = d_df.copy()
    d_df["_sort"] = d_df["condition_label"].map(order_map)
    d_df = d_df.sort_values("_sort").drop(columns="_sort")

    labels     = d_df["condition_label"].tolist()
    rates      = d_df["not_me_rate"].tolist()
    alphas     = d_df["alpha"].tolist()
    ns         = d_df["n"].tolist()

    colors = []
    for lbl in labels:
        if lbl == "baseline":
            colors.append("#95a5a6")
        elif lbl == "probe":
            colors.append("#e74c3c")
        else:
            colors.append("#3498db")

    x_pos = np.arange(len(labels))
    errs  = [_binom_ci(r, n) for r, n in zip(rates, ns)]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(x_pos, rates, color=colors, edgecolor="black", linewidth=0.5,
                  yerr=errs, capsize=4, alpha=0.85)

    baseline_rate = float(d_df[d_df["condition_label"] == "baseline"]["not_me_rate"].iloc[0])
    ax.axhline(baseline_rate, color="black", linestyle="--", linewidth=1.0, alpha=0.7,
               label=f"Baseline ({baseline_rate:.2f})")

    # Annotate with perturbation norm (= alpha, since vector is unit-norm)
    for bar, alpha_val in zip(bars, alphas):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"|Δ|={alpha_val:.3f}",
            ha="center", va="bottom", fontsize=7, rotation=45,
        )

    display_labels = []
    for lbl, frac in zip(labels, d_df["alpha_fraction"].tolist()):
        if lbl == "baseline":
            display_labels.append("Baseline")
        elif lbl == "probe":
            display_labels.append(f"Probe\nα={frac:.1f}×")
        else:
            display_labels.append(f"Random\nα={frac:.1f}×")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(display_labels, fontsize=9)
    ax.set_ylabel('"Not me" rate', fontsize=11)
    ax.set_ylim(0, min(1.05, max(rates) + 0.15))
    ax.legend(fontsize=9)
    ax.set_title(
        f"Analysis D: Magnitude vs. direction (layer {best_layer})\n"
        "Annotations show L2 norm of perturbation per token",
        fontsize=11,
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out = config.results_dir_ex6 / "fig4_magnitude_direction.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Figure 4 saved → {out}")


# ---------------------------------------------------------------------------
# Figure 5: Summary comparison
# ---------------------------------------------------------------------------

def figure5_summary(
    a_stats: dict,
    c_df: pd.DataFrame,
    d_df: pd.DataFrame,
    best_layer: int,
    config: Experiment6Config,
    exp5_probe_effect: float | None = None,
) -> None:
    # Collect the four summary numbers
    mean_random_effect = a_stats["random_mean"] - a_stats["baseline_not_me_rate"]
    probe_effect_a = a_stats["probe_not_me_rate"] - a_stats["baseline_not_me_rate"]

    best_row = c_df[c_df["layer"] == best_layer]
    best_probe_effect = float(best_row["probe_effect"].iloc[0]) if len(best_row) else float("nan")

    # Find the random alpha that most closely matches the probe rate in D
    probe_row  = d_df[d_df["condition_label"] == "probe"]
    probe_rate_d = float(probe_row["not_me_rate"].iloc[0]) if len(probe_row) > 0 else float("nan")
    random_d   = d_df[d_df["condition_label"].str.startswith("random_a")].copy()
    if not random_d.empty and not np.isnan(probe_rate_d):
        random_d["diff"] = (random_d["not_me_rate"] - probe_rate_d).abs()
        match_row = random_d.loc[random_d["diff"].idxmin()]
        match_frac = float(match_row["alpha_fraction"])
        match_label = f"Random α={match_frac:.1f}× matches probe"
    else:
        match_frac  = float("nan")
        match_label = "Random match: n/a"

    bar_labels = []
    bar_values = []
    bar_colors = []

    if exp5_probe_effect is not None:
        bar_labels.append("Exp 5\nprobe effect\n(layer 24)")
        bar_values.append(exp5_probe_effect)
        bar_colors.append("#e74c3c")

    bar_labels += [
        f"Analysis A\nprobe effect\n(layer 24, α={config.alpha_fractions[-1]}×)",
        f"Analysis A\nmean random effect\n(layer 24, α={config.alpha_fractions[-1]}×)",
        f"Analysis C\nbest probe effect\n(layer {best_layer})",
    ]
    bar_values += [probe_effect_a, mean_random_effect, best_probe_effect]
    bar_colors += ["#e74c3c", "#7f8c8d", "#2ecc71"]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(bar_labels))
    ax.bar(x, bar_values, color=bar_colors, edgecolor="black", linewidth=0.5, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, fontsize=8)
    ax.set_ylabel('"Not me" rate effect (vs. baseline)', fontsize=11)
    ax.set_title("Summary: Steering effect comparison across analyses", fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    ax.text(0.98, 0.02, match_label, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color="#3498db")

    fig.tight_layout()
    out = config.results_dir_ex6 / "fig5_summary.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Figure 5 saved → {out}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def write_summary_table(
    a_stats: dict,
    b_df: pd.DataFrame,
    c_df: pd.DataFrame,
    d_df: pd.DataFrame,
    best_layer: int,
    config: Experiment6Config,
) -> None:
    # Per-analysis CSVs
    b_df.to_csv(config.results_dir_ex6 / "analysis_b_rates.csv", index=False)
    c_df.to_csv(config.results_dir_ex6 / "analysis_c_rates.csv", index=False)
    d_df.to_csv(config.results_dir_ex6 / "analysis_d_rates.csv", index=False)

    # Top-level JSON summary
    probe_rate_d_row   = d_df[d_df["condition_label"] == "probe"]
    probe_rate_d       = float(probe_rate_d_row["not_me_rate"].iloc[0]) if len(probe_rate_d_row) > 0 else float("nan")
    random_d_sub       = d_df[d_df["condition_label"].str.startswith("random_a")].copy()

    match_frac = float("nan")
    if not random_d_sub.empty and not np.isnan(probe_rate_d):
        random_d_sub["diff"] = (random_d_sub["not_me_rate"] - probe_rate_d).abs()
        match_frac = float(random_d_sub.loc[random_d_sub["diff"].idxmin(), "alpha_fraction"])

    b_probe  = b_df[b_df["vector_type"] == "probe"].sort_values("alpha_fraction")
    b_random = b_df[b_df["vector_type"] == "random"].sort_values("alpha_fraction")
    baseline_b = float(
        b_df[b_df["vector_type"] == "baseline"]["not_me_rate"].iloc[0]
        if len(b_df[b_df["vector_type"] == "baseline"]) > 0
        else float("nan")
    )
    probe_threshold  = _first_nonzero_alpha(b_probe,  baseline_b)
    random_threshold = _first_nonzero_alpha(b_random, baseline_b)

    best_row   = c_df[c_df["layer"] == best_layer]
    best_spec  = float(best_row["specificity"].iloc[0]) if len(best_row) > 0 else float("nan")

    summary = {
        "analysis_a": {
            "probe_not_me_rate":   a_stats["probe_not_me_rate"],
            "random_mean":         a_stats["random_mean"],
            "random_std":          a_stats["random_std"],
            "probe_z_score":       a_stats["probe_z_score"],
            "baseline_not_me_rate": a_stats["baseline_not_me_rate"],
        },
        "analysis_b": {
            "lowest_alpha_probe_effect":  probe_threshold,
            "lowest_alpha_random_effect": random_threshold,
        },
        "analysis_c": {
            "best_layer":     best_layer,
            "best_specificity": best_spec,
            "per_layer": c_df.to_dict(orient="records"),
        },
        "analysis_d": {
            "probe_rate_at_alpha_1.0":        probe_rate_d,
            "random_alpha_matching_probe":    match_frac,
        },
    }

    out = config.results_dir_ex6 / "summary_ex6.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary JSON saved → {out}")


def _first_nonzero_alpha(df: pd.DataFrame, baseline: float, threshold: float = 0.05) -> float:
    """Return the smallest positive alpha_fraction where effect > threshold, or nan."""
    positive = df[df["alpha_fraction"] > 0].sort_values("alpha_fraction")
    for _, row in positive.iterrows():
        if (row["not_me_rate"] - baseline) > threshold:
            return float(row["alpha_fraction"])
    return float("nan")


# ---------------------------------------------------------------------------
# Interpretive checklist
# ---------------------------------------------------------------------------

def print_interpretive_checklist(
    a_stats: dict,
    b_df: pd.DataFrame,
    c_df: pd.DataFrame,
    d_df: pd.DataFrame,
    best_layer: int,
    config: Experiment6Config,
) -> str:
    z       = a_stats["probe_z_score"]
    r_mean  = a_stats["random_mean"]
    r_std   = a_stats["random_std"]
    p_rate  = a_stats["probe_not_me_rate"]

    best_row = c_df[c_df["layer"] == best_layer]
    best_spec = float(best_row["specificity"].iloc[0]) if len(best_row) > 0 else float("nan")

    depth_60_70_layers = [l for l in config.steering_layers if 48 <= l <= 56]
    spec_at_depth = c_df[c_df["layer"].isin(depth_60_70_layers)]["specificity"].max()

    probe_row_d  = d_df[d_df["condition_label"] == "probe"]
    probe_rate_d = float(probe_row_d["not_me_rate"].iloc[0]) if len(probe_row_d) > 0 else float("nan")
    random_d_sub = d_df[d_df["condition_label"].str.startswith("random_a")].copy()
    match_frac   = float("nan")
    if not random_d_sub.empty and not np.isnan(probe_rate_d):
        random_d_sub["diff"] = (random_d_sub["not_me_rate"] - probe_rate_d).abs()
        match_frac = float(random_d_sub.loc[random_d_sub["diff"].idxmin(), "alpha_fraction"])

    baseline_b = float(
        b_df[b_df["vector_type"] == "baseline"]["not_me_rate"].iloc[0]
        if len(b_df[b_df["vector_type"] == "baseline"]) > 0
        else float("nan")
    )
    probe_thresh  = _first_nonzero_alpha(b_df[b_df["vector_type"] == "probe"],  baseline_b)
    random_thresh = _first_nonzero_alpha(b_df[b_df["vector_type"] == "random"], baseline_b)

    lines = [
        "=" * 60,
        "EXPERIMENT 6 — STEERING CONTROL ANALYSIS",
        "=" * 60,
        "",
        "--- Analysis A: Multiple random vectors ---",
        f"  Probe 'not me' rate:      {p_rate:.3f}",
        f"  Random mean ± std:        {r_mean:.3f} ± {r_std:.3f}",
        f"  Probe z-score:            {z:.2f}",
        "",
        f"  [{'X' if not np.isnan(z) and z > 2.0 else ' '}] DIRECTIONALLY SPECIFIC (z > 2.0)",
        "      → Probe direction IS specifically causal.",
        f"  [{'X' if not np.isnan(z) and z < 1.0 else ' '}] NOT SPECIFIC (z < 1.0)",
        "      → Probe effect within normal random range.",
        f"  [{'X' if not np.isnan(z) and 1.0 <= z <= 2.0 else ' '}] MARGINAL (1.0 ≤ z ≤ 2.0)",
        "      → Weak evidence of specificity.",
        "",
        "--- Analysis B: Alpha sweep ---",
        f"  Lowest alpha with probe effect (>5pp):  {probe_thresh:.2f}×",
        f"  Lowest alpha with random effect (>5pp): {random_thresh:.2f}×",
        "",
        f"  [{'X' if not np.isnan(probe_thresh) and (np.isnan(random_thresh) or probe_thresh < random_thresh) else ' '}] PROBE-SPECIFIC AT LOW ALPHA",
        "      → Probe shows dose-response below random threshold.",
        f"  [{'X' if not np.isnan(probe_thresh) and not np.isnan(random_thresh) and abs(probe_thresh - random_thresh) < 0.15 else ' '}] BOTH RESPOND SIMILARLY",
        "      → No directional specificity at any magnitude.",
        "",
        "--- Analysis C: Layer sweep ---",
        f"  Best specificity at layer:  {best_layer}",
        f"  Specificity value:          {best_spec:.3f}",
        f"  Max specificity at 60-70%:  {spec_at_depth:.3f}",
        "",
        f"  [{'X' if not np.isnan(best_spec) and best_spec > 0.15 else ' '}] LAYER-SPECIFIC CAUSALITY (specificity > 0.15)",
        "      → Probe direction is causal at the right layer.",
        f"  [{'X' if not np.isnan(spec_at_depth) and spec_at_depth > 0.10 else ' '}] SPECIFICITY PEAKS AT 60-70% DEPTH",
        "      → Consistent with Macar et al. detection circuit.",
        f"  [{'X' if np.isnan(best_spec) or best_spec <= 0.05 else ' '}] NO LAYER SHOWS SPECIFICITY",
        "      → General fragility across all layers.",
        "",
        "--- Analysis D: Magnitude vs. direction ---",
        f"  Probe 'not me' rate (α=1.0):  {probe_rate_d:.3f}",
        f"  Random α to match probe:       {match_frac:.2f}×",
        "",
        f"  [{'X' if not np.isnan(match_frac) and match_frac <= 1.1 else ' '}] PURE ANOMALY DETECTOR (random at 1.0× matches probe)",
        "      → Consistent with Macar et al. evidence carrier mechanism.",
        f"  [{'X' if not np.isnan(match_frac) and 1.1 < match_frac < 2.5 else ' '}] PROBE MORE EFFICIENT (random needs 1.5-2× magnitude)",
        "      → Modest directional specificity; fundamentally magnitude-based.",
        f"  [{'X' if np.isnan(match_frac) or match_frac >= 2.5 else ' '}] PROBE UNIQUELY PRIVILEGED (random cannot match even at 2.5×)",
        "      → Strong directional specificity.",
        "",
        "--- Overall conclusion ---",
        f"  [{'X' if not np.isnan(z) and z > 2.0 or (not np.isnan(best_spec) and best_spec > 0.15) else ' '}] (A) PROBE IS SPECIFICALLY CAUSAL",
        "      → At least one analysis shows clear directional specificity.",
        f"  [{'X' if (np.isnan(z) or z < 1.0) and (np.isnan(best_spec) or best_spec <= 0.05) else ' '}] (B) GENERAL FRAGILITY / ANOMALY DETECTION",
        "      → Substrate exists but not directionally connected to behaviour.",
        f"  [{'X' if not np.isnan(z) and 1.0 <= z <= 2.0 else ' '}] (C) INCONCLUSIVE",
        "      → Mixed evidence. Report all analyses transparently.",
        "=" * 60,
    ]

    checklist = "\n".join(lines)
    print(checklist)

    out = config.results_dir_ex6 / "summary_ex6_checklist.txt"
    out.write_text(checklist)
    print(f"  Checklist saved → {out}")

    return checklist


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def generate_all_figures(
    results_a: list[dict],
    results_b: list[dict],
    results_c: list[dict],
    results_d: list[dict],
    config: Experiment6Config,
    best_layer: int | None = None,
    exp5_probe_effect: float | None = None,
) -> None:
    """Compute all statistics, generate all figures, write summary table and checklist."""

    print("\n  Computing Analysis A statistics...")
    a_stats = compute_analysis_a(results_a)
    print(f"    Probe rate={a_stats['probe_not_me_rate']:.3f}  "
          f"random mean={a_stats['random_mean']:.3f}  z={a_stats['probe_z_score']:.2f}")

    print("\n  Computing Analysis B statistics...")
    b_df = compute_analysis_b(results_b)

    print("\n  Computing Analysis C statistics...")
    c_df = compute_analysis_c(results_c, config.steering_layers)

    if best_layer is None:
        best_layer = find_best_layer(c_df)
    print(f"    Best layer (max specificity): {best_layer}")

    print("\n  Computing Analysis D statistics...")
    d_df = compute_analysis_d(results_d, config.alpha_fractions_d)

    print("\n  Generating figures...")
    figure1_random_histogram(a_stats, config)
    figure2_alpha_sweep(b_df, config)
    figure3_layer_sweep(c_df, config)
    figure4_magnitude_direction(d_df, best_layer, config)
    figure5_summary(a_stats, c_df, d_df, best_layer, config, exp5_probe_effect)

    print("\n  Writing summary table...")
    write_summary_table(a_stats, b_df, c_df, d_df, best_layer, config)

    print("\n  Interpretive checklist:")
    print_interpretive_checklist(a_stats, b_df, c_df, d_df, best_layer, config)
