"""
Figure generation for Experiment 9 (Probe Transfer and Representational Geometry).

Figure 1 — cosine_heatmap.png
    Pairwise cosine similarity heatmap across all named directions
    (prefill, truth_cities, truth_larger_than, eval/deploy, per-model directions)
    plus a random-baseline row for visual reference.

Figure 2 — pca_per_model.png
    2D PCA scatter of the four per-model probe directions, with the shared
    PC1 and PC2 axes annotated.

Figure 3 — cross_application_auroc.png
    Grouped bar chart of cross-application AUROCs and in-domain baselines.

Figure 4 — pc1_variance.png
    Bar chart of the fraction of each per-model direction's variance explained
    by the shared PC1.

Summary table written to results_dir_ex9 / "summary_ex9.json" and
"summary_ex9.txt".
"""

from __future__ import annotations

import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import Experiment9Config
from .probe_ex9 import ProbeBundle, cosine_sim, person_vector_analysis


# ---------------------------------------------------------------------------
# Figure 1: Cosine similarity heatmap
# ---------------------------------------------------------------------------

def figure1_cosine_heatmap(results: dict, config: Experiment9Config) -> None:
    cm = results.get("cosine_matrix", {})
    labels    = cm.get("labels", [])
    values    = cm.get("values", [])
    rand_row  = cm.get("random_mean_row", [])

    if not labels or not values:
        print("  Figure 1: skipped (no cosine matrix data).")
        return

    matrix = np.array(values, dtype=float)

    # Append random-baseline row for visual reference
    display_matrix = np.vstack([matrix, rand_row]) if rand_row else matrix
    display_labels = labels + ["random\nbaseline"] if rand_row else labels

    n_rows = display_matrix.shape[0]
    n_cols = display_matrix.shape[1]

    # Pretty labels
    def _pretty(lbl: str) -> str:
        mapping = {
            "prefill":            "Prefill (Ex1)",
            "truth_cities":       "Truth/Cities",
            "truth_larger_than":  "Truth/Larger-Than",
            "eval_deploy":        "Eval↔Deploy",
            "permodel_gemma":     "Gemma 9B",
            "permodel_mistral":   "Qwen 7B",
            "permodel_altered_self":    "Altered-Self",
            "permodel_style_imitated":  "Style-Imitated",
            "random\nbaseline":   "Random\nbaseline",
        }
        return mapping.get(lbl, lbl)

    x_labels = [_pretty(l) for l in labels]
    y_labels = [_pretty(l) for l in display_labels]

    fig_w = max(8, n_cols * 1.2)
    fig_h = max(6, n_rows * 1.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # The full matrix rows include the random row (no column for random)
    im = ax.imshow(
        display_matrix[:, :n_cols], vmin=-0.3, vmax=0.3,
        cmap="RdBu_r", aspect="auto",
    )
    plt.colorbar(im, ax=ax, label="Cosine similarity", shrink=0.8)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(x_labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(y_labels, fontsize=9)

    # Draw separator before random-baseline row
    if rand_row:
        ax.axhline(n_rows - 1.5, color="black", linewidth=1.5, linestyle="--", alpha=0.5)

    for r in range(n_rows):
        for c in range(n_cols):
            v = display_matrix[r, c]
            if np.isfinite(v):
                color = "white" if abs(v) > 0.18 else "black"
                ax.text(c, r, f"{v:+.3f}", ha="center", va="center",
                        fontsize=7.5, color=color)

    ax.set_title(
        "Experiment 9: Pairwise cosine similarities between probe directions\n"
        "(|cos| > 0.05 notable; > 0.10 significant in R⁴⁰⁹⁶)",
        fontsize=11,
    )
    fig.tight_layout()

    out = config.results_dir_ex9 / "fig1_cosine_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Figure 1 saved → {out}")


# ---------------------------------------------------------------------------
# Figure 2: PCA of per-model probe directions
# ---------------------------------------------------------------------------

def figure2_pca_per_model(results: dict, config: Experiment9Config) -> None:
    pva = results.get("person_vector_analysis", {})
    if not pva or not pva.get("labels"):
        print("  Figure 2: skipped (no person-vector analysis data).")
        return

    labels    = pva["labels"]
    pairwise  = np.array(pva["pairwise_cosines"])
    sv        = pva.get("singular_values", [])
    pc1_frac  = pva.get("pc1_variance_frac", {})

    # Build the matrix of directions (we can't recover the originals here, so
    # use the cosine matrix to project into 2D via classical MDS / eigen-decomp)
    # Alternatively, load saved direction numpy files.
    per_model_dir_files = [
        config.results_dir_ex9 / f"permodel_direction_{cond}.npy"
        for cond in labels
    ]

    if not all(f.exists() for f in per_model_dir_files):
        print("  Figure 2: skipped (per-model direction .npy files not found).")
        return

    dirs = np.stack([np.load(f).astype(np.float32) for f in per_model_dir_files])

    # Project into 2D using SVD
    U, S, Vt = np.linalg.svd(dirs, full_matrices=False)
    coords_2d = U[:, :2] * S[:2]  # (n_models, 2)

    total_var = float(np.sum(S ** 2))
    pct1 = 100.0 * S[0] ** 2 / total_var if total_var > 0 else 0.0
    pct2 = 100.0 * S[1] ** 2 / total_var if total_var > 0 else 0.0

    cond_labels_pretty = {
        "altered_self":   "Altered-Self",
        "gemma":          "Gemma 9B",
        "mistral":        "Qwen 7B",
        "style_imitated": "Style-Imitated",
    }

    colors = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(7, 6))
    for i, cond in enumerate(labels):
        x, y = coords_2d[i]
        ax.scatter(x, y, color=colors(i), s=120, zorder=5, edgecolors="black", linewidths=0.8)
        ax.annotate(cond_labels_pretty.get(cond, cond), (x, y),
                    textcoords="offset points", xytext=(8, 5), fontsize=10)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)

    ax.set_xlabel(f"PC1 ({pct1:.1f}% var)", fontsize=11)
    ax.set_ylabel(f"PC2 ({pct2:.1f}% var)", fontsize=11)
    ax.set_title(
        "PCA of per-model probe directions\n"
        "(PC1 = shared 'not-self'; spread along PC2 = model-specific)",
        fontsize=11,
    )
    ax.grid(True, alpha=0.3)

    if sv:
        sv_str = " | ".join([f"σ{i+1}={s:.2f}" for i, s in enumerate(sv[:4])])
        ax.text(0.02, 0.02, sv_str, transform=ax.transAxes,
                fontsize=8, va="bottom", color="gray")

    fig.tight_layout()
    out = config.results_dir_ex9 / "fig2_pca_per_model.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Figure 2 saved → {out}")


# ---------------------------------------------------------------------------
# Figure 3: Cross-application AUROC bars
# ---------------------------------------------------------------------------

def figure3_cross_application_auroc(results: dict, config: Experiment9Config) -> None:
    ca    = results.get("cross_application_auroc", {})
    truth_aurocs = results.get("truth_probe_aurocs", {})

    if not ca:
        print("  Figure 3: skipped (no cross-application AUROC data).")
        return

    # Build bar groups:
    #   In-domain baselines: prefill probe on prefill data, truth probes on truth data
    #   Cross-application:   prefill on truth, truth on prefill, eval/deploy on prefill
    prefill_indomain = results.get("prefill_probe_auroc", float("nan"))

    bar_data = []

    # Prefill in-domain (from analysis)
    bar_data.append({
        "label": "Prefill probe\n(in-domain)",
        "auroc": prefill_indomain,
        "color": "#2ecc71",
        "group": "In-domain",
    })

    # Truth in-domain
    for ds in config.got_datasets:
        layer = config.primary_layer
        auroc = float("nan")
        if ds in truth_aurocs and str(layer) in truth_aurocs[ds]:
            auroc = truth_aurocs[ds][str(layer)]
        bar_data.append({
            "label": f"Truth_{ds} probe\n(in-domain)",
            "auroc": auroc,
            "color": "#3498db",
            "group": "In-domain",
        })

    # Cross-application
    for ds in config.got_datasets:
        auroc = ca.get(f"prefill_on_{ds}", float("nan"))
        bar_data.append({
            "label": f"Prefill probe\non {ds}",
            "auroc": auroc,
            "color": "#e74c3c",
            "group": "Cross",
        })
        auroc2 = ca.get(f"{ds}_on_prefill", float("nan"))
        bar_data.append({
            "label": f"Truth_{ds} probe\non prefill",
            "auroc": auroc2,
            "color": "#e67e22",
            "group": "Cross",
        })

    auroc_ed = ca.get("eval_deploy_on_prefill", float("nan"))
    bar_data.append({
        "label": "Eval↔Deploy\ndir on prefill",
        "auroc": auroc_ed,
        "color": "#9b59b6",
        "group": "Cross",
    })

    valid = [d for d in bar_data if not math.isnan(d["auroc"])]
    if not valid:
        print("  Figure 3: skipped (all AUROCs are NaN).")
        return

    x = np.arange(len(valid))
    labels  = [d["label"]  for d in valid]
    aurocs  = [d["auroc"]  for d in valid]
    colors  = [d["color"]  for d in valid]

    fig, ax = plt.subplots(figsize=(max(9, len(valid) * 1.4), 5))
    bars = ax.bar(x, aurocs, color=colors, edgecolor="black", linewidth=0.5, alpha=0.85)

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.0, label="Chance (0.5)")

    # Add separator between in-domain and cross-application groups
    n_indomain = sum(1 for d in valid if d["group"] == "In-domain")
    if 0 < n_indomain < len(valid):
        ax.axvline(n_indomain - 0.5, color="black", linestyle="--",
                   linewidth=1.0, alpha=0.5)
        ax.text(n_indomain / 2 - 0.5, 0.52, "In-domain", fontsize=8,
                ha="center", color="gray")
        ax.text((n_indomain + len(valid)) / 2 - 0.5, 0.52, "Cross-application",
                fontsize=8, ha="center", color="gray")

    for bar, auroc_val in zip(bars, aurocs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{auroc_val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("AUROC", fontsize=11)
    ax.set_ylim(0.3, 1.05)
    ax.set_title(
        "Cross-application AUROC: probe transfer across domains\n"
        "(High cross-app AUROC → directions share meaning across concepts)",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out = config.results_dir_ex9 / "fig3_cross_application_auroc.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Figure 3 saved → {out}")


# ---------------------------------------------------------------------------
# Figure 4: PC1 variance explained (per-model)
# ---------------------------------------------------------------------------

def figure4_pc1_variance(results: dict, config: Experiment9Config) -> None:
    pva = results.get("person_vector_analysis", {})
    pc1_frac = pva.get("pc1_variance_frac", {})
    sv = pva.get("singular_values", [])

    if not pc1_frac:
        print("  Figure 4: skipped (no person-vector analysis data).")
        return

    cond_labels_pretty = {
        "altered_self":   "Altered-Self",
        "gemma":          "Gemma 9B",
        "mistral":        "Qwen 7B",
        "style_imitated": "Style-Imitated",
    }

    conds  = list(pc1_frac.keys())
    fracs  = [pc1_frac[c] for c in conds]
    labels = [cond_labels_pretty.get(c, c) for c in conds]

    total_var = sum(s**2 for s in sv) if sv else 1.0
    pc1_total = sv[0]**2 / total_var * 100 if sv else float("nan")

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(7, 8),
        gridspec_kw={"height_ratios": [2, 1]},
    )

    # Top: PC1 variance per model
    colors = plt.get_cmap("tab10")
    bar_colors = [colors(i) for i in range(len(conds))]
    bars = ax_top.bar(range(len(conds)), fracs, color=bar_colors,
                      edgecolor="black", linewidth=0.5, alpha=0.85)
    for bar, frac in zip(bars, fracs):
        ax_top.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{frac:.3f}", ha="center", va="bottom", fontsize=10)

    ax_top.set_xticks(range(len(conds)))
    ax_top.set_xticklabels(labels, fontsize=11)
    ax_top.set_ylabel("Fraction of direction\nexplained by shared PC1", fontsize=10)
    ax_top.set_ylim(0, 1.1)
    title_str = f"PC1 variance = {pc1_total:.1f}% of total" if not math.isnan(pc1_total) else ""
    ax_top.set_title(
        f"Variance in per-model probe directions explained by shared PC1\n{title_str}",
        fontsize=11,
    )
    ax_top.grid(axis="y", alpha=0.3)

    # Bottom: singular value spectrum
    if sv:
        sv_vals = np.array(sv[:min(len(conds), 4)])
        sv_pct  = 100.0 * sv_vals**2 / (total_var or 1.0)
        ax_bot.bar(range(len(sv_vals)), sv_pct, color="steelblue",
                   edgecolor="black", linewidth=0.5, alpha=0.85)
        for i, pct in enumerate(sv_pct):
            ax_bot.text(i, pct + 0.3, f"{pct:.1f}%", ha="center", va="bottom", fontsize=9)
        ax_bot.set_xticks(range(len(sv_vals)))
        ax_bot.set_xticklabels([f"PC{i+1}" for i in range(len(sv_vals))], fontsize=10)
        ax_bot.set_ylabel("% variance", fontsize=10)
        ax_bot.set_title("Singular value spectrum (variance per PC)", fontsize=10)
        ax_bot.grid(axis="y", alpha=0.3)
    else:
        ax_bot.axis("off")

    fig.tight_layout()
    out = config.results_dir_ex9 / "fig4_pc1_variance.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Figure 4 saved → {out}")


# ---------------------------------------------------------------------------
# Summary table / text
# ---------------------------------------------------------------------------

def write_summary(results: dict, config: Experiment9Config) -> None:
    # Save full results to JSON (convert np arrays to lists if needed)
    json_safe = _make_json_safe(results)
    out_json = config.results_dir_ex9 / "analysis_results.json"
    with open(out_json, "w") as f:
        json.dump(json_safe, f, indent=2)
    print(f"  Full results saved → {out_json}")

    # Print interpretive summary
    lines = [
        "=" * 65,
        "  EXPERIMENT 9 — REPRESENTATIONAL GEOMETRY SUMMARY",
        "=" * 65,
        "",
        "9.1 — Truth probe AUROCs (Llama 8B, layer 30):",
    ]
    for ds, layer_map in results.get("truth_probe_aurocs", {}).items():
        lines.append(f"  {ds}: " + ", ".join(
            f"layer {l}={v:.4f}" for l, v in sorted(layer_map.items(), key=lambda x: int(x[0]))
        ))

    lines += [
        "",
        "9.1 — Cosine similarity: prefill ↔ truth (layer 30):",
    ]
    for ds, sim in results.get("cosine_prefill_truth", {}).items():
        significance = (
            "SIGNIFICANT" if abs(sim) > 0.10 else
            "NOTABLE"     if abs(sim) > 0.05 else
            "~random"
        )
        lines.append(f"  prefill vs truth_{ds}: {sim:+.4f}  [{significance}]")

    rand_info = results.get("random_baseline", {})
    expected  = rand_info.get("expected_magnitude_4096d", float("nan"))
    lines += [
        "",
        f"  Random baseline (n={rand_info.get('n', '?')}): "
        f"expected |cos|≈{expected:.4f} in R^4096",
        f"  vs prefill: mean={rand_info.get('vs_prefill', {}).get('mean', float('nan')):+.4f} "
        f"± {rand_info.get('vs_prefill', {}).get('std', float('nan')):.4f}",
    ]

    lines += [
        "",
        "9.2 — Key cosines (prefill, truth, eval/deploy):",
    ]
    for k, v in results.get("key_cosines", {}).items():
        significance = (
            "SIGNIFICANT" if abs(v) > 0.10 else
            "NOTABLE"     if abs(v) > 0.05 else
            "~random"
        )
        lines.append(f"  {k}: {v:+.4f}  [{significance}]")

    lines += [
        "",
        "9.1–9.2 — Cross-application AUROC:",
    ]
    for k, v in results.get("cross_application_auroc", {}).items():
        lines.append(f"  {k}: {v:.4f}")

    pva = results.get("person_vector_analysis", {})
    if pva:
        pc1_frac = pva.get("pc1_variance_frac", {})
        mean_frac = float(np.mean(list(pc1_frac.values()))) if pc1_frac else float("nan")
        lines += [
            "",
            "9.3 — Per-model person vector analysis:",
            f"  Mean PC1 variance fraction: {mean_frac:.3f}",
            f"  (>0.90 → strong shared 'not-self'; <0.70 → model-specific components)",
        ]
        for cond, frac in pc1_frac.items():
            lines.append(f"  {cond}: PC1 fraction={frac:.3f}")

    lines += ["", "=" * 65]

    summary = "\n".join(lines)
    print(summary)

    out_txt = config.results_dir_ex9 / "summary_ex9.txt"
    out_txt.write_text(summary)
    print(f"  Text summary saved → {out_txt}")


def _make_json_safe(obj):
    """Recursively convert np arrays and np scalars for JSON serialisation."""
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    return obj


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_all_figures_ex9(results: dict, config: Experiment9Config) -> None:
    print("  Figure 1: cosine similarity heatmap...")
    figure1_cosine_heatmap(results, config)

    print("  Figure 2: PCA of per-model probe directions...")
    figure2_pca_per_model(results, config)

    print("  Figure 3: cross-application AUROC bars...")
    figure3_cross_application_auroc(results, config)

    print("  Figure 4: PC1 variance explained...")
    figure4_pc1_variance(results, config)

    print("  Writing summary table...")
    write_summary(results, config)
