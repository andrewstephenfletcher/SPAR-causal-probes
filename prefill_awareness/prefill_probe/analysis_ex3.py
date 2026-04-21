"""
Plotting and interpretive summary for Experiment 3.

Figures:
  1 — base_probe_aurocs.png      Grouped bar chart by condition × subset
  2 — cross_source_heatmap.png   4 × 4 cross-source transfer matrix
  3 — cross_topic_heatmaps.png   2 × 2 grid of 3 × 3 cross-topic matrices
  4 — perplexity_distributions.png  Log-scale density per condition
  5 — perp_matched_analysis.png  Full vs. matched vs. baseline AUROCs

Plus:
  results_dir_ex3 / "summary_ex3.csv"
  stdout: interpretive summary (printed and written to summary_ex3.txt)
"""

import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from .config import Experiment3Config
from .probe_ex3 import CROSS_CONDITIONS, DATASETS, CONDITIONS

_COND_LABELS = {
    "altered_self":   "Altered-self",
    "gemma":          "Gemma 9B",
    "mistral":        "Mistral 7B",
    "style_imitated": "Style-imitated",
}
_DS_LABELS = {"alpaca": "Alpaca", "oasst1": "OASST1", "mmlu": "MMLU"}


def _nan(x):
    try:
        return math.isnan(float(x))
    except (TypeError, ValueError):
        return True


# ---------------------------------------------------------------------------
# Figure 1: Base probe AUROCs — grouped bars
# ---------------------------------------------------------------------------

def _figure_base_probes(results: dict, ex3_config: Experiment3Config) -> None:
    base = results.get("base_probes", {})
    conds = CROSS_CONDITIONS
    n = len(conds)
    x = np.arange(n)
    width = 0.20

    full_auroc     = [base.get(f"self_vs_{c}", {}).get("full", {}).get("auroc", float("nan"))
                      for c in conds]
    oe_auroc       = [base.get(f"self_vs_{c}", {}).get("outlier_excluded", {}).get("auroc", float("nan"))
                      for c in conds]
    pm_auroc       = [base.get(f"self_vs_{c}", {}).get("perp_matched", {}).get("auroc", float("nan"))
                      for c in conds]
    ppl_full_auroc = [base.get(f"self_vs_{c}", {}).get("full", {}).get("ppl_baseline_auroc", float("nan"))
                      for c in conds]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width,     full_auroc, width, label="Probe (full data)",        color="steelblue")
    ax.bar(x,             oe_auroc,   width, label="Probe (outlier-excluded)",  color="cornflowerblue")
    ax.bar(x + width,     pm_auroc,   width, label="Probe (perp-matched)",      color="lightblue")

    # Overlay perplexity baseline as markers
    ax.scatter(x - width, ppl_full_auroc, color="darkorange", zorder=5,
               s=50, marker="D", label="Perp baseline (full)")

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([_COND_LABELS.get(c, c) for c in conds], fontsize=11)
    ax.set_ylabel("AUROC (test set)", fontsize=12)
    ax.set_ylim(0.3, 1.05)
    ax.set_title("Probe AUROC by condition and analysis subset (self = label 0)", fontsize=13)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = ex3_config.results_dir_ex3 / "base_probe_aurocs.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


# ---------------------------------------------------------------------------
# Figure 2: Cross-source transfer heatmap
# ---------------------------------------------------------------------------

def _figure_cross_source(results: dict, ex3_config: Experiment3Config) -> None:
    csm = results.get("cross_source_matrix", {})
    labels = csm.get("labels", CROSS_CONDITIONS)
    values = csm.get("values", [])
    if not values:
        print("  Skipping cross-source heatmap: no data.")
        return

    matrix = np.array(values, dtype=float)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, vmin=0.4, vmax=1.0, cmap="RdYlGn", aspect="auto")
    plt.colorbar(im, ax=ax, label="AUROC")

    tick_labels = [_COND_LABELS.get(l, l) for l in labels]
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=10)
    ax.set_yticklabels(tick_labels, fontsize=10)
    ax.set_xlabel("Eval condition", fontsize=11)
    ax.set_ylabel("Train condition", fontsize=11)
    ax.set_title("Cross-source transfer: AUROC (train condition → eval condition)", fontsize=12)

    for r in range(len(labels)):
        for c in range(len(labels)):
            v = matrix[r, c]
            if np.isfinite(v):
                color = "white" if v < 0.6 or v > 0.88 else "black"
                ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=9, color=color)

    plt.tight_layout()
    out = ex3_config.results_dir_ex3 / "cross_source_heatmap.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


# ---------------------------------------------------------------------------
# Figure 3: Cross-topic transfer heatmaps (2 × 2 grid)
# ---------------------------------------------------------------------------

def _figure_cross_topic(results: dict, ex3_config: Experiment3Config) -> None:
    ctm = results.get("cross_topic_matrices", {})
    conds = [c for c in CROSS_CONDITIONS if c in ctm]
    if not conds:
        print("  Skipping cross-topic heatmaps: no data.")
        return

    n_plots = len(conds)
    ncols = 2
    nrows = math.ceil(n_plots / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5.5, nrows * 4.5))
    axes = np.array(axes).flatten()

    for i, cond in enumerate(conds):
        ax = axes[i]
        data = ctm[cond]
        labels = data.get("labels", DATASETS)
        matrix = np.array(data.get("values", []), dtype=float)
        if matrix.size == 0:
            ax.axis("off")
            continue

        im = ax.imshow(matrix, vmin=0.4, vmax=1.0, cmap="RdYlGn", aspect="auto")
        plt.colorbar(im, ax=ax, label="AUROC", shrink=0.8)

        tick_labels = [_DS_LABELS.get(l, l) for l in labels]
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(tick_labels, rotation=20, ha="right", fontsize=9)
        ax.set_yticklabels(tick_labels, fontsize=9)
        ax.set_xlabel("Eval dataset", fontsize=10)
        ax.set_ylabel("Train dataset", fontsize=10)
        ax.set_title(f"{_COND_LABELS.get(cond, cond)}", fontsize=11)

        for r in range(len(labels)):
            for c in range(len(labels)):
                v = matrix[r, c]
                if np.isfinite(v):
                    color = "white" if v < 0.6 or v > 0.88 else "black"
                    ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                            fontsize=8, color=color)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Cross-topic transfer: AUROC by train/eval dataset", fontsize=13, y=1.01)
    plt.tight_layout()
    out = ex3_config.results_dir_ex3 / "cross_topic_heatmaps.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ---------------------------------------------------------------------------
# Figure 4: Perplexity distributions (log scale)
# ---------------------------------------------------------------------------

def _figure_perplexity_distributions(
    responses: list[dict],
    ex3_config: Experiment3Config,
) -> None:
    ppl_path = ex3_config.activations_dir_ex3 / "perplexity_all.json"
    if not ppl_path.exists():
        print("  Skipping perplexity distribution figure: perplexity_all.json not found.")
        return

    with open(ppl_path) as f:
        records = json.load(f)

    ppl_by_cond: dict[str, list[float]] = {c: [] for c in CONDITIONS}
    for rec in records:
        cond = rec["condition"]
        ppl = rec.get("perplexity", float("nan"))
        if np.isfinite(ppl) and ppl > 0:
            ppl_by_cond[cond].append(ppl)

    fig, ax = plt.subplots(figsize=(11, 5))
    colors = plt.get_cmap("tab10")
    bins = np.logspace(np.log10(0.5), np.log10(600), 60)

    for i, cond in enumerate(CONDITIONS):
        vals = ppl_by_cond[cond]
        if not vals:
            continue
        label = _COND_LABELS.get(cond, cond) if cond != "self" else "Self (Llama 8B)"
        ax.hist(vals, bins=bins, alpha=0.5, density=True, label=f"{label}  (n={len(vals)})",
                color=colors(i))

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x:.0f}" if x >= 1 else f"{x:.1f}"
    ))
    ax.set_xlabel("Perplexity under Llama 8B (log scale)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Response perplexity distributions by condition", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = ex3_config.results_dir_ex3 / "perplexity_distributions.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


# ---------------------------------------------------------------------------
# Figure 5: Perplexity-matched analysis
# ---------------------------------------------------------------------------

def _figure_perp_matched(results: dict, ex3_config: Experiment3Config) -> None:
    base = results.get("base_probes", {})
    conds = CROSS_CONDITIONS
    n = len(conds)
    x = np.arange(n)
    width = 0.22

    full_auroc  = [base.get(f"self_vs_{c}", {}).get("full", {}).get("auroc", float("nan"))
                   for c in conds]
    pm_auroc    = [base.get(f"self_vs_{c}", {}).get("perp_matched", {}).get("auroc", float("nan"))
                   for c in conds]
    ppl_full    = [base.get(f"self_vs_{c}", {}).get("full", {}).get("ppl_baseline_auroc", float("nan"))
                   for c in conds]
    ppl_matched = [base.get(f"self_vs_{c}", {}).get("perp_matched", {}).get("ppl_baseline_auroc",
                                                                             float("nan"))
                   for c in conds]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width * 1.5, full_auroc,  width, label="Probe AUROC (full)",     color="steelblue")
    ax.bar(x - width * 0.5, pm_auroc,    width, label="Probe AUROC (ppl-matched)", color="lightblue")
    ax.bar(x + width * 0.5, ppl_full,    width, label="Perp baseline (full)",   color="darkorange",
           alpha=0.7)
    ax.bar(x + width * 1.5, ppl_matched, width, label="Perp baseline (matched)", color="moccasin",
           alpha=0.9)

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([_COND_LABELS.get(c, c) for c in conds], fontsize=11)
    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_ylim(0.3, 1.05)
    ax.set_title("Perplexity-matched analysis: probe vs. baseline (key result)", fontsize=13)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = ex3_config.results_dir_ex3 / "perp_matched_analysis.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


# ---------------------------------------------------------------------------
# Summary CSV
# ---------------------------------------------------------------------------

def _save_summary_csv(results: dict, ex3_config: Experiment3Config) -> None:
    import csv
    base = results.get("base_probes", {})
    rows = []
    for cond in CROSS_CONDITIONS:
        key = f"self_vs_{cond}"
        d = base.get(key, {})
        full = d.get("full", {})
        oe   = d.get("outlier_excluded", {})
        pm   = d.get("perp_matched", {})

        def _fmt(v):
            return "" if _nan(v) else str(round(float(v), 4))

        rows.append({
            "condition":                cond,
            "n_test_full":              full.get("n_test", ""),
            "probe_auroc_full":         _fmt(full.get("auroc")),
            "ppl_baseline_auroc_full":  _fmt(full.get("ppl_baseline_auroc")),
            "gap_full":                 _fmt(
                float(full.get("auroc", 0)) - float(full.get("ppl_baseline_auroc", 0))
                if not _nan(full.get("auroc")) and not _nan(full.get("ppl_baseline_auroc"))
                else float("nan")
            ),
            "n_test_outlier_ex":        oe.get("n_test", ""),
            "probe_auroc_outlier_ex":   _fmt(oe.get("auroc")),
            "ppl_baseline_outlier_ex":  _fmt(oe.get("ppl_baseline_auroc")),
            "n_test_ppl_matched":       pm.get("n_matched", ""),
            "probe_auroc_ppl_matched":  _fmt(pm.get("auroc")),
            "ppl_baseline_matched":     _fmt(pm.get("ppl_baseline_auroc")),
        })

    # Unified probe row
    unif = results.get("unified_probe", {}).get("full", {})
    rows.append({
        "condition": "all_others (unified)",
        "n_test_full": unif.get("n_test", ""),
        "probe_auroc_full": str(round(float(unif.get("auroc", float("nan"))), 4))
        if not _nan(unif.get("auroc")) else "",
        "ppl_baseline_auroc_full": "",
        "gap_full": "", "n_test_outlier_ex": "", "probe_auroc_outlier_ex": "",
        "ppl_baseline_outlier_ex": "", "n_test_ppl_matched": "",
        "probe_auroc_ppl_matched": "", "ppl_baseline_matched": "",
    })

    out = ex3_config.results_dir_ex3 / "summary_ex3.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Summary saved → {out}")


# ---------------------------------------------------------------------------
# Interpretive summary
# ---------------------------------------------------------------------------

def _print_interpretive_summary(results: dict, ex3_config: Experiment3Config) -> None:
    base = results.get("base_probes", {})

    # Signal strength
    aurocs = {
        c: base.get(f"self_vs_{c}", {}).get("full", {}).get("auroc", float("nan"))
        for c in CROSS_CONDITIONS
    }
    valid_aurocs = {c: v for c, v in aurocs.items() if not _nan(v)}

    weakest = min(valid_aurocs, key=valid_aurocs.get) if valid_aurocs else "N/A"
    strongest = max(valid_aurocs, key=valid_aurocs.get) if valid_aurocs else "N/A"

    # Cross-source transfer
    csm = results.get("cross_source_matrix", {})
    mat = np.array(csm.get("values", []), dtype=float)
    if mat.size > 0:
        diagonal = np.diag(mat)
        off_diag = mat[~np.eye(mat.shape[0], dtype=bool)]
        mean_diag  = np.nanmean(diagonal)
        mean_offdiag = np.nanmean(off_diag)
        transfer_gap = mean_diag - mean_offdiag
    else:
        mean_diag = mean_offdiag = transfer_gap = float("nan")

    if not _nan(transfer_gap):
        if transfer_gap < 0.05:
            src_class = "[ X ] STRONG TRANSFER (gap < 0.05): probe learns general \"not-self\""
        elif transfer_gap < 0.15:
            src_class = "[ X ] MODERATE TRANSFER (gap 0.05-0.15): partial generalization"
        else:
            src_class = "[ X ] WEAK TRANSFER (gap > 0.15): source-specific features"
    else:
        src_class = "[   ] UNABLE TO CLASSIFY (insufficient data)"

    # Cross-topic transfer
    all_topic_diag, all_topic_offdiag = [], []
    for cond, data in results.get("cross_topic_matrices", {}).items():
        m = np.array(data.get("values", []), dtype=float)
        if m.size == 0:
            continue
        all_topic_diag.extend(np.diag(m).tolist())
        all_topic_offdiag.extend(m[~np.eye(m.shape[0], dtype=bool)].tolist())
    mean_topic_diag    = np.nanmean(all_topic_diag) if all_topic_diag else float("nan")
    mean_topic_offdiag = np.nanmean(all_topic_offdiag) if all_topic_offdiag else float("nan")
    topic_gap = mean_topic_diag - mean_topic_offdiag if not (_nan(mean_topic_diag) or _nan(mean_topic_offdiag)) else float("nan")

    if not _nan(topic_gap):
        if topic_gap < 0.05:
            topic_class = "[ X ] STRONG TRANSFER: learns topic-independent authorship features"
        elif topic_gap < 0.15:
            topic_class = "[ X ] MODERATE TRANSFER: partial topic generalization"
        else:
            topic_class = "[ X ] WEAK TRANSFER: topic-conditional features"
    else:
        topic_class = "[   ] UNABLE TO CLASSIFY"

    # Perplexity-controlled signal
    pm_aurocs = [
        base.get(f"self_vs_{c}", {}).get("perp_matched", {}).get("auroc", float("nan"))
        for c in CROSS_CONDITIONS
    ]
    mean_pm = np.nanmean([v for v in pm_aurocs if not _nan(v)]) if pm_aurocs else float("nan")
    if not _nan(mean_pm):
        if mean_pm > 0.65:
            ppl_class = "[ X ] STRONG RESIDUAL SIGNAL (> 0.65): beyond surprisal"
        elif mean_pm > 0.55:
            ppl_class = "[ X ] WEAK RESIDUAL SIGNAL (0.55-0.65): some info beyond surprisal"
        else:
            ppl_class = "[ X ] NO RESIDUAL SIGNAL (< 0.55): probe is largely surprisal"
    else:
        ppl_class = "[   ] UNABLE TO CLASSIFY"

    # Altered-self
    asc = results.get("altered_self_comparison", {})
    as_auroc = asc.get("self_vs_altered_self", {}).get("auroc", float("nan"))
    gm_auroc = asc.get("self_vs_gemma", {}).get("auroc", float("nan"))
    ratio = float(as_auroc) / float(gm_auroc) if not (_nan(as_auroc) or _nan(gm_auroc) or gm_auroc == 0) else float("nan")
    if not _nan(ratio):
        as_class = (
            "[ X ] CLEAR DISTINCTION: probe separates self-variation from other-source"
            if ratio < 0.85
            else "[ X ] BLURRED DISTINCTION: probe mostly detects \"unusual for me\""
        )
    else:
        as_class = "[   ] UNABLE TO CLASSIFY"

    def _f(v):
        return f"{float(v):.4f}" if not _nan(v) else "N/A"

    summary = f"""
{'=' * 65}
  EXPERIMENT 3 — INTERPRETIVE SUMMARY
{'=' * 65}

Signal strength by source:
  Weakest (lowest probe AUROC):   {_COND_LABELS.get(weakest, weakest)}  ({_f(valid_aurocs.get(weakest, float('nan')))})
  Strongest (highest probe AUROC):{_COND_LABELS.get(strongest, strongest)}  ({_f(valid_aurocs.get(strongest, float('nan')))})
  AUROCs: {", ".join(f"{_COND_LABELS.get(c,c)}={_f(v)}" for c,v in aurocs.items())}

Cross-source transfer:
  Mean on-diagonal AUROC:  {_f(mean_diag)}
  Mean off-diagonal AUROC: {_f(mean_offdiag)}
  Transfer gap:            {_f(transfer_gap)}
  {src_class}

Cross-topic transfer:
  Mean on-diagonal AUROC:  {_f(mean_topic_diag)}
  Mean off-diagonal AUROC: {_f(mean_topic_offdiag)}
  Transfer gap:            {_f(topic_gap)}
  {topic_class}

Perplexity-controlled signal:
  Mean probe AUROC (perp-matched subsets): {_f(mean_pm)}
  {ppl_class}

Altered-self vs. cross-model:
  self vs. altered_self AUROC: {_f(as_auroc)}
  self vs. gemma AUROC:        {_f(gm_auroc)}
  Ratio (altered/gemma):       {_f(ratio)}
  {as_class}
{'=' * 65}"""

    print(summary)
    txt_path = ex3_config.results_dir_ex3 / "summary_ex3.txt"
    with open(txt_path, "w") as f:
        f.write(summary)
    print(f"  Interpretive summary saved → {txt_path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_all_figures_ex3(
    results: dict,
    responses: list[dict],
    ex3_config: Experiment3Config,
) -> None:
    """Generate all 5 figures, print interpretive summary, save CSVs."""
    print("  Figure 1: base probe AUROCs...")
    _figure_base_probes(results, ex3_config)

    print("  Figure 2: cross-source transfer heatmap...")
    _figure_cross_source(results, ex3_config)

    print("  Figure 3: cross-topic transfer heatmaps...")
    _figure_cross_topic(results, ex3_config)

    print("  Figure 4: perplexity distributions...")
    _figure_perplexity_distributions(responses, ex3_config)

    print("  Figure 5: perplexity-matched analysis...")
    _figure_perp_matched(results, ex3_config)

    _save_summary_csv(results, ex3_config)
    _print_interpretive_summary(results, ex3_config)
