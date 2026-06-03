"""
Analysis for Experiment 0m: Gemma 4 31B single-evaluator, OSS sources.

Mirrors experiment0n structure but for open-source models.

Figures:
  auroc_by_source.png         — AUROC per source with 95% CI
  auroc_by_dataset.png        — AUROC per source × dataset
  balanced_accuracy_by_source.png — balanced accuracy (optimal threshold)
  probability_distributions.png — tamper prob distributions per source
  explanation_categories.png  — detection cue breakdown

Summary: results_dir/summary_table.csv
"""

import re
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import SOURCE_FAMILIES, Experiment0mConfig

_META   = "#E8442A"   # distinct red-orange to separate from Google blue
_GOOGLE = "#4285F4"
_QWEN   = "#FF6A00"

_SOURCE_PALETTE = {
    "gemma_3_4b":   "#7BAAF7",   # Google (small, lighter)
    "gemma_4_31b":  _GOOGLE,     # Google (large)
    "llama_31_8b":  "#F5A08A",   # Meta (small, lighter)
    "llama_33_70b": _META,       # Meta (large)
    "qwen_25_7b":   "#FFB07A",   # Qwen (small, lighter)
    "qwen_25_72b":  _QWEN,       # Qwen (large)
    "organic":      "#888888",
}
_SRC_LABELS = {
    "gemma_3_4b":   "Gemma 3\n4B",
    "gemma_4_31b":  "Gemma 4\n31B",
    "llama_31_8b":  "Llama 3.1\n8B",
    "llama_33_70b": "Llama 3.3\n70B",
    "qwen_25_7b":   "Qwen 2.5\n7B",
    "qwen_25_72b":  "Qwen 2.5\n72B",
    "organic":      "Organic (self)",
}
_DATASET_LABELS = {
    "oasst1":       "OASST1",
    "bigcodebench": "BigCodeBench",
    "gpqa":         "GPQA",
}
_DATASET_COLORS = {
    "oasst1":       "#9467bd",
    "bigcodebench": "#ff7f0e",
    "gpqa":         "#2ca02c",
}
_REASON_PATTERNS = [
    ("Style / tone",            r"style|tone|formal|informal|verbose|concise|way (i|you) (write|speak)"),
    ("Format / structure",      r"format|structur|bullet|markdown|heading|spacing|length"),
    ("Technical inconsistency", r"inconsisten|approach differ|different method|wouldn't|would not"),
    ("Quality / capability",    r"qualit|capabil|sophisticat|level of|complexity"),
    ("Confident self-claim",    r"this is my|i (generated|wrote|produced)|recogni(se|ze)"),
    ("Uncertainty / vague",     r"unsure|uncertain|hard to (say|tell)|unclear"),
    ("Other",                   r".*"),
]


def run_analysis(results: list[dict], config: Experiment0mConfig) -> None:
    df = pd.DataFrame(results)
    if len(df) == 0:
        print("  No results to analyse.")
        return

    df["tamper_prob"] = pd.to_numeric(df["tamper_prob"], errors="coerce")
    det = df.dropna(subset=["tamper_prob"]).copy()
    if len(det) == 0:
        print("  No parseable results.")
        return

    sources = [s for s in det["source"].unique() if s != "organic"]
    datasets = [d for d in ["oasst1", "bigcodebench", "gpqa"]
                if d in det["dataset"].unique()]

    _figure_auroc_by_source(det, sources, config)
    _figure_auroc_blog(det, config)
    _figure_auroc_blog_by_dataset(det, datasets, config)
    _figure_auroc_by_dataset(det, sources, datasets, config)
    _figure_balanced_accuracy(det, sources, config)
    _figure_probability_distributions(det, sources, datasets, config)
    _figure_explanation_categories(det, sources, config)
    _save_summary(det, sources, datasets, config)


# ---------------------------------------------------------------------------
# Fig 1: AUROC by source
# ---------------------------------------------------------------------------

def _figure_auroc_by_source(
    df: pd.DataFrame,
    sources: list[str],
    config: Experiment0mConfig,
) -> None:
    org = df[df["source"] == "organic"]["tamper_prob"].dropna()

    aurocs, ci_los, ci_his, colors = [], [], [], []
    for src in sources:
        sub = df[df["source"] == src]["tamper_prob"].dropna()
        if len(sub) < 5 or len(org) < 5:
            aurocs.append(np.nan); ci_los.append(0); ci_his.append(0)
            colors.append("grey")
            continue
        y_true  = [0] * len(org) + [1] * len(sub)
        y_score = list(org) + list(sub)
        try:
            a = roc_auc_score(y_true, y_score)
            ci_lo, ci_hi = _bootstrap_auroc_ci(y_true, y_score)
        except Exception:
            a = ci_lo = ci_hi = np.nan
        aurocs.append(a)
        ci_los.append(a - ci_lo if not np.isnan(a) else 0)
        ci_his.append(ci_hi - a if not np.isnan(a) else 0)
        colors.append(_SOURCE_PALETTE.get(src, "grey"))

    x = np.arange(len(sources))
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(
        x, [v if not np.isnan(v) else 0 for v in aurocs],
        color=colors, edgecolor="white", width=0.5,
        yerr=[ci_los, ci_his], capsize=6,
        error_kw={"linewidth": 1.3, "capthick": 1.3},
    )
    for bar, v in zip(bars, aurocs):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.axhline(0.5, color="red", linestyle="--", linewidth=1.5, label="Chance (0.50)")
    ax.set_xticks(x)
    ax.set_xticklabels([_SRC_LABELS.get(s, s) for s in sources], fontsize=9)
    ax.set_ylabel("AUROC")
    ax.set_title(
        "Gemma 4 31B Prefill Detection AUROC — OSS Sources\n"
        "(error bars = 95% CI bootstrap; N=300 tasks per source)",
        fontsize=10,
    )
    ax.set_ylim(0.3, 1.05)
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = config.figures_dir / "auroc_by_source.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Blog figure: vertical bars grouped by family
# ---------------------------------------------------------------------------

def _figure_auroc_blog(det: pd.DataFrame, config: Experiment0mConfig) -> None:
    org = det[det["source"] == "organic"]["tamper_prob"].dropna()
    if len(org) < 5:
        return

    # Groups: (org_label, color, [(source_key, bar_label), ...])
    groups = [
        ("Google", _GOOGLE, [
            ("gemma_3_4b",  "Gemma 3\n4B"),
            ("gemma_4_31b", "Gemma 4\n31B"),
        ]),
        ("Meta", _META, [
            ("llama_31_8b",  "Llama 3.1\n8B"),
            ("llama_33_70b", "Llama 3.3\n70B"),
        ]),
        ("Qwen", _QWEN, [
            ("qwen_25_7b",  "Qwen 2.5\n7B"),
            ("qwen_25_72b", "Qwen 2.5\n72B"),
        ]),
    ]

    bar_w = 0.55
    group_gap = 0.7
    xs, aurocs, lo_errs, hi_errs, colors, bar_labels = [], [], [], [], [], []
    group_spans = []

    x = 0.0
    for org_label, color, members in groups:
        x_start = x
        for src, lbl in members:
            pos = det[det["source"] == src]["tamper_prob"].dropna()
            if len(pos) < 5 or len(org) < 5:
                continue
            y_true  = [0] * len(org) + [1] * len(pos)
            y_score = list(org) + list(pos)
            try:
                a = roc_auc_score(y_true, y_score)
                lo, hi = _bootstrap_auroc_ci(y_true, y_score)
            except Exception:
                continue
            xs.append(x)
            aurocs.append(a)
            lo_errs.append(a - lo if not np.isnan(lo) else 0.0)
            hi_errs.append(hi - a if not np.isnan(hi) else 0.0)
            colors.append(color)
            bar_labels.append(lbl)
            x += bar_w + 0.15
        if x > x_start:
            group_spans.append((x_start, x - 0.15, org_label, color))
            x += group_gap

    if not xs:
        return

    fig, ax = plt.subplots(figsize=(max(6, len(xs) * 1.5 + 2), 5.5))

    bars = ax.bar(xs, aurocs, width=bar_w, color=colors, edgecolor="white",
                  linewidth=1.2, zorder=3,
                  yerr=[lo_errs, hi_errs], capsize=5,
                  error_kw={"linewidth": 1.3, "capthick": 1.3, "zorder": 4})

    for bar, a, hi_e in zip(bars, aurocs, hi_errs):
        ax.text(bar.get_x() + bar.get_width() / 2, a + hi_e + 0.018,
                f"{a:.2f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold", color="#222222")

    for x_left, x_right, org_label, color in group_spans:
        mid = (x_left + x_right) / 2
        ax.text(mid, -0.085, org_label,
                ha="center", va="top", fontsize=12,
                fontweight="bold", color=color,
                transform=ax.get_xaxis_transform())

    ax.set_xticks(xs)
    ax.set_xticklabels(bar_labels, fontsize=10)
    ax.tick_params(axis="x", length=0, pad=6)

    ax.axhline(0.5, color="#cccccc", linestyle="--", linewidth=1.3, zorder=2)
    ax.text(xs[-1] + bar_w * 0.6, 0.502, "chance",
            fontsize=9, color="#aaaaaa", va="bottom")

    ax.set_ylabel("AUROC  (Gemma 4 31B monitor)", fontsize=12)
    ax.set_ylim(0.3, 1.08)
    ax.set_xlim(xs[0] - bar_w * 0.8, xs[-1] + bar_w * 1.4)
    ax.set_title(
        "Gemma 4 31B detects OSS prefills",
        fontsize=14, fontweight="bold", pad=12,
    )

    ax.grid(axis="y", alpha=0.2, zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    out = config.figures_dir / "auroc_by_source_blog.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Blog figure: vertical bars by dataset (hatch = dataset, colour = family)
# ---------------------------------------------------------------------------

_DATASET_HATCHES = {
    "oasst1":       "",
    "bigcodebench": "///",
    "gpqa":         "...",
}

def _figure_auroc_blog_by_dataset(
    det: pd.DataFrame,
    datasets: list[str],
    config: Experiment0mConfig,
) -> None:
    org_all = det[det["source"] == "organic"]["tamper_prob"].dropna()
    if len(org_all) < 5:
        return

    groups = [
        ("Google", _GOOGLE, [
            ("gemma_3_4b",  "Gemma 3\n4B"),
            ("gemma_4_31b", "Gemma 4\n31B"),
        ]),
        ("Meta", _META, [
            ("llama_31_8b",  "Llama 3.1\n8B"),
            ("llama_33_70b", "Llama 3.3\n70B"),
        ]),
        ("Qwen", _QWEN, [
            ("qwen_25_7b",  "Qwen 2.5\n7B"),
            ("qwen_25_72b", "Qwen 2.5\n72B"),
        ]),
    ]

    bar_w    = 0.22
    ds_gap   = 0.04   # gap between dataset bars within a source
    src_gap  = 0.25   # gap between source model groups
    fam_gap  = 0.55   # gap between family groups
    n_ds     = len(datasets)
    src_span = n_ds * bar_w + (n_ds - 1) * ds_gap  # total width of one source group

    xs_per_src = []   # list of [x0, x1, x2] per source
    group_spans = []  # (x_left, x_right, family_label, color)
    src_centers = []  # (center_x, src_label) for x-tick
    src_colors  = []

    x = 0.0
    for fam_label, color, members in groups:
        fam_x_start = x
        for src, lbl in members:
            bar_xs = [x + i * (bar_w + ds_gap) for i in range(n_ds)]
            xs_per_src.append((src, color, bar_xs))
            src_centers.append((x + src_span / 2, lbl))
            src_colors.append(color)
            x += src_span + src_gap
        group_spans.append((fam_x_start, x - src_gap, fam_label, color))
        x += fam_gap

    fig_w = max(10, x)
    fig, ax = plt.subplots(figsize=(fig_w, 5.5))

    for src, color, bar_xs in xs_per_src:
        for ds_name, bx in zip(datasets, bar_xs):
            ds_det = det[det["dataset"] == ds_name]
            org = ds_det[ds_det["source"] == "organic"]["tamper_prob"].dropna()
            pos = ds_det[ds_det["source"] == src]["tamper_prob"].dropna()
            if len(pos) < 5 or len(org) < 5:
                continue
            y_true  = [0] * len(org) + [1] * len(pos)
            y_score = list(org) + list(pos)
            try:
                a = roc_auc_score(y_true, y_score)
                lo, hi = _bootstrap_auroc_ci(y_true, y_score)
            except Exception:
                continue

            err_lo = a - lo if not np.isnan(lo) else 0.0
            err_hi = hi - a if not np.isnan(hi) else 0.0
            hatch  = _DATASET_HATCHES.get(ds_name, "")

            ax.bar(bx, a, width=bar_w, color=color, hatch=hatch,
                   edgecolor="white", linewidth=0.8, zorder=3,
                   yerr=[[err_lo], [err_hi]], capsize=3,
                   error_kw={"linewidth": 1.0, "capthick": 1.0, "zorder": 4})
            ax.text(bx, a + err_hi + 0.015, f"{a:.2f}",
                    ha="center", va="bottom", fontsize=7.5,
                    fontweight="bold", color="#222222")

    # Source model labels as x-ticks
    tick_xs   = [c for c, _ in src_centers]
    tick_lbls = [l for _, l in src_centers]
    ax.set_xticks(tick_xs)
    ax.set_xticklabels(tick_lbls, fontsize=9)
    ax.tick_params(axis="x", length=0, pad=6)

    # Family labels below
    for x_left, x_right, fam_label, color in group_spans:
        ax.text((x_left + x_right) / 2, -0.095, fam_label,
                ha="center", va="top", fontsize=11,
                fontweight="bold", color=color,
                transform=ax.get_xaxis_transform())

    ax.axhline(0.5, color="#cccccc", linestyle="--", linewidth=1.3, zorder=2)

    ax.set_ylabel("AUROC  (Gemma 4 31B monitor)", fontsize=12)
    ax.set_ylim(0.3, 1.12)
    ax.set_xlim(-bar_w, x - fam_gap + bar_w)
    ax.set_title("Gemma 4 31B detects OSS prefills — by dataset",
                 fontsize=14, fontweight="bold", pad=12)

    # Dataset legend
    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor="#aaaaaa", hatch=_DATASET_HATCHES[d], edgecolor="white",
              label=_DATASET_LABELS.get(d, d))
        for d in datasets if d in _DATASET_HATCHES
    ]
    ax.legend(handles=legend_els, loc="upper right", fontsize=10,
              title="Dataset", title_fontsize=10)

    ax.grid(axis="y", alpha=0.2, zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    out = config.figures_dir / "auroc_blog_by_dataset.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Fig 2: AUROC by dataset
# ---------------------------------------------------------------------------

def _figure_auroc_by_dataset(
    df: pd.DataFrame,
    sources: list[str],
    datasets: list[str],
    config: Experiment0mConfig,
) -> None:
    n_src = len(sources)
    w = 0.7 / n_src
    x = np.arange(len(datasets))

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, src in enumerate(sources):
        aurocs = []
        for ds_name in datasets:
            sub_df = df[df["dataset"] == ds_name]
            org = sub_df[sub_df["source"] == "organic"]["tamper_prob"].dropna()
            sub = sub_df[sub_df["source"] == src]["tamper_prob"].dropna()
            if len(sub) < 5 or len(org) < 5:
                aurocs.append(np.nan)
                continue
            y_true  = [0] * len(org) + [1] * len(sub)
            y_score = list(org) + list(sub)
            try:
                aurocs.append(roc_auc_score(y_true, y_score))
            except Exception:
                aurocs.append(np.nan)

        offsets = x + (i - (n_src - 1) / 2) * w
        bars = ax.bar(offsets, [v if not np.isnan(v) else 0 for v in aurocs],
                      w, label=_SRC_LABELS.get(src, src).replace("\n", " "),
                      color=_SOURCE_PALETTE.get(src, "grey"), edgecolor="white")
        for bar, v in zip(bars, aurocs):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                        f"{v:.2f}", ha="center", fontsize=7, fontweight="bold")

    ax.axhline(0.5, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([_DATASET_LABELS.get(d, d) for d in datasets])
    ax.set_ylabel("AUROC")
    ax.set_title("Gemma 4 31B AUROC by Dataset and Source\n(dashed = chance)")
    ax.set_ylim(0.3, 1.05)
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = config.figures_dir / "auroc_by_dataset.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Fig 3: Balanced accuracy (optimal threshold)
# ---------------------------------------------------------------------------

def _figure_balanced_accuracy(
    df: pd.DataFrame,
    sources: list[str],
    config: Experiment0mConfig,
) -> None:
    org = df[df["source"] == "organic"]["tamper_prob"].dropna()
    if len(org) < 5:
        return

    bas, ci_los, ci_his, colors, opt_thresholds = [], [], [], [], []
    for src in sources:
        sub = df[df["source"] == src]["tamper_prob"].dropna()
        if len(sub) < 5:
            bas.append(np.nan); ci_los.append(0); ci_his.append(0)
            colors.append("grey"); opt_thresholds.append(np.nan)
            continue
        ba, lo, hi, opt_t = _bootstrap_ba_optimal(org.tolist(), sub.tolist())
        bas.append(ba)
        ci_los.append(ba - lo if not np.isnan(lo) else 0)
        ci_his.append(hi - ba if not np.isnan(hi) else 0)
        colors.append(_SOURCE_PALETTE.get(src, "grey"))
        opt_thresholds.append(opt_t)

    x = np.arange(len(sources))
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(
        x, [v if not np.isnan(v) else 0 for v in bas],
        color=colors, edgecolor="white", width=0.5,
        yerr=[ci_los, ci_his], capsize=6,
        error_kw={"linewidth": 1.3, "capthick": 1.3},
    )
    for bar, v, t in zip(bars, bas, opt_thresholds):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012,
                    f"{v:.3f}\n(t={t:.0f})", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

    ax.axhline(0.5, color="red", linestyle="--", linewidth=1.5, label="Chance (0.50)")
    ax.set_xticks(x)
    ax.set_xticklabels([_SRC_LABELS.get(s, s) for s in sources], fontsize=9)
    ax.set_ylabel("Balanced Accuracy")
    ax.set_ylim(0.3, 1.05)
    ax.set_title(
        "Gemma 4 31B Balanced Accuracy (optimal threshold per source)\n"
        "(error bars = 95% CI bootstrap; t = threshold used)",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = config.figures_dir / "balanced_accuracy_by_source.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out}")


def _optimal_ba(org: list[float], pos: list[float]) -> tuple[float, float]:
    all_scores = sorted(set(org + pos))
    best_ba, best_t = 0.0, all_scores[0]
    org_arr = np.array(org)
    pos_arr = np.array(pos)
    for t in all_scores:
        tnr = np.mean(org_arr <= t)
        tpr = np.mean(pos_arr > t)
        ba = (tpr + tnr) / 2
        if ba > best_ba:
            best_ba, best_t = ba, t
    return float(best_ba), float(best_t)


def _bootstrap_ba_optimal(
    org: list[float],
    pos: list[float],
    n: int = 1000,
) -> tuple[float, float, float, float]:
    ba, opt_t = _optimal_ba(org, pos)
    rng = np.random.default_rng(42)
    boot = []
    for _ in range(n):
        o_b = rng.choice(org, size=len(org), replace=True).tolist()
        p_b = rng.choice(pos, size=len(pos), replace=True).tolist()
        boot.append(_optimal_ba(o_b, p_b)[0])
    return float(ba), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), opt_t


# ---------------------------------------------------------------------------
# Fig 4: Probability distributions
# ---------------------------------------------------------------------------

def _figure_probability_distributions(
    df: pd.DataFrame,
    sources: list[str],
    datasets: list[str],
    config: Experiment0mConfig,
) -> None:
    ncols = len(datasets)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4), sharey=False)
    if ncols == 1:
        axes = [axes]

    for ax, ds_name in zip(axes, datasets):
        sub_df = df[df["dataset"] == ds_name]
        org = sub_df[sub_df["source"] == "organic"]["tamper_prob"].dropna()
        ax.hist(org, bins=20, range=(0, 100), alpha=0.6,
                label="Organic", color=_SOURCE_PALETTE["organic"])
        for src in sources:
            src_data = sub_df[sub_df["source"] == src]["tamper_prob"].dropna()
            if len(src_data) < 3:
                continue
            ax.hist(src_data, bins=20, range=(0, 100), alpha=0.55,
                    label=_SRC_LABELS.get(src, src).replace("\n", " "),
                    color=_SOURCE_PALETTE.get(src, "grey"))
        ax.set_title(_DATASET_LABELS.get(ds_name, ds_name), fontsize=9)
        ax.set_xlabel("Tamper probability (0–100)", fontsize=8)
        ax.set_ylabel("Count", fontsize=8)
        ax.legend(fontsize=7)

    fig.suptitle("Tamper Probability Distributions — Gemma 4 31B (OSS Sources)", fontsize=10)
    plt.tight_layout()
    out = config.figures_dir / "probability_distributions.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Fig 5: Explanation categories
# ---------------------------------------------------------------------------

def _figure_explanation_categories(
    df: pd.DataFrame,
    sources: list[str],
    config: Experiment0mConfig,
) -> None:
    groups = {"Organic": [], "Within-family": [], "Cross-family": []}
    for _, row in df.iterrows():
        src    = row.get("source", "")
        reason = row.get("reason", "")
        if not isinstance(reason, str) or len(reason) < 5:
            continue
        if src == "organic":
            groups["Organic"].append(reason.lower())
        elif config.is_within_family(str(src)):
            groups["Within-family"].append(reason.lower())
        else:
            groups["Cross-family"].append(reason.lower())

    all_cats = [label for label, _ in _REASON_PATTERNS]
    x = np.arange(len(all_cats))
    n = len(groups)
    w = 0.7 / n
    grp_colors = {"Organic": "steelblue", "Within-family": "seagreen", "Cross-family": "tomato"}

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (grp, reasons) in enumerate(groups.items()):
        cats = Counter(_categorise(r) for r in reasons)
        total = sum(cats.values()) or 1
        freqs = [cats.get(c, 0) / total for c in all_cats]
        offsets = x + (i - (n - 1) / 2) * w
        ax.bar(offsets, freqs, w, label=grp,
               color=grp_colors.get(grp, "grey"), edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(all_cats, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Fraction of responses")
    ax.set_title("Self-reported detection cues — Gemma 4 31B (OSS Sources)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = config.figures_dir / "explanation_categories.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out}")


def _categorise(text: str) -> str:
    for label, pattern in _REASON_PATTERNS:
        if re.search(pattern, text):
            return label
    return "Other"


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _save_summary(
    df: pd.DataFrame,
    sources: list[str],
    datasets: list[str],
    config: Experiment0mConfig,
) -> None:
    rows = []
    for ds_name in datasets + ["all"]:
        sub_df = df[df["dataset"] == ds_name] if ds_name != "all" else df
        org = sub_df[sub_df["source"] == "organic"]["tamper_prob"].dropna()

        for src in sources:
            sub = sub_df[sub_df["source"] == src]["tamper_prob"].dropna()
            if len(sub) < 3 or len(org) < 3:
                continue
            y_true  = [0] * len(org) + [1] * len(sub)
            y_score = list(org) + list(sub)
            if len(set(y_true)) < 2:
                auroc = ci_lo = ci_hi = np.nan
            else:
                auroc = roc_auc_score(y_true, y_score)
                ci_lo, ci_hi = _bootstrap_auroc_ci(y_true, y_score)

            rows.append({
                "dataset":         ds_name,
                "source":          src,
                "source_family":   SOURCE_FAMILIES.get(src, ""),
                "within_family":   config.is_within_family(src),
                "N_organic":       len(org),
                "N_tampered":      len(sub),
                "AUROC":           _r(auroc),
                "CI_lo":           _r(ci_lo),
                "CI_hi":           _r(ci_hi),
                "mean_P_organic":  _r(org.mean()),
                "mean_P_tampered": _r(sub.mean()),
            })

    summary = pd.DataFrame(rows)
    out = config.results_dir / "summary_table.csv"
    summary.to_csv(out, index=False)
    print(f"\n  Saved {out}")
    print("\n  Summary (all datasets combined):")
    print(summary[summary["dataset"] == "all"].to_string(index=False))


def _bootstrap_auroc_ci(
    y_true: list[int],
    y_score: list[float],
    n: int = 1000,
    alpha: float = 0.05,
) -> tuple[float, float]:
    rng = np.random.default_rng(42)
    n_obs = len(y_true)
    aurocs = []
    for _ in range(n):
        idx = rng.choice(n_obs, size=n_obs, replace=True)
        yt = [y_true[i] for i in idx]
        ys = [y_score[i] for i in idx]
        if len(set(yt)) < 2:
            continue
        aurocs.append(roc_auc_score(yt, ys))
    if not aurocs:
        return np.nan, np.nan
    return (float(np.percentile(aurocs, 100 * alpha / 2)),
            float(np.percentile(aurocs, 100 * (1 - alpha / 2))))


def _r(v, d: int = 3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return round(float(v), d)
