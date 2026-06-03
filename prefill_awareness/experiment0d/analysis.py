"""
Analysis for Experiment 0d: Full Evaluator Sweep.

Figures:
  Fig 1: auroc_by_evaluator.png       — headline AUROC per evaluator (grouped by family)
  Fig 2: auroc_heatmap.png            — evaluator × source AUROC matrix
  Fig 3: within_vs_cross_family.png   — paired within/cross-family bars per evaluator
  Fig 4: roc_curves_by_evaluator.png  — ROC curves, one panel per evaluator
  Fig 5: mean_tamper_by_source.png    — mean tamper_prob per source per evaluator
  Fig 6: explanation_categories.png   — cue distribution, within vs cross family

Summary: results_dir/summary_table.csv
"""

import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from .config import EVALUATOR_FAMILIES, SOURCE_FAMILIES, Experiment0dConfig

_FAMILY_COLORS = {"claude": "steelblue", "openai": "tomato", "google": "seagreen"}
_SOURCE_PALETTE = {
    "sonnet_45":    "#4878CF",
    "gpt_4o_mini":  "#D65F5F",
    "gemini_25pro": "#59A14F",
    "self":         "#888888",
}

_EVAL_ORDER = ["opus_45", "opus_46", "sonnet_45", "gpt_4o_mini", "gemini_25pro"]
_EVAL_LABELS = {
    "opus_45":      "Opus 4.5",
    "opus_46":      "Opus 4.6",
    "sonnet_45":    "Sonnet 4.5",
    "gpt_4o_mini":  "GPT-4o-mini",
    "gemini_25pro": "Gemini 2.5 Pro",
}
_SRC_LABELS = {
    "sonnet_45":    "Sonnet 4.5",
    "gpt_4o_mini":  "GPT-4o-mini",
    "gemini_25pro": "Gemini 2.5 Pro",
    "self":         "Self (organic)",
}

_REASON_PATTERNS = [
    ("Style / tone",             r"style|tone|formal|informal|verbose|concise|way (i|you) (write|speak)"),
    ("Format / structure",       r"format|structur|bullet|markdown|heading|spacing|length"),
    ("Technical inconsistency",  r"inconsisten|approach differ|different method|wouldn't|would not"),
    ("Quality / capability",     r"qualit|capabil|sophisticat|level of|complexity"),
    ("Confident self-claim",     r"this is my|i (generated|wrote|produced)|recogni(se|ze)"),
    ("Uncertainty / vague",      r"unsure|uncertain|hard to (say|tell)|unclear"),
    ("Other",                    r".*"),
]


def run_analysis(results: list[dict], config: Experiment0dConfig) -> None:
    df = pd.DataFrame(results)
    if len(df) == 0:
        print("  No results to analyse.")
        return

    df["tamper_prob"] = pd.to_numeric(df["tamper_prob"], errors="coerce")
    det = df.dropna(subset=["tamper_prob"]).copy()
    if len(det) == 0:
        print("  No parseable results.")
        return

    evaluators = [e for e in _EVAL_ORDER if e in det["evaluator"].unique()]
    sources_present = [s for s in det["source"].unique() if s != "self"]

    _figure_auroc_by_evaluator(det, evaluators, config)
    _figure_auroc_heatmap(det, evaluators, sources_present, config)
    _figure_within_vs_cross(det, evaluators, config)
    _figure_roc_curves(det, evaluators, sources_present, config)
    _figure_mean_tamper_by_source(det, evaluators, config)
    _figure_explanation_categories(det, config)
    _save_summary(det, evaluators, sources_present, config)


# ---------------------------------------------------------------------------
# Fig 1: AUROC by evaluator (headline)
# ---------------------------------------------------------------------------

def _figure_auroc_by_evaluator(
    df: pd.DataFrame,
    evaluators: list[str],
    config: Experiment0dConfig,
) -> None:
    organic_by_ev: dict[str, pd.Series] = {
        ev: df[(df["evaluator"] == ev) & (df["source"] == "self")]["tamper_prob"].dropna()
        for ev in evaluators
    }
    sources = [s for s in df["source"].unique() if s != "self"]

    mean_aurocs, ci_los, ci_his, colors = [], [], [], []
    for ev in evaluators:
        org = organic_by_ev[ev]
        ev_aurocs = []
        for src in sources:
            sub = df[(df["evaluator"] == ev) & (df["source"] == src)]["tamper_prob"].dropna()
            if len(sub) < 5 or len(org) < 5:
                continue
            y_true  = [0] * len(org) + [1] * len(sub)
            y_score = list(org) + list(sub)
            if len(set(y_true)) < 2:
                continue
            try:
                ev_aurocs.append(roc_auc_score(y_true, y_score))
            except Exception:
                pass
        if not ev_aurocs:
            mean_aurocs.append(np.nan); ci_los.append(0); ci_his.append(0)
            colors.append("grey")
            continue
        mean_a = float(np.mean(ev_aurocs))
        # Bootstrap CI over mean of per-source AUROCs
        rng = np.random.default_rng(42)
        boot = [np.mean(rng.choice(ev_aurocs, size=len(ev_aurocs), replace=True))
                for _ in range(1000)]
        ci_lo, ci_hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
        mean_aurocs.append(mean_a); ci_los.append(mean_a - ci_lo); ci_his.append(ci_hi - mean_a)
        fam = EVALUATOR_FAMILIES.get(ev, "unknown")
        colors.append(_FAMILY_COLORS.get(fam, "grey"))

    x = np.arange(len(evaluators))
    fig, ax = plt.subplots(figsize=(max(8, len(evaluators) * 2), 5))
    bars = ax.bar(x, [v if not np.isnan(v) else 0 for v in mean_aurocs],
                  color=colors, edgecolor="white", width=0.6,
                  yerr=[ci_los, ci_his], capsize=5,
                  error_kw={"linewidth": 1.3, "capthick": 1.3})

    for bar, v in zip(bars, mean_aurocs):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.axhline(0.5, color="red", linestyle="--", linewidth=1.5, label="Chance (0.50)")

    # Family legend patches
    for fam, col in _FAMILY_COLORS.items():
        ax.bar([], [], color=col, label=fam.capitalize())

    ax.set_xticks(x)
    ax.set_xticklabels([_EVAL_LABELS.get(e, e) for e in evaluators], rotation=20, ha="right")
    ax.set_ylabel("Mean AUROC across 3 cross-family sources")
    ax.set_title(
        "Prefill Detection AUROC by Evaluator — Full Sweep\n"
        "(interleaved protocol; error bars = 95% CI; colour = model family)",
        fontsize=11,
    )
    ax.set_ylim(0.3, 1.05)
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = config.figures_dir / "auroc_by_evaluator.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Fig 2: AUROC heatmap
# ---------------------------------------------------------------------------

def _figure_auroc_heatmap(
    df: pd.DataFrame,
    evaluators: list[str],
    sources: list[str],
    config: Experiment0dConfig,
) -> None:
    matrix = np.full((len(evaluators), len(sources)), np.nan)

    for i, ev in enumerate(evaluators):
        org = df[(df["evaluator"] == ev) & (df["source"] == "self")]["tamper_prob"].dropna()
        for j, src in enumerate(sources):
            sub = df[(df["evaluator"] == ev) & (df["source"] == src)]["tamper_prob"].dropna()
            if len(sub) < 5 or len(org) < 5:
                continue
            y_true  = [0] * len(org) + [1] * len(sub)
            y_score = list(org) + list(sub)
            if len(set(y_true)) < 2:
                continue
            try:
                matrix[i, j] = roc_auc_score(y_true, y_score)
            except Exception:
                pass

    fig, ax = plt.subplots(figsize=(max(6, len(sources) * 2), max(4, len(evaluators) * 1.2)))
    cmap = plt.cm.RdYlGn
    im = ax.imshow(matrix, cmap=cmap, vmin=0.4, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(sources)))
    ax.set_xticklabels([_SRC_LABELS.get(s, s) for s in sources], rotation=20, ha="right")
    ax.set_yticks(range(len(evaluators)))
    ax.set_yticklabels([_EVAL_LABELS.get(e, e) for e in evaluators])

    # Annotate cells
    for i in range(len(evaluators)):
        for j in range(len(sources)):
            v = matrix[i, j]
            if not np.isnan(v):
                # Mark within-family cells with a border
                is_within = config.is_within_family(evaluators[i], sources[j])
                txt = f"{v:.2f}"
                weight = "bold" if is_within else "normal"
                color = "white" if abs(v - 0.7) > 0.2 else "black"
                ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                        color=color, fontweight=weight)

    plt.colorbar(im, ax=ax, label="AUROC", fraction=0.03)
    ax.set_title(
        "AUROC Heatmap — Evaluator × Source\n"
        "(bold = within-family; rows = evaluator; cols = source)",
        fontsize=10,
    )
    plt.tight_layout()
    out = config.figures_dir / "auroc_heatmap.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Fig 3: Within vs cross-family
# ---------------------------------------------------------------------------

def _figure_within_vs_cross(
    df: pd.DataFrame,
    evaluators: list[str],
    config: Experiment0dConfig,
) -> None:
    within_aurocs, cross_aurocs = [], []
    ev_labels = []

    for ev in evaluators:
        org = df[(df["evaluator"] == ev) & (df["source"] == "self")]["tamper_prob"].dropna()
        within, cross = [], []
        for src in df[df["evaluator"] == ev]["source"].unique():
            if src == "self":
                continue
            sub = df[(df["evaluator"] == ev) & (df["source"] == src)]["tamper_prob"].dropna()
            if len(sub) < 5 or len(org) < 5:
                continue
            y_true  = [0] * len(org) + [1] * len(sub)
            y_score = list(org) + list(sub)
            if len(set(y_true)) < 2:
                continue
            try:
                a = roc_auc_score(y_true, y_score)
                if config.is_within_family(ev, src):
                    within.append(a)
                else:
                    cross.append(a)
            except Exception:
                pass
        within_aurocs.append(float(np.mean(within)) if within else np.nan)
        cross_aurocs.append(float(np.mean(cross)) if cross else np.nan)
        ev_labels.append(_EVAL_LABELS.get(ev, ev))

    x = np.arange(len(evaluators))
    w = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(evaluators) * 2), 5))
    ax.bar(x - w/2, [v if not np.isnan(v) else 0 for v in within_aurocs], w,
           label="Within-family source", color="steelblue", edgecolor="white")
    ax.bar(x + w/2, [v if not np.isnan(v) else 0 for v in cross_aurocs], w,
           label="Cross-family source", color="tomato", edgecolor="white")

    # Mark missing within-family bars with "N/A"
    for i, v in enumerate(within_aurocs):
        if np.isnan(v):
            ax.text(x[i] - w/2, 0.52, "N/A", ha="center", fontsize=8, color="grey")

    ax.axhline(0.5, color="red", linestyle="--", linewidth=1.5, label="Chance (0.50)")
    ax.set_xticks(x)
    ax.set_xticklabels(ev_labels, rotation=20, ha="right")
    ax.set_ylabel("Mean AUROC")
    ax.set_title("Within-Family vs Cross-Family Detection\n"
                 "(within = same model family; cross = different family)")
    ax.set_ylim(0.3, 1.05)
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = config.figures_dir / "within_vs_cross_family.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Fig 4: ROC curves by evaluator
# ---------------------------------------------------------------------------

def _figure_roc_curves(
    df: pd.DataFrame,
    evaluators: list[str],
    sources: list[str],
    config: Experiment0dConfig,
) -> None:
    n = len(evaluators)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for idx, ev in enumerate(evaluators):
        ax = axes[idx // ncols][idx % ncols]
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
        org = df[(df["evaluator"] == ev) & (df["source"] == "self")]["tamper_prob"].dropna()

        for src in sources:
            sub = df[(df["evaluator"] == ev) & (df["source"] == src)]["tamper_prob"].dropna()
            if len(sub) < 5 or len(org) < 5:
                continue
            y_true  = [0] * len(org) + [1] * len(sub)
            y_score = list(org) + list(sub)
            if len(set(y_true)) < 2:
                continue
            try:
                auroc = roc_auc_score(y_true, y_score)
                fpr, tpr, _ = roc_curve(y_true, y_score)
                within = config.is_within_family(ev, src)
                ls = "--" if within else "-"
                ax.plot(fpr, tpr, color=_SOURCE_PALETTE.get(src, "grey"),
                        linewidth=2, linestyle=ls,
                        label=f"{_SRC_LABELS.get(src, src)} ({auroc:.2f})")
            except Exception:
                pass

        ax.set_title(_EVAL_LABELS.get(ev, ev), fontsize=9)
        ax.set_xlabel("FPR", fontsize=8)
        ax.set_ylabel("TPR", fontsize=8)
        ax.legend(fontsize=7, loc="lower right")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # Hide unused panels
    for idx in range(len(evaluators), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("ROC Curves by Evaluator\n(dashed = within-family source)", fontsize=11)
    plt.tight_layout()
    out = config.figures_dir / "roc_curves_by_evaluator.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Fig 5: Mean tamper probability by source
# ---------------------------------------------------------------------------

def _figure_mean_tamper_by_source(
    df: pd.DataFrame,
    evaluators: list[str],
    config: Experiment0dConfig,
) -> None:
    all_sources = ["self"] + [s for s in df["source"].unique() if s != "self"]
    n = len(evaluators)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows),
                             sharey=True, squeeze=False)

    for idx, ev in enumerate(evaluators):
        ax = axes[idx // ncols][idx % ncols]
        means = []
        srcs  = []
        colors = []
        for src in all_sources:
            sub = df[(df["evaluator"] == ev) & (df["source"] == src)]["tamper_prob"].dropna()
            if len(sub) == 0:
                continue
            means.append(sub.mean())
            srcs.append(src)
            colors.append(_SOURCE_PALETTE.get(src, "grey"))

        x = np.arange(len(srcs))
        bars = ax.bar(x, means, color=colors, edgecolor="white")
        ax.axhline(50, color="grey", linestyle=":", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels([_SRC_LABELS.get(s, s) for s in srcs],
                           rotation=20, ha="right", fontsize=7)
        ax.set_title(_EVAL_LABELS.get(ev, ev), fontsize=9)
        ax.set_ylim(0, 100)
        if idx % ncols == 0:
            ax.set_ylabel("Mean tamper prob", fontsize=8)

        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, m + 1.5,
                    f"{m:.0f}", ha="center", fontsize=7)

    for idx in range(len(evaluators), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("Mean Tamper Probability by Source\n"
                 "(grey = organic/self; organic should be low)", fontsize=10)
    plt.tight_layout()
    out = config.figures_dir / "mean_tamper_by_source.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Fig 6: Explanation categories (within vs cross family)
# ---------------------------------------------------------------------------

def _figure_explanation_categories(
    df: pd.DataFrame,
    config: Experiment0dConfig,
) -> None:
    groups = {"Within-family": [], "Cross-family": [], "Organic": []}
    for _, row in df.iterrows():
        src  = row.get("source", "")
        ev   = row.get("evaluator", "")
        reason = row.get("reason", "")
        if not isinstance(reason, str) or len(reason) < 5:
            continue
        if src == "self":
            groups["Organic"].append(reason.lower())
        elif config.is_within_family(ev, src):
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
    ax.set_title("Self-reported detection cues by source type\n"
                 "(normalised within each group; aggregated across all evaluators)")
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
    evaluators: list[str],
    sources: list[str],
    config: Experiment0dConfig,
) -> None:
    rows = []
    for ev in evaluators:
        org = df[(df["evaluator"] == ev) & (df["source"] == "self")]["tamper_prob"].dropna()
        for src in sources:
            sub = df[(df["evaluator"] == ev) & (df["source"] == src)]["tamper_prob"].dropna()
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
                "evaluator":       ev,
                "source":          src,
                "family":          "within" if config.is_within_family(ev, src) else "cross",
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
    print("\n  Summary:")
    print(summary.to_string(index=False))


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
    return float(np.percentile(aurocs, 100 * alpha / 2)), \
           float(np.percentile(aurocs, 100 * (1 - alpha / 2)))


def _r(v, d: int = 3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return round(float(v), d)
