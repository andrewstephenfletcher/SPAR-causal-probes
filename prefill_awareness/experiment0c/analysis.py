"""
Analysis for Experiment 0c.

Figures:
  Fig 1: auroc_by_condition.png         — headline AUROC per condition
  Fig 2: probability_distributions.png — tamper_prob histograms per condition
  Fig 3: style_alteration_detail.png   — Condition D mean tamper_prob per style
  Fig 4: explanation_categories.png    — self-reported cues per condition

Summary: results_dir/summary_table.csv
"""

import json
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import STYLE_ALTERATIONS, Experiment0cConfig

_COND_ORDER  = ["organic", "A", "B", "C", "D"]
_COND_LABELS = {
    "organic": "Organic\n(self)",
    "A":       "A: Within-\nfamily\n(Sonnet 4.5)",
    "B":       "B: Cross-fam\nfrontier\n(GPT-4o)",
    "C":       "C: Style\nimitation\n(GPT-4o+few)",
    "D":       "D: Self\naltered\n(Opus 4.6)",
}
_COND_COLORS = {
    "organic": "steelblue",
    "A":       "seagreen",
    "B":       "tomato",
    "C":       "darkorange",
    "D":       "goldenrod",
}

_REASON_PATTERNS = [
    ("Style / tone",             r"style|tone|formal|informal|verbose|concise|way (i|you) (write|speak)"),
    ("Format / structure",       r"format|structur|bullet|markdown|heading|spacing|length"),
    ("Technical inconsistency",  r"inconsisten|approach differ|different method|wouldn't|would not"),
    ("Quality / capability",     r"qualit|capabil|sophisticat|level of|complexity"),
    ("Confident self-claim",     r"this is my|i (generated|wrote|produced)|recogni(se|ze)"),
    ("Uncertainty / vague",      r"unsure|uncertain|hard to (say|tell)|unclear|not certain"),
    ("Other",                    r".*"),
]


def run_analysis(
    results: list[dict],
    config: Experiment0cConfig,
    exp0b_auroc: float | None = None,
) -> None:
    df = pd.DataFrame(results)
    if len(df) == 0:
        print("  No results to analyse.")
        return

    df["tamper_prob"] = pd.to_numeric(df["tamper_prob"], errors="coerce")
    det = df.dropna(subset=["tamper_prob"]).copy()

    if len(det) == 0:
        print("  No parseable detection results.")
        return

    conditions = [c for c in _COND_ORDER if c in det["condition"].unique()]
    tampered   = [c for c in conditions if c != "organic"]

    _figure_auroc_by_condition(det, tampered, config, exp0b_auroc)
    _figure_probability_distributions(det, conditions, config)
    if "D" in conditions:
        _figure_style_alteration_detail(det, config)
    _figure_explanation_categories(det, conditions, config)
    _save_summary(det, tampered, config, exp0b_auroc)


# ---------------------------------------------------------------------------
# Figure 1: AUROC by condition
# ---------------------------------------------------------------------------

def _figure_auroc_by_condition(
    df: pd.DataFrame,
    tampered_conditions: list[str],
    config: Experiment0cConfig,
    exp0b_auroc: float | None,
) -> None:
    organic = df[df["condition"] == "organic"]["tamper_prob"].dropna()
    aurocs, ci_los, ci_his, labels, colors = [], [], [], [], []

    for cond in tampered_conditions:
        sub = df[df["condition"] == cond]["tamper_prob"].dropna()
        if len(sub) < 5 or len(organic) < 5:
            continue
        y_true  = [0] * len(organic) + [1] * len(sub)
        y_score = list(organic) + list(sub)
        if len(set(y_true)) < 2:
            continue
        auroc = roc_auc_score(y_true, y_score)
        ci_lo, ci_hi = _bootstrap_auroc_ci(y_true, y_score)
        aurocs.append(auroc)
        ci_los.append(auroc - ci_lo)
        ci_his.append(ci_hi - auroc)
        labels.append(_COND_LABELS.get(cond, cond))
        colors.append(_COND_COLORS.get(cond, "grey"))

    if not aurocs:
        return

    x = np.arange(len(aurocs))
    fig, ax = plt.subplots(figsize=(max(6, len(aurocs) * 2.2), 5))
    bars = ax.bar(x, aurocs, color=colors, edgecolor="white", width=0.55,
                  yerr=[ci_los, ci_his], capsize=5,
                  error_kw={"linewidth": 1.3, "capthick": 1.3})

    for bar, v in zip(bars, aurocs):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.014,
                f"{v:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.axhline(0.5, color="red", linestyle="--", linewidth=1.5, label="Chance (0.50)")
    if exp0b_auroc is not None:
        ax.axhline(exp0b_auroc, color="black", linestyle=":", linewidth=1.5,
                   label=f"Exp 0b reference ({exp0b_auroc:.2f})")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("AUROC (tampered vs organic)", fontsize=11)
    ax.set_title(
        "AUROC by Condition — What Is Opus 4.6 Actually Detecting?\n"
        "(error bars = 95% bootstrap CI; organic = Opus 4.6 self-condition)",
        fontsize=11,
    )
    ax.set_ylim(0.3, 1.05)
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = config.figures_dir / "auroc_by_condition.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure 2: Probability distributions
# ---------------------------------------------------------------------------

def _figure_probability_distributions(
    df: pd.DataFrame,
    conditions: list[str],
    config: Experiment0cConfig,
) -> None:
    # For condition D, group all style alterations together
    plot_conditions = [c for c in conditions if c != "D"]
    if "D" in conditions:
        plot_conditions.append("D")

    n = len(plot_conditions)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    organic = df[df["condition"] == "organic"]["tamper_prob"].dropna()

    for ax, cond in zip(axes, plot_conditions):
        if cond == "organic":
            ax.hist(organic, bins=range(0, 105, 10), alpha=0.8,
                    color="steelblue", label="Organic", density=True)
        else:
            sub = df[df["condition"] == cond]["tamper_prob"].dropna()
            ax.hist(organic, bins=range(0, 105, 10), alpha=0.45,
                    color="steelblue", label="Organic", density=True)
            ax.hist(sub, bins=range(0, 105, 10), alpha=0.65,
                    color=_COND_COLORS.get(cond, "grey"),
                    label=f"Cond {cond}", density=True)

        ax.axvline(50, color="grey", linestyle=":", linewidth=1)
        ax.set_title(_COND_LABELS.get(cond, cond), fontsize=8)
        ax.set_xlabel("Tamper prob", fontsize=8)
        ax.set_ylabel("Density" if cond == plot_conditions[0] else "", fontsize=8)
        ax.legend(fontsize=7)

    fig.suptitle(
        "Tamper probability distributions by condition\n"
        "(organic shown in blue for reference)",
        y=1.02,
    )
    plt.tight_layout()
    out = config.figures_dir / "probability_distributions.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure 3: Condition D — per style alteration
# ---------------------------------------------------------------------------

def _figure_style_alteration_detail(
    df: pd.DataFrame,
    config: Experiment0cConfig,
) -> None:
    organic_mean = df[df["condition"] == "organic"]["tamper_prob"].mean()
    cond_d = df[df["condition"] == "D"]
    style_names = list(STYLE_ALTERATIONS.keys())

    means, errs_lo, errs_hi = [], [], []
    for style in style_names:
        sub = cond_d[cond_d["source_name"] == style]["tamper_prob"].dropna()
        if len(sub) == 0:
            means.append(np.nan); errs_lo.append(0); errs_hi.append(0)
            continue
        m = sub.mean()
        rng = np.random.default_rng(42)
        boot = [rng.choice(sub.values, size=len(sub), replace=True).mean()
                for _ in range(1000)]
        ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
        means.append(m)
        errs_lo.append(m - ci_lo)
        errs_hi.append(ci_hi - m)

    x = np.arange(len(style_names))
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(x, [m if not np.isnan(m) else 0 for m in means],
                  color=_COND_COLORS["D"], edgecolor="white",
                  yerr=[errs_lo, errs_hi], capsize=5,
                  error_kw={"linewidth": 1.3, "capthick": 1.3})

    ax.axhline(50, color="grey", linestyle=":", linewidth=1, label="Neutral (50)")
    ax.axhline(organic_mean, color="steelblue", linestyle="--", linewidth=1.5,
               label=f"Organic mean ({organic_mean:.1f})")

    for bar, m in zip(bars, means):
        if not np.isnan(m):
            ax.text(bar.get_x() + bar.get_width() / 2, m + 1.5,
                    f"{m:.1f}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(style_names, fontsize=10)
    ax.set_ylabel("Mean tamper probability")
    ax.set_title(
        "Condition D: Altered Style on Same Model (Opus 4.6)\n"
        "Does changing surface style on self trigger 'not me'?",
    )
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = config.figures_dir / "style_alteration_detail.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure 4: Explanation categories by condition
# ---------------------------------------------------------------------------

def _figure_explanation_categories(
    df: pd.DataFrame,
    conditions: list[str],
    config: Experiment0cConfig,
) -> None:
    cat_by_cond: dict[str, Counter] = {}
    for cond in conditions:
        sub = df[df["condition"] == cond]
        cats: Counter = Counter()
        for _, row in sub.iterrows():
            reason = row.get("reason", "")
            if isinstance(reason, str) and len(reason) > 5:
                cats[_categorise(reason.lower())] += 1
        cat_by_cond[cond] = cats

    all_cats = [label for label, _ in _REASON_PATTERNS]
    x = np.arange(len(all_cats))
    n = len(conditions)
    w = 0.7 / n

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, cond in enumerate(conditions):
        counts = [cat_by_cond[cond].get(cat, 0) for cat in all_cats]
        total = sum(counts) or 1
        freqs = [c / total for c in counts]
        offsets = x + (i - (n - 1) / 2) * w
        ax.bar(offsets, freqs, w,
               label=_COND_LABELS.get(cond, cond).replace("\n", " "),
               color=_COND_COLORS.get(cond, "grey"),
               edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(all_cats, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Fraction of responses")
    ax.set_title("Self-reported detection cues by condition (normalised within condition)")
    ax.legend(fontsize=8, ncol=3)
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
    tampered_conditions: list[str],
    config: Experiment0cConfig,
    exp0b_auroc: float | None,
) -> None:
    organic = df[df["condition"] == "organic"]["tamper_prob"].dropna()
    rows = []

    if exp0b_auroc is not None:
        rows.append({
            "condition": "0b_reference", "source": "mixed OSS",
            "N_organic": "—", "N_tampered": "—",
            "AUROC": _r(exp0b_auroc), "CI_lo": "—", "CI_hi": "—",
            "mean_P_organic": "—", "mean_P_tampered": "—", "delta_from_0b": "—",
        })

    for cond in tampered_conditions:
        sources = df[df["condition"] == cond]["source_name"].unique()

        # One row per source, plus a pooled row for multi-source conditions
        row_sources = list(sources)
        if len(sources) > 1:
            row_sources.append("_all_")

        for src in row_sources:
            if src == "_all_":
                sub = df[df["condition"] == cond]["tamper_prob"].dropna()
                src_label = "all"
            else:
                sub = df[(df["condition"] == cond) & (df["source_name"] == src)]["tamper_prob"].dropna()
                src_label = src

            if len(sub) < 3 or len(organic) < 3:
                continue

            y_true  = [0] * len(organic) + [1] * len(sub)
            y_score = list(organic) + list(sub)

            if len(set(y_true)) < 2:
                auroc = ci_lo = ci_hi = np.nan
            else:
                auroc = roc_auc_score(y_true, y_score)
                ci_lo, ci_hi = _bootstrap_auroc_ci(y_true, y_score)

            delta = (auroc - exp0b_auroc) if (exp0b_auroc is not None and not np.isnan(auroc)) else np.nan

            rows.append({
                "condition":      cond,
                "source":         src_label,
                "N_organic":      len(organic),
                "N_tampered":     len(sub),
                "AUROC":          _r(auroc),
                "CI_lo":          _r(ci_lo),
                "CI_hi":          _r(ci_hi),
                "mean_P_organic": _r(organic.mean()),
                "mean_P_tampered":_r(sub.mean()),
                "delta_from_0b":  _r(delta),
            })

    summary = pd.DataFrame(rows)
    out = config.results_dir / "summary_table.csv"
    summary.to_csv(out, index=False)
    print(f"\n  Saved {out}")
    print("\n  Summary:")
    print(summary.to_string(index=False))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bootstrap_auroc_ci(
    y_true: list[int],
    y_score: list[float],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
) -> tuple[float, float]:
    rng = np.random.default_rng(42)
    n = len(y_true)
    aurocs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        yt = [y_true[i] for i in idx]
        ys = [y_score[i] for i in idx]
        if len(set(yt)) < 2:
            continue
        aurocs.append(roc_auc_score(yt, ys))
    if not aurocs:
        return np.nan, np.nan
    return (
        float(np.percentile(aurocs, 100 * alpha / 2)),
        float(np.percentile(aurocs, 100 * (1 - alpha / 2))),
    )


def _r(v, decimals: int = 3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return round(float(v), decimals)


def load_exp0b_auroc(config: Experiment0cConfig) -> float | None:
    """Load Opus 4.6 agentic AUROC from Experiment 0b as the reference baseline."""
    path = config.exp0b_results_path
    if not path.exists():
        print(f"  Exp 0b results not found at {path}.")
        return None
    try:
        with open(path) as f:
            records = json.load(f)
        df = pd.DataFrame(records)
        df = df[
            (df["evaluator"] == "opus_46") &
            (df["dataset"] == "agentic") &
            (df["task_type"] == "detection")
        ].copy()
        df["tamper_prob"] = pd.to_numeric(df["tamper_prob"], errors="coerce")
        df = df.dropna(subset=["tamper_prob"])
        if len(df) < 5 or df["is_tampered"].nunique() < 2:
            return None
        return float(roc_auc_score(df["is_tampered"].astype(int), df["tamper_prob"]))
    except Exception as e:
        print(f"  Could not load Exp 0b AUROC: {e}")
        return None
