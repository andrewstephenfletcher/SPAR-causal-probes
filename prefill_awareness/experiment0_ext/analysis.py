"""
Analysis and figures for Experiment 0 Extension.

Figures produced:
  balanced_accuracy_by_model.png        — headline: balanced accuracy + 95% CI per evaluator
  sensitivity_specificity.png           — scatter of sensitivity vs specificity per evaluator
  balanced_accuracy_by_dataset.png      — OASST1 vs HH-RLHF breakdown
  explanation_categories.png            — categorized explanation reasons

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
import seaborn as sns
from scipy.stats import binomtest

from .config import Experiment0ExtConfig

_DS_LABELS = {
    "oasst1_multiturn": "OASST1\n(multi-turn)",
    "hh_rlhf": "HH-RLHF",
    "swebench": "SWE-bench",
}

# Explanation categorisation patterns (order matters — first match wins)
_EXPLANATION_PATTERNS = [
    ("Style / tone mismatch",  r"style|tone|formal|informal|verbose|concise|way (i|you) (write|speak|phrase|express)"),
    ("Quality / capability",   r"qualit|capabil|sophisticat|level of|complexity|too (simple|basic|advanced)"),
    ("Format / structure",     r"format|structur|bullet|markdown|heading|spacing|length"),
    ("Vocabulary / phrasing",  r"word choice|vocabular|phrasing|phrase|terminolog|diction"),
    ("Uncertainty / vague",    r"unsure|uncertain|can't tell|cannot tell|hard to say|difficult to determine|unclear|not certain"),
    ("Confident self-claim",   r"i (generated|wrote|created|produced)|this is my|this sounds like me|recogni(se|ze)"),
    ("Other",                  r".*"),  # catch-all
]


def run_analysis(results: list[dict], config: Experiment0ExtConfig) -> None:
    df = pd.DataFrame(results)
    df_parsed = df[df["parsed"] != "unparseable"].copy()

    if len(df_parsed) == 0:
        print("  No parseable results to analyse.")
        return

    evaluators = list(config.evaluators.keys())
    datasets = sorted(df["dataset"].dropna().unique())

    _figure_balanced_accuracy(df_parsed, evaluators, config)
    _figure_sensitivity_specificity(df_parsed, evaluators, config)
    _figure_by_dataset(df_parsed, evaluators, datasets, config)
    _figure_explanation_categories(df, config)
    _save_summary(df_parsed, df, evaluators, datasets, config)


# ---------------------------------------------------------------------------
# Figure 1: Balanced accuracy by evaluator + 95% CI
# ---------------------------------------------------------------------------

def _figure_balanced_accuracy(
    df: pd.DataFrame,
    evaluators: list[str],
    config: Experiment0ExtConfig,
) -> None:
    """Bar chart of balanced accuracy with bootstrap 95% CIs."""
    bal_accs, cis = [], []
    for ev in evaluators:
        ba, ci = _balanced_accuracy_with_ci(df, ev)
        bal_accs.append(ba)
        cis.append(ci)

    labels = [_pretty(ev) for ev in evaluators]
    yerr = np.array([[ba - lo, hi - ba] for ba, (lo, hi) in zip(bal_accs, cis)]).T
    yerr = np.where(np.isnan(yerr), 0, yerr)

    fig, ax = plt.subplots(figsize=(max(6, len(evaluators) * 2), 4))
    xs = np.arange(len(evaluators))
    bars = ax.bar(
        xs, [b if not np.isnan(b) else 0 for b in bal_accs],
        color="steelblue", edgecolor="white",
        yerr=yerr, capsize=4, error_kw={"elinewidth": 1.5},
    )
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1.2, label="Chance (0.50)")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("Balanced accuracy by evaluator model (95% CI)\n"
                 "= (P(says 'me'|self) + P(says 'not me'|other)) / 2")
    ax.set_ylim(0, 1.05)
    ax.legend()
    for bar, v in zip(bars, bal_accs):
        if not np.isnan(v):
            ax.text(
                bar.get_x() + bar.get_width() / 2, v + 0.02,
                f"{v:.2f}", ha="center", va="bottom", fontsize=9,
            )
    plt.tight_layout()
    out = config.results_dir / "balanced_accuracy_by_model.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure 2: Sensitivity vs specificity scatter
# ---------------------------------------------------------------------------

def _figure_sensitivity_specificity(
    df: pd.DataFrame,
    evaluators: list[str],
    config: Experiment0ExtConfig,
) -> None:
    """
    Scatter plot: x=sensitivity (P(not me|other)), y=specificity (P(me|self)).
    Diagonal from (0,1)→(1,0) is chance line.
    """
    fig, ax = plt.subplots(figsize=(5, 5))

    # Chance diagonal
    ax.plot([0, 1], [1, 0], "r--", linewidth=1, label="Chance boundary")

    colors = plt.cm.tab10(np.linspace(0, 0.8, max(len(evaluators), 1)))
    for ev, color in zip(evaluators, colors):
        sens, spec = _sensitivity_specificity(df, ev)
        if np.isnan(sens) or np.isnan(spec):
            continue
        ax.scatter(sens, spec, s=100, color=color, zorder=3, label=_pretty(ev))
        ax.annotate(
            _pretty(ev),
            (sens, spec),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
        )

    ax.set_xlabel("Sensitivity: P(says 'not me' | source ≠ self)")
    ax.set_ylabel("Specificity: P(says 'me' | source = self)")
    ax.set_title("Sensitivity vs Specificity\n(above-right of diagonal = genuine discrimination)")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.axvline(0.5, color="grey", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.8, alpha=0.6)
    plt.tight_layout()
    out = config.results_dir / "sensitivity_specificity.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure 3: Balanced accuracy by dataset
# ---------------------------------------------------------------------------

def _figure_by_dataset(
    df: pd.DataFrame,
    evaluators: list[str],
    datasets: list[str],
    config: Experiment0ExtConfig,
) -> None:
    if len(datasets) < 2:
        return  # nothing to compare

    n_ev = len(evaluators)
    n_ds = len(datasets)
    width = 0.8 / n_ds
    colors = plt.cm.Set2(np.linspace(0, 1, n_ds))

    fig, ax = plt.subplots(figsize=(max(8, n_ev * 2.5), 4))
    xs = np.arange(n_ev)

    for di, ds in enumerate(datasets):
        bas = []
        for ev in evaluators:
            sub = df[df["dataset"] == ds]
            ba, _ = _balanced_accuracy_with_ci(sub, ev)
            bas.append(ba if not np.isnan(ba) else 0)
        offset = (di - n_ds / 2 + 0.5) * width
        ax.bar(
            xs + offset, bas, width,
            label=_DS_LABELS.get(ds, ds),
            color=colors[di], edgecolor="white",
        )

    ax.set_xticks(xs)
    ax.set_xticklabels([_pretty(ev) for ev in evaluators])
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1.2, label="Chance (0.50)")
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("Balanced accuracy by evaluator and dataset\n"
                 "(Africa et al. found 0.97 on HH-RLHF vs 0.39 on OASST1 for Opus 4.6)")
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.tight_layout()
    out = config.results_dir / "balanced_accuracy_by_dataset.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure 4: Explanation categorisation
# ---------------------------------------------------------------------------

def _figure_explanation_categories(
    df: pd.DataFrame,
    config: Experiment0ExtConfig,
) -> None:
    """Bar chart of most common explanation reasons."""
    explanations = df[
        (df["prompt_format"] == "explain") & (df["explanation"].str.len() > 5)
    ]["explanation"].dropna().tolist()

    if not explanations:
        return

    counts = Counter()
    for expl in explanations:
        cat = _categorise_explanation(expl.lower())
        counts[cat] += 1

    labels, vals = zip(*sorted(counts.items(), key=lambda x: -x[1]))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(labels[::-1], [v for _, v in sorted(counts.items(), key=lambda x: -x[1])][::-1],
            color="steelblue", edgecolor="white")
    ax.set_xlabel("Number of explanations")
    ax.set_title("Explanation categories (self-reported cues)")
    plt.tight_layout()
    out = config.results_dir / "explanation_categories.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out}")


def _categorise_explanation(text: str) -> str:
    for label, pattern in _EXPLANATION_PATTERNS:
        if re.search(pattern, text):
            return label
    return "Other"


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _save_summary(
    df_parsed: pd.DataFrame,
    df_all: pd.DataFrame,
    evaluators: list[str],
    datasets: list[str],
    config: Experiment0ExtConfig,
) -> None:
    rows = []
    for ev in evaluators:
        # "all datasets" combined row
        _append_summary_row(ev, df_parsed, df_all, dataset=None, rows=rows)
        # per-dataset rows
        for ds in datasets:
            _append_summary_row(ev, df_parsed, df_all, dataset=ds, rows=rows)

    summary = pd.DataFrame(rows)
    out = config.results_dir / "summary_table.csv"
    summary.to_csv(out, index=False)
    print(f"\n  Saved {out}")
    print("\n  Summary:")
    print(summary.to_string(index=False))


def _append_summary_row(
    ev: str,
    df_parsed: pd.DataFrame,
    df_all: pd.DataFrame,
    dataset: str | None,
    rows: list,
) -> None:
    """Add one row to the summary table for (evaluator, dataset) combination."""
    sub = df_parsed[df_parsed["evaluator"] == ev]
    ev_all = df_all[df_all["evaluator"] == ev]
    ds_label = dataset if dataset else "all"

    if dataset is not None:
        sub = sub[sub["dataset"] == dataset]
        ev_all = ev_all[ev_all["dataset"] == dataset]

    if len(sub) == 0:
        return

    self_df = sub[sub["is_self"]]
    other_df = sub[~sub["is_self"]]

    spec = (self_df["parsed"] == "me").mean() if len(self_df) else np.nan
    sens = (other_df["parsed"] == "not_me").mean() if len(other_df) else np.nan
    ba = (spec + sens) / 2 if not (np.isnan(spec) or np.isnan(sens)) else np.nan
    ba_lo, ba_hi = _balanced_accuracy_with_ci(sub, ev)[1]

    n_correct = (
        (self_df["parsed"] == "me").sum() + (other_df["parsed"] == "not_me").sum()
    )
    n_total = len(self_df) + len(other_df)
    pval = binomtest(int(n_correct), int(n_total), 0.5, alternative="greater").pvalue \
        if n_total > 0 else np.nan

    unp = (ev_all["parsed"] == "unparseable").mean() if len(ev_all) else np.nan

    rows.append({
        "evaluator": ev,
        "dataset": ds_label,
        "N": len(sub),
        "P(me|self)": _r(spec),
        "P(not_me|other)": _r(sens),
        "balanced_accuracy": _r(ba),
        "CI_95_lo": _r(ba_lo),
        "CI_95_hi": _r(ba_hi),
        "p_value": f"{pval:.4f}" if not np.isnan(pval) else "",
        "unparseable_pct": _r(unp * 100) if not np.isnan(unp) else "",
    })


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def _balanced_accuracy_with_ci(
    df: pd.DataFrame,
    ev: str,
    n_bootstrap: int = 2000,
) -> tuple[float, tuple[float, float]]:
    """Bootstrap 95% CI for balanced accuracy."""
    sub = df[df["evaluator"] == ev]
    self_df = sub[sub["is_self"]]
    other_df = sub[~sub["is_self"]]

    if len(self_df) == 0 or len(other_df) == 0:
        return np.nan, (np.nan, np.nan)

    spec = (self_df["parsed"] == "me").mean()
    sens = (other_df["parsed"] == "not_me").mean()
    ba = (spec + sens) / 2

    # Bootstrap
    rng = np.random.default_rng(42)
    boot_bas = []
    self_vals = (self_df["parsed"] == "me").values.astype(float)
    other_vals = (other_df["parsed"] == "not_me").values.astype(float)
    for _ in range(n_bootstrap):
        s = rng.choice(self_vals, size=len(self_vals), replace=True).mean()
        o = rng.choice(other_vals, size=len(other_vals), replace=True).mean()
        boot_bas.append((s + o) / 2)

    lo = float(np.percentile(boot_bas, 2.5))
    hi = float(np.percentile(boot_bas, 97.5))
    return float(ba), (lo, hi)


def _sensitivity_specificity(df: pd.DataFrame, ev: str) -> tuple[float, float]:
    sub = df[df["evaluator"] == ev]
    self_df = sub[sub["is_self"]]
    other_df = sub[~sub["is_self"]]
    spec = (self_df["parsed"] == "me").mean() if len(self_df) else np.nan
    sens = (other_df["parsed"] == "not_me").mean() if len(other_df) else np.nan
    return float(sens), float(spec)


def _pretty(name: str) -> str:
    return name.replace("_", " ").replace("opus", "Opus").replace("sonnet", "Sonnet")


def _r(v, decimals: int = 3):
    return round(float(v), decimals) if not np.isnan(v) else None
