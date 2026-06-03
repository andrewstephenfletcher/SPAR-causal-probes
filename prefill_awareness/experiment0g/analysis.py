"""
Analysis for Experiment 0g: Style transfer against Opus 4.5 detection.

Figures:
  Fig 1: auroc_by_method.png         — AUROC per method, faceted by target
  Fig 2: probability_distributions.png — tamper prob distributions per method/target
  Fig 3: explanation_comparison.png  — detection cue breakdown per method

Summary: results_dir/summary_table.csv
"""

import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import Experiment0gConfig

_METHOD_ORDER = ["A", "B", "C", "AC"]

_METHOD_PALETTE = {
    "A":       "#4878CF",
    "B":       "#59A14F",
    "C":       "#F28E2B",
    "AC":      "#E15759",
    "organic": "#888888",
}
_METHOD_LABELS = {
    "A":  "A: Fewshot",
    "B":  "B: Rewrite",
    "C":  "C: Style",
    "AC": "A+C: Combined",
}
_TARGET_LABELS = {
    "gemini_pro": "Gemini 2.5 Pro",
    "gpt_5":      "GPT-5",
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


def run_analysis(results: list[dict], config: Experiment0gConfig) -> None:
    df = pd.DataFrame(results)
    if len(df) == 0:
        print("  No results to analyse.")
        return

    df["tamper_prob"] = pd.to_numeric(df["tamper_prob"], errors="coerce")
    det = df.dropna(subset=["tamper_prob"]).copy()
    if len(det) == 0:
        print("  No parseable results.")
        return

    targets  = [t for t in config.target_sources if
                any(det["condition"].str.startswith(t))]
    methods  = [m for m in _METHOD_ORDER if
                any(det["method"] == m)]

    _figure_auroc_by_method(det, targets, methods, config)
    _figure_probability_distributions(det, targets, methods, config)
    _figure_explanation_comparison(det, targets, methods, config)
    _save_summary(det, targets, methods, config)


# ---------------------------------------------------------------------------
# Fig 1: AUROC by method, faceted by target (side-by-side subplots)
# ---------------------------------------------------------------------------

def _figure_auroc_by_method(
    df: pd.DataFrame,
    targets: list[str],
    methods: list[str],
    config: Experiment0gConfig,
) -> None:
    org = df[df["condition"] == "organic"]["tamper_prob"].dropna()
    n_targets = len(targets)
    fig, axes = plt.subplots(1, n_targets, figsize=(6 * n_targets, 5), sharey=True)
    if n_targets == 1:
        axes = [axes]

    baseline_aurocs = _get_0f_baseline_aurocs(config)

    for ax, target_name in zip(axes, targets):
        aurocs, ci_los, ci_his, colors = [], [], [], []
        for method in methods:
            cond = f"{target_name}_{method}"
            sub = df[df["condition"] == cond]["tamper_prob"].dropna()
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
            colors.append(_METHOD_PALETTE.get(method, "grey"))

        x = np.arange(len(methods))
        bars = ax.bar(
            x, [v if not np.isnan(v) else 0 for v in aurocs],
            color=colors, edgecolor="white", width=0.5,
            yerr=[ci_los, ci_his], capsize=6,
            error_kw={"linewidth": 1.3, "capthick": 1.3},
        )
        for bar, v in zip(bars, aurocs):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012,
                        f"{v:.3f}", ha="center", va="bottom",
                        fontsize=10, fontweight="bold")

        ax.axhline(0.5, color="red", linestyle="--", linewidth=1.5, label="Chance")
        bl = baseline_aurocs.get(target_name)
        if bl is not None:
            ax.axhline(bl, color="#555555", linestyle=":", linewidth=2.0,
                       label=f"0f baseline: {bl:.3f}")

        ax.set_xticks(x)
        ax.set_xticklabels([_METHOD_LABELS.get(m, m) for m in methods], fontsize=9)
        ax.set_title(_TARGET_LABELS.get(target_name, target_name), fontsize=11)
        ax.set_ylim(0.2, 1.12)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("AUROC")
    fig.suptitle(
        "Opus 4.5 Detection AUROC — Style Transfer Methods\n"
        "(BigCodeBench; error bars = 95% CI bootstrap)",
        fontsize=11,
    )
    plt.tight_layout()
    out = config.figures_dir / "auroc_by_method.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Fig 2: Probability distributions per target
# ---------------------------------------------------------------------------

def _figure_probability_distributions(
    df: pd.DataFrame,
    targets: list[str],
    methods: list[str],
    config: Experiment0gConfig,
) -> None:
    n_targets = len(targets)
    fig, axes = plt.subplots(1, n_targets, figsize=(6 * n_targets, 4), sharey=False)
    if n_targets == 1:
        axes = [axes]

    org = df[df["condition"] == "organic"]["tamper_prob"].dropna()

    for ax, target_name in zip(axes, targets):
        ax.hist(org, bins=20, range=(0, 100), alpha=0.6,
                label="Organic", color=_METHOD_PALETTE["organic"])
        for method in methods:
            cond = f"{target_name}_{method}"
            data = df[df["condition"] == cond]["tamper_prob"].dropna()
            if len(data) < 3:
                continue
            ax.hist(data, bins=20, range=(0, 100), alpha=0.55,
                    label=_METHOD_LABELS.get(method, method),
                    color=_METHOD_PALETTE.get(method, "grey"))
        ax.set_title(_TARGET_LABELS.get(target_name, target_name), fontsize=10)
        ax.set_xlabel("Tamper probability (0–100)", fontsize=8)
        ax.set_ylabel("Count", fontsize=8)
        ax.legend(fontsize=7)

    fig.suptitle(
        "Tamper Probability Distributions — Style Transfer Methods\n"
        "(BigCodeBench, Opus 4.5 evaluator)",
        fontsize=10,
    )
    plt.tight_layout()
    out = config.figures_dir / "probability_distributions.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Fig 3: Explanation cue breakdown (organic + per-method, collapsed across targets)
# ---------------------------------------------------------------------------

def _figure_explanation_comparison(
    df: pd.DataFrame,
    targets: list[str],
    methods: list[str],
    config: Experiment0gConfig,
) -> None:
    # Collapse across targets: group by method
    groups: dict[str, list[str]] = {"organic": []}
    for method in methods:
        groups[method] = []

    for _, row in df.iterrows():
        cond   = row.get("condition", "")
        reason = row.get("reason", "")
        if not isinstance(reason, str) or len(reason) < 5:
            continue
        method = row.get("method", "")
        if cond == "organic":
            groups["organic"].append(reason.lower())
        elif method in groups:
            groups[method].append(reason.lower())

    all_cats = [label for label, _ in _REASON_PATTERNS]
    x = np.arange(len(all_cats))
    show_groups = ["organic"] + methods
    n = len(show_groups)
    w = 0.75 / n

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, grp in enumerate(show_groups):
        reasons = groups.get(grp, [])
        cats = Counter(_categorise(r) for r in reasons)
        total = sum(cats.values()) or 1
        freqs = [cats.get(c, 0) / total for c in all_cats]
        offsets = x + (i - (n - 1) / 2) * w
        label = "Organic" if grp == "organic" else _METHOD_LABELS.get(grp, grp)
        ax.bar(offsets, freqs, w, label=label,
               color=_METHOD_PALETTE.get(grp, "grey"), edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(all_cats, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Fraction of responses")
    ax.set_title(
        "Self-reported detection cues by method (collapsed across targets)\n"
        "(normalised within each group)",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = config.figures_dir / "explanation_comparison.png"
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
    targets: list[str],
    methods: list[str],
    config: Experiment0gConfig,
) -> None:
    org = df[df["condition"] == "organic"]["tamper_prob"].dropna()
    rows = []

    for target_name in targets:
        for method in methods:
            cond = f"{target_name}_{method}"
            sub = df[df["condition"] == cond]["tamper_prob"].dropna()
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
                "target":          target_name,
                "method":          method,
                "description":     _METHOD_LABELS.get(method, method),
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


def _get_0f_baseline_aurocs(config: Experiment0gConfig) -> dict[str, float]:
    baselines: dict[str, float] = {}
    try:
        import json
        path = config.experiment0f_dir / "results" / "detection_results.json"
        if not path.exists():
            return baselines
        with open(path) as f:
            records = json.load(f)
        df = pd.DataFrame(records)
        df = df[df["dataset"] == "bigcodebench"].copy()
        df["tamper_prob"] = pd.to_numeric(df["tamper_prob"], errors="coerce")
        df = df.dropna(subset=["tamper_prob"])
        org = df[df["source"] == "organic"]["tamper_prob"]

        for target_name in config.target_sources:
            sub = df[df["source"] == target_name]["tamper_prob"]
            if len(sub) < 5 or len(org) < 5:
                continue
            y = [0] * len(org) + [1] * len(sub)
            s = list(org) + list(sub)
            try:
                baselines[target_name] = float(roc_auc_score(y, s))
            except Exception:
                pass
    except Exception:
        pass
    return baselines


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
