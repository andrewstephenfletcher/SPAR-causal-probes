"""
Analysis and figure generation for Experiment 5 (Causal Steering).

Figure 1 — attribution_bars.png
    Grouped bar chart: "not me" rate by prefill source (self / within_family /
    cross_family) and steering direction (steer-self / no-steer / steer-not-self).

Figure 2 — dose_response.png
    Dose-response: x=alpha (negative=self, positive=not-self), y="not me" rate,
    one line per prefill source.  Only plotted if multiple alpha values were run.

Figure 3 — sentiment_bars.png
    Grouped bar chart: mean numerical rating by text source (self / other) and
    steering direction.  Error bars = SEM.

Figure 4 — criticism_distribution.png
    Stacked bar chart: LLM-judge criticism level (none/mild/moderate/harsh)
    by (text_source, steering_direction).

Summary tables are also written as CSV.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .config import Experiment5Config

# ---------------------------------------------------------------------------
# Colour / label constants
# ---------------------------------------------------------------------------

_SOURCE_LABELS = {
    "self":         "Self (Llama 70B)",
    "within_family": "Within-family (Llama 8B)",
    "cross_family":  "Cross-family (Gemma 9B)",
}
_ALPHA_LABELS = {
    "positive": "Steer → not-self",
    "zero":     "No steering",
    "negative": "Steer → self",
}
_CRITICISM_ORDER = ["none", "mild", "moderate", "harsh"]
_CRITICISM_COLORS = {
    "none": "#4caf50", "mild": "#ff9800", "moderate": "#f44336", "harsh": "#9c27b0"
}


def _alpha_label(alpha: float, alpha_moderate: float) -> str:
    if abs(alpha) < 1e-6:
        return "zero"
    return "positive" if alpha > 0 else "negative"


# ---------------------------------------------------------------------------
# 5A analysis helpers
# ---------------------------------------------------------------------------

def compute_5a_rates(results_5a: list[dict], alpha_moderate: float) -> pd.DataFrame:
    """
    Returns DataFrame with columns:
      prefill_source, alpha_direction, not_me_rate, me_rate, unparseable_rate, n
    """
    rows = []
    for r in results_5a:
        r["alpha_dir"] = _alpha_label(r["alpha"], alpha_moderate)
    df = pd.DataFrame(results_5a)
    for (src, adir), grp in df.groupby(["prefill_source", "alpha_dir"]):
        n = len(grp)
        rows.append({
            "prefill_source": src,
            "alpha_direction": adir,
            "not_me_rate": (grp["parsed"] == "not_me").mean(),
            "me_rate":     (grp["parsed"] == "me").mean(),
            "unparseable_rate": (grp["parsed"] == "unparseable").mean(),
            "n": n,
        })
    return pd.DataFrame(rows)


def compute_steering_effect(rates_df: pd.DataFrame) -> pd.DataFrame:
    """
    Steering effect = not_me_rate(steer-not-self) − not_me_rate(steer-self).
    """
    rows = []
    for src, grp in rates_df.groupby("prefill_source"):
        pos = grp[grp["alpha_direction"] == "positive"]["not_me_rate"].values
        neg = grp[grp["alpha_direction"] == "negative"]["not_me_rate"].values
        zer = grp[grp["alpha_direction"] == "zero"]["not_me_rate"].values
        effect = (pos[0] - neg[0]) if len(pos) and len(neg) else float("nan")
        rows.append({
            "prefill_source": src,
            "effect": effect,
            "not_me_rate_positive": pos[0] if len(pos) else float("nan"),
            "not_me_rate_zero": zer[0] if len(zer) else float("nan"),
            "not_me_rate_negative": neg[0] if len(neg) else float("nan"),
        })
    return pd.DataFrame(rows)


def _binomial_ci(p: float, n: int, z: float = 1.96) -> float:
    """Half-width of Wald 95% CI for a proportion."""
    if n == 0:
        return 0.0
    return z * np.sqrt(p * (1 - p) / n)


# ---------------------------------------------------------------------------
# Figure 1: Attribution bars
# ---------------------------------------------------------------------------

def figure1_attribution_bars(
    rates_df: pd.DataFrame,
    config: Experiment5Config,
) -> None:
    sources = ["self", "within_family", "cross_family"]
    adirs = ["positive", "zero", "negative"]
    colors = {"positive": "#e74c3c", "zero": "#95a5a6", "negative": "#3498db"}

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(sources))
    width = 0.25

    for i, adir in enumerate(adirs):
        sub = rates_df[rates_df["alpha_direction"] == adir].set_index("prefill_source")
        heights, errs = [], []
        for src in sources:
            if src in sub.index:
                p = sub.loc[src, "not_me_rate"]
                n = sub.loc[src, "n"]
                heights.append(p)
                errs.append(_binomial_ci(p, n))
            else:
                heights.append(0.0)
                errs.append(0.0)

        ax.bar(
            x + (i - 1) * width, heights, width,
            yerr=errs, capsize=4,
            color=colors[adir], label=_ALPHA_LABELS[adir],
            alpha=0.85, edgecolor="black", linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([_SOURCE_LABELS[s] for s in sources], fontsize=10)
    ax.set_ylabel('"Not me" rate', fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.5,
               label="50% chance")
    ax.legend(fontsize=9)
    ax.set_title("Experiment 5A: Attribution steering effect", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out = config.results_dir_ex5 / "fig1_attribution_bars.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Figure 1 saved → {out}")


# ---------------------------------------------------------------------------
# Figure 2: Dose-response (multi-alpha)
# ---------------------------------------------------------------------------

def figure2_dose_response(
    results_5a: list[dict],
    config: Experiment5Config,
) -> None:
    df = pd.DataFrame(results_5a)
    alphas_sorted = sorted(df["alpha"].unique())
    if len(alphas_sorted) < 3:
        print("  Skipping Figure 2: fewer than 3 distinct alpha values.")
        return

    sources = ["self", "within_family", "cross_family"]
    colors = {"self": "#e74c3c", "within_family": "#f39c12", "cross_family": "#2ecc71"}

    fig, ax = plt.subplots(figsize=(7, 5))

    for src in sources:
        sub = df[df["prefill_source"] == src]
        xs, ys = [], []
        for alpha in alphas_sorted:
            grp = sub[sub["alpha"] == alpha]
            if len(grp):
                xs.append(alpha)
                ys.append((grp["parsed"] == "not_me").mean())
        ax.plot(xs, ys, marker="o", label=_SOURCE_LABELS[src], color=colors[src])

    ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Steering alpha (negative = toward self, positive = toward not-self)",
                  fontsize=10)
    ax.set_ylabel('"Not me" rate', fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.set_title("Experiment 5A: Dose-response", fontsize=12)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = config.results_dir_ex5 / "fig2_dose_response.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Figure 2 saved → {out}")


# ---------------------------------------------------------------------------
# Figure 3: Sentiment bars (mean rating)
# ---------------------------------------------------------------------------

def figure3_sentiment_bars(
    judged: list[dict],
    alpha_moderate: float,
    config: Experiment5Config,
) -> None:
    for r in judged:
        r["alpha_dir"] = _alpha_label(r["alpha"], alpha_moderate)
        # Prefer regex rating (direct parse), fall back to judge
        r["rating"] = r.get("regex_rating") or r.get("judge_numerical_rating")

    df = pd.DataFrame([r for r in judged if r["rating"] is not None])
    if df.empty:
        print("  Skipping Figure 3: no ratings extracted.")
        return

    sources = ["self", "other"]
    adirs = ["positive", "zero", "negative"]
    colors = {"positive": "#e74c3c", "zero": "#95a5a6", "negative": "#3498db"}

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(sources))
    width = 0.25

    for i, adir in enumerate(adirs):
        sub = df[df["alpha_dir"] == adir]
        heights, errs = [], []
        for src in sources:
            grp = sub[sub["text_source"] == src]["rating"]
            if len(grp):
                heights.append(grp.mean())
                errs.append(stats.sem(grp))
            else:
                heights.append(0.0)
                errs.append(0.0)

        ax.bar(
            x + (i - 1) * width, heights, width,
            yerr=errs, capsize=4,
            color=colors[adir], label=_ALPHA_LABELS[adir],
            alpha=0.85, edgecolor="black", linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(["Self-generated text", "Other-generated text"], fontsize=11)
    ax.set_ylabel("Mean quality rating (1-10)", fontsize=11)
    ax.set_ylim(0, 10.5)
    ax.legend(fontsize=9)
    ax.set_title("Experiment 5B: Sentiment steering effect", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out = config.results_dir_ex5 / "fig3_sentiment_bars.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Figure 3 saved → {out}")


# ---------------------------------------------------------------------------
# Figure 4: Criticism distribution (stacked bars)
# ---------------------------------------------------------------------------

def figure4_criticism_distribution(
    judged: list[dict],
    alpha_moderate: float,
    config: Experiment5Config,
) -> None:
    for r in judged:
        r["alpha_dir"] = _alpha_label(r["alpha"], alpha_moderate)

    df = pd.DataFrame([r for r in judged if r.get("judge_criticism")])
    if df.empty:
        print("  Skipping Figure 4: no judge_criticism data.")
        return

    # Columns: (text_source, alpha_dir) — rows
    group_keys = [
        ("self", "positive"), ("self", "zero"), ("self", "negative"),
        ("other", "positive"), ("other", "zero"), ("other", "negative"),
    ]
    x_labels = [
        f"Self\n{_ALPHA_LABELS[d]}" for (_, d) in group_keys[:3]
    ] + [
        f"Other\n{_ALPHA_LABELS[d]}" for (_, d) in group_keys[3:]
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    bottoms = np.zeros(len(group_keys))

    for crit in _CRITICISM_ORDER:
        fracs = []
        for src, adir in group_keys:
            grp = df[(df["text_source"] == src) & (df["alpha_dir"] == adir)]
            frac = (grp["judge_criticism"] == crit).mean() if len(grp) else 0.0
            fracs.append(frac)
        ax.bar(
            np.arange(len(group_keys)), fracs, bottom=bottoms,
            color=_CRITICISM_COLORS[crit], label=crit.capitalize(),
        )
        bottoms += np.array(fracs)

    # Separator between self / other groups
    ax.axvline(2.5, color="black", linewidth=1.2, linestyle=":")

    ax.set_xticks(np.arange(len(group_keys)))
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_ylabel("Fraction of responses", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(title="Criticism level", fontsize=9, loc="upper right")
    ax.set_title("Experiment 5B: Criticism level distribution", fontsize=12)
    fig.tight_layout()

    out = config.results_dir_ex5 / "fig4_criticism_distribution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Figure 4 saved → {out}")


# ---------------------------------------------------------------------------
# Figure 5: Random-vector control comparison
# ---------------------------------------------------------------------------

def figure5_random_control(
    results_5a: list[dict],
    random_control: list[dict],
    config: Experiment5Config,
) -> None:
    """
    Side-by-side comparison of probe steering vs. random-vector control,
    both on the self-prefill condition.  A systematic effect in the random
    arm would indicate alpha is too large; no effect is the desired outcome.
    """
    if not config.load_alphas_from_calibration():
        print("  Skipping Figure 5: calibration not found.")
        return
    alpha_moderate = config.alpha_moderate

    adirs = ["negative", "zero", "positive"]
    colors = {"positive": "#e74c3c", "zero": "#95a5a6", "negative": "#3498db"}

    # Restrict main 5A to self condition only
    probe_df = pd.DataFrame([r for r in results_5a if r["prefill_source"] == "self"])
    rand_df = pd.DataFrame(random_control)

    if probe_df.empty or rand_df.empty:
        print("  Skipping Figure 5: missing data.")
        return

    for df in (probe_df, rand_df):
        df["alpha_dir"] = df["alpha"].apply(lambda a: _alpha_label(a, alpha_moderate))

    groups = ["Probe\n(self prefill)", "Random vector\n(self prefill)"]
    dfs = [probe_df, rand_df]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(groups))
    width = 0.22

    for i, adir in enumerate(adirs):
        heights, errs = [], []
        for df in dfs:
            grp = df[df["alpha_dir"] == adir]
            p = (grp["parsed"] == "not_me").mean() if len(grp) else 0.0
            n = len(grp)
            heights.append(p)
            errs.append(_binomial_ci(p, n))
        ax.bar(
            x + (i - 1) * width, heights, width,
            yerr=errs, capsize=4,
            color=colors[adir], label=_ALPHA_LABELS[adir],
            alpha=0.85, edgecolor="black", linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=11)
    ax.set_ylabel('"Not me" rate', fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.5,
               label="50% chance")
    ax.legend(fontsize=9)
    ax.set_title("Experiment 5A: Probe vs. random-vector control", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out = config.results_dir_ex5 / "fig5_random_control.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Figure 5 saved → {out}")


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------

def write_summary_tables(
    rates_df: pd.DataFrame,
    effects_df: pd.DataFrame,
    judged: list[dict],
    alpha_moderate: float,
    config: Experiment5Config,
) -> None:
    rates_df.to_csv(config.results_dir_ex5 / "summary_5a_rates.csv", index=False)
    effects_df.to_csv(config.results_dir_ex5 / "summary_5a_effects.csv", index=False)

    # 5B summary table
    for r in judged:
        r["alpha_dir"] = _alpha_label(r["alpha"], alpha_moderate)
        r["rating"] = r.get("regex_rating") or r.get("judge_numerical_rating")

    df5b = pd.DataFrame(judged)
    if "rating" in df5b.columns and df5b["rating"].notna().any():
        tbl = (
            df5b[df5b["rating"].notna()]
            .groupby(["text_source", "alpha_dir"])["rating"]
            .agg(["mean", "sem", "count"])
            .reset_index()
        )
        tbl.to_csv(config.results_dir_ex5 / "summary_5b_ratings.csv", index=False)

    print(f"  Summary tables saved → {config.results_dir_ex5}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_all_figures(
    results_5a: list[dict],
    judged_5b: list[dict],
    config: Experiment5Config,
) -> None:
    """Generate all four figures and summary tables."""
    if not config.load_alphas_from_calibration():
        raise ValueError("Calibration not found. Run --from-step calibrate first.")
    alpha_moderate = config.alpha_moderate

    # 5A analysis
    print("\n  Computing 5A attribution rates...")
    rates_df = compute_5a_rates(results_5a, alpha_moderate)
    effects_df = compute_steering_effect(rates_df)

    print("\n  Steering effects (not_me_rate[+] − not_me_rate[-]):")
    for _, row in effects_df.iterrows():
        print(f"    {row['prefill_source']:15s}: effect={row['effect']:+.3f}  "
              f"(+alpha={row['not_me_rate_positive']:.3f}, "
              f"0={row['not_me_rate_zero']:.3f}, "
              f"-alpha={row['not_me_rate_negative']:.3f})")

    figure1_attribution_bars(rates_df, config)
    figure2_dose_response(results_5a, config)

    if judged_5b:
        figure3_sentiment_bars(judged_5b, alpha_moderate, config)
        figure4_criticism_distribution(judged_5b, alpha_moderate, config)

    random_path = config.generations_dir_ex5 / "random_control_5a.json"
    if random_path.exists():
        with open(random_path) as f:
            random_control = json.load(f)
        figure5_random_control(results_5a, random_control, config)
    else:
        print("  Skipping Figure 5: random_control_5a.json not found.")

    write_summary_tables(rates_df, effects_df, judged_5b, alpha_moderate, config)
