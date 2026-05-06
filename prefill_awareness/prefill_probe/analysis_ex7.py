"""
Figure generation for Experiment 7 (Implicit Behavioral Measures).

Figures:
  1. Box plot — 0-100 self-ratings by condition (llama70b, llama8b, gemma9b),
     with pairwise significance annotations.
  2. Rating vs. judge quality scatter — does self-rating track actual quality,
     or is there a systematic upward bias for own responses?
  3. Defense rate bar chart — fraction of challenge prompts where 70B defended
     vs. revised, broken down by initial rating quartile.
  4. Rating vs. NLL scatter — self-assigned rating vs. perplexity (NLL) of
     the response under 70B (lower NLL = more "native" to the model).

Saves figures to results_dir_ex7/figures/.
Saves summary_ex7.txt with key statistics.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats

from .config import Experiment7Config


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_ratings(config: Experiment7Config) -> pd.DataFrame | None:
    path = config.results_dir_ex7 / "ratings_ex7.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return pd.DataFrame(data)


def _load_quality(config: Experiment7Config) -> pd.DataFrame | None:
    path = config.results_dir_ex7 / "judge_quality_ex7.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return pd.DataFrame(data)


def _load_perplexity(config: Experiment7Config) -> pd.DataFrame | None:
    path = config.results_dir_ex7 / "perplexity_ex7.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return pd.DataFrame(data)


def _load_challenge(config: Experiment7Config) -> pd.DataFrame | None:
    path = config.results_dir_ex7 / "challenge_ex7.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return pd.DataFrame(data)


def _figures_dir(config: Experiment7Config) -> Path:
    d = config.results_dir_ex7 / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


_CONDITION_LABELS = {
    "llama70b": "Llama 70B\n(own)",
    "llama8b":  "Llama 8B\n(foreign)",
    "gemma9b":  "Gemma 9B\n(foreign)",
}
_CONDITION_COLORS = {
    "llama70b": "#2196F3",
    "llama8b":  "#FF9800",
    "gemma9b":  "#4CAF50",
}


# ---------------------------------------------------------------------------
# Figure 1: Box plot of ratings by condition
# ---------------------------------------------------------------------------

def figure1_rating_boxplot(ratings: pd.DataFrame, config: Experiment7Config) -> None:
    figs_dir = _figures_dir(config)
    conditions = ["llama70b", "llama8b", "gemma9b"]
    data_by_cond = [
        ratings[ratings["condition"] == c]["score"].dropna().tolist()
        for c in conditions
    ]

    fig, ax = plt.subplots(figsize=(8, 6))
    positions = range(len(conditions))
    bp = ax.boxplot(data_by_cond, positions=list(positions), patch_artist=True, widths=0.5)

    for patch, cond in zip(bp["boxes"], conditions):
        patch.set_facecolor(_CONDITION_COLORS[cond])
        patch.set_alpha(0.7)

    # Overlay individual points
    for i, (data, cond) in enumerate(zip(data_by_cond, conditions)):
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(data))
        ax.scatter(
            [i + 1 + j for j in jitter], data,
            color=_CONDITION_COLORS[cond], alpha=0.3, s=15, zorder=2,
        )

    # Print means above boxes
    for i, data in enumerate(data_by_cond):
        if data:
            ax.text(i + 1, max(data) + 2, f"μ={np.mean(data):.1f}",
                    ha="center", va="bottom", fontsize=9)

    # Pairwise Mann-Whitney U tests
    pairs = [(0, 1), (0, 2), (1, 2)]
    y_top = 105
    for (a_idx, b_idx) in pairs:
        if data_by_cond[a_idx] and data_by_cond[b_idx]:
            _, p = stats.mannwhitneyu(
                data_by_cond[a_idx], data_by_cond[b_idx], alternative="two-sided"
            )
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            x1, x2 = a_idx + 1, b_idx + 1
            ax.plot([x1, x1, x2, x2], [y_top, y_top + 2, y_top + 2, y_top],
                    lw=1.0, color="black")
            ax.text((x1 + x2) / 2, y_top + 2.5, sig, ha="center", va="bottom", fontsize=9)
            y_top += 8

    ax.set_xticks(range(1, len(conditions) + 1))
    ax.set_xticklabels([_CONDITION_LABELS[c] for c in conditions])
    ax.set_ylabel("Self-Assigned Rating (0–100)", fontsize=11)
    ax.set_title("Experiment 7: Llama 70B Self-Rating by Response Condition", fontsize=12)
    ax.set_ylim(-5, y_top + 5)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out = figs_dir / "figure1_rating_boxplot.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure 1 saved → {out}")


# ---------------------------------------------------------------------------
# Figure 2: Rating vs. judge quality scatter
# ---------------------------------------------------------------------------

def figure2_rating_vs_judge(
    ratings: pd.DataFrame,
    quality: pd.DataFrame,
    config: Experiment7Config,
) -> None:
    figs_dir = _figures_dir(config)
    merged = ratings.merge(
        quality.rename(columns={"overall_score": "judge_score"}),
        on=["prompt_id", "condition"],
    ).dropna(subset=["score", "judge_score"])

    if merged.empty:
        print("  Skipping Figure 2: no merged rating+judge data.")
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    for cond in ["llama70b", "llama8b", "gemma9b"]:
        sub = merged[merged["condition"] == cond]
        ax.scatter(
            sub["judge_score"], sub["score"],
            color=_CONDITION_COLORS[cond], alpha=0.5, s=25,
            label=_CONDITION_LABELS[cond].replace("\n", " "),
        )

    # Diagonal (perfect calibration)
    ax.plot([0, 100], [0, 100], "k--", linewidth=0.8, label="Perfect calibration")

    # Per-condition regression lines
    for cond in ["llama70b", "llama8b", "gemma9b"]:
        sub = merged[merged["condition"] == cond]
        if len(sub) >= 5:
            m, b, r, p, _ = stats.linregress(sub["judge_score"], sub["score"])
            x_range = np.linspace(sub["judge_score"].min(), sub["judge_score"].max(), 50)
            ax.plot(x_range, m * x_range + b, color=_CONDITION_COLORS[cond], linewidth=1.5)

    ax.set_xlabel("Judge Quality Score (0–100)", fontsize=11)
    ax.set_ylabel("Self-Rating (0–100)", fontsize=11)
    ax.set_title("Experiment 7: Self-Rating vs. Independent Quality Score", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = figs_dir / "figure2_rating_vs_judge.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure 2 saved → {out}")


# ---------------------------------------------------------------------------
# Figure 3: Defense rate in challenge experiment
# ---------------------------------------------------------------------------

def figure3_defense_rates(challenge: pd.DataFrame, config: Experiment7Config) -> None:
    figs_dir = _figures_dir(config)
    if challenge.empty or "defended" not in challenge.columns:
        print("  Skipping Figure 3: no challenge data.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Left: overall defense vs. revision
    n_defended = int(challenge["defended"].sum())
    n_revised  = int(challenge["revised"].sum())
    n_neither  = len(challenge) - n_defended - n_revised
    ax = axes[0]
    bars = ax.bar(
        ["Defended", "Revised", "Neither"],
        [n_defended, n_revised, n_neither],
        color=["#2196F3", "#F44336", "#9E9E9E"],
    )
    for bar, val in zip(bars, [n_defended, n_revised, n_neither]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            str(val),
            ha="center", va="bottom", fontsize=10,
        )
    ax.set_title("Challenge Outcomes (all prompts)", fontsize=11)
    ax.set_ylabel("Count", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    # Right: defense rate by initial rating quartile
    ax = axes[1]
    valid = challenge.dropna(subset=["initial_score"])
    if not valid.empty:
        valid = valid.copy()
        valid["quartile"] = pd.qcut(valid["initial_score"], q=4, labels=["Q1\n(Low)", "Q2", "Q3", "Q4\n(High)"])
        defense_by_q = valid.groupby("quartile")["defended"].mean()
        ax.bar(
            range(len(defense_by_q)),
            defense_by_q.values,
            color="#2196F3", alpha=0.8,
        )
        ax.set_xticks(range(len(defense_by_q)))
        ax.set_xticklabels(defense_by_q.index.tolist())
        ax.set_xlabel("Initial Rating Quartile", fontsize=10)
        ax.set_ylabel("Defense Rate", fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.set_title("Defense Rate by Initial Rating Quartile", fontsize=11)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Experiment 7: Challenge Experiment Results", fontsize=12)
    plt.tight_layout()
    out = figs_dir / "figure3_defense_rates.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure 3 saved → {out}")


# ---------------------------------------------------------------------------
# Figure 4: Rating vs. perplexity (NLL)
# ---------------------------------------------------------------------------

def figure4_rating_vs_perplexity(
    ratings: pd.DataFrame,
    perplexity: pd.DataFrame,
    config: Experiment7Config,
) -> None:
    figs_dir = _figures_dir(config)
    merged = ratings.merge(perplexity, on=["prompt_id", "condition"]).dropna(subset=["score", "nll"])

    if merged.empty:
        print("  Skipping Figure 4: no merged rating+perplexity data.")
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    for cond in ["llama70b", "llama8b", "gemma9b"]:
        sub = merged[merged["condition"] == cond]
        ax.scatter(
            sub["nll"], sub["score"],
            color=_CONDITION_COLORS[cond], alpha=0.5, s=25,
            label=_CONDITION_LABELS[cond].replace("\n", " "),
        )

    ax.set_xlabel("NLL under Llama 70B (nats)", fontsize=11)
    ax.set_ylabel("Self-Rating (0–100)", fontsize=11)
    ax.set_title("Experiment 7: Self-Rating vs. Response Perplexity", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = figs_dir / "figure4_rating_vs_perplexity.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure 4 saved → {out}")


# ---------------------------------------------------------------------------
# Summary text
# ---------------------------------------------------------------------------

def write_summary(config: Experiment7Config) -> None:
    ratings   = _load_ratings(config)
    quality   = _load_quality(config)
    perplexity = _load_perplexity(config)
    challenge = _load_challenge(config)

    lines: list[str] = ["=== Experiment 7 Summary ===\n"]

    if ratings is not None:
        for cond in ["llama70b", "llama8b", "gemma9b"]:
            sub = ratings[ratings["condition"] == cond]["score"].dropna()
            if not sub.empty:
                lines.append(
                    f"{cond}: mean={sub.mean():.1f}, median={sub.median():.1f}, "
                    f"n={len(sub)}"
                )
        # Self-rating bias test
        own   = ratings[ratings["condition"] == "llama70b"]["score"].dropna()
        cross = ratings[ratings["condition"] != "llama70b"]["score"].dropna()
        if len(own) >= 5 and len(cross) >= 5:
            _, p = stats.mannwhitneyu(own, cross, alternative="greater")
            lines.append(f"\nSelf-rating bias (own > cross): p={p:.4f} (one-sided MWU)")

    if challenge is not None and not challenge.empty:
        n_def = int(challenge["defended"].sum())
        n_rev = int(challenge["revised"].sum())
        n_tot = len(challenge)
        lines.append(f"\nChallenge: {n_def}/{n_tot} defended, {n_rev}/{n_tot} revised")

    out = config.results_dir_ex7 / "summary_ex7.txt"
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Summary → {out}")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def generate_all_figures_ex7(config: Experiment7Config) -> None:
    """Generate all 4 figures and summary for Experiment 7."""
    ratings    = _load_ratings(config)
    quality    = _load_quality(config)
    perplexity = _load_perplexity(config)
    challenge  = _load_challenge(config)

    if ratings is None:
        print("  No rating data found — skipping figures.")
        return

    figure1_rating_boxplot(ratings, config)
    if quality is not None:
        figure2_rating_vs_judge(ratings, quality, config)
    else:
        print("  Skipping Figure 2: judge_quality_ex7.json not found.")
    if challenge is not None:
        figure3_defense_rates(challenge, config)
    else:
        print("  Skipping Figure 3: challenge_ex7.json not found.")
    if perplexity is not None:
        figure4_rating_vs_perplexity(ratings, perplexity, config)
    else:
        print("  Skipping Figure 4: perplexity_ex7.json not found.")

    write_summary(config)
