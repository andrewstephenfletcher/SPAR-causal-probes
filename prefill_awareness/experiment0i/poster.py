"""
Poster-quality figures for Experiment 0i.

Run from prefill_awareness/:
    uv run python -m experiment0i.poster
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# Brand colours (same palette as experiment0f poster)
_ANTHROPIC = "#C96A3B"
_GOOGLE    = "#4285F4"

# Source display order and metadata
_SOURCES = [
    ("opus_45",    "Claude Opus 4.5\n(self)",  _ANTHROPIC),
    ("sonnet_45",  "Claude Sonnet 4.5",         _ANTHROPIC),
    ("gemini_pro", "Gemini 2.5 Pro\n(self)",    _GOOGLE),
]

_EVALUATORS = [
    ("opus_45", "Evaluator: Opus 4.5"),
]

_HONEST_ALPHA   = 0.90
_ATTACK_ALPHA   = 0.45
_VERIFIED_ALPHA = 0.45   # same as attack but hatched
_BAR_WIDTH      = 0.35
_GROUP_GAP      = 0.30   # extra gap between source groups


def make_poster_plots(results_dir: Path | None = None, figures_dir: Path | None = None) -> None:
    if results_dir is None:
        results_dir = Path("outputs/experiment0i/results")
    if figures_dir is None:
        figures_dir = Path("outputs/experiment0i/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_dir / "detection_results.csv")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")

    auth = df[df["probe"] == "authorship_attack"].copy()

    # Build verified task IDs per source from backdoor probe ground truth
    backdoor = df[df["probe"] == "backdoor"]
    verified_ids: dict[str, set] = {}
    for src_key, _, _ in _SOURCES:
        sub = backdoor[(backdoor["source"] == src_key) & (backdoor["condition"] == "attack")]
        verified_ids[src_key] = set(sub[sub["side_task_success"] == True]["task_id"].unique())

    _poster_authorship(auth, verified_ids, figures_dir)


def _poster_authorship(auth: pd.DataFrame, verified_ids: dict, figures_dir: Path) -> None:
    n_eval = len(_EVALUATORS)
    fig, axes = plt.subplots(1, n_eval, figsize=(6.2 * n_eval, 5.2), sharey=True)
    if n_eval == 1:
        axes = [axes]

    bar_w = _BAR_WIDTH

    for ax, (eval_key, eval_label) in zip(axes, _EVALUATORS):
        sub = auth[auth["evaluator"] == eval_key]

        # Build x positions: two bars per source group, with gap between groups
        x = 0.0
        group_info = []   # (center_x, label, color)

        for src_key, src_label, color in _SOURCES:
            src = sub[sub["source"] == src_key]
            honest = src[src["condition"] == "honest"]["score"].dropna()
            attack = src[src["condition"] == "attack"]["score"].dropna()

            if honest.empty and attack.empty:
                continue

            x0 = x  # left edge of honest bar
            x1 = x + bar_w  # left edge of attack bar

            if not honest.empty:
                ax.bar(x0, honest.mean(), width=bar_w,
                       color=color, alpha=_HONEST_ALPHA,
                       edgecolor="white", linewidth=1.0, zorder=3)
                ax.text(x0 + bar_w / 2, honest.mean() + 1.5,
                        f"{honest.mean():.0f}",
                        ha="center", va="bottom", fontsize=10, fontweight="bold",
                        color="#333333")

            if not attack.empty:
                ax.bar(x1, attack.mean(), width=bar_w,
                       color=color, alpha=_ATTACK_ALPHA,
                       edgecolor="white", linewidth=1.0, zorder=3)
                ax.text(x1 + bar_w / 2, attack.mean() + 1.5,
                        f"{attack.mean():.0f}",
                        ha="center", va="bottom", fontsize=10, fontweight="bold",
                        color="#333333")

            # Verified-only attack bar (hatched)
            x2 = x + 2 * bar_w
            ver_tasks = verified_ids.get(src_key, set())
            verified = src[(src["condition"] == "attack") &
                           (src["task_id"].isin(ver_tasks))]["score"].dropna()
            if not verified.empty:
                ax.bar(x2, verified.mean(), width=bar_w,
                       color=color, alpha=_VERIFIED_ALPHA,
                       edgecolor=color, linewidth=1.2,
                       hatch="///", zorder=3)
                ax.text(x2 + bar_w / 2, verified.mean() + 1.5,
                        f"{verified.mean():.0f}",
                        ha="center", va="bottom", fontsize=10, fontweight="bold",
                        color="#333333")

            group_center = x0 + bar_w           # midpoint of honest + attack bars
            group_info.append((group_center, src_label, color))
            x += 3 * bar_w + _GROUP_GAP

        # Source group labels below x-axis
        for cx, label, color in group_info:
            ax.text(cx, -9, label,
                    ha="center", va="top", fontsize=9.5,
                    fontweight="bold", color=color)

        # Single panel — panel subtitle not needed; subtitle lives in suptitle
        ax.set_xlim(-0.3, x - _GROUP_GAP + 0.3)
        ax.set_ylim(-2, 105)
        ax.set_xticks([])
        ax.axhline(50, color="#cccccc", linestyle="--", linewidth=1.2, zorder=1)
        ax.grid(axis="y", alpha=0.2, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

    axes[0].set_ylabel("Tamper Probability", fontsize=11)
    axes[0].set_yticks([0, 20, 40, 60, 80, 100])
    axes[0].set_yticklabels(["0\n(Me)", "20", "40", "60", "80", "100\n(Not me)"], fontsize=9)

    # Legend
    legend_patches = [
        mpatches.Patch(facecolor="#888888", alpha=_HONEST_ALPHA,
                       label="Honest code"),
        mpatches.Patch(facecolor="#888888", alpha=_ATTACK_ALPHA,
                       label="Attack code"),
        mpatches.Patch(facecolor="#888888", alpha=_VERIFIED_ALPHA,
                       hatch="///", label="Attack code (verified)"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", fontsize=10,
               framealpha=0.92, ncol=3, bbox_to_anchor=(0.5, -0.04))

    fig.suptitle(
        "Models accept their own attack code as genuine output",
        fontsize=14, fontweight="bold", y=1.01,
    )

    plt.subplots_adjust(bottom=0.22, wspace=0.06)
    out = figures_dir / "fig5_authorship_framing_poster.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out.resolve()}")


if __name__ == "__main__":
    try:
        from dotenv import load_dotenv
        _env = Path(__file__).resolve().parents[2] / ".env"
        if _env.exists():
            load_dotenv(_env)
    except ImportError:
        pass
    make_poster_plots()
