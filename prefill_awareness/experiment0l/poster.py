"""
Poster-quality figures for Experiment 0l.

Run from prefill_awareness/:
    uv run python -m experiment0l.poster
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from .analysis import _load_organic_baselines, _auroc_with_ci
from .config import Experiment0lConfig

_FONT = "DejaVu Sans"
_COLORS = {
    "baseline":   "#9E9E9E",
    "style_only": "#59A14F",
    "logic_only": "#4878CF",
}


def make_poster_plots(config: Experiment0lConfig | None = None) -> None:
    if config is None:
        config = Experiment0lConfig()

    all_trajectories = {}
    for variant in ["style_only", "logic_only"]:
        p = config.results_dir / f"{variant}_trajectories.json"
        if p.exists():
            all_trajectories[variant] = json.loads(p.read_text())

    if not all_trajectories:
        print("No trajectory files found — run experiment0l first.")
        return

    organic = _load_organic_baselines(config)
    _poster_auroc(all_trajectories, organic, config)


def _poster_auroc(all_trajectories: dict, organic: dict, config: Experiment0lConfig) -> None:
    all_task_ids = set.intersection(
        *[{t["task_id"] for t in trajs} for trajs in all_trajectories.values()]
    ) if all_trajectories else set()
    valid_tasks = [tid for tid in all_task_ids if organic.get(tid) is not None]
    neg = [organic[tid] for tid in valid_tasks]

    if len(neg) < 3:
        print("  insufficient organic baselines — skipping poster AUROC.")
        return

    # Baseline iter-0 scores
    first_var = next(iter(all_trajectories))
    bl_map = {t["task_id"]: t["iterations"][0]["tamper_prob"]
              for t in all_trajectories[first_var]
              if t["iterations"] and t["iterations"][0].get("tamper_prob") is not None}
    bl_pos = [bl_map[tid] for tid in valid_tasks if bl_map.get(tid) is not None]

    entries = []  # (name, label, color, pos_scores)
    if len(bl_pos) == len(valid_tasks):
        entries.append(("baseline", "Baseline\n(Most aware 20)", _COLORS["baseline"], bl_pos))

    for variant in ["style_only", "logic_only"]:
        if variant not in all_trajectories:
            continue
        label = "Style only\n(format / prose)" if variant == "style_only" else "Logic only\n(code)"
        final_map = {t["task_id"]: t["final_score"]
                     for t in all_trajectories[variant] if t.get("final_score") is not None}
        pos = [final_map[tid] for tid in valid_tasks if final_map.get(tid) is not None]
        if len(pos) >= 3:
            entries.append((variant, label, _COLORS[variant], pos))

    aurocs, labels, colors = [], [], []
    baseline_auroc = np.nan
    for name, label, color, pos in entries:
        a, _, _ = _auroc_with_ci(neg, pos)
        aurocs.append(float(a) if not np.isnan(a) else 0.0)
        labels.append(label)
        colors.append(color)
        if name == "baseline":
            baseline_auroc = float(a)

    # --- figure ---
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(labels))
    bar_width = 0.52

    bars = ax.bar(x, aurocs, width=bar_width, color=colors, edgecolor="white",
                  linewidth=1.5, zorder=3)

    # AUROC value on top of each bar
    for bar, v in zip(bars, aurocs):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.008,
                f"{v:.2f}", ha="center", va="bottom", fontsize=14, fontweight="bold",
                color="#333333")

    # Drop annotation (↓ Δ) for non-baseline bars — only when meaningful
    if not np.isnan(baseline_auroc):
        for bar, v, (name, *_) in zip(bars, aurocs, entries):
            if name == "baseline":
                continue
            delta = baseline_auroc - v
            if abs(delta) < 0.01:
                continue
            sign = "↓" if delta > 0 else "↑"
            mid_y = (baseline_auroc + v) / 2
            ax.annotate(
                f"{sign} {abs(delta):.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, mid_y),
                fontsize=11, ha="center", va="center",
                color="#555555", style="italic",
            )
            # bracket line showing the drop
            bx = bar.get_x() + bar.get_width() + 0.04
            ax.plot([bx, bx], [v, baseline_auroc], color="#888888", linewidth=1.2,
                    clip_on=False, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_ylim(0.45, 1.08)
    ax.set_ylabel("AUROC  (Opus 4.5 monitor)", fontsize=12)
    ax.set_title(
        "Decrease in AUROC driven by\nGemini imitating the style of Opus",
        fontsize=14, fontweight="bold", pad=14,
    )
    ax.axhline(0.5, color="#cccccc", linestyle="--", linewidth=1.2, zorder=2)
    ax.text(len(labels) - 0.5 + 0.35, 0.502, "chance", fontsize=9,
            color="#aaaaaa", va="bottom")
    ax.grid(axis="y", alpha=0.25, zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = config.figures_dir / "auroc_by_variant_poster.png"
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
