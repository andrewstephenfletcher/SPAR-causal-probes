"""
Poster-quality figures for Experiment 0j.

Run from prefill_awareness/:
    uv run python -m experiment0j.poster
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ANTHROPIC = "#C96A3B"   # Claude-style features (reduce detection)
_BACKFIRE  = "#C0392B"   # Feature that increases detection

_FEATURE_LABELS = {
    "code_first":        "Code First",
    "concise_prose":     "Concise Prose",
    "no_bold_headers":   "No Bold Headers",
    "minimal_docstring": "Minimal Docstring",
    "no_numbered_lists": "No Numbered Lists",
}


def make_poster_plots(results_dir: Path | None = None, figures_dir: Path | None = None) -> None:
    if results_dir is None:
        results_dir = Path("outputs/experiment0j/results")
    if figures_dir is None:
        figures_dir = Path("outputs/experiment0j/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_dir / "feature_importance_table.csv")
    fwd = df[df["direction"] == "forward"].copy()
    fwd = fwd.sort_values("mean_delta")   # most negative first

    _poster_feature_importance(fwd, figures_dir)


def _poster_feature_importance(fwd: pd.DataFrame, figures_dir: Path) -> None:
    labels  = [_FEATURE_LABELS.get(r, r) for r in fwd["feature"]]
    deltas  = fwd["mean_delta"].tolist()
    colors  = [_BACKFIRE if d > 0 else _ANTHROPIC for d in deltas]

    n = len(labels)
    fig, ax = plt.subplots(figsize=(10, 0.75 * n + 1.6))

    y = np.arange(n)
    ax.barh(y, deltas, height=0.55, color=colors, alpha=0.85,
            edgecolor="white", linewidth=1.2, zorder=3)

    # Value labels — just outside the bar tip
    for yi, d in zip(y, deltas):
        sign = "+" if d > 0 else ""
        if d >= 0:
            ax.text(d + 0.4, yi, f"{sign}{d:.1f}",
                    va="center", ha="left", fontsize=11, fontweight="bold", color="#222222")
        else:
            ax.text(d - 0.4, yi, f"{sign}{d:.1f}",
                    va="center", ha="right", fontsize=11, fontweight="bold", color="#222222")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=12)
    ax.axvline(0, color="#555555", linewidth=1.0, zorder=5)
    ax.set_xlabel("Mean change in Tamper Probability", fontsize=11)
    ax.set_xlim(min(deltas) * 1.25, max(deltas) * 1.35)
    # Annotations at axis ends
    xlim = ax.get_xlim()
    ax.text(xlim[0] + 0.5, -0.65, "← Harder to detect",
            fontsize=9, color=_ANTHROPIC, va="top", style="italic")
    ax.text(xlim[1] - 0.5, -0.65, "Easier to detect →",
            fontsize=9, color=_BACKFIRE, va="top", ha="right", style="italic")

    ax.grid(axis="x", alpha=0.2, zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()

    fig.tight_layout()
    out = figures_dir / "feature_importance_poster.png"
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
