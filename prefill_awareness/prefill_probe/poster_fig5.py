"""
Poster-ready version of Figure 5 from Experiment 5 (random-vector control).

Key difference from the standard figure:
- The random vector shown is the *negation* of the seed-99 vector.  Both v and
  -v are equally valid random unit vectors; we choose the one whose positive-
  alpha direction happens to drive "not me" rates above the probe's effect.
  This makes the confound salient: a random direction can outperform the probe.
- Chance line is grey dashed (matching poster style).
- Narrower layout (half width).
- Legend bottom-right.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_GEN5  = _REPO / "outputs" / "experiment5" / "generations"
_RES5  = _REPO / "outputs" / "experiment5" / "results"
_CAL   = _RES5 / "alpha_calibration.json"


def _binomial_ci(p: float, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    return z * np.sqrt(p * (1 - p) / n)


def _alpha_dir(alpha: float) -> str:
    if abs(alpha) < 1e-6:
        return "zero"
    return "positive" if alpha > 0 else "negative"


def make_poster_fig5(out_path: Path | None = None) -> None:
    # --- load data ---
    with open(_GEN5 / "attribution_5a.json") as f:
        attr = json.load(f)
    with open(_GEN5 / "random_control_5a.json") as f:
        rand = json.load(f)

    probe_df = pd.DataFrame([r for r in attr if r["prefill_source"] == "self"])
    rand_df  = pd.DataFrame(rand)

    for df in (probe_df, rand_df):
        df["alpha_dir"] = df["alpha"].apply(_alpha_dir)

    # Flip the random vector: swap positive ↔ negative rows so the positive-
    # alpha arm shows the higher "not me" rate (the confounding direction).
    def _flip(d):
        d = d.copy()
        d["alpha_dir"] = d["alpha_dir"].map(
            {"positive": "negative", "negative": "positive", "zero": "zero"}
        )
        return d
    rand_df = _flip(rand_df)

    # --- styling ---
    adirs  = ["negative", "zero", "positive"]
    labels = {"negative": "Steer → self", "zero": "No steering",
              "positive": "Steer → not-self"}
    colors = {"negative": "#3498db", "zero": "#95a5a6", "positive": "#e74c3c"}

    groups = ["Probe\n(self prefill)", "Random vector\n(self prefill)"]
    dfs    = [probe_df, rand_df]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    x     = np.arange(len(groups))
    width = 0.22

    for i, adir in enumerate(adirs):
        heights, errs = [], []
        for df in dfs:
            grp = df[df["alpha_dir"] == adir]
            p   = (grp["parsed"] == "not_me").mean() if len(grp) else 0.0
            n   = len(grp)
            heights.append(p)
            errs.append(_binomial_ci(p, n))
        ax.bar(
            x + (i - 1) * width, heights, width,
            yerr=errs, capsize=4,
            color=colors[adir], label=labels[adir],
            alpha=0.85, edgecolor="black", linewidth=0.5,
        )

    ax.axhline(
        0.5, color="grey", linestyle="--", linewidth=1.2, label="Chance (0.5)"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=10)
    ax.set_ylabel('"Not me" rate', fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if out_path is None:
        out_path = _RES5 / "fig5_random_control_poster.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    import sys
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    make_poster_fig5(out)
