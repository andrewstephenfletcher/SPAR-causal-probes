"""
Poster-ready version of Figure 1 from Experiment 4.

Changes from the standard figure:
- Narrower layout (half width)
- Chance line is dashed grey (matching poster style)
- Llama models share a purple colour family (8B = light, 70B = dark)
- Gemma keeps its own colour (teal/green family)
- Perplexity baselines removed
- Legend moved to lower right
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------------------------------------------------------------------------
# Resolve paths relative to this file so the script can be run from anywhere
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent  # prefill_awareness/
_RESULTS_EX4 = _REPO_ROOT / "outputs" / "experiment4" / "results"
_RESULTS_EX1 = _REPO_ROOT / "outputs" / "experiment1" / "results"

MODEL_DISPLAY = {
    "llama8b":   "Llama 3.1 8B",
    "llama70b":  "Llama 3.3 70B",
    "gemma4b":   "Gemma 4 4B",
    "gemma31b":  "Gemma 4 31B",
    "mistral7b":  "Mistral 7B",
    "mistral24b": "Mistral Small 24B",
}

# Purple for Llama, green for Gemma, orange/amber for Mistral
MODEL_COLORS = {
    "llama8b":   "#9B59B6",   # medium purple
    "llama70b":  "#4A235A",   # dark purple
    "gemma4b":   "#27AE60",   # medium green
    "gemma31b":  "#145A32",   # dark green
    "mistral7b":  "#E67E22",  # medium orange
    "mistral24b": "#784212",  # dark orange/brown
}

# Primary cross-family condition used for the main scaling comparison
PRIMARY_CROSS = {
    "llama8b":   "cross_gemma9b",
    "llama70b":  "cross_gemma9b",
    "gemma4b":   "cross_llama8b",
    "gemma31b":  "cross_llama8b",
    "mistral7b":  "cross_llama8b",
    "mistral24b": "cross_llama8b",
}


def _load_results() -> dict[str, dict]:
    paths = {
        "llama8b":  _RESULTS_EX1 / "probe_results_llama8b.json",
        "llama70b": _RESULTS_EX4  / "probe_results_llama70b.json",
        "gemma31b": _RESULTS_EX4  / "probe_results_gemma31b.json",
    }
    results = {}
    for name, path in paths.items():
        if path.exists():
            with open(path) as f:
                results[name] = json.load(f)
        else:
            print(f"  WARNING: {name} not found at {path}")
    return results


def _layer_auroc_curve(model_results: dict, condition_name: str):
    layer_results = model_results["cross_conditions"][condition_name]["layer_results"]
    layers = sorted(int(k) for k in layer_results)
    aurocs = [layer_results[str(l)]["test_auroc"] for l in layers]
    return layers, aurocs


def make_poster_fig1(out_path: Path | None = None) -> None:
    all_results = _load_results()

    # Half the original width (11 → 5.5), keep height
    fig, ax = plt.subplots(figsize=(5, 4))

    threshold = 0.95
    crossings = {}  # model_name -> relative depth of first 0.95 crossing

    for model_name in ("llama8b", "llama70b", "gemma4b", "gemma31b", "mistral7b", "mistral24b"):
        model_r = all_results.get(model_name)
        if model_r is None:
            continue
        cond_name = PRIMARY_CROSS[model_name]
        if cond_name not in model_r.get("cross_conditions", {}):
            continue

        n_layers = model_r["n_layers"]
        layers, aurocs = _layer_auroc_curve(model_r, cond_name)
        rel_layers = [l / (n_layers - 1) for l in layers]

        ax.plot(
            rel_layers, aurocs,
            color=MODEL_COLORS[model_name],
            linewidth=2.0,
            label=MODEL_DISPLAY[model_name],
        )

        crossing = next(
            (x for x, y in zip(rel_layers, aurocs) if y >= threshold), None
        )
        if crossing is not None:
            crossings[model_name] = crossing
            ax.axvline(
                crossing,
                color=MODEL_COLORS[model_name],
                linestyle=":",
                linewidth=1.2,
                alpha=0.7,
            )

    # Chance line: dashed grey, matching poster style
    ax.axhline(
        0.5,
        color="grey",
        linestyle="--",
        linewidth=1.2,
        label="Chance (0.5)",
    )

    # Legend entry for the 0.95 threshold lines
    import matplotlib.lines as mlines
    threshold_handle = mlines.Line2D(
        [], [], color="grey", linestyle=":", linewidth=1.2,
        label="First layer ≥ 0.95 AUROC",
    )

    # Arrow between earliest and latest crossing to show larger models detect earlier
    x_large = min(crossings.values()) if crossings else None
    x_small = max(crossings.values()) if crossings else None
    if x_large is not None and x_small is not None:
        arrow_y = 0.57
        ax.annotate(
            "",
            xy=(x_large, arrow_y),
            xytext=(x_small, arrow_y),
            arrowprops=dict(
                arrowstyle="->",
                color="dimgrey",
                lw=1.4,
                connectionstyle="arc3,rad=0",
            ),
        )
        ax.text(
            (x_large + x_small) / 2, arrow_y + 0.018,
            "larger model detects earlier",
            ha="center", va="bottom", fontsize=7.5, color="dimgrey",
        )

    ax.set_xlabel("Relative layer depth", fontsize=11)
    ax.set_ylabel("AUROC (test set)", fontsize=11)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.35, 1.05)
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [threshold_handle], fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if out_path is None:
        out_path = _RESULTS_EX4 / "fig1_normalized_auroc_poster.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    make_poster_fig1(out)
