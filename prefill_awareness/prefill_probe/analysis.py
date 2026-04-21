"""
Plotting and summary statistics for Experiment 1.

Produces:
  Figure 1 — layer_accuracy.png    (balanced accuracy per layer)
  Figure 2 — layer_auroc.png       (AUROC per layer vs. perplexity baseline)
  Figure 3 — perplexity_distributions.png
  results/summary.csv
"""

import csv
import json

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (safe for servers without display)
import matplotlib.pyplot as plt
import numpy as np

from .config import Config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_perplexity_data(config: Config) -> list[dict]:
    with open(config.activations_dir / "perplexity.json") as f:
        return json.load(f)


def _split_perplexity_by_condition(perplexity_data: list[dict]) -> tuple[list, list]:
    """Returns (self_perplexities, cross_perplexities)."""
    self_ppls = [d["perplexity"] for d in perplexity_data if d["condition"] == "self"]
    cross_ppls = [d["perplexity"] for d in perplexity_data
                  if d["condition"] == "cross_gemma"]
    return self_ppls, cross_ppls


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _figure_layer_accuracy(
    layers: list[int],
    accs: list[float],
    ppl_baseline_auroc: float,
    config: Config,
) -> None:
    """Figure 1: balanced accuracy per layer."""
    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(layers, accs, marker="o", linewidth=1.5, markersize=4,
            label="Probe balanced accuracy")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1,
               label="Chance (50%)")
    ax.axhline(
        ppl_baseline_auroc, color="darkorange", linestyle="--", linewidth=1,
        label=f"Perplexity baseline AUROC ({ppl_baseline_auroc:.3f})"
    )

    ax.set_xlabel("Layer index", fontsize=12)
    ax.set_ylabel("Balanced accuracy (test set)", fontsize=12)
    ax.set_title(
        "Probe accuracy: self-generated vs. cross-model prefill by layer",
        fontsize=13,
    )
    ax.set_xlim(-0.5, max(layers) + 0.5)
    ax.set_ylim(0.3, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = config.results_dir / "layer_accuracy.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


def _figure_layer_auroc(
    layers: list[int],
    aurocs: list[float],
    ppl_baseline_auroc: float,
    config: Config,
) -> None:
    """Figure 2: AUROC per layer vs. perplexity baseline."""
    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(layers, aurocs, marker="s", linewidth=1.5, markersize=4,
            color="steelblue", label="Probe AUROC")
    ax.axhline(ppl_baseline_auroc, color="darkorange", linestyle="--", linewidth=1.5,
               label=f"Perplexity baseline AUROC ({ppl_baseline_auroc:.3f})")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="Chance (0.5)")

    ax.set_xlabel("Layer index", fontsize=12)
    ax.set_ylabel("AUROC (test set)", fontsize=12)
    ax.set_title(
        "AUROC comparison: activation probe vs. perplexity baseline",
        fontsize=13,
    )
    ax.set_xlim(-0.5, max(layers) + 0.5)
    ax.set_ylim(0.3, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = config.results_dir / "layer_auroc.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


def _figure_perplexity_distributions(
    self_ppls: list[float],
    cross_ppls: list[float],
    config: Config,
) -> None:
    """Figure 3: Perplexity distribution by condition."""
    # Clip extreme values for legibility
    clip_ppl = 500.0
    self_clipped = [min(p, clip_ppl) for p in self_ppls if np.isfinite(p)]
    cross_clipped = [min(p, clip_ppl) for p in cross_ppls if np.isfinite(p)]

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, clip_ppl, 50)

    ax.hist(self_clipped, bins=bins, alpha=0.6, density=True,
            label=f"Self-generated (Llama)  n={len(self_clipped)}", color="steelblue")
    ax.hist(cross_clipped, bins=bins, alpha=0.6, density=True,
            label=f"Cross-model (Gemma)  n={len(cross_clipped)}", color="tomato")

    ax.set_xlabel("Perplexity (clipped at 500)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Perplexity under target model by response source", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = config.results_dir / "perplexity_distributions.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def generate_all_figures(
    probe_results: dict,
    responses: list[dict],
    config: Config,
) -> float:
    """
    Generate all three figures.  Returns the perplexity baseline AUROC.
    """
    layer_results = probe_results["layer_results"]
    layers = sorted(layer_results.keys())
    accs = [layer_results[l]["test_balanced_accuracy"] for l in layers]
    aurocs = [layer_results[l]["test_auroc"] for l in layers]

    ppl_baseline_auroc = probe_results.get("perplexity_baseline_auroc")
    if ppl_baseline_auroc is None or not np.isfinite(ppl_baseline_auroc):
        print("  WARNING: perplexity baseline AUROC unavailable; using 0.5 for plots.")
        ppl_baseline_auroc = 0.5

    _figure_layer_accuracy(layers, accs, ppl_baseline_auroc, config)
    _figure_layer_auroc(layers, aurocs, ppl_baseline_auroc, config)

    ppl_data = _load_perplexity_data(config)
    self_ppls, cross_ppls = _split_perplexity_by_condition(ppl_data)
    _figure_perplexity_distributions(self_ppls, cross_ppls, config)

    return ppl_baseline_auroc


def print_summary_table(
    probe_results: dict,
    responses: list[dict],
    config: Config,
) -> None:
    """Print summary table to stdout and save as CSV."""
    layer_results = probe_results["layer_results"]
    ppl_baseline_auroc = probe_results.get("perplexity_baseline_auroc", float("nan"))

    # Split counts from the response list
    split_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    for r in responses:
        split_counts[r["split"]] += 1

    # Best layer metrics
    if layer_results:
        best_layer = max(
            layer_results,
            key=lambda l: layer_results[l]["test_balanced_accuracy"],
        )
        best_acc = layer_results[best_layer]["test_balanced_accuracy"]
        best_auroc = layer_results[best_layer]["test_auroc"]
    else:
        best_layer = best_acc = best_auroc = float("nan")

    auroc_gap = (best_auroc - ppl_baseline_auroc
                 if isinstance(best_auroc, float) and isinstance(ppl_baseline_auroc, float)
                 else float("nan"))

    sanity_acc = probe_results.get("sanity_check_acc", float("nan"))
    sanity_train_acc = probe_results.get("sanity_check", {}).get("train_acc", float("nan"))

    # Perplexity means
    ppl_path = config.activations_dir / "perplexity.json"
    mean_ppl_self = mean_ppl_cross = float("nan")
    if ppl_path.exists():
        import json
        with open(ppl_path) as f:
            ppl_data = json.load(f)
        self_ppls = [d["perplexity"] for d in ppl_data
                     if d["condition"] == "self" and np.isfinite(d["perplexity"])]
        cross_ppls = [d["perplexity"] for d in ppl_data
                      if d["condition"] == "cross_gemma" and np.isfinite(d["perplexity"])]
        if self_ppls:
            mean_ppl_self = float(np.mean(self_ppls))
        if cross_ppls:
            mean_ppl_cross = float(np.mean(cross_ppls))

    rows = [
        ("N prompts (after filtering)",
         str(len(responses))),
        ("N train / val / test",
         f"{split_counts['train']} / {split_counts['val']} / {split_counts['test']}"),
        ("Best probe layer",
         str(best_layer)),
        ("Best probe balanced accuracy",
         f"{best_acc:.4f}" if isinstance(best_acc, float) else str(best_acc)),
        ("Best probe AUROC",
         f"{best_auroc:.4f}" if isinstance(best_auroc, float) else str(best_auroc)),
        ("Perplexity baseline AUROC",
         f"{ppl_baseline_auroc:.4f}" if np.isfinite(ppl_baseline_auroc) else "N/A"),
        ("Probe - perplexity AUROC gap",
         f"{auroc_gap:.4f}" if np.isfinite(auroc_gap) else "N/A"),
        ("Sanity check test acc (should ~50%)",
         f"{sanity_acc:.4f}" if isinstance(sanity_acc, float) else str(sanity_acc)),
        ("Sanity check train acc (should ~50%)",
         f"{sanity_train_acc:.4f}" if np.isfinite(sanity_train_acc) else "N/A"),
        ("Mean perplexity (self)",
         f"{mean_ppl_self:.2f}" if np.isfinite(mean_ppl_self) else "N/A"),
        ("Mean perplexity (cross-model)",
         f"{mean_ppl_cross:.2f}" if np.isfinite(mean_ppl_cross) else "N/A"),
    ]

    # Console output
    w_metric = 42
    w_value = 20
    sep = "-" * (w_metric + w_value + 3)
    print("\n" + "=" * (w_metric + w_value + 3))
    print("  EXPERIMENT 1 — PREFILL DETECTION SUMMARY")
    print("=" * (w_metric + w_value + 3))
    print(f"  {'Metric':<{w_metric}} {'Value':<{w_value}}")
    print(sep)
    for metric, value in rows:
        print(f"  {metric:<{w_metric}} {value:<{w_value}}")
    print("=" * (w_metric + w_value + 3))

    # CSV
    csv_path = config.results_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerows(rows)
    print(f"\n  Summary saved → {csv_path}")
