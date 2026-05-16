"""
PCA visualization of layer-30 activations from Experiment 3.

Two figures:

  pca_by_condition_and_dataset.png
    4 × 3 grid (condition × dataset).  Each panel projects self activations
    (blue circles) and cross-model activations (coloured triangles) onto the
    top-2 principal components of their joint pool.

  pca_all_conditions.png
    Single scatter showing all 5 conditions in the joint PCA space, coloured
    and shaped by condition.

PCA is computed via eigendecomposition of the sample covariance matrix,
following the same approach as get_pca_components in dct_probe_analysis.ipynb.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import torch
from sklearn.metrics import roc_auc_score

from .config import Experiment3Config
from .probe import apply_normaliser, train_probe
from .probe_ex3 import CONDITIONS, CROSS_CONDITIONS, DATASETS, build_dataset, load_all_activations

_COND_COLORS = {
    "self":           "#1f77b4",  # blue
    "altered_self":   "#2ca02c",  # green
    "gemma":          "#d62728",  # red
    "mistral":        "#ff7f0e",  # orange
    "style_imitated": "#9467bd",  # purple
}
_COND_LABELS = {
    "self":           "Self (Llama 8B)",
    "altered_self":   "Altered-self",
    "gemma":          "Gemma 9B",
    "mistral":        "Qwen 7B",
    "style_imitated": "Style-imitated",
}
_COND_MARKERS = {
    "self": "o", "altered_self": "s", "gemma": "^", "mistral": "D", "style_imitated": "P",
}
_DS_LABELS = {"alpaca": "Alpaca", "oasst1": "OASST1", "mmlu": "MMLU"}


# ---------------------------------------------------------------------------
# PCA helpers (eigendecomposition of sample covariance)
# ---------------------------------------------------------------------------

def get_pca_components(
    activations: np.ndarray,
    k: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the top-k principal components via eigendecomposition.

    Equivalent to dct_probe_analysis.ipynb's get_pca_components but in numpy.

    Args:
        activations: shape (n, d_model)
        k:           number of components

    Returns:
        components:    shape (d_model, k)  — top-k eigenvectors as columns
        mean:          shape (d_model,)    — training-set mean
        explained_var: shape (k,)          — fraction of total variance per PC
    """
    X = activations.astype(np.float64)
    mean = X.mean(axis=0)
    X -= mean

    cov = X.T @ X / max(X.shape[0] - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # eigh returns ascending order; reverse for descending
    idx = np.argsort(eigenvalues)[::-1]
    components = eigenvectors[:, idx[:k]].astype(np.float32)
    total_var = eigenvalues.sum()
    explained = eigenvalues[idx[:k]] / total_var if total_var > 0 else np.zeros(k)
    return components, mean.astype(np.float32), explained


def _get_X(acts_list: list[dict], dataset: str | None = None) -> np.ndarray | None:
    """Stack activation vectors, optionally filtered by dataset."""
    vecs = [
        d["activation"].astype(np.float32)
        for d in acts_list
        if dataset is None or d.get("dataset") == dataset
    ]
    return np.stack(vecs) if len(vecs) >= 3 else None


# ---------------------------------------------------------------------------
# Figure 1: 4 × 3 grid (condition × dataset)
# ---------------------------------------------------------------------------

def figure_pca_grid(
    all_acts: dict[str, list[dict]],
    ex3_config: Experiment3Config,
) -> None:
    """
    Rows = cross-model conditions, columns = datasets.
    Each panel: joint PCA of self + condition activations for that dataset.
    Self shown as blue circles; condition as coloured triangles.
    """
    nrows, ncols = len(CROSS_CONDITIONS), len(DATASETS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.2, nrows * 3.5))

    for r, cond in enumerate(CROSS_CONDITIONS):
        for c, ds in enumerate(DATASETS):
            ax = axes[r, c]

            self_X  = _get_X(all_acts["self"],  dataset=ds)
            other_X = _get_X(all_acts[cond], dataset=ds)

            if self_X is None or other_X is None:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, color="gray", fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                pool = np.concatenate([self_X, other_X], axis=0)
                comps, mean, expl = get_pca_components(pool, k=2)

                self_proj  = (self_X  - mean) @ comps
                other_proj = (other_X - mean) @ comps

                ax.scatter(self_proj[:, 0],  self_proj[:, 1],
                           c=_COND_COLORS["self"], alpha=0.55, s=14,
                           marker="o", rasterized=True, linewidths=0)
                ax.scatter(other_proj[:, 0], other_proj[:, 1],
                           c=_COND_COLORS[cond], alpha=0.55, s=14,
                           marker="^", rasterized=True, linewidths=0)
                ax.set_xlabel(f"PC1 ({expl[0]:.1%})", fontsize=8)
                ax.set_ylabel(f"PC2 ({expl[1]:.1%})", fontsize=8)

            ax.tick_params(labelsize=7)
            if r == 0:
                ax.set_title(_DS_LABELS.get(ds, ds), fontsize=11, fontweight="bold", pad=6)
            if c == 0:
                ax.annotate(
                    _COND_LABELS.get(cond, cond), xy=(-0.35, 0.5),
                    xycoords="axes fraction", ha="right", va="center",
                    fontsize=9, rotation=0,
                )

    # Legend
    handles = [
        Line2D([0], [0], marker="o", linestyle="", color=_COND_COLORS["self"],
               markersize=7, label=_COND_LABELS["self"]),
    ] + [
        Line2D([0], [0], marker="^", linestyle="", color=_COND_COLORS[c],
               markersize=7, label=_COND_LABELS[c])
        for c in CROSS_CONDITIONS
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        "Layer-30 residual stream activations: self vs. cross-model (PCA per panel)",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()
    out = ex3_config.results_dir_ex3 / "pca_by_condition_and_dataset.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ---------------------------------------------------------------------------
# Figure 2: All conditions in a single joint PCA space
# ---------------------------------------------------------------------------

def figure_pca_all_conditions(
    all_acts: dict[str, list[dict]],
    ex3_config: Experiment3Config,
) -> None:
    """
    All 5 conditions projected onto the top-2 PCs of the pooled activation matrix.
    Colour = condition; shape = condition.  Dataset not distinguished here.
    """
    parts = [_get_X(all_acts[c]) for c in CONDITIONS]
    available = [(c, X) for c, X in zip(CONDITIONS, parts) if X is not None]
    if not available:
        print("  No activation data; skipping all-conditions PCA.")
        return

    pool = np.concatenate([X for _, X in available], axis=0)
    comps, mean, expl = get_pca_components(pool, k=2)

    fig, ax = plt.subplots(figsize=(10, 7))

    for cond, X in available:
        proj = (X - mean) @ comps
        ax.scatter(
            proj[:, 0], proj[:, 1],
            c=_COND_COLORS[cond],
            alpha=0.45, s=20,
            marker=_COND_MARKERS.get(cond, "o"),
            label=f"{_COND_LABELS.get(cond, cond)} (n={len(X)})",
            rasterized=True,
            linewidths=0,
        )

    ax.set_xlabel(f"PC1 ({expl[0]:.2%} variance explained)", fontsize=12)
    ax.set_ylabel(f"PC2 ({expl[1]:.2%} variance explained)", fontsize=12)
    ax.set_title(
        "Layer-30 activations: all 5 conditions in joint PCA space\n"
        "(Llama 8B reading its own vs. foreign responses)",
        fontsize=13,
    )
    ax.legend(fontsize=10, markerscale=1.8)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out = ex3_config.results_dir_ex3 / "pca_all_conditions.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


# ---------------------------------------------------------------------------
# Figure 3: Per-dataset scatter (all conditions, one subplot per dataset)
# ---------------------------------------------------------------------------

def figure_pca_per_dataset(
    all_acts: dict[str, list[dict]],
    ex3_config: Experiment3Config,
) -> None:
    """
    One panel per dataset.  All 5 conditions projected onto the joint PCA
    for that dataset.  Useful for checking dataset-specific clustering.
    """
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(len(DATASETS) * 5.5, 5))

    for ax, ds in zip(axes, DATASETS):
        parts = [(c, _get_X(all_acts[c], dataset=ds)) for c in CONDITIONS]
        available = [(c, X) for c, X in parts if X is not None]

        if not available:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
            ax.set_title(_DS_LABELS.get(ds, ds))
            continue

        pool = np.concatenate([X for _, X in available], axis=0)
        comps, mean, expl = get_pca_components(pool, k=2)

        for cond, X in available:
            proj = (X - mean) @ comps
            ax.scatter(
                proj[:, 0], proj[:, 1],
                c=_COND_COLORS[cond],
                alpha=0.5, s=18,
                marker=_COND_MARKERS.get(cond, "o"),
                label=_COND_LABELS.get(cond, cond),
                rasterized=True,
                linewidths=0,
            )

        ax.set_xlabel(f"PC1 ({expl[0]:.1%})", fontsize=10)
        ax.set_ylabel(f"PC2 ({expl[1]:.1%})", fontsize=10)
        ax.set_title(_DS_LABELS.get(ds, ds), fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=8, markerscale=1.4)

    fig.suptitle("Layer-30 activations per dataset: all conditions (PCA)", fontsize=13)
    plt.tight_layout()
    out = ex3_config.results_dir_ex3 / "pca_per_dataset.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


# ---------------------------------------------------------------------------
# Probe direction helpers
# ---------------------------------------------------------------------------

def _get_probe_scores(
    self_acts: list[dict],
    other_acts: list[dict],
    ex3_config: Experiment3Config,
    dataset: str | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Train a linear probe on the train+val split, then return the raw logit
    scores on the TEST split, separately for self (label 0) and other (label 1).

    Scores are signed distances from the decision hyperplane:
      score > 0  →  predicted as cross-model
      score < 0  →  predicted as self
    Decision boundary is at score = 0.
    """
    X_tr, y_tr = build_dataset(self_acts, other_acts, split="train", dataset=dataset)
    X_v,  y_v  = build_dataset(self_acts, other_acts, split="val",   dataset=dataset)

    if len(X_tr) < 4 or len(X_v) < 2 or len(np.unique(y_tr)) < 2:
        return None

    probe, mean, std, _, _ = train_probe(X_tr, y_tr, X_v, y_v,
                                         ex3_config.probe_regularisation_grid)

    # Collect test activations separately per class
    self_by_pid  = {d["prompt_id"]: d for d in self_acts}
    other_by_pid = {d["prompt_id"]: d for d in other_acts}

    self_test:  list[np.ndarray] = []
    other_test: list[np.ndarray] = []
    for pid, self_item in self_by_pid.items():
        other_item = other_by_pid.get(pid)
        if other_item is None:
            continue
        if self_item.get("split") != "test":
            continue
        if dataset is not None and self_item.get("dataset") != dataset:
            continue
        self_test.append(self_item["activation"].astype(np.float32))
        other_test.append(other_item["activation"].astype(np.float32))

    if len(self_test) < 2 or len(other_test) < 2:
        return None

    probe.eval()
    with torch.no_grad():
        self_scores = probe(
            torch.tensor(apply_normaliser(np.stack(self_test), mean, std))
        ).numpy()
        other_scores = probe(
            torch.tensor(apply_normaliser(np.stack(other_test), mean, std))
        ).numpy()

    return self_scores, other_scores


# ---------------------------------------------------------------------------
# Figure 4: 1-D separation in the probe's decision direction
# ---------------------------------------------------------------------------

def figure_probe_direction(
    all_acts: dict[str, list[dict]],
    ex3_config: Experiment3Config,
) -> None:
    """
    4 × 3 grid (cross-condition × dataset).

    Each panel shows the distribution of probe logit scores for self (blue)
    and cross-model (coloured) on the TEST split.  The vertical dashed line
    marks the decision boundary (score = 0).  AUROC is annotated per panel.

    This is the complement to the PCA grid: PCA shows directions of maximum
    variance; the probe direction shows the most discriminative direction.
    """
    nrows, ncols = len(CROSS_CONDITIONS), len(DATASETS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.2, nrows * 3.2))

    for r, cond in enumerate(CROSS_CONDITIONS):
        for c, ds in enumerate(DATASETS):
            ax = axes[r, c]

            result = _get_probe_scores(
                all_acts["self"], all_acts[cond], ex3_config, dataset=ds
            )

            if result is None:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, color="gray", fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                self_scores, other_scores = result

                all_scores = np.concatenate([self_scores, other_scores])
                labels     = np.concatenate([
                    np.zeros(len(self_scores)), np.ones(len(other_scores))
                ])
                auroc = roc_auc_score(labels, all_scores)

                lo = float(np.percentile(all_scores, 1))
                hi = float(np.percentile(all_scores, 99))
                bins = np.linspace(lo, hi, 35)

                ax.hist(self_scores,  bins=bins, density=True, alpha=0.6,
                        color=_COND_COLORS["self"],  label="Self")
                ax.hist(other_scores, bins=bins, density=True, alpha=0.6,
                        color=_COND_COLORS[cond], label=_COND_LABELS.get(cond, cond))
                ax.axvline(0, color="black", linestyle="--", linewidth=1.0,
                           label="Decision boundary")
                ax.text(0.97, 0.95, f"AUROC {auroc:.3f}",
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=8, bbox=dict(boxstyle="round,pad=0.2",
                                             fc="white", alpha=0.7))
                ax.set_xlabel("Probe logit score", fontsize=8)
                ax.set_ylabel("Density", fontsize=8)

            ax.tick_params(labelsize=7)
            if r == 0:
                ax.set_title(_DS_LABELS.get(ds, ds), fontsize=11,
                             fontweight="bold", pad=6)
            if c == 0:
                ax.annotate(
                    _COND_LABELS.get(cond, cond), xy=(-0.38, 0.5),
                    xycoords="axes fraction", ha="right", va="center", fontsize=9,
                )

    # Shared legend
    handles = [
        Line2D([0], [0], color=_COND_COLORS["self"], linewidth=6, alpha=0.6,
               label=_COND_LABELS["self"]),
    ] + [
        Line2D([0], [0], color=_COND_COLORS[c], linewidth=6, alpha=0.6,
               label=_COND_LABELS[c])
        for c in CROSS_CONDITIONS
    ] + [
        Line2D([0], [0], color="black", linestyle="--", linewidth=1,
               label="Decision boundary (score=0)")
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.03))

    fig.suptitle(
        "Separation in the probe's decision direction (layer-30 activations, test split)",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()
    out = ex3_config.results_dir_ex3 / "probe_direction_distributions.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate_pca_figures(
    ex3_config: Experiment3Config,
    responses: list[dict] | None = None,
) -> None:
    """
    Load activations and produce all four PCA/probe figures.

    If `responses` is provided, split/dataset overrides are applied to the
    activation records before plotting (ensures reassigned splits are used).
    """
    from .probe_ex3 import _apply_split_overrides

    print("  Loading activations for PCA...")
    all_acts = load_all_activations(ex3_config)

    if responses is not None:
        _apply_split_overrides(all_acts, responses)

    n_loaded = {c: len(v) for c, v in all_acts.items()}
    print(f"  Activation counts per condition: {n_loaded}")

    print("  PCA Figure 1: condition × dataset grid...")
    figure_pca_grid(all_acts, ex3_config)

    print("  PCA Figure 2: all conditions, joint space...")
    figure_pca_all_conditions(all_acts, ex3_config)

    print("  PCA Figure 3: per-dataset, all conditions...")
    figure_pca_per_dataset(all_acts, ex3_config)

    print("  PCA Figure 4: probe direction distributions...")
    figure_probe_direction(all_acts, ex3_config)
