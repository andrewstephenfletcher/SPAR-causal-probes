"""
Analysis for Experiment 0k.

Fig 1: convergence_curves.png
       Per-variant convergence lines (green=converged, red=plateaued) + mean.
       Mirrors experiment0j fig1, but one panel per variant.

Fig 2: auroc_by_variant.png
       AUROC for baseline vs final score of each variant.
       Negative class = organic Opus responses from 0f.

Fig 3: score_distributions.png
       tamper_prob histograms: baseline vs final per variant.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.metrics import roc_auc_score

from .config import Experiment0kConfig

_VARIANT_LABELS = {
    "style_only": "Style only\n(format/prose)",
    "logic_only": "Logic only\n(code)",
    "both":       "Both\n(unconstrained)",
}
_VARIANT_COLORS = {
    "style_only": "#59A14F",
    "logic_only": "#4878CF",
    "both":       "#F28E2B",
}


def run_analysis(all_trajectories: dict[str, list[dict]], config: Experiment0kConfig) -> None:
    organic = _load_organic_baselines(config)

    _fig1_convergence(all_trajectories, config)
    _fig2_auroc(all_trajectories, organic, config)
    _fig3_distributions(all_trajectories, config)
    _fig4_mean_tamper(all_trajectories, config)
    _save_summary(all_trajectories, organic, config)

    print(f"  Figures → {config.figures_dir.resolve()}")


# ---------------------------------------------------------------------------
# Fig 1: Convergence curves — one panel per variant
# ---------------------------------------------------------------------------

def _fig1_convergence(all_trajectories: dict, config: Experiment0kConfig) -> None:
    variants = [v for v in config.variants if v in all_trajectories]
    fig, axes = plt.subplots(1, len(variants), figsize=(6 * len(variants), 5), sharey=True)
    if len(variants) == 1:
        axes = [axes]

    for ax, variant in zip(axes, variants):
        trajectories = all_trajectories[variant]
        max_iter = max(
            (it["iter"] for t in trajectories for it in t["iterations"]), default=0
        )
        mean_by_iter: dict[int, list] = {}
        for traj in trajectories:
            color = "#2ca02c" if traj["converged"] else "#d62728"
            iters = [it["iter"] for it in traj["iterations"]]
            scores = [it.get("tamper_prob") or float("nan") for it in traj["iterations"]]
            ax.plot(iters, scores, color=color, alpha=0.35, linewidth=1)
            for it in traj["iterations"]:
                if it.get("tamper_prob") is not None:
                    mean_by_iter.setdefault(it["iter"], []).append(it["tamper_prob"])

        if mean_by_iter:
            xs = sorted(mean_by_iter)
            ys = [np.mean(mean_by_iter[x]) for x in xs]
            ax.plot(xs, ys, color="black", linewidth=2.5, zorder=5)

        ax.axhline(config.convergence_threshold, color="grey", linestyle="--",
                   linewidth=1, label=f"Threshold ({config.convergence_threshold})")

        n_conv = sum(1 for t in trajectories if t["converged"])
        ax.set_title(
            f"{_VARIANT_LABELS.get(variant, variant).replace(chr(10), ' ')}\n"
            f"{n_conv}/{len(trajectories)} converged",
            fontsize=10,
        )
        ax.set_xlabel("Iteration", fontsize=10)
        ax.set_xlim(-0.2, max_iter + 0.2)
        ax.set_ylim(-5, 105)
        ax.set_xticks(range(max_iter + 1))
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("tamper_prob (Opus score)", fontsize=10)

    legend_elems = [
        Line2D([0], [0], color="#2ca02c", alpha=0.6, label="Converged"),
        Line2D([0], [0], color="#d62728", alpha=0.6, label="Plateaued"),
        Line2D([0], [0], color="black", linewidth=2.5, label="Mean"),
        Line2D([0], [0], color="grey", linestyle="--",
               label=f"Threshold ({config.convergence_threshold})"),
    ]
    fig.legend(handles=legend_elems, loc="upper right", fontsize=9)
    fig.suptitle(
        "Experiment 0k — Iterative Convergence by Variant\n"
        "Each iteration: Gemini sees its previous response + Opus reference + monitor reason",
        fontsize=11,
    )
    fig.tight_layout()
    out = config.figures_dir / "convergence_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ---------------------------------------------------------------------------
# Fig 2: AUROC — baseline vs final score per variant
# ---------------------------------------------------------------------------

def _fig2_auroc(
    all_trajectories: dict,
    organic: dict[str, float],
    config: Experiment0kConfig,
) -> None:
    # Collect task_ids common to all variants (and with organic baseline)
    all_task_ids = set.intersection(
        *[{t["task_id"] for t in trajs} for trajs in all_trajectories.values()]
    ) if all_trajectories else set()
    valid_tasks = [tid for tid in all_task_ids if organic.get(tid) is not None]
    neg = [organic[tid] for tid in valid_tasks]

    if len(neg) < 3:
        print("  [fig2] insufficient organic baselines — skipping.")
        return

    # Build series: baseline (iter-0) + final per variant
    series: list[tuple[str, str, list[float]]] = []

    # Baseline from iter-0 of any variant (all share the same 0f starting point)
    first_variant = next(iter(all_trajectories))
    bl_map = {t["task_id"]: t["iterations"][0]["tamper_prob"]
              for t in all_trajectories[first_variant]
              if t["iterations"] and t["iterations"][0].get("tamper_prob") is not None}
    bl_pos = [bl_map[tid] for tid in valid_tasks if bl_map.get(tid) is not None]
    if len(bl_pos) == len(valid_tasks):
        series.append(("baseline", "#808080", bl_pos))

    for variant, trajectories in all_trajectories.items():
        final_map = {t["task_id"]: t["final_score"]
                     for t in trajectories if t.get("final_score") is not None}
        pos = [final_map[tid] for tid in valid_tasks if final_map.get(tid) is not None]
        if len(pos) >= 3:
            series.append((variant, _VARIANT_COLORS.get(variant, "grey"), pos))

    labels, aurocs, lo_errs, hi_errs, colors = [], [], [], [], []
    baseline_auroc = np.nan

    for i, (name, color, pos) in enumerate(series):
        a, lo, hi = _auroc_with_ci(neg, pos)
        labels.append("Baseline\n(0f raw)" if name == "baseline"
                       else _VARIANT_LABELS.get(name, name))
        aurocs.append(a if not np.isnan(a) else 0.0)
        lo_errs.append(a - lo if not np.isnan(lo) else 0.0)
        hi_errs.append(hi - a if not np.isnan(hi) else 0.0)
        colors.append(color)
        if name == "baseline":
            baseline_auroc = a

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, aurocs, color=colors, alpha=0.85, edgecolor="white",
                  yerr=[lo_errs, hi_errs], capsize=5,
                  error_kw={"linewidth": 1.2, "capthick": 1.2})
    for bar, v in zip(bars, aurocs):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.axhline(0.5, color="red", linestyle="--", linewidth=1.5, label="Chance (0.5)")
    if not np.isnan(baseline_auroc):
        ax.axhline(baseline_auroc, color="grey", linestyle=":", linewidth=1.5,
                   label=f"Baseline: {baseline_auroc:.3f}")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0.2, 1.1)
    ax.set_ylabel("AUROC (Opus 4.5 evaluator)", fontsize=11)
    ax.set_title(
        "Opus 4.5 Detection AUROC — Final Score by Variant\n"
        "(error bars = 95% CI bootstrap; negative class = organic Opus from 0f)",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = config.figures_dir / "auroc_by_variant.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ---------------------------------------------------------------------------
# Fig 3: Score distributions — baseline vs final per variant
# ---------------------------------------------------------------------------

def _fig3_distributions(all_trajectories: dict, config: Experiment0kConfig) -> None:
    variants = list(all_trajectories.keys())
    fig, axes = plt.subplots(1, len(variants), figsize=(5 * len(variants), 4), sharey=True)
    if len(variants) == 1:
        axes = [axes]

    bins = np.linspace(0, 100, 21)
    for ax, variant in zip(axes, variants):
        trajectories = all_trajectories[variant]
        bl_scores = [t["iterations"][0]["tamper_prob"] for t in trajectories
                     if t["iterations"] and t["iterations"][0].get("tamper_prob") is not None]
        final_scores = [t["final_score"] for t in trajectories
                        if t.get("final_score") is not None]
        color = _VARIANT_COLORS.get(variant, "grey")
        ax.hist(bl_scores, bins=bins, alpha=0.5, color="#808080",
                label=f"Baseline (mean={np.mean(bl_scores):.0f})")
        ax.hist(final_scores, bins=bins, alpha=0.7, color=color,
                label=f"Final (mean={np.mean(final_scores):.0f})")
        ax.set_title(_VARIANT_LABELS.get(variant, variant).replace("\n", " "), fontsize=10)
        ax.set_xlabel("tamper_prob", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Count", fontsize=10)
    fig.suptitle("Score Distributions: Baseline vs Final Iteration", fontsize=11)
    fig.tight_layout()
    out = config.figures_dir / "score_distributions.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ---------------------------------------------------------------------------
# Fig 4: Mean tamper_prob ± 95% CI — Baseline vs each variant
# ---------------------------------------------------------------------------

def _fig4_mean_tamper(all_trajectories: dict, config: Experiment0kConfig) -> None:
    series: list[tuple[str, str, list[float]]] = []

    first_variant = next(iter(all_trajectories), None)
    if first_variant:
        bl_scores = [
            t["iterations"][0]["tamper_prob"]
            for t in all_trajectories[first_variant]
            if t["iterations"] and t["iterations"][0].get("tamper_prob") is not None
        ]
        if bl_scores:
            series.append(("Baseline\n(0f raw)", "#808080", bl_scores))

    for variant in config.variants:
        if variant not in all_trajectories:
            continue
        scores = [t["final_score"] for t in all_trajectories[variant]
                  if t.get("final_score") is not None]
        if scores:
            series.append((
                _VARIANT_LABELS.get(variant, variant),
                _VARIANT_COLORS.get(variant, "grey"),
                scores,
            ))

    if not series:
        return

    labels, means, lo_errs, hi_errs, colors = [], [], [], [], []
    for label, color, scores in series:
        m, lo, hi = _mean_ci(scores)
        labels.append(label)
        means.append(m)
        lo_errs.append(m - lo)
        hi_errs.append(hi - m)
        colors.append(color)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, color=colors, alpha=0.85, edgecolor="white",
                  yerr=[lo_errs, hi_errs], capsize=6,
                  error_kw={"linewidth": 1.3, "capthick": 1.3})
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, m + max(hi_errs) * 0.15 + 0.5,
                f"{m:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Mean Tamper Probability (Opus 4.5)", fontsize=11)
    ax.set_ylim(0, max(m + e for m, e in zip(means, hi_errs)) * 1.25)
    ax.set_title(
        "Mean Tamper Probability by Rewrite Variant\n"
        "(error bars = 95% CI bootstrap)",
        fontsize=11,
    )
    ax.axhline(config.convergence_threshold, color="grey", linestyle="--",
               linewidth=1.3, label=f"Convergence threshold ({config.convergence_threshold})")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = config.figures_dir / "mean_tamper_by_variant.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out.name}")


def _mean_ci(scores: list[float], n_bootstrap: int = 1000) -> tuple[float, float, float]:
    arr = np.array(scores)
    m = float(arr.mean())
    rng = np.random.default_rng(42)
    boot = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_bootstrap)]
    return m, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _save_summary(
    all_trajectories: dict,
    organic: dict[str, float],
    config: Experiment0kConfig,
) -> None:
    valid_tasks = [tid for tid in organic]
    neg = [organic[tid] for tid in valid_tasks]
    rows = []

    for variant, trajectories in all_trajectories.items():
        bl_scores = [t["iterations"][0]["tamper_prob"] for t in trajectories
                     if t["iterations"] and t["iterations"][0].get("tamper_prob") is not None]
        final_scores = [t["final_score"] for t in trajectories
                        if t.get("final_score") is not None]
        n_conv = sum(1 for t in trajectories if t["converged"])
        avg_iters = np.mean([t["n_iters"] for t in trajectories]) if trajectories else 0

        pos_final = [t["final_score"] for t in trajectories
                     if t.get("final_score") is not None
                     and organic.get(t["task_id"]) is not None]
        a, lo, hi = _auroc_with_ci(neg, pos_final) if len(pos_final) >= 3 else (np.nan,)*3

        rows.append({
            "variant":         variant,
            "n_tasks":         len(trajectories),
            "n_converged":     n_conv,
            "avg_iters":       round(float(avg_iters), 1),
            "mean_bl":         round(float(np.mean(bl_scores)), 1) if bl_scores else None,
            "mean_final":      round(float(np.mean(final_scores)), 1) if final_scores else None,
            "auroc_final":     round(float(a), 3) if not np.isnan(a) else None,
            "auroc_ci_lo":     round(float(lo), 3) if not np.isnan(lo) else None,
            "auroc_ci_hi":     round(float(hi), 3) if not np.isnan(hi) else None,
        })

    summary = pd.DataFrame(rows)
    out = config.results_dir / "summary.csv"
    summary.to_csv(out, index=False)
    print(f"\n  Summary:\n{summary.to_string(index=False)}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_organic_baselines(config: Experiment0kConfig) -> dict[str, float]:
    det_path = config.experiment0f_dir / "results" / "detection_results.json"
    if not det_path.exists():
        return {}
    detections = json.loads(det_path.read_text())
    return {
        r["task_id"]: r["tamper_prob"]
        for r in detections
        if r.get("source") == "organic"
        and r.get("dataset") == "bigcodebench"
        and r.get("tamper_prob") is not None
    }


def _auroc_with_ci(neg, pos, n_bootstrap=1000):
    if len(neg) < 3 or len(pos) < 3:
        return np.nan, np.nan, np.nan
    yt = [0]*len(neg) + [1]*len(pos)
    ys = list(neg) + list(pos)
    try:
        a = roc_auc_score(yt, ys)
        rng = np.random.default_rng(42)
        boot = []
        for _ in range(n_bootstrap):
            idx = rng.choice(len(yt), size=len(yt), replace=True)
            yt_b = [yt[i] for i in idx]; ys_b = [ys[i] for i in idx]
            if len(set(yt_b)) < 2: continue
            boot.append(roc_auc_score(yt_b, ys_b))
        if not boot: return float(a), np.nan, np.nan
        return float(a), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    except Exception:
        return np.nan, np.nan, np.nan
