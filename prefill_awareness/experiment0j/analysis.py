"""
Analysis for Experiment 0j.

Produces three figures and a summary CSV:

  Fig 1: iterative_convergence.png
         Thin lines per task (green=converged, red=plateaued), thick mean line.
         X = iteration 0-5, Y = tamper_prob.

  Fig 2: feature_importance.png
         Two-panel bar chart. Left: forward direction (Gemini→Claude, Δ<0 = good).
         Right: reverse direction (Claude→Gemini, Δ>0 = expected).
         Sorted by absolute effect within each panel.

  Fig 3: feature_stacking.png
         Individual top features vs combined top-2 and top-3.
         Shows whether effects are additive or subadditive.

  summary_tables.csv — all per-task and per-feature stats.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import Experiment0jConfig


def run_analysis(
    trajectories: list[dict],
    phase2_scores: list[dict],
    config: Experiment0jConfig,
) -> None:
    # ------------------------------------------------------------------ #
    # 0f baselines (iter-0 Gemini scores and organic Opus scores)         #
    # ------------------------------------------------------------------ #
    forward_baselines = _load_forward_baselines(trajectories)
    reverse_baselines = _load_reverse_baselines(config)

    # ------------------------------------------------------------------ #
    # Fig 1: Iterative convergence                                         #
    # ------------------------------------------------------------------ #
    _fig1_convergence(trajectories, config)

    # ------------------------------------------------------------------ #
    # Fig 2 & 3: Feature importance and stacking                          #
    # ------------------------------------------------------------------ #
    if phase2_scores:
        df2 = pd.DataFrame(phase2_scores)
        df2["tamper_prob"] = pd.to_numeric(df2["tamper_prob"], errors="coerce")
        df2 = df2.dropna(subset=["tamper_prob"])

        feature_effects = _fig2_feature_importance(
            df2, forward_baselines, reverse_baselines, config
        )
        _fig3_stacking(df2, feature_effects, forward_baselines, config)
    else:
        print("  [analysis] No phase2 scores — skipping Figs 2 & 3.")
        df2 = pd.DataFrame()
        feature_effects = {}

    # ------------------------------------------------------------------ #
    # Fig 4: AUROC comparison across methods (mirrors 0g auroc_by_method) #
    # ------------------------------------------------------------------ #
    _fig4_auroc_comparison(trajectories, df2, config)

    # ------------------------------------------------------------------ #
    # Summary tables                                                       #
    # ------------------------------------------------------------------ #
    _save_summary(trajectories, phase2_scores, forward_baselines, reverse_baselines,
                  feature_effects, config)

    print(f"  Figures → {config.figures_dir.resolve()}")
    print(f"  Results → {config.results_dir.resolve()}")


# ---------------------------------------------------------------------------
# Figure 1: Iterative convergence curves
# ---------------------------------------------------------------------------

def _fig1_convergence(trajectories: list[dict], config: Experiment0jConfig) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    all_iters = []
    for traj in trajectories:
        iters = [it["iter"] for it in traj["iterations"]]
        scores = [it.get("tamper_prob") for it in traj["iterations"]]
        scores_clean = [s if s is not None else float("nan") for s in scores]

        color = "#2ca02c" if traj["converged"] else "#d62728"
        ax.plot(iters, scores_clean, color=color, alpha=0.35, linewidth=1)
        all_iters.append((iters, scores_clean))

    # Compute mean trajectory at each iter position
    max_iter = max(
        (it["iter"] for t in trajectories for it in t["iterations"]), default=0
    )
    mean_by_iter = {}
    for i in range(max_iter + 1):
        vals = []
        for traj in trajectories:
            for it in traj["iterations"]:
                if it["iter"] == i and it.get("tamper_prob") is not None:
                    vals.append(it["tamper_prob"])
        if vals:
            mean_by_iter[i] = np.mean(vals)

    if mean_by_iter:
        xs = sorted(mean_by_iter)
        ys = [mean_by_iter[x] for x in xs]
        ax.plot(xs, ys, color="black", linewidth=2.5, label="Mean", zorder=5)

    ax.axhline(config.convergence_threshold, color="grey", linestyle="--",
               linewidth=1, label=f"Threshold ({config.convergence_threshold})")

    n_conv = sum(1 for t in trajectories if t["converged"])
    n_total = len(trajectories)
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("tamper_prob (Opus score)", fontsize=12)
    ax.set_title(
        f"Iterative Refinement: Convergence Curves\n"
        f"{n_conv}/{n_total} tasks converged (tamper_prob < {config.convergence_threshold})",
        fontsize=12,
    )
    ax.set_xlim(-0.2, max_iter + 0.2)
    ax.set_ylim(-5, 105)
    ax.set_xticks(range(max_iter + 1))

    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color="#2ca02c", alpha=0.6, label="Converged"),
        Line2D([0], [0], color="#d62728", alpha=0.6, label="Plateaued"),
        Line2D([0], [0], color="black", linewidth=2.5, label="Mean"),
        Line2D([0], [0], color="grey", linestyle="--", label=f"Threshold ({config.convergence_threshold})"),
    ]
    ax.legend(handles=legend_elems, fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(config.figures_dir / "iterative_convergence.png", dpi=150)
    plt.close(fig)
    print("  Saved: iterative_convergence.png")


# ---------------------------------------------------------------------------
# Figure 2: Feature importance
# ---------------------------------------------------------------------------

def _fig2_feature_importance(
    df: pd.DataFrame,
    forward_baselines: dict[str, float],
    reverse_baselines: dict[str, float],
    config: Experiment0jConfig,
) -> dict[str, float]:
    """Returns {feature_name: mean_delta} for forward features (sorted by effect)."""
    forward_df = df[df["direction"] == "forward"]
    reverse_df = df[df["direction"] == "reverse"]

    def _mean_delta(sub_df, baselines: dict[str, float]) -> dict[str, float]:
        effects = {}
        for feature, grp in sub_df.groupby("feature"):
            deltas = []
            for _, row in grp.iterrows():
                bl = baselines.get(row["task_id"])
                if bl is not None:
                    deltas.append(row["tamper_prob"] - bl)
            if deltas:
                effects[feature] = float(np.mean(deltas))
        return effects

    fwd_effects = _mean_delta(forward_df, forward_baselines)
    rev_effects = _mean_delta(reverse_df, reverse_baselines)

    if not fwd_effects and not rev_effects:
        print("  [analysis] No feature data to plot for Fig 2.")
        return {}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

    def _plot_panel(ax, effects: dict, title: str, expected_sign: str) -> None:
        if not effects:
            ax.set_visible(False)
            return
        sorted_items = sorted(effects.items(), key=lambda x: abs(x[1]), reverse=True)
        names = [_short_name(k) for k, _ in sorted_items]
        vals = [v for _, v in sorted_items]
        colors = [
            "#2ca02c" if (expected_sign == "neg" and v < 0) or (expected_sign == "pos" and v > 0)
            else "#d62728"
            for v in vals
        ]
        bars = ax.barh(names, vals, color=colors, alpha=0.8, edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Mean Δ tamper_prob", fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="x", alpha=0.3)
        for bar, val in zip(bars, vals):
            ax.text(
                val + (1 if val >= 0 else -1), bar.get_y() + bar.get_height() / 2,
                f"{val:+.1f}", va="center", ha="left" if val >= 0 else "right",
                fontsize=9,
            )

    _plot_panel(
        axes[0], fwd_effects,
        "Forward: Gemini → Claude style\n(negative = fewer detections)",
        "neg",
    )
    _plot_panel(
        axes[1], rev_effects,
        "Reverse: Claude → Gemini style\n(positive = more detections)",
        "pos",
    )

    fig.suptitle("Feature Importance: Mean Change in Opus tamper_prob", fontsize=13)
    fig.tight_layout()
    fig.savefig(config.figures_dir / "feature_importance.png", dpi=150)
    plt.close(fig)
    print("  Saved: feature_importance.png")

    return fwd_effects


# ---------------------------------------------------------------------------
# Figure 3: Feature stacking
# ---------------------------------------------------------------------------

def _fig3_stacking(
    df: pd.DataFrame,
    forward_effects: dict[str, float],
    forward_baselines: dict[str, float],
    config: Experiment0jConfig,
) -> None:
    stack_df = df[df["direction"] == "stacking"]
    if stack_df.empty or not forward_effects:
        print("  [analysis] No stacking data — skipping Fig 3.")
        return

    sorted_features = sorted(forward_effects, key=lambda k: forward_effects[k])  # most negative first
    top_features = sorted_features[:3]

    # Collect effects
    labels = []
    means = []
    sems = []

    for f in top_features[:2]:
        labels.append(_short_name(f))
        means.append(forward_effects.get(f, float("nan")))
        sems.append(0.0)

    for n in [2, 3]:
        combo_key = f"top{n}_" + "_".join(top_features[:n])
        sub = stack_df[stack_df["feature"] == combo_key]
        if sub.empty:
            continue
        deltas = [
            row["tamper_prob"] - forward_baselines.get(row["task_id"], row["tamper_prob"])
            for _, row in sub.iterrows()
            if forward_baselines.get(row["task_id"]) is not None
        ]
        if not deltas:
            continue
        labels.append(f"Top-{n} combined")
        means.append(float(np.mean(deltas)))
        sems.append(float(np.std(deltas) / np.sqrt(len(deltas))))

    if len(top_features) >= 3:
        labels.insert(2, _short_name(top_features[2]))
        means.insert(2, forward_effects.get(top_features[2], float("nan")))
        sems.insert(2, 0.0)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#1f77b4"] * min(len(top_features), len(labels)) + ["#ff7f0e", "#ff7f0e"]
    colors = colors[: len(labels)]
    bars = ax.bar(range(len(labels)), means, color=colors, alpha=0.8, edgecolor="white",
                  yerr=sems, capsize=4)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=10)
    ax.axhline(0, color="black", linewidth=0.8)

    # Expected additive sum lines
    if len(means) >= 4:
        indiv_sum2 = means[0] + means[1]
        ax.axhline(indiv_sum2, color="#1f77b4", linestyle="--", linewidth=1,
                   label="Additive sum (top-2)")
    if len(means) >= 5:
        indiv_sum3 = means[0] + means[1] + means[2]
        ax.axhline(indiv_sum3, color="#9467bd", linestyle="--", linewidth=1,
                   label="Additive sum (top-3)")

    ax.set_ylabel("Mean Δ tamper_prob", fontsize=11)
    ax.set_title("Feature Stacking: Individual vs Combined Effects", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.figures_dir / "feature_stacking.png", dpi=150)
    plt.close(fig)
    print("  Saved: feature_stacking.png")


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------

def _save_summary(
    trajectories: list[dict],
    phase2_scores: list[dict],
    forward_baselines: dict[str, float],
    reverse_baselines: dict[str, float],
    feature_effects: dict[str, float],
    config: Experiment0jConfig,
) -> None:
    rows = []

    # Convergence table
    for traj in trajectories:
        iter_scores = {it["iter"]: it.get("tamper_prob") for it in traj["iterations"]}
        row = {
            "task_id": traj["task_id"],
            "converged": traj["converged"],
            "convergence_iter": traj.get("convergence_iter"),
            "final_score": traj.get("final_score"),
            "n_iters": traj.get("n_iters"),
        }
        for i in range(config.max_iterations + 1):
            row[f"iter{i}"] = iter_scores.get(i)
        rows.append(row)

    conv_df = pd.DataFrame(rows)
    conv_df.to_csv(config.results_dir / "convergence_table.csv", index=False)

    # Feature importance table
    if phase2_scores:
        df = pd.DataFrame(phase2_scores)
        df["tamper_prob"] = pd.to_numeric(df["tamper_prob"], errors="coerce")
        feat_rows = []
        for (direction, feature), grp in df.groupby(["direction", "feature"]):
            baselines = forward_baselines if direction in ("forward", "stacking") else reverse_baselines
            deltas = [
                row["tamper_prob"] - baselines.get(row["task_id"], float("nan"))
                for _, row in grp.iterrows()
                if not np.isnan(baselines.get(row["task_id"], float("nan")))
                and not np.isnan(row["tamper_prob"])
            ]
            if deltas:
                feat_rows.append({
                    "direction": direction,
                    "feature": feature,
                    "N": len(deltas),
                    "mean_delta": round(float(np.mean(deltas)), 2),
                    "se": round(float(np.std(deltas) / np.sqrt(len(deltas))), 2),
                    "mean_score": round(float(grp["tamper_prob"].mean()), 2),
                })
        if feat_rows:
            pd.DataFrame(feat_rows).to_csv(
                config.results_dir / "feature_importance_table.csv", index=False
            )

    print(f"  Saved: convergence_table.csv, feature_importance_table.csv")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_forward_baselines(trajectories: list[dict]) -> dict[str, float]:
    """Iter-0 Gemini scores from 0f, keyed by task_id."""
    return {
        t["task_id"]: t["iterations"][0]["tamper_prob"]
        for t in trajectories
        if t["iterations"] and t["iterations"][0].get("tamper_prob") is not None
    }


def _load_reverse_baselines(config: Experiment0jConfig) -> dict[str, float]:
    """0f organic (Opus self) scores for the same tasks, keyed by task_id."""
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


def _fig4_auroc_comparison(
    trajectories: list[dict],
    df2: pd.DataFrame,
    config: Experiment0jConfig,
) -> None:
    """
    AUROC for each 0j method (negative class = organic Opus from 0f).
    Mirrors experiment0g's auroc_by_method.png.
    Skips methods with no data; safe to call with partial results.
    """
    organic = _load_reverse_baselines(config)
    task_ids = [t["task_id"] for t in trajectories]
    valid_tasks = [tid for tid in task_ids if organic.get(tid) is not None]
    neg = [organic[tid] for tid in valid_tasks]

    if len(neg) < 3:
        print("  [analysis] Insufficient organic baselines for AUROC comparison — skipping Fig 4.")
        return

    # Collect (label, pos_scores) for each method
    methods: list[tuple[str, list[float]]] = []

    # 0f baseline — iter-0 scores (raw Gemini, unmodified)
    bl_map = {t["task_id"]: t["iterations"][0]["tamper_prob"]
              for t in trajectories
              if t["iterations"] and t["iterations"][0].get("tamper_prob") is not None}
    bl_pos = [bl_map[tid] for tid in valid_tasks if bl_map.get(tid) is not None]
    if len(bl_pos) == len(valid_tasks):
        methods.append(("0f Baseline\n(raw Gemini)", bl_pos))

    # Phase 1 final refined scores
    p1_map = {t["task_id"]: t["final_score"]
              for t in trajectories if t.get("final_score") is not None}
    p1_pos = [p1_map[tid] for tid in valid_tasks if p1_map.get(tid) is not None]
    if len(p1_pos) >= 3:
        methods.append(("P1: Refined\n(iterative)", p1_pos))

    # Phase 2 forward features
    if not df2.empty:
        fwd_df = df2[df2["direction"] == "forward"]
        for feature in config.forward_features:
            sub = fwd_df[fwd_df["feature"] == feature].set_index("task_id")
            pos = [sub.loc[tid, "tamper_prob"]
                   for tid in valid_tasks
                   if tid in sub.index and not np.isnan(sub.loc[tid, "tamper_prob"])]
            if len(pos) >= 3:
                methods.append((f"Fwd:\n{_short_name(feature)}", pos))

        # Stacking
        stack_df = df2[df2["direction"] == "stacking"]
        for combo_key in sorted(stack_df["feature"].unique()):
            sub = stack_df[stack_df["feature"] == combo_key].set_index("task_id")
            pos = [sub.loc[tid, "tamper_prob"]
                   for tid in valid_tasks
                   if tid in sub.index and not np.isnan(sub.loc[tid, "tamper_prob"])]
            n = int(combo_key[3]) if combo_key.startswith("top") else 0
            if len(pos) >= 3:
                methods.append((f"Stack:\nTop-{n}", pos))

    if not methods:
        print("  [analysis] No method data for Fig 4.")
        return

    labels, aurocs, lo_errs, hi_errs = [], [], [], []
    baseline_auroc = np.nan

    for i, (label, pos) in enumerate(methods):
        a, lo, hi = _auroc_with_ci(neg, pos)
        labels.append(label)
        aurocs.append(a if not np.isnan(a) else 0.0)
        lo_errs.append(a - lo if not np.isnan(a) else 0.0)
        hi_errs.append(hi - a if not np.isnan(a) else 0.0)
        if i == 0:
            baseline_auroc = a  # 0f baseline becomes the reference line

    colors = (
        ["#808080"]            # 0f baseline (grey)
        + ["#4878CF"]          # Phase 1 (blue)
        + ["#59A14F"] * len(config.forward_features)  # forward features (green)
        + ["#F28E2B", "#E15759"][:max(0, len(methods) - 2 - len(config.forward_features))]
    )
    colors = colors[: len(labels)]

    fig, ax = plt.subplots(figsize=(max(9, 2 * len(labels)), 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, aurocs, color=colors, alpha=0.85, edgecolor="white",
                  yerr=[lo_errs, hi_errs], capsize=5,
                  error_kw={"linewidth": 1.2, "capthick": 1.2})

    for bar, v in zip(bars, aurocs):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.axhline(0.5, color="red", linestyle="--", linewidth=1.5, label="Chance (0.5)")
    if not np.isnan(baseline_auroc):
        ax.axhline(baseline_auroc, color="grey", linestyle=":", linewidth=1.5,
                   label=f"0f baseline: {baseline_auroc:.3f}")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0.2, 1.1)
    ax.set_ylabel("AUROC (Opus 4.5 evaluator)", fontsize=11)
    ax.set_title(
        "Opus 4.5 Detection AUROC — Experiment 0j Methods\n"
        "(BigCodeBench; error bars = 95% CI bootstrap; negative class = organic Opus from 0f)",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.figures_dir / "auroc_by_method.png", dpi=150)
    plt.close(fig)
    print("  Saved: auroc_by_method.png")


def _auroc_with_ci(
    neg: list[float],
    pos: list[float],
    n_bootstrap: int = 1000,
) -> tuple[float, float, float]:
    if len(neg) < 3 or len(pos) < 3:
        return np.nan, np.nan, np.nan
    yt = [0] * len(neg) + [1] * len(pos)
    ys = list(neg) + list(pos)
    if len(set(yt)) < 2:
        return np.nan, np.nan, np.nan
    try:
        a = roc_auc_score(yt, ys)
        rng = np.random.default_rng(42)
        n_obs = len(yt)
        boot = []
        for _ in range(n_bootstrap):
            idx = rng.choice(n_obs, size=n_obs, replace=True)
            yt_b = [yt[i] for i in idx]
            ys_b = [ys[i] for i in idx]
            if len(set(yt_b)) < 2:
                continue
            boot.append(roc_auc_score(yt_b, ys_b))
        if not boot:
            return float(a), np.nan, np.nan
        return float(a), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    except Exception:
        return np.nan, np.nan, np.nan


def _short_name(feature: str) -> str:
    return feature.replace("_", " ").replace("no ", "no-").title()
