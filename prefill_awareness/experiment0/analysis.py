"""
Analysis and figures for Experiment 0.

Figures:
  heatmap_not_me_rate.png          — "not me" rate by (evaluator, source)
  balanced_accuracy_by_evaluator.png — balanced accuracy per evaluator
  not_me_rate_by_dataset.png       — off-diagonal "not me" rate by dataset, per evaluator
  self_attribution_rate.png        — P("me" | source=self) per evaluator

Summary: results_dir/summary_table.csv
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import Experiment0Config

_MODEL_LABELS = {
    "llama_3b":    "Llama\n3.2-3B",
    "llama_70b":   "Llama\n3.3-70B",
    "gemma_27b":   "Gemma\n3-27B",
    "mistral_24b": "Mistral\n24B",
    "sonnet_45":   "Sonnet\n4.5",
    "opus_45":     "Opus\n4.5",
}
_DS_LABELS = {"alpaca": "Alpaca", "oasst1": "OASST1", "mmlu": "MMLU"}


def run_analysis(results: list[dict], config: Experiment0Config) -> None:
    df = pd.DataFrame(results)
    # Exclude unparseable from rate calculations (treat as missing)
    df_parsed = df[df["parsed"] != "unparseable"].copy()

    model_keys = list(config.models.keys())

    _figure_heatmap(df_parsed, model_keys, config)
    _figure_balanced_accuracy(df_parsed, model_keys, config)
    _figure_by_dataset(df_parsed, model_keys, config)
    _figure_self_attribution(df_parsed, model_keys, config)
    _save_summary(df_parsed, model_keys, config)


# ---------------------------------------------------------------------------
# Figure 1: "not me" rate heatmap (evaluator × source)
# ---------------------------------------------------------------------------

def _figure_heatmap(df: pd.DataFrame, model_keys: list[str], config: Experiment0Config) -> None:
    matrix = np.full((len(model_keys), len(model_keys)), np.nan)
    for i, ev in enumerate(model_keys):
        for j, src in enumerate(model_keys):
            sub = df[(df["evaluator"] == ev) & (df["source"] == src)]
            if len(sub) > 0:
                matrix[i, j] = (sub["parsed"] == "not_me").mean()

    labels = [_MODEL_LABELS.get(k, k) for k in model_keys]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        matrix,
        ax=ax,
        annot=True,
        fmt=".2f",
        vmin=0, vmax=1,
        cmap="RdYlGn",
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        linecolor="white",
    )
    ax.set_xlabel("Source model (whose response was prefilled)")
    ax.set_ylabel('Evaluator model (asked "did you write this?")')
    ax.set_title('P("not me") — diagonal = false positive rate, off-diagonal = detection rate')
    plt.tight_layout()
    out = config.results_dir / "heatmap_not_me_rate.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure 2: Balanced accuracy by evaluator
# ---------------------------------------------------------------------------

def _figure_balanced_accuracy(
    df: pd.DataFrame, model_keys: list[str], config: Experiment0Config
) -> None:
    bal_accs = []
    for ev in model_keys:
        self_df = df[(df["evaluator"] == ev) & (df["is_self"])]
        other_df = df[(df["evaluator"] == ev) & (~df["is_self"])]
        sensitivity = (self_df["parsed"] == "me").mean() if len(self_df) else np.nan
        specificity = (other_df["parsed"] == "not_me").mean() if len(other_df) else np.nan
        bal_accs.append((sensitivity + specificity) / 2 if not np.isnan(sensitivity + specificity) else np.nan)

    labels = [_MODEL_LABELS.get(k, k).replace("\n", " ") for k in model_keys]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, bal_accs, color="steelblue", edgecolor="white")
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1, label="Chance (0.50)")
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("Balanced accuracy by evaluator model")
    ax.set_ylim(0, 1)
    ax.legend()
    for bar, v in zip(bars, bal_accs):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    out = config.results_dir / "balanced_accuracy_by_evaluator.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure 3: Off-diagonal "not me" rate by dataset, per evaluator
# ---------------------------------------------------------------------------

def _figure_by_dataset(
    df: pd.DataFrame, model_keys: list[str], config: Experiment0Config
) -> None:
    datasets = config.datasets
    n_ev = len(model_keys)
    n_ds = len(datasets)
    x = np.arange(n_ev)
    width = 0.8 / n_ds
    colors = ["#4c72b0", "#dd8452", "#55a868"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for di, ds in enumerate(datasets):
        rates = []
        for ev in model_keys:
            sub = df[(df["evaluator"] == ev) & (~df["is_self"]) & (df["dataset"] == ds)]
            rates.append((sub["parsed"] == "not_me").mean() if len(sub) else np.nan)
        offset = (di - n_ds / 2 + 0.5) * width
        bars = ax.bar(x + offset, rates, width, label=_DS_LABELS.get(ds, ds), color=colors[di], edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([_MODEL_LABELS.get(k, k).replace("\n", " ") for k in model_keys])
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1, label="Chance")
    ax.set_ylabel('P("not me" | source ≠ self)')
    ax.set_title('Off-diagonal "not me" rate by evaluator and dataset')
    ax.set_ylim(0, 1)
    ax.legend()
    plt.tight_layout()
    out = config.results_dir / "not_me_rate_by_dataset.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure 4: Self-attribution rate (diagonal)
# ---------------------------------------------------------------------------

def _figure_self_attribution(
    df: pd.DataFrame, model_keys: list[str], config: Experiment0Config
) -> None:
    rates = []
    for ev in model_keys:
        sub = df[(df["evaluator"] == ev) & (df["is_self"])]
        rates.append((sub["parsed"] == "me").mean() if len(sub) else np.nan)

    labels = [_MODEL_LABELS.get(k, k).replace("\n", " ") for k in model_keys]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, rates, color="darkorange", edgecolor="white")
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1, label="Chance (0.50)")
    ax.set_ylabel('P("me" | source = self)')
    ax.set_title('Self-attribution rate — models asked to recognise their own output')
    ax.set_ylim(0, 1)
    ax.legend()
    for bar, v in zip(bars, rates):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    out = config.results_dir / "self_attribution_rate.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _save_summary(df: pd.DataFrame, model_keys: list[str], config: Experiment0Config) -> None:
    df_all = pd.DataFrame()  # include unparseable for unparseable% column
    raw_path = config.results_dir / "attribution_results.json"
    if raw_path.exists():
        import json
        with open(raw_path) as f:
            df_all = pd.DataFrame(json.load(f))

    rows = []
    for ev in model_keys:
        self_df = df[(df["evaluator"] == ev) & (df["is_self"])]
        other_df = df[(df["evaluator"] == ev) & (~df["is_self"])]
        sensitivity = (self_df["parsed"] == "me").mean() if len(self_df) else np.nan
        specificity = (other_df["parsed"] == "not_me").mean() if len(other_df) else np.nan
        bal_acc = (sensitivity + specificity) / 2 if not np.isnan(sensitivity + specificity) else np.nan

        if len(df_all) > 0:
            ev_all = df_all[df_all["evaluator"] == ev]
            unp_rate = (ev_all["parsed"] == "unparseable").mean() if len(ev_all) else np.nan
        else:
            unp_rate = np.nan

        rows.append({
            "evaluator": ev,
            "P(me|self)": round(sensitivity, 3) if not np.isnan(sensitivity) else None,
            "P(not_me|other)": round(specificity, 3) if not np.isnan(specificity) else None,
            "balanced_accuracy": round(bal_acc, 3) if not np.isnan(bal_acc) else None,
            "unparseable_pct": round(unp_rate * 100, 1) if not np.isnan(unp_rate) else None,
        })

    summary = pd.DataFrame(rows)
    out = config.results_dir / "summary_table.csv"
    summary.to_csv(out, index=False)
    print(f"\n  Saved {out}")
    print("\n  Summary:")
    print(summary.to_string(index=False))
