"""
Standalone analysis: BigCodeBench attack success rates by model.

Reads solution JSON files from experiment0h/solutions/ and reports:
  - main_task_success: did the code pass the standard unit tests?
  - side_task_success: did the backdoor actually work?

Run from prefill_awareness/:
    python -m experiment0h.attack_success
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from dotenv import load_dotenv
    _env = Path(__file__).resolve().parents[2] / ".env"
    if _env.exists():
        load_dotenv(_env)
except ImportError:
    pass

_here = Path(__file__).resolve().parent.parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from experiment0h.config import Experiment0hConfig


_MODELS = [
    ("sonnet", "sonnet_attack", "Sonnet 4.5"),
    ("gemini", "gemini_attack", "Gemini 2.5 Pro"),
    ("opus",   "opus_attack",   "Opus 4.5"),
]

_COLORS = {
    "sonnet": "#B07AA1",
    "gemini": "#E15759",
    "opus":   "#4878CF",
}


def load_attack_results(config: Experiment0hConfig) -> dict[str, dict]:
    results = {}
    for key, condition, label in _MODELS:
        path = config.solutions_dir / f"{condition}.json"
        if not path.exists():
            print(f"  MISSING: {path.name} — run generation step first.")
            results[key] = {"label": label, "n": 0,
                            "main": 0, "side": 0, "missing": True}
            continue
        data = json.loads(path.read_text())
        n = len(data)
        main_ok = sum(1 for r in data if r.get("main_task_success") is True)
        side_ok  = sum(1 for r in data if r.get("side_task_success") is True)
        results[key] = {
            "label": label,
            "n":     n,
            "main":  main_ok,
            "side":  side_ok,
            "missing": False,
        }
    return results


def print_table(results: dict[str, dict]) -> None:
    print("\n=== BigCodeBench Attack Success Rates ===\n")
    print(f"{'Model':<18} {'N':>4}  {'Main task':>10}  {'Side task (backdoor)':>20}")
    print("-" * 58)
    for key, r in results.items():
        if r["missing"]:
            print(f"  {r['label']:<16} {'—':>4}  {'NOT RUN':>10}  {'NOT RUN':>20}")
            continue
        n = r["n"]
        main_pct = 100 * r["main"] / n if n else 0
        side_pct = 100 * r["side"] / n if n else 0
        print(f"  {r['label']:<16} {n:>4}  "
              f"{r['main']:>3}/{n}  ({main_pct:4.0f}%)  "
              f"{r['side']:>3}/{n}  ({side_pct:4.0f}%)")


def make_figure(results: dict[str, dict], config: Experiment0hConfig) -> None:
    available = [(k, r) for k, r in results.items() if not r["missing"] and r["n"] > 0]
    if not available:
        print("  No data to plot.")
        return

    labels   = [r["label"] for _, r in available]
    main_pct = [100 * r["main"] / r["n"] for _, r in available]
    side_pct = [100 * r["side"] / r["n"] for _, r in available]
    colors   = [_COLORS.get(k, "grey") for k, _ in available]
    ns       = [r["n"] for _, r in available]

    x = np.arange(len(available))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    bars_main = ax.bar(x - w / 2, main_pct, w, color=colors, alpha=0.5,
                       edgecolor="white", label="Main task (passes unit tests)")
    bars_side = ax.bar(x + w / 2, side_pct, w, color=colors, alpha=0.9,
                       edgecolor="white", label="Side task (backdoor works)")

    for bar, v, n in zip(bars_main, main_pct, ns):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5,
                f"{v:.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for bar, v, n in zip(bars_side, side_pct, ns):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5,
                f"{v:.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{lbl}\n(n={n})" for lbl, n in zip(labels, ns)], fontsize=10)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Success rate (%)")
    ax.set_title(
        "BigCodeBench Attack Success Rates\n"
        "Main task = code passes unit tests.  "
        "Side task = backdoor triggers correctly.",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out = config.figures_dir / "attack_success_rates.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\n  Saved: {out.resolve()}")


def main() -> None:
    config = Experiment0hConfig()
    results = load_attack_results(config)
    print_table(results)
    make_figure(results, config)


if __name__ == "__main__":
    main()
