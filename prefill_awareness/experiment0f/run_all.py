"""
Experiment 0f: Opus 4.5 single-evaluator on real benchmarks, clean prompts.

Isolates the impact of role personas and response truncation on AUROC by
running Opus 4.5 against three source models (Sonnet 4.5, GPT-4o-mini,
Gemini Flash) with no persona instructions and no max_tokens truncation.

Run from prefill_awareness/:
    python -m experiment0f.run_all
    python -m experiment0f.run_all --from-step detection
    python -m experiment0f.run_all --from-step analysis
    python -m experiment0f.run_all --skip-verify
"""

import argparse
import json
import sys
from pathlib import Path

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

from experiment0f.analysis import run_analysis
from experiment0f.config import Experiment0fConfig
from experiment0f.benchmarks import load_all_tasks
from experiment0f.generate import generate_all
from experiment0f.run_detection import run_detection
from experiment0f.utils import CostTracker, verify_all_models

_STEP_ORDER = ["generate", "detection", "analysis"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 0f: Opus 4.5 single-evaluator, clean prompts"
    )
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing responses and results.")
    parser.add_argument("--from-step", choices=_STEP_ORDER, default=None,
                        metavar="STEP",
                        help=f"Start from this step: {', '.join(_STEP_ORDER)}")
    parser.add_argument("--skip-verify", action="store_true",
                        help="Skip OpenRouter model ID verification.")
    parser.add_argument("--n-tasks", type=int, default=None,
                        help="Override n_tasks_per_dataset (for quick tests).")
    parser.add_argument("--datasets", nargs="+", default=None,
                        metavar="DATASET",
                        help="Restrict to specific datasets. Choices: swebench bigcodebench gpqa")
    return parser.parse_args()


def _should_run(step: str, from_step: str | None) -> bool:
    if from_step is None:
        return True
    return _STEP_ORDER.index(step) >= _STEP_ORDER.index(from_step)


def main() -> None:
    args = parse_args()
    config = Experiment0fConfig()
    if args.n_tasks is not None:
        config.n_tasks_per_dataset = args.n_tasks

    cost_tracker = CostTracker(config, label="experiment0f")

    print(f"\n=== Experiment 0f ===")
    print(f"  Evaluator:         {config.evaluator_name} ({config.evaluator_id})")
    print(f"  Sources:           {list(config.sources.keys())}")
    print(f"  Tasks per dataset: {config.n_tasks_per_dataset}")
    print(f"  Max tokens (gen):  {config.max_tokens_generation}")
    print(f"  Cost cap:          ${config.cost_cap:.2f}")

    if not args.skip_verify:
        print("\n=== Verifying model IDs on OpenRouter ===")
        if not verify_all_models(config):
            print("\nUpdate config.py with correct IDs, then re-run.")
            sys.exit(1)

    print("\n=== Loading datasets ===")
    tasks = load_all_tasks(config.n_tasks_per_dataset)
    if args.datasets:
        tasks = [t for t in tasks if t["dataset"] in args.datasets]
        print(f"  Filtered to {args.datasets}: {len(tasks)} tasks")

    # ------------------------------------------------------------------ #
    # Step 1: Generate responses
    # ------------------------------------------------------------------ #
    if _should_run("generate", args.from_step):
        print(f"\n=== Step 1: Generating responses ===")
        generate_all(tasks, config, cost_tracker, force=args.force)
    else:
        print("\n[Skip Step 1]")

    # ------------------------------------------------------------------ #
    # Step 2: Detection
    # ------------------------------------------------------------------ #
    if _should_run("detection", args.from_step):
        print(f"\n=== Step 2: Running detection ===")
        results = run_detection(tasks, config, cost_tracker, force=args.force)
    else:
        results_path = config.results_dir / "detection_results.json"
        print(f"\n[Skip Step 2] Loading from {results_path}")
        with open(results_path) as f:
            results = json.load(f)

    print(f"  {len(results)} detection records.")

    # ------------------------------------------------------------------ #
    # Step 3: Analysis
    # ------------------------------------------------------------------ #
    if _should_run("analysis", args.from_step):
        print("\n=== Step 3: Analysis and figures ===")
        run_analysis(results, config)

    print(f"\n=== Experiment 0f complete ===")
    print(f"  Results: {config.results_dir.resolve()}")
    print(f"  Figures: {config.figures_dir.resolve()}")
    print(cost_tracker.report())
    _print_auroc_summary(results, config)


def _print_auroc_summary(results: list[dict], config: Experiment0fConfig) -> None:
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    df = pd.DataFrame(results)
    if len(df) == 0:
        return
    df["tamper_prob"] = pd.to_numeric(df["tamper_prob"], errors="coerce")
    df = df.dropna(subset=["tamper_prob"])

    org = df[df["source"] == "organic"]["tamper_prob"]
    print("\n  AUROC per source (all datasets):")
    for src in config.sources:
        sub = df[df["source"] == src]["tamper_prob"].dropna()
        if len(sub) < 5 or len(org) < 5:
            print(f"    {src}: insufficient data")
            continue
        y = [0] * len(org) + [1] * len(sub)
        s = list(org) + list(sub)
        if len(set(y)) < 2:
            continue
        try:
            fam = "within" if config.is_within_family(src) else "cross"
            a = roc_auc_score(y, s)
            print(f"    {src:15s} ({fam}): AUROC={a:.3f}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
