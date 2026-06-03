"""
Experiment 0g: Style transfer — can Gemini Pro / GPT-5 responses defeat
Opus 4.5 detection on BigCodeBench?

Tests four style-transfer methods for each target source:
  A   — fewshot imitation (target re-generates given Sonnet examples)
  B   — Sonnet rewrite of existing target responses
  C   — style instruction system prompt for target
  AC  — fewshot + style instructions combined

Condition names: "{target}_{method}" e.g. "gemini_pro_B", "gpt_5_AC"
Organic baseline (Opus 4.5 self-responses) is loaded from experiment 0f.

Run from prefill_awareness/:
    python -m experiment0g.run_all
    python -m experiment0g.run_all --from-step detection
    python -m experiment0g.run_all --from-step analysis
    python -m experiment0g.run_all --skip-verify
    python -m experiment0g.run_all --methods B --targets gemini_pro
    python -m experiment0g.run_all --n-tasks 10 --methods B
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

from experiment0g.analysis import run_analysis
from experiment0g.benchmarks import load_tasks
from experiment0g.config import Experiment0gConfig
from experiment0g.generate import generate_all
from experiment0g.run_detection import run_detection
from experiment0g.utils import CostTracker, verify_models

_STEP_ORDER = ["generate", "detection", "analysis"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 0g: Style transfer to defeat Opus 4.5 detection"
    )
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing responses and results.")
    parser.add_argument("--from-step", choices=_STEP_ORDER, default=None,
                        metavar="STEP",
                        help=f"Start from this step: {', '.join(_STEP_ORDER)}")
    parser.add_argument("--skip-verify", action="store_true",
                        help="Skip OpenRouter model ID verification.")
    parser.add_argument("--methods", nargs="+", default=None,
                        metavar="METHOD",
                        help="Restrict to specific methods. Choices: A B C AC")
    parser.add_argument("--targets", nargs="+", default=None,
                        metavar="TARGET",
                        help="Restrict to specific targets. Choices: gemini_pro gpt_5")
    parser.add_argument("--n-tasks", type=int, default=None,
                        help="Override n_test_tasks (for quick tests).")
    return parser.parse_args()


def _should_run(step: str, from_step: str | None) -> bool:
    if from_step is None:
        return True
    return _STEP_ORDER.index(step) >= _STEP_ORDER.index(from_step)


def main() -> None:
    args = parse_args()
    config = Experiment0gConfig()

    if args.methods:
        config.methods = [m for m in args.methods if m in ["A", "B", "C", "AC"]]
    if args.targets:
        config.target_sources = {
            k: v for k, v in config.target_sources.items() if k in args.targets
        }
    if args.n_tasks is not None:
        config.n_test_tasks = args.n_tasks

    cost_tracker = CostTracker(config, label="experiment0g")

    print(f"\n=== Experiment 0g ===")
    print(f"  Evaluator:    {config.evaluator_name} ({config.evaluator_id})")
    print(f"  Targets:      {list(config.target_sources.keys())}")
    print(f"  Rewriter:     {config.rewriter_name} ({config.rewriter_id})")
    print(f"  Methods:      {config.methods}")
    print(f"  Test tasks:   {config.n_test_tasks}")
    print(f"  Fewshot pool: {config.n_fewshot_tasks}")
    print(f"  Cost cap:     ${config.cost_cap:.2f}")
    print(f"  0f data:      {config.experiment0f_dir.resolve()}")

    if not args.skip_verify:
        print("\n=== Verifying model IDs on OpenRouter ===")
        if not verify_models(config):
            print("\nUpdate config.py with correct IDs, then re-run.")
            sys.exit(1)

    print("\n=== Loading BigCodeBench tasks ===")
    test_tasks, fewshot_tasks = load_tasks(config.n_test_tasks, config.n_fewshot_tasks)

    # ------------------------------------------------------------------ #
    # Step 1: Generate
    # ------------------------------------------------------------------ #
    if _should_run("generate", args.from_step):
        print("\n=== Step 1: Generating responses ===")
        generate_all(test_tasks, fewshot_tasks, config, cost_tracker, force=args.force)
    else:
        print("\n[Skip Step 1]")

    # ------------------------------------------------------------------ #
    # Step 2: Detection
    # ------------------------------------------------------------------ #
    if _should_run("detection", args.from_step):
        print("\n=== Step 2: Running detection ===")
        results = run_detection(test_tasks, config, cost_tracker, force=args.force)
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

    print(f"\n=== Experiment 0g complete ===")
    print(f"  Results: {config.results_dir.resolve()}")
    print(f"  Figures: {config.figures_dir.resolve()}")
    print(cost_tracker.report())
    _print_auroc_summary(results, config)


def _print_auroc_summary(results: list[dict], config: Experiment0gConfig) -> None:
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    df = pd.DataFrame(results)
    if len(df) == 0:
        return
    df["tamper_prob"] = pd.to_numeric(df["tamper_prob"], errors="coerce")
    df = df.dropna(subset=["tamper_prob"])
    org = df[df["condition"] == "organic"]["tamper_prob"]

    print("\n  AUROC per target × method:")
    for target_name in config.target_sources:
        for method in config.methods:
            cond = f"{target_name}_{method}"
            sub = df[df["condition"] == cond]["tamper_prob"].dropna()
            if len(sub) < 5 or len(org) < 5:
                print(f"    {cond}: insufficient data")
                continue
            y = [0] * len(org) + [1] * len(sub)
            s = list(org) + list(sub)
            if len(set(y)) < 2:
                continue
            try:
                a = roc_auc_score(y, s)
                print(f"    {cond:20s}: AUROC={a:.3f}")
            except Exception:
                pass


if __name__ == "__main__":
    main()
