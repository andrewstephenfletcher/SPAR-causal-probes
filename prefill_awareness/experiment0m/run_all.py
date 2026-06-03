"""
Experiment 0m: OSS prefill detection, Gemma 4 31B as single evaluator.

Sources (2 per family, 6 total):
  Google: gemma_4_4b, gemma_4_31b
  Meta:   llama_31_8b, llama_33_70b
  Qwen:   qwen_25_7b, qwen_25_32b

Datasets: OASST1, BigCodeBench, GPQA.

Run from prefill_awareness/:
    uv run python -m experiment0m.run_all
    uv run python -m experiment0m.run_all --from-step detection
    uv run python -m experiment0m.run_all --from-step analysis
    uv run python -m experiment0m.run_all --skip-verify
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

from experiment0m.analysis import run_analysis
from experiment0m.benchmarks import load_all_tasks
from experiment0m.config import Experiment0mConfig
from experiment0m.generate import generate_all
from experiment0m.run_detection import run_detection
from experiment0m.utils import CostTracker, verify_all_models

_STEP_ORDER = ["generate", "detection", "analysis"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 0m: Gemma 4 31B evaluator, OSS sources, OASST1 + BCB + GPQA"
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
                        help="Restrict to specific datasets. Choices: oasst1 bigcodebench gpqa")
    return parser.parse_args()


def _should_run(step: str, from_step: str | None) -> bool:
    if from_step is None:
        return True
    return _STEP_ORDER.index(step) >= _STEP_ORDER.index(from_step)


def main() -> None:
    args = parse_args()
    config = Experiment0mConfig()
    if args.n_tasks is not None:
        config.n_tasks_per_dataset = args.n_tasks

    cost_tracker = CostTracker(config, label="experiment0m")

    print(f"\n=== Experiment 0m ===")
    print(f"  Evaluator:         {config.evaluator_name} ({config.evaluator_id})")
    print(f"  Sources:           {list(config.sources.keys())}")
    print(f"  Tasks per dataset: {config.n_tasks_per_dataset}")
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

    if _should_run("generate", args.from_step):
        print(f"\n=== Step 1: Generating responses ===")
        generate_all(tasks, config, cost_tracker, force=args.force)
    else:
        print("\n[Skip Step 1]")

    if _should_run("detection", args.from_step):
        print(f"\n=== Step 2: Running detection ===")
        results = run_detection(tasks, config, cost_tracker, force=args.force)
    else:
        results_path = config.results_dir / "detection_results.json"
        print(f"\n[Skip Step 2] Loading from {results_path}")
        with open(results_path) as f:
            results = json.load(f)

    print(f"  {len(results)} detection records.")

    if _should_run("analysis", args.from_step):
        print("\n=== Step 3: Analysis and figures ===")
        run_analysis(results, config)

    print(f"\n=== Experiment 0m complete ===")
    print(f"  Results: {config.results_dir.resolve()}")
    print(f"  Figures: {config.figures_dir.resolve()}")
    print(cost_tracker.report())


if __name__ == "__main__":
    main()
