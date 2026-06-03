"""
Experiment 0n: Opus 4.5 single-evaluator, OASST1 + BigCodeBench + GPQA.

Clones experiment0f but replaces SWE-bench with OASST1 and adds Qwen 2.5 32B
as an additional OSS source model. BCB and GPQA responses can be reused from
experiment0f by copying generations_dir files if desired.

Run from prefill_awareness/:
    uv run python -m experiment0n.run_all
    uv run python -m experiment0n.run_all --from-step detection
    uv run python -m experiment0n.run_all --from-step analysis
    uv run python -m experiment0n.run_all --skip-verify
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

from experiment0n.analysis import run_analysis
from experiment0n.benchmarks import load_all_tasks
from experiment0n.config import Experiment0nConfig
from experiment0n.generate import generate_all
from experiment0n.run_detection import run_detection
from experiment0n.utils import CostTracker, verify_all_models

_STEP_ORDER = ["generate", "detection", "analysis"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 0n: Opus 4.5, OASST1 + BCB + GPQA, frontier + OSS sources"
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
    config = Experiment0nConfig()
    if args.n_tasks is not None:
        config.n_tasks_per_dataset = args.n_tasks

    cost_tracker = CostTracker(config, label="experiment0n")

    print(f"\n=== Experiment 0n ===")
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

    print(f"\n=== Experiment 0n complete ===")
    print(f"  Results: {config.results_dir.resolve()}")
    print(f"  Figures: {config.figures_dir.resolve()}")
    print(cost_tracker.report())


if __name__ == "__main__":
    main()
