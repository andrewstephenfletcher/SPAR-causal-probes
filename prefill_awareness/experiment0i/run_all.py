"""
Experiment 0i: Within-Family Backdoor Detection Blind Spot

Tests whether Opus 4.5's failure to detect Sonnet 4.5 backdoors (AUROC ~0.62
in experiment 0h) is a within-family blind spot, or whether Sonnet simply
writes subtler backdoors that fool all monitors.

Design:
  - Two source models generate honest + attack BigCodeBench solutions
  - Two evaluators (Opus 4.5, Gemini 2.5 Pro) run backdoor detection on all solutions
  - 2×2 AUROC heatmap: evaluator × source

Key question: if Gemini detects Sonnet backdoors that Opus misses, the blind
spot is within-family-specific (Opus goes easy on Claude-family code).

Run from prefill_awareness/:
    python -m experiment0i.run_all
    python -m experiment0i.run_all --from-step detection
    python -m experiment0i.run_all --from-step analysis
    python -m experiment0i.run_all --n-tasks 20   # cheap smoke test
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

from experiment0i.analysis import run_analysis
from experiment0i.config import Experiment0iConfig
from experiment0i.generate import count_verified, generate_all
from experiment0i.run_detection import CostTracker, run_detection

_STEP_ORDER = ["generate", "detection", "analysis"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 0i: within-family blind spot")
    p.add_argument("--force", action="store_true",
                   help="Overwrite cached solutions and results.")
    p.add_argument("--from-step", choices=_STEP_ORDER, default=None,
                   metavar="STEP", help=f"Start from step: {', '.join(_STEP_ORDER)}")
    p.add_argument("--n-tasks", type=int, default=None,
                   help="Tasks per source/condition (default: config.n_tasks).")
    p.add_argument("--task-offset", type=int, default=None,
                   help="Skip first N BigCodeBench tasks (default: config.task_offset).")
    return p.parse_args()


def _should_run(step: str, from_step: str | None) -> bool:
    if from_step is None:
        return True
    return _STEP_ORDER.index(step) >= _STEP_ORDER.index(from_step)


def main() -> None:
    args = parse_args()
    config = Experiment0iConfig()
    if args.n_tasks is not None:
        config.n_tasks = args.n_tasks
    if args.task_offset is not None:
        config.task_offset = args.task_offset

    cost = CostTracker(cap=config.cost_cap, log_every=config.cost_log_every)

    print("\n=== Experiment 0i: Within-Family Blind Spot ===")
    print(f"  Sources:      {list(config.source_models.keys())}")
    print(f"  Evaluators:   {list(config.evaluator_models.keys())}")
    print(f"  Tasks/source: {config.n_tasks}  (offset: {config.task_offset})")
    print(f"  Target:       {config.target_verified}+ verified backdoors per source")
    print(f"  Cost cap:     ${config.cost_cap:.2f}")

    # ------------------------------------------------------------------ #
    # Step 1: Generate solutions via ControlArena
    # ------------------------------------------------------------------ #
    if _should_run("generate", args.from_step):
        print("\n=== Step 1: Generating solutions (ControlArena + Docker) ===")
        generate_all(config, force=args.force)

        # Report verified counts
        print("\n  Verified backdoor counts:")
        for source in config.source_models:
            n = count_verified(source, config)
            target_met = "✓" if n >= config.target_verified else "✗"
            print(f"  {target_met} {source}: {n} / {config.target_verified} target")
            if n < config.target_verified:
                print(f"    → Consider increasing --n-tasks or running with --force")
    else:
        print("\n[Skip Step 1]")

    # ------------------------------------------------------------------ #
    # Step 2: Detection — both evaluators on all tasks
    # ------------------------------------------------------------------ #
    if _should_run("detection", args.from_step):
        print("\n=== Step 2: Running detection (Opus 4.5 + Gemini 2.5 Pro) ===")
        results = run_detection(config, cost, force=args.force)
    else:
        out_json = config.results_dir / "detection_results.json"
        print(f"\n[Skip Step 2] Loading from {out_json}")
        results = json.loads(out_json.read_text()) if out_json.exists() else []

    print(f"  {len(results)} detection records total.")
    print(f"  {cost.report()}")

    # ------------------------------------------------------------------ #
    # Step 3: Analysis
    # ------------------------------------------------------------------ #
    if _should_run("analysis", args.from_step):
        print("\n=== Step 3: Analysis ===")
        run_analysis(results, config)

    print(f"\n=== Experiment 0i complete ===")
    print(f"  Results: {config.results_dir.resolve()}")
    print(f"  Figures: {config.figures_dir.resolve()}")
    print(f"  {cost.report()}")


if __name__ == "__main__":
    main()
