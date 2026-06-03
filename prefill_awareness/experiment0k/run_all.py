"""
Experiment 0k: Comparative Rewrite — Style vs Logic vs Both

Shows Gemini its own response and Opus 4.5's response to the same task.
Asks Gemini to rewrite its answer to be less detectable, under three constraints:
  style_only  — change only formatting, prose, variable names, comments
  logic_only  — change only the algorithm and code structure
  both        — unconstrained

Run from prefill_awareness/:
    python -m experiment0k.run_all
    python -m experiment0k.run_all --from-step detection
    python -m experiment0k.run_all --from-step analysis
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

from experiment0k.analysis import run_analysis
from experiment0k.config import Experiment0kConfig
from experiment0k.generate import load_tasks, run_iterative
from experiment0k.helpers import CostTracker

_STEP_ORDER = ["generate", "analysis"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 0k: comparative rewrite")
    p.add_argument("--force", action="store_true",
                   help="Overwrite cached responses and results.")
    p.add_argument("--from-step", choices=_STEP_ORDER, default=None,
                   help=f"Start from: {', '.join(_STEP_ORDER)}")
    p.add_argument("--n-tasks", type=int, default=None,
                   help="Override n_tasks for quick tests.")
    return p.parse_args()


def _should_run(step: str, from_step: str | None) -> bool:
    if from_step is None:
        return True
    return _STEP_ORDER.index(step) >= _STEP_ORDER.index(from_step)


def main() -> None:
    args = parse_args()
    config = Experiment0kConfig()
    if args.n_tasks is not None:
        config.n_tasks = args.n_tasks

    cost = CostTracker(cap=config.cost_cap, log_every=config.cost_log_every)

    print("\n=== Experiment 0k: Comparative Rewrite (Style vs Logic vs Both) ===")
    print(f"  Evaluator:  {config.evaluator_id}")
    print(f"  Generator:  {config.gemini_id}")
    print(f"  Tasks:      {config.n_tasks} (top-N Gemini from 0f)")
    print(f"  Variants:   {list(config.variants.keys())}")
    print(f"  Cost cap:   ${config.cost_cap:.2f}  (estimated ~$1.70)")

    print("\n  Loading tasks from experiment 0f…")
    tasks = load_tasks(config)
    print(f"  {len(tasks)} tasks loaded "
          f"(tamper_prob {tasks[-1]['baseline_tamper_prob']}–{tasks[0]['baseline_tamper_prob']})")

    if _should_run("generate", args.from_step):
        print("\n=== Step 1: Iterative rewrites (Gemini) + detection (Opus) ===")
        all_trajectories = run_iterative(tasks, config, cost, force=args.force)
    else:
        # Load cached trajectories
        all_trajectories = {}
        for variant in config.variants:
            p = config.results_dir / f"{variant}_trajectories.json"
            if p.exists():
                all_trajectories[variant] = json.loads(p.read_text())
        print(f"\n[Skip Step 1] Loaded {sum(len(v) for v in all_trajectories.values())} cached trajectories.")

    print(f"  {cost.report()}")

    if _should_run("analysis", args.from_step):
        print("\n=== Step 2: Analysis ===")
        run_analysis(all_trajectories, config)

    print(f"\n=== Experiment 0k complete ===")
    print(f"  Results: {config.results_dir.resolve()}")
    print(f"  Figures: {config.figures_dir.resolve()}")
    print(f"  {cost.report()}")


if __name__ == "__main__":
    main()
