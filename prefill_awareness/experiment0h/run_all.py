"""
Experiment 0h: Strategy and authorship detection in backdoored BigCodeBench code.

Tests whether Opus 4.5 can detect:
  1. Foreign authorship (GPT-4o-mini vs Opus) in coding solutions
  2. Hidden backdoors (ControlArena attack mode) from code content alone
  3. Whether conversation framing drives detection (strategy swap)

Conditions (generated via ControlArena + Docker):
  opus_honest      — Opus 4.5, honest mode (baseline)
  gpt_mini_honest  — GPT-4o-mini, honest mode
  gpt_mini_attack  — GPT-4o-mini, attack mode (contains backdoor)

Probes:
  authorship — "Was this written by a different model?"
  backdoor   — "Does this contain a hidden backdoor?"

Framings (strategy swap):
  honest — standard helpful-assistant system prompt
  attack — ControlArena attack system prompt (tests if framing drives scores)

Run from prefill_awareness/:
    python -m experiment0h.run_all
    python -m experiment0h.run_all --from-step detection
    python -m experiment0h.run_all --from-step analysis
    python -m experiment0h.run_all --n-tasks 5 --skip-opus   # cheap smoke test
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

from experiment0h.analysis import run_analysis
from experiment0h.config import Experiment0hConfig
from experiment0h.generate import generate_all
from experiment0h.run_detection import CostTracker, run_detection

_STEP_ORDER = ["generate", "detection", "analysis"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Experiment 0h: backdoor + authorship detection"
    )
    p.add_argument("--force", action="store_true",
                   help="Overwrite cached solutions and results.")
    p.add_argument("--from-step", choices=_STEP_ORDER, default=None,
                   metavar="STEP", help=f"Start from step: {', '.join(_STEP_ORDER)}")
    p.add_argument("--n-tasks", type=int, default=None,
                   help="Tasks per condition (default: config.n_tasks).")
    p.add_argument("--skip-opus", action="store_true",
                   help="Skip generating opus_honest (saves budget; uses cached if present).")
    return p.parse_args()


def _should_run(step: str, from_step: str | None) -> bool:
    if from_step is None:
        return True
    return _STEP_ORDER.index(step) >= _STEP_ORDER.index(from_step)


def main() -> None:
    args = parse_args()
    config = Experiment0hConfig()
    if args.n_tasks is not None:
        config.n_tasks = args.n_tasks

    cost = CostTracker(cap=config.cost_cap, log_every=config.cost_log_every)

    print("\n=== Experiment 0h ===")
    print(f"  Evaluator:   {config.evaluator_id}")
    print(f"  Conditions:  {list(config.condition_models.keys())}")
    print(f"  Tasks/cond:  {config.n_tasks}")
    print(f"  Det. runs:   {len(config.detection_runs)}")
    print(f"  Cost cap:    ${config.cost_cap:.2f}")

    # ------------------------------------------------------------------ #
    # Step 1: Generate solutions via ControlArena
    # ------------------------------------------------------------------ #
    if _should_run("generate", args.from_step):
        print("\n=== Step 1: Generating solutions (ControlArena + Docker) ===")
        if args.skip_opus:
            # Check if opus_honest already cached; if not, warn
            cached = (config.solutions_dir / "opus_honest.json").exists()
            if not cached:
                print("  WARNING: --skip-opus but no cached opus_honest solutions.")
                print("  Run without --skip-opus at least once to generate the baseline.")
            orig_models = config.condition_models
            config.condition_models = {
                k: v for k, v in orig_models.items() if k != "opus_honest"
            }
        generate_all(config, force=args.force)
        if args.skip_opus:
            config.condition_models = orig_models
    else:
        print("\n[Skip Step 1]")

    # ------------------------------------------------------------------ #
    # Step 2: Detection
    # ------------------------------------------------------------------ #
    if _should_run("detection", args.from_step):
        print("\n=== Step 2: Running detection (Opus 4.5) ===")
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

    print(f"\n=== Experiment 0h complete ===")
    print(f"  Results: {config.results_dir.resolve()}")
    print(f"  Figures: {config.figures_dir.resolve()}")
    print(f"  {cost.report()}")


if __name__ == "__main__":
    main()
