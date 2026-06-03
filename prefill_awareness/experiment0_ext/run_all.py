"""
Experiment 0 Extension: SWE-Bench Prefill Awareness Replication.

Replicates Africa et al. (2026) finding that Opus 4.5 detects prefilled
assistant turns in multi-turn conversations (OASST1/HH-RLHF or SWE-bench).

Run from prefill_awareness/:
    python -m experiment0_ext.run_all              # all models from config, cap=$15
    python -m experiment0_ext.run_all --phase 1   # cap=$3
    python -m experiment0_ext.run_all --phase 2   # cap=$5
    python -m experiment0_ext.run_all --phase 3   # cap=$5
    python -m experiment0_ext.run_all --force      # re-run all steps
    python -m experiment0_ext.run_all --from-step attribution

The --phase flag only controls the cost cap; the evaluator set is always
taken from config.evaluators. Expand that dict to add models.

Steps:
  1. data        — load/cache 30 multi-turn conversations
  2. generate    — generate replacement turns from each model
  3. attribution — present prefilled conversations and record attribution answers
  4. analysis    — compute balanced accuracy, figures, summary CSV
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

from experiment0_ext.analysis import run_analysis
from experiment0_ext.config import Experiment0ExtConfig
from experiment0_ext.data import load_conversations
from experiment0_ext.generate_responses import generate_all_replacements
from experiment0_ext.run_attribution import run_all_attribution
from experiment0_ext.utils import CostTracker, verify_model_ids

_STEP_ORDER = ["data", "generate", "attribution", "analysis"]

# Phase caps only control spending; evaluator set always comes from config.evaluators
_PHASE_CAPS = {1: 3.0, 2: 5.0, 3: 5.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 0 Extension: SWE-Bench Prefill Awareness Replication"
    )
    parser.add_argument(
        "--phase", type=int, choices=[1, 2, 3], default=None,
        help="Cost cap phase: 1=$3, 2=$5, 3=$5. Omit to use hard ceiling ($15).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run all steps, overwriting existing outputs.",
    )
    parser.add_argument(
        "--from-step", choices=_STEP_ORDER, default=None, metavar="STEP",
        help=f"Start from this step (earlier outputs must exist). One of: {', '.join(_STEP_ORDER)}",
    )
    parser.add_argument(
        "--skip-verify", action="store_true",
        help="Skip OpenRouter model ID verification.",
    )
    parser.add_argument(
        "--simple-format", action="store_true",
        help="Also run the simple (one-word) attribution prompt alongside the explain format.",
    )
    return parser.parse_args()


def _should_run(step: str, from_step: str | None) -> bool:
    if from_step is None:
        return True
    return _STEP_ORDER.index(step) >= _STEP_ORDER.index(from_step)


def main() -> None:
    args = parse_args()

    config = Experiment0ExtConfig()
    if args.simple_format:
        config.run_simple_format = True

    # Phase flag controls only the cost cap; evaluators come from config.evaluators
    if args.phase is not None:
        cap = _PHASE_CAPS[args.phase]
        phase_label = f"phase{args.phase}"
    else:
        cap = config.hard_ceiling
        phase_label = "full"

    cost_tracker = CostTracker(
        phase_cap=cap,
        hard_ceiling=config.hard_ceiling,
        phase_name=phase_label,
    )

    print(f"\n=== Experiment 0 Extension — {phase_label} ===")
    print(f"  Evaluators: {list(config.evaluators.keys())}")
    print(f"  Sources:    {list(config.sources.keys())}")
    print(f"  Conversations: {config.n_conversations}")
    print(f"  Cost cap: ${cap:.2f}  |  Hard ceiling: ${config.hard_ceiling:.2f}")

    # ------------------------------------------------------------------ #
    # Verify model IDs
    # ------------------------------------------------------------------ #
    if not args.skip_verify:
        print("\n=== Verifying model IDs on OpenRouter ===")
        try:
            verify_model_ids(config)
        except AssertionError as e:
            print(f"\nERROR: {e}")
            print("Fix config.py then re-run, or pass --skip-verify to bypass.")
            sys.exit(1)

    # ------------------------------------------------------------------ #
    # Step 1: Load conversations
    # ------------------------------------------------------------------ #
    if _should_run("data", args.from_step):
        print("\n=== Step 1: Loading multi-turn conversations ===")
        conversations = load_conversations(config)
    else:
        conv_path = config.generations_dir / "conversations.json"
        print(f"\n[Skip Step 1] Loading from {conv_path}")
        with open(conv_path) as f:
            conversations = json.load(f)

    by_ds: dict[str, int] = {}
    for c in conversations:
        by_ds[c["dataset"]] = by_ds.get(c["dataset"], 0) + 1
    print(f"  Conversations: {', '.join(f'{ds}={n}' for ds, n in sorted(by_ds.items()))}")

    # ------------------------------------------------------------------ #
    # Step 2: Generate replacement turns
    # ------------------------------------------------------------------ #
    if _should_run("generate", args.from_step):
        print("\n=== Step 2: Generating replacement turns ===")
        print(f"  Models: {list(config.all_generation_models.keys())}")
        replacements = generate_all_replacements(conversations, config, force=args.force)
    else:
        repl_path = config.generations_dir / "replacements.json"
        print(f"\n[Skip Step 2] Loading from {repl_path}")
        with open(repl_path) as f:
            replacements = json.load(f)

    n_complete = sum(
        1 for r in replacements
        if len(r.get("replacements", {})) == len(config.all_generation_models)
    )
    print(f"  {n_complete}/{len(replacements)} conversations have all replacements.")

    # ------------------------------------------------------------------ #
    # Step 3: Attribution testing
    # ------------------------------------------------------------------ #
    if _should_run("attribution", args.from_step):
        print("\n=== Step 3: Running attribution tests ===")
        results = run_all_attribution(
            conversations, replacements, config, cost_tracker, force=args.force
        )
    else:
        results_path = config.results_dir / "attribution_results.json"
        print(f"\n[Skip Step 3] Loading from {results_path}")
        with open(results_path) as f:
            results = json.load(f)

    print(f"  {len(results)} attribution records.")

    # ------------------------------------------------------------------ #
    # Step 4: Analysis
    # ------------------------------------------------------------------ #
    if _should_run("analysis", args.from_step):
        print("\n=== Step 4: Analysis and figures ===")
        run_analysis(results, config)

    print(f"\n=== Experiment 0 Extension — {phase_label} complete. ===")
    print(f"  Results: {config.results_dir.resolve()}")
    print(cost_tracker.report())


if __name__ == "__main__":
    main()
