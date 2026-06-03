"""
Experiment 0b: Probability-Based Prefill Detection.

Tests whether models can detect a tampered (off-policy) assistant turn in a
conversation, using a 0-100 tamper probability rating → AUROC as primary metric.

Two datasets:
  - agentic (20 synthetic coding tasks, 3-turn trajectories)
  - oasst1  (15 multi-turn conversations, negative control from first run)

Run from prefill_awareness/:
    python -m experiment0b.run_all              # all evaluators from config
    python -m experiment0b.run_all --phase 1   # cap=$18 (Opus 4.5/4.6 + Sonnet 4.5)
    python -m experiment0b.run_all --phase 2   # cap=$5  (add Opus 4.7)
    python -m experiment0b.run_all --force     # overwrite all cached outputs
    python -m experiment0b.run_all --from-step detection
    python -m experiment0b.run_all --skip-verify

Steps:
  1. data       — load/cache agentic + oasst1 conversation metadata
  2. generate   — generate 3-turn agentic trajectories for all models × tasks
  3. detection  — run detection (+ optional binary) tasks for all evaluators
  4. analysis   — compute AUROC, d', figures, summary CSV
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

from experiment0b.analysis import run_analysis
from experiment0b.config import Experiment0bConfig
from experiment0b.data import load_all_conversations, load_oasst1_replacements
from experiment0b.generate_trajectories import generate_all_trajectories, load_trajectories
from experiment0b.run_detection import run_all_detection
from experiment0b.utils import CostTracker, verify_model_ids

_STEP_ORDER = ["data", "generate", "detection", "analysis"]

# Phase caps: phase 1 covers main evaluators, phase 2 adds Opus 4.7
_PHASE_CAPS = {1: 18.0, 2: 5.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 0b: Probability-Based Prefill Detection"
    )
    parser.add_argument(
        "--phase", type=int, choices=[1, 2], default=None,
        help="Cost cap phase: 1=$18 (main evaluators), 2=$5 (add Opus 4.7). "
             "Omit to use hard ceiling ($25).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run all steps, overwriting existing outputs.",
    )
    parser.add_argument(
        "--from-step", choices=_STEP_ORDER, default=None, metavar="STEP",
        help=f"Start from this step (earlier outputs must exist). "
             f"One of: {', '.join(_STEP_ORDER)}",
    )
    parser.add_argument(
        "--skip-verify", action="store_true",
        help="Skip OpenRouter model ID verification.",
    )
    return parser.parse_args()


def _should_run(step: str, from_step: str | None) -> bool:
    if from_step is None:
        return True
    return _STEP_ORDER.index(step) >= _STEP_ORDER.index(from_step)


def main() -> None:
    args = parse_args()
    config = Experiment0bConfig()

    if args.phase is not None:
        cap = _PHASE_CAPS[args.phase]
        phase_label = f"phase{args.phase}"
    else:
        cap = config.hard_ceiling
        phase_label = "full"

    cost_tracker = CostTracker(config, phase_cap=cap, phase_label=phase_label)

    print(f"\n=== Experiment 0b — {phase_label} ===")
    print(f"  Evaluators:    {list(config.evaluators.keys())}")
    print(f"  Sources:       {list(config.sources.keys())}")
    print(f"  Agentic tasks: {config.n_agentic_tasks}")
    print(f"  OASST1 convos: {config.n_oasst1_convos}")
    print(f"  Cost cap: ${cap:.2f}  |  Hard ceiling: ${config.hard_ceiling:.2f}")
    print(f"  Binary task:   {config.run_binary_task}")

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
        print("\n=== Step 1: Loading conversations ===")
        conversations = load_all_conversations(config)
    else:
        conv_path = config.generations_dir / "conversations.json"
        print(f"\n[Skip Step 1] Loading from {conv_path}")
        with open(conv_path) as f:
            conversations = json.load(f)

    by_ds: dict[str, int] = {}
    for c in conversations:
        by_ds[c["dataset"]] = by_ds.get(c["dataset"], 0) + 1
    print(f"  Loaded: {', '.join(f'{ds}={n}' for ds, n in sorted(by_ds.items()))}")

    # ------------------------------------------------------------------ #
    # Step 2: Generate trajectories
    # ------------------------------------------------------------------ #
    if _should_run("generate", args.from_step):
        print("\n=== Step 2: Generating agentic trajectories ===")
        print(f"  Models: {list(config.all_generation_models.keys())}")
        trajectories = generate_all_trajectories(config, force=args.force)
    else:
        print("\n[Skip Step 2] Loading trajectories from disk")
        trajectories = load_trajectories(config)

    n_traj = sum(len(v) for v in trajectories.values())
    print(f"  Trajectories loaded: {n_traj} (across {len(trajectories)} models)")

    # ------------------------------------------------------------------ #
    # Step 3: Load OASST1 replacements (for tampered condition)
    # ------------------------------------------------------------------ #
    print("\n  Loading OASST1 replacements from first run...")
    oasst1_replacements = load_oasst1_replacements(config)
    n_repl = sum(len(v) for v in oasst1_replacements.values())
    print(f"  OASST1 replacements: {n_repl} total entries across {len(oasst1_replacements)} convos")

    # ------------------------------------------------------------------ #
    # Step 4: Detection testing
    # ------------------------------------------------------------------ #
    if _should_run("detection", args.from_step):
        print("\n=== Step 3: Running detection tasks ===")
        results = run_all_detection(
            conversations,
            trajectories,
            oasst1_replacements,
            config,
            cost_tracker,
            force=args.force,
        )
    else:
        results_path = config.results_dir / "detection_results.json"
        print(f"\n[Skip Step 3] Loading from {results_path}")
        with open(results_path) as f:
            results = json.load(f)

    print(f"  {len(results)} detection records.")

    # ------------------------------------------------------------------ #
    # Step 5: Analysis
    # ------------------------------------------------------------------ #
    if _should_run("analysis", args.from_step):
        print("\n=== Step 4: Analysis and figures ===")
        run_analysis(results, config)

    print(f"\n=== Experiment 0b — {phase_label} complete. ===")
    print(f"  Results: {config.results_dir.resolve()}")
    print(cost_tracker.report())


if __name__ == "__main__":
    main()
