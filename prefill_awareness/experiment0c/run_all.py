"""
Experiment 0c: What Is Opus 4.6 Actually Detecting?

Ablation study using the interleaved turn generation protocol to disentangle
style signal from trajectory coherence in prefill detection.

Run from prefill_awareness/:
    python -m experiment0c.run_all --conditions organic A      # ~$2.50
    python -m experiment0c.run_all --conditions B              # ~$1.60
    python -m experiment0c.run_all --conditions C              # ~$1.80
    python -m experiment0c.run_all --conditions D              # ~$3.60
    python -m experiment0c.run_all --from-step detection       # re-run detection only
    python -m experiment0c.run_all --from-step analysis        # analysis only
    python -m experiment0c.run_all --force                     # overwrite everything

Conditions:
  organic — Opus 4.6 all turns (false-alarm baseline)
  A       — Sonnet 4.5 turn 2 (within-family detection)
  B       — GPT-4o turn 2 (cross-family frontier, no style matching)
  C       — GPT-4o turn 2 + Opus 4.6 style fewshot (style imitation)
  D       — Opus 4.6 turn 2 with altered style prompt (3 alterations)

Steps:
  1. generate   — interleaved conversation generation
  2. detection  — tamper probability rating by Opus 4.6
  3. analysis   — AUROC figures and summary table
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

from experiment0b.task_scenarios import AGENTIC_TASKS
from experiment0c.analysis import load_exp0b_auroc, run_analysis
from experiment0c.config import Experiment0cConfig
from experiment0c.generate import generate_all, load_conversations
from experiment0c.run_detection import run_all_detection
from experiment0c.utils import CostTracker, verify_all_models

_ALL_CONDITIONS = ["organic", "A", "B", "C", "D"]
_STEP_ORDER = ["generate", "detection", "analysis"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 0c: What Is Opus 4.6 Actually Detecting?"
    )
    parser.add_argument(
        "--conditions", nargs="+",
        choices=_ALL_CONDITIONS, default=None,
        help=(
            "Conditions to run. Default: all. "
            "Recommended: start with 'organic A', then add B, C, D sequentially."
        ),
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing generated conversations and detection results.",
    )
    parser.add_argument(
        "--from-step", choices=_STEP_ORDER, default=None, metavar="STEP",
        help=f"Start from this step. One of: {', '.join(_STEP_ORDER)}",
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
    config = Experiment0cConfig()
    conditions = args.conditions or _ALL_CONDITIONS
    cost_tracker = CostTracker(config, label="experiment0c")

    print(f"\n=== Experiment 0c ===")
    print(f"  Evaluator:  {config.evaluator_name} ({config.evaluator_id})")
    print(f"  Conditions: {conditions}")
    print(f"  Tasks:      {config.n_tasks}")
    print(f"  Cost ceiling: ${config.hard_ceiling:.2f}")

    # ------------------------------------------------------------------ #
    # Exp 0b reference AUROC
    # ------------------------------------------------------------------ #
    exp0b_auroc = load_exp0b_auroc(config)
    if exp0b_auroc is not None:
        print(f"  Exp 0b reference AUROC (Opus 4.6, agentic): {exp0b_auroc:.3f}")

    # ------------------------------------------------------------------ #
    # Verify model IDs
    # ------------------------------------------------------------------ #
    if not args.skip_verify:
        print("\n=== Verifying model IDs on OpenRouter ===")
        if not verify_all_models(config):
            print("\nFix config.py then re-run, or pass --skip-verify to bypass.")
            sys.exit(1)

    # ------------------------------------------------------------------ #
    # Load tasks (reused from experiment0b)
    # ------------------------------------------------------------------ #
    tasks = AGENTIC_TASKS[: config.n_tasks]
    print(f"\n  {len(tasks)} tasks loaded from experiment0b.task_scenarios")

    # ------------------------------------------------------------------ #
    # Step 1: Generate conversations
    # ------------------------------------------------------------------ #
    if _should_run("generate", args.from_step):
        print(f"\n=== Step 1: Generating conversations ({conditions}) ===")
        all_conversations = generate_all(
            tasks, conditions, config, cost_tracker, force=args.force
        )
    else:
        print("\n[Skip Step 1] Loading conversations from disk")
        all_conversations = load_conversations(config)

    by_cond: dict[str, int] = {}
    for c in all_conversations:
        by_cond[c["condition"]] = by_cond.get(c["condition"], 0) + 1
    print(f"  On disk: {', '.join(f'{k}={v}' for k, v in sorted(by_cond.items()))}")

    # Only pass conversations for the requested conditions to detection
    to_detect = [c for c in all_conversations if c["condition"] in conditions]

    # ------------------------------------------------------------------ #
    # Step 2: Detection
    # ------------------------------------------------------------------ #
    if _should_run("detection", args.from_step):
        print(f"\n=== Step 2: Running detection ({len(to_detect)} conversations) ===")
        results = run_all_detection(to_detect, config, cost_tracker, force=args.force)
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
        run_analysis(results, config, exp0b_auroc=exp0b_auroc)

    print(f"\n=== Experiment 0c complete ===")
    print(f"  Results: {config.results_dir.resolve()}")
    print(f"  Figures: {config.figures_dir.resolve()}")
    print(cost_tracker.report())

    # ------------------------------------------------------------------ #
    # Decision gate after Organic + A
    # ------------------------------------------------------------------ #
    if set(conditions) <= {"organic", "A"} and results:
        _print_decision_gate(results, exp0b_auroc)


def _print_decision_gate(results: list[dict], exp0b_auroc: float | None) -> None:
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    df = pd.DataFrame(results)
    df["tamper_prob"] = pd.to_numeric(df["tamper_prob"], errors="coerce")
    org = df[df["condition"] == "organic"]["tamper_prob"].dropna()
    cA  = df[df["condition"] == "A"]["tamper_prob"].dropna()

    if len(org) < 5 or len(cA) < 5:
        return

    y = [0] * len(org) + [1] * len(cA)
    s = list(org) + list(cA)
    try:
        auroc_A = roc_auc_score(y, s)
    except Exception:
        return

    print(f"\n  *** Decision gate (Organic + A) ***")
    print(f"  Condition A (Sonnet 4.5) AUROC = {auroc_A:.3f}")
    if exp0b_auroc is not None:
        drop = exp0b_auroc - auroc_A
        print(f"  Drop from Exp 0b baseline: {drop:+.3f}")

    if auroc_A < 0.55:
        print(
            "\n  AUROC < 0.55: interleaved protocol substantially reduced or eliminated\n"
            "  within-family detection signal. The 0b result may have been coherence-based.\n"
            "  Still worth running B and C to check cross-family frontier signal."
        )
    else:
        print(
            f"\n  AUROC >= 0.55: within-family style signal survives the coherence fix.\n"
            "  Proceed to Condition B:\n"
            "    python -m experiment0c.run_all --conditions B"
        )


if __name__ == "__main__":
    main()
