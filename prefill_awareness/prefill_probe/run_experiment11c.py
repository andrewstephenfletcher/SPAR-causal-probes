"""
Experiment 11c: Norm-Controlled Probes — orchestrator.

Run from the prefill_awareness/ directory:

    # Run all analyses in order:
    python -m prefill_probe.run_experiment11c

    # Run specific analyses by number:
    python -m prefill_probe.run_experiment11c --analyses 0 1

    # Re-run, overwriting cached results:
    python -m prefill_probe.run_experiment11c --force

Analysis IDs (execution order):
  0  norm_profiles           — mean L2 norm per layer for self and each cross condition
  1  depth_curves_normalized — per-source depth curves with L2-normalised activations
  2  comparison_and_step     — unnorm vs. norm AUROC overlay + step retention ratios
  3  cosine_similarity       — DIM direction cosine similarity between adjacent layers

Outputs go to outputs/experiment11c/ (results/, figures/).
No model inference; CPU-only.

Prerequisite: Experiment 11b analysis 9 (per_source_depth_curves) must have been run
before analysis 2 (comparison_and_step).
"""

import argparse
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent.parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from prefill_probe.analysis_ex11c import (
    analysis_comparison_and_step,
    analysis_cosine_similarity,
    analysis_depth_curves_normalized,
    analysis_norm_profiles,
)
from prefill_probe.config import Experiment11cConfig


_EXECUTION_ORDER = [0, 1, 2, 3]

_ANALYSIS_NAMES = {
    0: "norm_profiles",
    1: "depth_curves_normalized",
    2: "comparison_and_step",
    3: "cosine_similarity",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 11c: Norm-Controlled Probes")
    parser.add_argument(
        "--analyses",
        nargs="+",
        type=int,
        choices=sorted(_ANALYSIS_NAMES.keys()),
        default=None,
        metavar="N",
        help=(
            "Which analyses to run (by number). Default: all, in correct order. "
            f"Available: {_ANALYSIS_NAMES}"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing results (default: skip if output JSON exists).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    config = Experiment11cConfig()
    print(f"\nExperiment 11c: Norm-Controlled Probes")
    print(f"  Results:  {config.results_dir.resolve()}")
    print(f"  Figures:  {config.figures_dir.resolve()}")
    print(f"  Force:    {args.force}")

    requested = set(args.analyses) if args.analyses else set(_ANALYSIS_NAMES.keys())
    to_run = [a for a in _EXECUTION_ORDER if a in requested]
    print(f"  Analyses: {[_ANALYSIS_NAMES[a] for a in to_run]}\n")

    for analysis_id in to_run:
        name = _ANALYSIS_NAMES[analysis_id]
        print(f"{'='*60}")
        print(f"Running analysis {analysis_id}: {name}")
        print(f"{'='*60}")

        if analysis_id == 0:
            analysis_norm_profiles(config, force=args.force)
        elif analysis_id == 1:
            analysis_depth_curves_normalized(config, force=args.force)
        elif analysis_id == 2:
            analysis_comparison_and_step(config, force=args.force)
        elif analysis_id == 3:
            analysis_cosine_similarity(config, force=args.force)

    print(f"\n=== Experiment 11c complete. ===")
    print(f"  Results: {config.results_dir.resolve()}")
    print(f"  Figures: {config.figures_dir.resolve()}")


if __name__ == "__main__":
    main()
