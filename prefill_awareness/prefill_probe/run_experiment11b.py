"""
Experiment 11b: Probe Training and Analysis — orchestrator.

Run from the prefill_awareness/ directory:

    # Run all analyses in the correct order:
    python -m prefill_probe.run_experiment11b

    # Run specific analyses by number (0=validate, 1=depth_curves, ...):
    python -m prefill_probe.run_experiment11b --analyses 0 1

    # Re-run specific analyses, overwriting cached results:
    python -m prefill_probe.run_experiment11b --analyses 1 --force

    # Re-run everything:
    python -m prefill_probe.run_experiment11b --force

Analysis IDs (in execution order):
  0  validate              — confirm all 72 activation files, log counts/shapes
  3  token_distributions   — pre-normalisation first/last token distributions
  1  depth_curves          — per-layer AUROC curves, all targets × datasets (pooled)
  9  per_source_depth_curves — per-source breakdown: self vs. each cross-source separately
  4  position0_diagnostic  — layer-0/1 probe vs. token-identity baseline
  2  last_token_confound   — pre-normalisation last-token overlap analysis
  5  cross_dataset         — 3×3 cross-dataset generalisation heatmaps
  6  cross_model           — 3×3 cross-source generalisation heatmaps
  7  token_position        — accumulation curves from token-position activations
  8  summary               — headline numbers aggregated from all prior results

Outputs go to outputs/experiment11b/ (results/, figures/, probes/).
No model inference; CPU-only.
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

_here = Path(__file__).resolve().parent.parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from prefill_probe.analysis_ex11b import (
    analysis_blog_heatmaps,
    analysis_cross_dataset,
    analysis_cross_model,
    analysis_depth_curves,
    analysis_last_token_confound,
    analysis_per_source_depth_curves,
    analysis_position0_diagnostic,
    analysis_summary,
    analysis_token_distributions,
    analysis_token_position,
    validate_data,
)
from prefill_probe.config import Experiment11bConfig


_EXECUTION_ORDER = [0, 3, 1, 9, 4, 2, 5, 6, 7, 8, 10]

_ANALYSIS_NAMES = {
    0:  "validate",
    1:  "depth_curves",
    2:  "last_token_confound",
    3:  "token_distributions",
    4:  "position0_diagnostic",
    5:  "cross_dataset",
    6:  "cross_model",
    7:  "token_position",
    8:  "summary",
    9:  "per_source_depth_curves",
    10: "blog_heatmaps",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 11b: Probe Training and Analysis")
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
    parser.add_argument(
        "--backup",
        action="store_true",
        help=(
            "Before running (especially with --force), copy existing results/, "
            "figures/ and probes/ to a timestamped backup directory."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    config = Experiment11bConfig()
    print(f"\nExperiment 11b: Probe Training and Analysis")
    print(f"  Results:  {config.results_dir.resolve()}")
    print(f"  Figures:  {config.figures_dir.resolve()}")
    print(f"  Force:    {args.force}")

    if args.backup:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_root = config.output_dir.parent / f"experiment11b_backup_{ts}"
        backup_root.mkdir(parents=True, exist_ok=True)
        for subdir in ("results", "figures", "probes"):
            src = config.output_dir / subdir
            if src.exists():
                shutil.copytree(src, backup_root / subdir)
                print(f"  Backed up {subdir}/ → {backup_root / subdir}")
        print(f"  Backup complete → {backup_root}")

    # Determine which analyses to run, in execution order
    requested = set(args.analyses) if args.analyses else set(_ANALYSIS_NAMES.keys())
    to_run = [a for a in _EXECUTION_ORDER if a in requested]
    print(f"  Analyses: {[_ANALYSIS_NAMES[a] for a in to_run]}\n")

    depth_results: dict | None = None  # passed to analyses 5 and 6

    for analysis_id in to_run:
        name = _ANALYSIS_NAMES[analysis_id]
        print(f"{'='*60}")
        print(f"Running analysis {analysis_id}: {name}")
        print(f"{'='*60}")

        if analysis_id == 0:
            validate_data(config, force=args.force)

        elif analysis_id == 3:
            analysis_token_distributions(config, force=args.force)

        elif analysis_id == 1:
            depth_results = analysis_depth_curves(config, force=args.force)
            # If we skipped depth_curves (loaded from cache), still populate depth_results
            if depth_results is None:
                depth_results = _load_depth_results(config)

        elif analysis_id == 9:
            analysis_per_source_depth_curves(config, force=args.force)

        elif analysis_id == 4:
            analysis_position0_diagnostic(config, force=args.force)

        elif analysis_id == 2:
            analysis_last_token_confound(config, force=args.force)

        elif analysis_id == 5:
            # Try to load depth_results if not already in memory
            if depth_results is None:
                depth_results = _load_depth_results(config)
            analysis_cross_dataset(config, depth_results=depth_results, force=args.force)

        elif analysis_id == 6:
            if depth_results is None:
                depth_results = _load_depth_results(config)
            analysis_cross_model(config, depth_results=depth_results, force=args.force)

        elif analysis_id == 7:
            analysis_token_position(config, force=args.force)

        elif analysis_id == 8:
            summary = analysis_summary(config, force=args.force)
            _print_headline_summary(summary, config)

        elif analysis_id == 10:
            analysis_blog_heatmaps(config, force=args.force)

    print(f"\n=== Experiment 11b complete. ===")
    print(f"  Results: {config.results_dir.resolve()}")
    print(f"  Figures: {config.figures_dir.resolve()}")


def _load_depth_results(config: Experiment11bConfig) -> dict:
    """Load cached depth curve results from disk."""
    depth_results: dict = {}
    dc_dir = config.results_dir / "depth_curves"
    if not dc_dir.exists():
        return depth_results
    for target in config.target_models:
        depth_results[target] = {}
        for ds in config.datasets:
            path = dc_dir / f"{target}_{ds}.json"
            if path.exists():
                with open(path) as f:
                    depth_results[target][ds] = json.load(f)
    return depth_results


def _print_headline_summary(summary: dict, config: Experiment11bConfig) -> None:
    print("\n" + "="*60)
    print("HEADLINE NUMBERS")
    print("="*60)
    for target in config.target_models:
        ts = summary.get(target, {})
        print(f"\n{target}:")
        for ds in config.datasets:
            ds_s = ts.get(ds, {})
            auroc = ds_s.get("best_lr_auroc")
            rel   = ds_s.get("rel_peak")
            first = ds_s.get("rel_depth_first_0.90")
            auroc_s = f"{auroc:.4f}"       if auroc is not None else "N/A"
            rel_s   = f"{100*rel:.0f}%"   if rel   is not None else "N/A"
            first_s = f"{100*first:.0f}%" if first is not None else "N/A"
            print(f"  {ds}: peak={auroc_s} at {rel_s} depth, first>0.90 at {first_s} depth")
        cd = ts.get("cross_dataset_mean_off_diagonal")
        cm = ts.get("cross_model_mean_off_diagonal")
        if cd is not None:
            print(f"  cross-dataset off-diag AUROC: {cd:.4f}")
        if cm is not None:
            print(f"  cross-model   off-diag AUROC: {cm:.4f}")


if __name__ == "__main__":
    main()
