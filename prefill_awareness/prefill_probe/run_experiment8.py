"""
Experiment 8: Cross-Architecture Probing — end-to-end orchestration.

Run from the prefill_awareness/ directory:

    # Full run (both models):
    python -m prefill_probe.run_experiment8

    # One model only:
    python -m prefill_probe.run_experiment8 --model mistral
    python -m prefill_probe.run_experiment8 --model gemma

    # Resume from a specific step:
    python -m prefill_probe.run_experiment8 --from-step extract

    # Force re-run a step:
    python -m prefill_probe.run_experiment8 --from-step generate --force

Steps:
  1. generate  — load each model, generate 300 responses for Ex1 Alpaca prompts
  2. extract   — extract 2D (layer × position) activations for both models
  3. probe     — train probes at all (layer, position) cells; compute perplexity baseline
  4. figures   — generate 5 figures comparing models

Prerequisites:
  - Experiment 1 responses: outputs/experiment1/generations/responses.json
  - HF_TOKEN set (Mistral 24B requires access; Gemma 31B is gated)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent.parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from prefill_probe.analysis_ex8 import generate_all_figures_ex8
from prefill_probe.config import Experiment8Config
from prefill_probe.extract_ex8 import run_extraction_ex8
from prefill_probe.generate_ex8 import run_generation_ex8
from prefill_probe.probe_ex8 import run_probing_ex8
from prefill_probe.utils import get_device

_STEP_ORDER = ["generate", "extract", "probe", "figures"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 8: Cross-Architecture Probing")
    parser.add_argument(
        "--model",
        choices=["mistral", "gemma", "both"],
        default="both",
        help="Which model(s) to run (default: both)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run the current step, overwriting existing outputs.",
    )
    parser.add_argument(
        "--from-step",
        choices=_STEP_ORDER,
        default=None,
        metavar="STEP",
        help=f"Skip steps before STEP. One of: {', '.join(_STEP_ORDER)}",
    )
    return parser.parse_args()


def _should_run(step: str, from_step: str | None) -> bool:
    if from_step is None:
        return True
    return _STEP_ORDER.index(step) >= _STEP_ORDER.index(from_step)


def main() -> None:
    args = parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    config = Experiment8Config()

    print("\nExperiment 8: Cross-Architecture Probing")
    print(f"  Device:         {get_device()}")
    print(f"  Model(s):       {args.model}")
    print(f"  Output dir:     {config.output_dir_ex8.resolve()}")
    print(f"  Positions:      {config.positions}")
    print(f"  Mistral layers: {config.mistral_n_layers} (all)")
    print(f"  Gemma layers:   {len(config.gemma_layers)} (every {config.gemma_extract_every})")

    # ------------------------------------------------------------------
    # Step 1: Generate responses
    # ------------------------------------------------------------------
    if _should_run("generate", args.from_step):
        print("\n=== Step 1: Generating responses ===")
        run_generation_ex8(config, model=args.model, force=args.force)
    else:
        print("\n[Skipping Step 1] Generation.")

    # ------------------------------------------------------------------
    # Step 2: Extract activations
    # ------------------------------------------------------------------
    if _should_run("extract", args.from_step):
        print("\n=== Step 2: Extracting 2D activations ===")
        run_extraction_ex8(config, model=args.model, force=args.force)
    else:
        print("\n[Skipping Step 2] Extraction.")

    # ------------------------------------------------------------------
    # Step 3: Train probes
    # ------------------------------------------------------------------
    if _should_run("probe", args.from_step):
        print("\n=== Step 3: Training probes ===")
        probe_results = run_probing_ex8(config, model=args.model, force=args.force)
    else:
        print("\n[Skipping Step 3] Loading existing probe results...")
        import pandas as pd
        probe_results = {}
        if args.model in ("mistral", "both"):
            p = config.results_dir_ex8 / "probe_results_mistral.csv"
            if p.exists():
                probe_results["mistral"] = pd.read_csv(p)
        if args.model in ("gemma", "both"):
            p = config.results_dir_ex8 / "probe_results_gemma31b.csv"
            if p.exists():
                probe_results["gemma31b"] = pd.read_csv(p)

    # ------------------------------------------------------------------
    # Step 4: Figures
    # ------------------------------------------------------------------
    if _should_run("figures", args.from_step):
        print("\n=== Step 4: Generating figures ===")
        generate_all_figures_ex8(probe_results, config)
    else:
        print("\n[Skipping Step 4] Figures.")

    print(f"\n=== Experiment 8 complete. ===")
    print(f"  Results in: {config.results_dir_ex8.resolve()}")


if __name__ == "__main__":
    main()
