"""
Experiment 9: Probe Transfer and Representational Geometry — end-to-end orchestration.

Run from the prefill_awareness/ directory:

    # Full run:
    python -m prefill_probe.run_experiment9

    # Resume from a specific step:
    python -m prefill_probe.run_experiment9 --from-step directions
    python -m prefill_probe.run_experiment9 --from-step analysis

    # Force re-run a step:
    python -m prefill_probe.run_experiment9 --from-step extract --force

Steps:
  1. extract       — load Llama 8B, extract truth activations (GoT) and
                     eval/deploy CAA direction at layer 30
  2. directions    — (CPU) re-train Ex1 prefill probe + Ex3 per-model probes
                     to extract their weight-space directions
  3. truth_probes  — (CPU) train truth probes on GoT activations at layers
                     [16, 24, 30] and cache their directions
  4. analysis      — compute all cosine similarities, cross-application AUROCs,
                     and person-vector SVD decomposition
  5. figures       — generate 4 figures and summary table

Prerequisites:
  - Experiment 1 activations: outputs/experiment1/activations/
  - Experiment 1 responses:   outputs/experiment1/generations/responses.json
  - Experiment 3 activations: outputs/experiment3/activations/
  - Geometry of Truth CSVs:   dct_probes/geometry-of-truth/datasets/
  - HF_TOKEN set for Llama 8B download (only for steps 1)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_here = Path(__file__).resolve().parent.parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from prefill_probe.analysis_ex9 import generate_all_figures_ex9
from prefill_probe.config import Experiment9Config
from prefill_probe.extract_ex9 import (
    load_model,
    run_eval_deploy_extraction,
    run_truth_extraction,
    unload_model,
)
from prefill_probe.probe_ex9 import (
    get_per_model_directions,
    get_prefill_probe_direction,
    get_truth_probe_directions,
    run_all_analyses,
)
from prefill_probe.utils import get_device

_STEP_ORDER = ["extract", "directions", "truth_probes", "analysis", "figures"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 9: Representational Geometry")
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
    parser.add_argument(
        "--skip-ex3", action="store_true",
        help="Skip Experiment 3 per-model directions (use if Ex3 has not been run).",
    )
    return parser.parse_args()


def _should_run(step: str, from_step: str | None) -> bool:
    if from_step is None:
        return True
    return _STEP_ORDER.index(step) >= _STEP_ORDER.index(from_step)


def _load_truth_acts_from_cache(config: Experiment9Config) -> dict[str, list[dict]]:
    result = {}
    for dataset_name in config.got_datasets:
        path = config.activations_dir_ex9 / f"truth_activations_{dataset_name}.pt"
        if path.exists():
            result[dataset_name] = torch.load(path, weights_only=False)
        else:
            print(f"  WARNING: truth activations for '{dataset_name}' not found at {path}.")
    return result


def _load_eval_deploy_direction(config: Experiment9Config) -> np.ndarray | None:
    path = config.results_dir_ex9 / "eval_deploy_direction.npy"
    if path.exists():
        return np.load(path).astype(np.float32)
    print(f"  WARNING: eval/deploy direction not found at {path}.")
    return None


def _load_analysis_results(config: Experiment9Config) -> dict | None:
    path = config.results_dir_ex9 / "analysis_results.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def main() -> None:
    args = parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    config = Experiment9Config()

    print("\nExperiment 9: Probe Transfer and Representational Geometry")
    print(f"  Device:           {get_device()}")
    print(f"  Output directory: {config.output_dir_ex9.resolve()}")
    print(f"  GoT datasets:     {config.got_datasets}")
    print(f"  Truth layers:     {config.truth_layers}")
    print(f"  Primary layer:    {config.primary_layer}")
    print(f"  Random vectors:   {config.n_random_vectors}")

    # ------------------------------------------------------------------
    # Step 1: Extract truth activations and eval/deploy direction
    #         Requires: Llama 8B loaded
    # ------------------------------------------------------------------
    if _should_run("extract", args.from_step):
        print("\n=== Step 1: Extracting truth activations and eval/deploy direction ===")
        model, tokenizer, device = load_model(config)

        all_truth_acts = run_truth_extraction(
            model, tokenizer, config, device, force=args.force
        )
        eval_deploy_dir = run_eval_deploy_extraction(
            model, tokenizer, config, device, force=args.force
        )

        unload_model(model)
    else:
        print("\n[Skipping Step 1] Loading truth activations and eval/deploy direction from cache...")
        all_truth_acts  = _load_truth_acts_from_cache(config)
        eval_deploy_dir = _load_eval_deploy_direction(config)
        if eval_deploy_dir is None:
            raise RuntimeError(
                "Eval/deploy direction missing. Run with --from-step extract."
            )

    print(f"  Truth datasets loaded: {list(all_truth_acts.keys())}")
    print(f"  Eval/deploy direction: shape={eval_deploy_dir.shape}")

    # ------------------------------------------------------------------
    # Step 2: Extract probe directions (Ex1 prefill + Ex3 per-model)
    #         CPU-only, uses saved activations
    # ------------------------------------------------------------------
    if _should_run("directions", args.from_step):
        print("\n=== Step 2: Extracting probe directions (CPU, from saved activations) ===")

        print("  2a. Prefill probe direction (Experiment 1, layer 30)...")
        prefill_bundle = get_prefill_probe_direction(config, force=args.force)
        print(f"      AUROC={prefill_bundle.test_auroc:.4f}  "
              f"n_train={prefill_bundle.n_train}")

        if not args.skip_ex3:
            print("  2b. Per-model probe directions (Experiment 3, layer 30)...")
            try:
                per_model_bundles = get_per_model_directions(config, force=args.force)
                print(f"      Conditions extracted: {list(per_model_bundles.keys())}")
            except FileNotFoundError as e:
                print(f"  WARNING: {e}")
                print("  Skipping per-model directions. Use --skip-ex3 to suppress this warning.")
                per_model_bundles = {}
        else:
            print("  2b. Skipping per-model directions (--skip-ex3 set).")
            per_model_bundles = {}
    else:
        print("\n[Skipping Step 2] Loading probe directions from cache...")
        prefill_bundle = get_prefill_probe_direction(config, force=False)

        per_model_bundles = {}
        if not args.skip_ex3:
            try:
                per_model_bundles = get_per_model_directions(config, force=False)
            except (FileNotFoundError, Exception) as e:
                print(f"  Per-model directions unavailable: {e}")

    # ------------------------------------------------------------------
    # Step 3: Train truth probes (GoT, layers [16, 24, 30])
    #         CPU-only
    # ------------------------------------------------------------------
    if _should_run("truth_probes", args.from_step):
        print("\n=== Step 3: Training truth probes on Geometry of Truth data ===")
        if not all_truth_acts:
            raise RuntimeError("No truth activations available. Run --from-step extract first.")
        truth_bundles = get_truth_probe_directions(all_truth_acts, config, force=args.force)
    else:
        print("\n[Skipping Step 3] Loading truth probe directions from cache...")
        # Reload truth acts if needed for later steps
        if not all_truth_acts:
            all_truth_acts = _load_truth_acts_from_cache(config)
        truth_bundles = get_truth_probe_directions(all_truth_acts, config, force=False)

    # ------------------------------------------------------------------
    # Step 4: Run all analyses
    # ------------------------------------------------------------------
    if _should_run("analysis", args.from_step):
        print("\n=== Step 4: Running geometric analyses ===")
        if not all_truth_acts:
            all_truth_acts = _load_truth_acts_from_cache(config)

        analysis_results = run_all_analyses(
            all_truth_acts,
            prefill_bundle,
            truth_bundles,
            eval_deploy_dir,
            per_model_bundles,
            config,
        )

        # Attach in-domain prefill AUROC to results
        analysis_results["prefill_probe_auroc"] = prefill_bundle.test_auroc

        # Save to JSON now so figures step can load without re-running
        import json as _json
        from prefill_probe.analysis_ex9 import _make_json_safe
        out_path = config.results_dir_ex9 / "analysis_results.json"
        with open(out_path, "w") as f:
            _json.dump(_make_json_safe(analysis_results), f, indent=2)
        print(f"  Analysis results saved → {out_path}")
    else:
        print("\n[Skipping Step 4] Loading analysis results from cache...")
        analysis_results = _load_analysis_results(config)
        if analysis_results is None:
            raise RuntimeError(
                "Analysis results not found. Run with --from-step analysis."
            )

    # ------------------------------------------------------------------
    # Step 5: Generate figures
    # ------------------------------------------------------------------
    if _should_run("figures", args.from_step):
        print("\n=== Step 5: Generating figures ===")
        if analysis_results is None:
            analysis_results = _load_analysis_results(config)
            if analysis_results is None:
                raise RuntimeError("Analysis results not found. Run step 4 first.")
        generate_all_figures_ex9(analysis_results, config)
    else:
        print("\n[Skipping Step 5] Figure generation skipped.")

    print(f"\n=== Experiment 9 complete. ===")
    print(f"  Results in: {config.results_dir_ex9.resolve()}")


if __name__ == "__main__":
    main()
