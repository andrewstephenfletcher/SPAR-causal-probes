"""
Experiment 3: Full Condition Matrix and Generalization — end-to-end orchestration.

Run from prefill_awareness/:
    python -m prefill_probe.run_experiment3
    python -m prefill_probe.run_experiment3 --force
    python -m prefill_probe.run_experiment3 --from-step extract

Steps:
  1. data        — load Alpaca (reuse Ex1) + OASST1 + MMLU prompts
  2. generate    — generate 5-condition responses (checkpoint-resumable)
  3. extract     — extract layer-30 activations for all 5 conditions
  4. perplexity  — compute mean log-prob under Llama 8B for all conditions
  5. probe       — train probes + transfer matrices + perplexity analyses
  6. analysis    — generate 5 figures, summary CSV, interpretive summary

Llama 70B is excluded so this runs on a single GPU / Apple Silicon Mac.
"""

import argparse
import json
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent.parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from prefill_probe.analysis_ex3 import generate_all_figures_ex3
from prefill_probe.analysis_pca_ex3 import generate_pca_figures
from prefill_probe.config import Config, Experiment3Config
from prefill_probe.data_ex3 import load_all_prompts
from prefill_probe.extract_ex3 import extract_all_activations_ex3
from prefill_probe.generate_ex3 import generate_all_responses_ex3
from prefill_probe.perplexity_ex3 import compute_all_perplexity_ex3
from prefill_probe.probe_ex3 import run_all_probe_analyses
from prefill_probe.utils import log_environment

_STEP_ORDER = ["data", "generate", "extract", "perplexity", "probe", "analysis", "visualize"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 3: Full Condition Matrix and Generalization"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run all steps, overwriting existing outputs.",
    )
    parser.add_argument(
        "--from-step",
        choices=_STEP_ORDER,
        default=None,
        metavar="STEP",
        help=(
            "Skip all steps before STEP (earlier outputs must already exist). "
            f"One of: {', '.join(_STEP_ORDER)}"
        ),
    )
    return parser.parse_args()


def _should_run(step: str, from_step: str | None) -> bool:
    if from_step is None:
        return True
    return _STEP_ORDER.index(step) >= _STEP_ORDER.index(from_step)


def main() -> None:
    args = parse_args()
    ex1_config = Config()
    ex3_config = Experiment3Config()

    log_environment(ex1_config)

    # ------------------------------------------------------------------ #
    # Step 1: Data
    # ------------------------------------------------------------------ #
    if _should_run("data", args.from_step):
        print("\n=== Step 1: Loading prompts (Alpaca + OASST1 + MMLU) ===")
        all_prompts = load_all_prompts(ex3_config, ex1_config)
    else:
        master_path = ex3_config.generations_dir_ex3 / "prompts_all.json"
        print(f"\n[Skipping Step 1] Loading prompts from {master_path}")
        with open(master_path) as f:
            all_prompts = json.load(f)

    by_ds = {}
    for p in all_prompts:
        by_ds[p["dataset"]] = by_ds.get(p["dataset"], 0) + 1
    print(f"  Prompts: " + ", ".join(f"{ds}={n}" for ds, n in sorted(by_ds.items())))

    # ------------------------------------------------------------------ #
    # Step 2: Generate responses
    # ------------------------------------------------------------------ #
    if _should_run("generate", args.from_step):
        print("\n=== Step 2: Generating 5-condition responses ===")
        responses = generate_all_responses_ex3(
            all_prompts, ex3_config, ex1_config, force=args.force
        )
    else:
        resp_path = ex3_config.generations_dir_ex3 / "responses_all.json"
        print(f"\n[Skipping Step 2] Loading responses from {resp_path}")
        with open(resp_path) as f:
            responses = json.load(f)

    print(f"  {len(responses)} prompts with complete 5-condition responses.")

    # ------------------------------------------------------------------ #
    # Step 3: Extract activations
    # ------------------------------------------------------------------ #
    if _should_run("extract", args.from_step):
        print("\n=== Step 3: Extracting layer-30 activations (5 conditions) ===")
        extract_all_activations_ex3(responses, ex3_config, ex1_config, force=args.force)
    else:
        print("\n[Skipping Step 3] Using existing activation files.")

    # ------------------------------------------------------------------ #
    # Step 4: Perplexity
    # ------------------------------------------------------------------ #
    if _should_run("perplexity", args.from_step):
        print("\n=== Step 4: Computing perplexity for all conditions ===")
        compute_all_perplexity_ex3(responses, ex3_config, ex1_config, force=args.force)
    else:
        print("\n[Skipping Step 4] Using existing perplexity file.")

    # ------------------------------------------------------------------ #
    # Step 5: Probe training & analyses
    # ------------------------------------------------------------------ #
    if _should_run("probe", args.from_step):
        print("\n=== Step 5: Training probes + running analyses ===")
        probe_results = run_all_probe_analyses(responses, ex3_config, force=args.force)
    else:
        results_path = ex3_config.results_dir_ex3 / "probe_results_ex3.json"
        print(f"\n[Skipping Step 5] Loading probe results from {results_path}")
        with open(results_path) as f:
            probe_results = json.load(f)

    # ------------------------------------------------------------------ #
    # Step 6: Analysis & figures
    # ------------------------------------------------------------------ #
    if _should_run("analysis", args.from_step):
        print("\n=== Step 6: Generating figures and interpretive summary ===")
        generate_all_figures_ex3(probe_results, responses, ex3_config)

    # ------------------------------------------------------------------ #
    # Step 7: PCA visualization
    # ------------------------------------------------------------------ #
    if _should_run("visualize", args.from_step):
        print("\n=== Step 7: PCA visualization ===")
        generate_pca_figures(ex3_config, responses=responses)

    print("\n=== Experiment 3 complete. ===")
    print(f"  Results in: {ex3_config.results_dir_ex3.resolve()}")


if __name__ == "__main__":
    main()
