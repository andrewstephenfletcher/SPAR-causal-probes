"""
Experiment 1: Prefill Detection — end-to-end orchestration.

Run from the project root:
    python -m prefill_probe.run_experiment1            # run all steps, skip completed
    python -m prefill_probe.run_experiment1 --force   # re-run everything
    python -m prefill_probe.run_experiment1 --from-step extract  # skip earlier steps

Steps:
  1. data       — load & filter Alpaca prompts
  2. generate   — generate responses from target (Llama) and source (Gemma)
  3. extract    — extract activations from target model (two conditions)
  4. perplexity — compute per-token log-probs under target model
  5. probe      — train & evaluate logistic regression probes
  6. analysis   — generate figures and summary table
"""

import argparse
import sys
from pathlib import Path

# Allow running as `python run_experiment1.py` from inside the package directory
# by inserting the parent (prefill_awareness/) into sys.path.
_here = Path(__file__).resolve().parent.parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from prefill_probe.analysis import generate_all_figures, print_summary_table
from prefill_probe.config import Config
from prefill_probe.data import load_and_filter_prompts
from prefill_probe.extract import extract_all_activations
from prefill_probe.generate import generate_all_responses
from prefill_probe.perplexity import compute_all_perplexity
from prefill_probe.probe import train_and_evaluate_all_probes
from prefill_probe.utils import log_environment

_STEP_ORDER = ["data", "generate", "extract", "perplexity", "probe", "analysis"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 1: Prefill Detection"
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override Config.output_dir (and derived subdirs).",
    )
    return parser.parse_args()


def _should_run(step: str, from_step: str | None) -> bool:
    if from_step is None:
        return True
    return _STEP_ORDER.index(step) >= _STEP_ORDER.index(from_step)


def main() -> None:
    args = parse_args()
    config = Config()

    if args.output_dir is not None:
        base = args.output_dir
        config.output_dir = base
        config.generations_dir = base / "generations"
        config.activations_dir = base / "activations"
        config.results_dir = base / "results"
        config.__post_init__()  # re-create directories

    log_environment(config)

    # ------------------------------------------------------------------ #
    # Step 1: Data
    # ------------------------------------------------------------------ #
    if _should_run("data", args.from_step):
        print("\n=== Step 1: Loading and filtering prompts ===")
        prompts = load_and_filter_prompts(config, force=args.force)
    else:
        import json
        prompts_path = config.generations_dir / "prompts.json"
        print(f"\n[Skipping Step 1] Loading prompts from {prompts_path}")
        with open(prompts_path) as f:
            prompts = json.load(f)

    # ------------------------------------------------------------------ #
    # Step 2: Generate responses
    # ------------------------------------------------------------------ #
    if _should_run("generate", args.from_step):
        print("\n=== Step 2: Generating responses from both models ===")
        responses = generate_all_responses(prompts, config, force=args.force)
    else:
        import json
        resp_path = config.generations_dir / "responses.json"
        print(f"\n[Skipping Step 2] Loading responses from {resp_path}")
        with open(resp_path) as f:
            responses = json.load(f)

    # ------------------------------------------------------------------ #
    # Step 3: Extract activations
    # ------------------------------------------------------------------ #
    if _should_run("extract", args.from_step):
        print("\n=== Step 3: Extracting activations from target model ===")
        extract_all_activations(responses, config, force=args.force)
    else:
        print("\n[Skipping Step 3] Using existing activation files.")

    # ------------------------------------------------------------------ #
    # Step 4: Perplexity
    # ------------------------------------------------------------------ #
    if _should_run("perplexity", args.from_step):
        print("\n=== Step 4: Computing perplexity scores ===")
        compute_all_perplexity(responses, config, force=args.force)
    else:
        print("\n[Skipping Step 4] Using existing perplexity file.")

    # ------------------------------------------------------------------ #
    # Step 5: Probe training & evaluation
    # ------------------------------------------------------------------ #
    if _should_run("probe", args.from_step):
        print("\n=== Step 5: Training and evaluating probes ===")
        probe_results = train_and_evaluate_all_probes(responses, config)
    else:
        print("\n[Skipping Step 5] probe results will not be computed.")
        probe_results = {}

    # ------------------------------------------------------------------ #
    # Step 6: Analysis
    # ------------------------------------------------------------------ #
    if _should_run("analysis", args.from_step) and probe_results:
        print("\n=== Step 6: Generating figures and summary ===")
        generate_all_figures(probe_results, responses, config)
        print_summary_table(probe_results, responses, config)

    print("\n=== Experiment 1 complete. ===")
    print(f"  Results in: {config.results_dir.resolve()}")


if __name__ == "__main__":
    main()
