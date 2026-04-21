"""
Experiment 2: Accumulation Analysis — end-to-end orchestration.

Run from the project root:
    python -m prefill_probe.run_experiment2            # run all steps, skip completed
    python -m prefill_probe.run_experiment2 --force   # re-run everything
    python -m prefill_probe.run_experiment2 --from-step extract  # skip earlier steps

Steps:
  1. data     — load & filter Experiment 1 responses (≥32 response tokens)
  2. extract  — extract activations at (layer, position) grid for both conditions
  3. probe    — train one LinearProbe per (layer, position) cell
  4. analysis — generate figures, interpretive checklist, and summary CSV
"""

import argparse
import json
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent.parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from prefill_probe.analysis_positions import generate_all_figures_ex2
from prefill_probe.config import Config, Experiment2Config
from prefill_probe.extract_positions import extract_all_position_activations
from prefill_probe.probe_positions import train_all_position_probes
from prefill_probe.utils import log_environment

_STEP_ORDER = ["data", "extract", "probe", "analysis"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 2: Accumulation Analysis"
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


def _load_responses(ex1_config: Config) -> list[dict]:
    resp_path = ex1_config.generations_dir / "responses.json"
    if not resp_path.exists():
        raise FileNotFoundError(
            f"Experiment 1 responses not found at {resp_path}. "
            "Run run_experiment1.py first (through at least the 'generate' step)."
        )
    with open(resp_path) as f:
        return json.load(f)


def _filter_responses(
    responses: list[dict],
    ex1_config: Config,
    ex2_config: Experiment2Config,
) -> list[dict]:
    """Keep only responses whose target is ≥ min_length_tokens under the target tokenizer."""
    from transformers import AutoTokenizer
    print(f"  Loading tokenizer ({ex1_config.target_model_id}) for length filtering...")
    tokenizer = AutoTokenizer.from_pretrained(ex1_config.target_model_id)
    min_len = ex2_config.min_length_tokens
    max_pos = max(ex2_config.positions)

    filtered = []
    for r in responses:
        toks = tokenizer(r["response_target"], add_special_tokens=False)["input_ids"]
        if len(toks) > max_pos:
            filtered.append(r)

    n_dropped = len(responses) - len(filtered)
    print(f"  Kept {len(filtered)} / {len(responses)} responses "
          f"({n_dropped} dropped — too short for max position {max_pos}).")
    return filtered


def main() -> None:
    args = parse_args()
    ex1_config = Config()
    ex2_config = Experiment2Config()

    log_environment(ex1_config)

    # ------------------------------------------------------------------ #
    # Step 1: Data filtering
    # ------------------------------------------------------------------ #
    if _should_run("data", args.from_step):
        print("\n=== Step 1: Loading and filtering Experiment 1 responses ===")
        responses = _load_responses(ex1_config)
        responses = _filter_responses(responses, ex1_config, ex2_config)
    else:
        print("\n[Skipping Step 1] Loading all Experiment 1 responses directly...")
        responses = _load_responses(ex1_config)

    # ------------------------------------------------------------------ #
    # Step 2: Activation extraction
    # ------------------------------------------------------------------ #
    if _should_run("extract", args.from_step):
        print(
            f"\n=== Step 2: Extracting activations at "
            f"{len(ex2_config.layers)} layers × {len(ex2_config.positions)} positions ==="
        )
        extract_all_position_activations(
            responses, ex2_config, ex1_config, force=args.force
        )
    else:
        print("\n[Skipping Step 2] Using existing Experiment 2 activation files.")

    # ------------------------------------------------------------------ #
    # Step 3: Probe training
    # ------------------------------------------------------------------ #
    if _should_run("probe", args.from_step):
        print(
            f"\n=== Step 3: Training "
            f"{len(ex2_config.layers) * len(ex2_config.positions)} probes ==="
        )
        probe_results_df = train_all_position_probes(
            responses, ex2_config, force=args.force
        )
    else:
        import pandas as pd
        csv_path = ex2_config.results_dir_ex2 / "probe_results.csv"
        print(f"\n[Skipping Step 3] Loading probe results from {csv_path}")
        probe_results_df = pd.read_csv(csv_path)

    # ------------------------------------------------------------------ #
    # Step 4: Analysis
    # ------------------------------------------------------------------ #
    if _should_run("analysis", args.from_step):
        print("\n=== Step 4: Generating figures and summary ===")
        generate_all_figures_ex2(probe_results_df, ex2_config)

    print("\n=== Experiment 2 complete. ===")
    print(f"  Results in: {ex2_config.results_dir_ex2.resolve()}")


if __name__ == "__main__":
    main()
