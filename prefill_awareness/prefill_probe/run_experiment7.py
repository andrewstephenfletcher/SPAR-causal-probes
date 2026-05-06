"""
Experiment 7: Implicit Behavioral Measures — end-to-end orchestration.

Run from the prefill_awareness/ directory:

    # Full run:
    python -m prefill_probe.run_experiment7

    # Resume from a specific step:
    python -m prefill_probe.run_experiment7 --from-step rate

    # Skip judging (e.g., no OpenRouter key):
    python -m prefill_probe.run_experiment7 --skip-judge

    # Force re-run a step:
    python -m prefill_probe.run_experiment7 --from-step rate --force

Steps:
  1. prepare  — assemble responses from Ex1 + Ex4; generate Llama 70B if needed
  2. rate     — Llama 70B rates own + foreign responses (0-100); challenge experiment
  3. judge    — independent quality judging (OpenRouter) + perplexity under Llama 70B
  4. figures  — 4 figures + summary statistics

Prerequisites:
  - Experiment 1 responses: outputs/experiment1/generations/responses.json
  - Experiment 4 responses: outputs/experiment4/generations/responses.json
    (strongly preferred — if absent, Llama 70B is loaded to generate responses first)
  - OPENROUTER_API_KEY set (required for Step 3 quality judging)
  - HF_TOKEN set for Llama 70B (if generating responses from scratch)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent.parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from prefill_probe.analysis_ex7 import generate_all_figures_ex7
from prefill_probe.config import Experiment7Config
from prefill_probe.generate_ex7 import prepare_ex7_data
from prefill_probe.judge_ex7 import run_perplexity_computation, run_quality_judging
from prefill_probe.rate_ex7 import run_challenge_experiment, run_rating_experiment
from prefill_probe.utils import get_device

_STEP_ORDER = ["prepare", "rate", "judge", "figures"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 7: Implicit Behavioral Measures")
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
        "--skip-judge", action="store_true",
        help="Skip Step 3 quality judging (requires OPENROUTER_API_KEY).",
    )
    parser.add_argument(
        "--skip-perplexity", action="store_true",
        help="Skip perplexity computation in Step 3 (very slow on CPU).",
    )
    return parser.parse_args()


def _should_run(step: str, from_step: str | None) -> bool:
    if from_step is None:
        return True
    return _STEP_ORDER.index(step) >= _STEP_ORDER.index(from_step)


def _load_records(config: Experiment7Config) -> list[dict]:
    path = config.generations_dir_ex7 / "responses_ex7.json"
    if not path.exists():
        raise RuntimeError(
            "responses_ex7.json not found. Run with --from-step prepare first."
        )
    with open(path) as f:
        return json.load(f)


def main() -> None:
    args = parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    config = Experiment7Config()

    print("\nExperiment 7: Implicit Behavioral Measures")
    print(f"  Device:           {get_device()}")
    print(f"  Rating model:     {config.llama70b_model_id}")
    print(f"  Judge model:      {config.judge_model_id}")
    print(f"  n_prompts:        {config.n_prompts}")
    print(f"  n_challenge:      {config.n_prompts_challenge}")
    print(f"  Output dir:       {config.output_dir_ex7.resolve()}")

    # ------------------------------------------------------------------
    # Step 1: Prepare / assemble responses
    # ------------------------------------------------------------------
    if _should_run("prepare", args.from_step):
        print("\n=== Step 1: Preparing response data ===")
        records = prepare_ex7_data(config, force=args.force)
        print(f"  {len(records)} records ready.")
    else:
        print("\n[Skipping Step 1] Loading responses from cache...")
        records = _load_records(config)
        print(f"  Loaded {len(records)} records.")

    # ------------------------------------------------------------------
    # Step 2: Rate responses + challenge experiment
    # ------------------------------------------------------------------
    if _should_run("rate", args.from_step):
        print("\n=== Step 2: Rating responses (0-100 scale) ===")
        ratings = run_rating_experiment(records, config, force=args.force)
        print(f"  Ratings complete: {len(ratings)} records.")

        print("\n  Running challenge experiment...")
        challenge = run_challenge_experiment(records, config, force=args.force)
        print(f"  Challenge complete: {len(challenge)} records.")
    else:
        print("\n[Skipping Step 2] Rating/challenge.")

    # ------------------------------------------------------------------
    # Step 3: Judge quality + perplexity
    # ------------------------------------------------------------------
    if _should_run("judge", args.from_step):
        if not args.skip_judge:
            print("\n=== Step 3a: Quality judging via OpenRouter ===")
            try:
                run_quality_judging(records, config, force=args.force)
            except EnvironmentError as e:
                print(f"  {e}")
                print("  Use --skip-judge to skip this step.")
        else:
            print("\n[Skipping Step 3a] Quality judging (--skip-judge).")

        if not args.skip_perplexity:
            print("\n=== Step 3b: Perplexity under Llama 70B ===")
            run_perplexity_computation(records, config, force=args.force)
        else:
            print("\n[Skipping Step 3b] Perplexity (--skip-perplexity).")
    else:
        print("\n[Skipping Step 3] Judging and perplexity.")

    # ------------------------------------------------------------------
    # Step 4: Figures
    # ------------------------------------------------------------------
    if _should_run("figures", args.from_step):
        print("\n=== Step 4: Generating figures ===")
        generate_all_figures_ex7(config)
    else:
        print("\n[Skipping Step 4] Figures.")

    print(f"\n=== Experiment 7 complete. ===")
    print(f"  Results in: {config.results_dir_ex7.resolve()}")


if __name__ == "__main__":
    main()
