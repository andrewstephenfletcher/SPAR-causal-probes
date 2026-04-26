"""
Experiment 0: Verbalized Prefill Awareness — orchestration.

Run from prefill_awareness/:
    python -m experiment0.run_all
    python -m experiment0.run_all --force
    python -m experiment0.run_all --from-step attribution

Steps:
  1. data        — load 150 prompts (50 per dataset)
  2. generate    — generate responses from all models via OpenRouter
  3. attribution — present prefilled conversations and record answers
  4. analysis    — compute metrics and produce 4 figures + summary CSV
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure dotenv is loaded before anything reads os.environ
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

from experiment0.analysis import run_analysis
from experiment0.config import Experiment0Config
from experiment0.data import load_prompts
from experiment0.generate_responses import generate_all_responses
from experiment0.run_attribution import run_all_attribution
from experiment0.utils import verify_model_ids

_STEP_ORDER = ["data", "generate", "attribution", "analysis"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 0: Verbalized Prefill Awareness"
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
            "Skip all steps before STEP (earlier outputs must exist). "
            f"One of: {', '.join(_STEP_ORDER)}"
        ),
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip OpenRouter model ID verification (use if IDs are known good).",
    )
    return parser.parse_args()


def _should_run(step: str, from_step: str | None) -> bool:
    if from_step is None:
        return True
    return _STEP_ORDER.index(step) >= _STEP_ORDER.index(from_step)


def main() -> None:
    args = parse_args()
    config = Experiment0Config()

    # ------------------------------------------------------------------ #
    # Verify model IDs against OpenRouter
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
    # Step 1: Data
    # ------------------------------------------------------------------ #
    if _should_run("data", args.from_step):
        print("\n=== Step 1: Loading prompts ===")
        prompts = load_prompts(config)
    else:
        prompts_path = config.generations_dir / "prompts.json"
        print(f"\n[Skipping Step 1] Loading prompts from {prompts_path}")
        with open(prompts_path) as f:
            prompts = json.load(f)

    by_ds = {}
    for p in prompts:
        by_ds[p["dataset"]] = by_ds.get(p["dataset"], 0) + 1
    print(f"  Prompts: {', '.join(f'{ds}={n}' for ds, n in sorted(by_ds.items()))}")

    # ------------------------------------------------------------------ #
    # Step 2: Generate responses
    # ------------------------------------------------------------------ #
    if _should_run("generate", args.from_step):
        print("\n=== Step 2: Generating responses from all models ===")
        responses = generate_all_responses(prompts, config, force=args.force)
    else:
        resp_path = config.generations_dir / "responses.json"
        print(f"\n[Skipping Step 2] Loading responses from {resp_path}")
        with open(resp_path) as f:
            responses = json.load(f)

    n_complete = sum(
        1 for r in responses
        if len(r.get("responses", {})) == len(config.models)
    )
    print(f"  {n_complete}/{len(responses)} prompts have responses from all models.")

    # ------------------------------------------------------------------ #
    # Step 3: Attribution
    # ------------------------------------------------------------------ #
    if _should_run("attribution", args.from_step):
        print("\n=== Step 3: Running attribution tests ===")
        results = run_all_attribution(prompts, responses, config, force=args.force)
    else:
        results_path = config.results_dir / "attribution_results.json"
        print(f"\n[Skipping Step 3] Loading results from {results_path}")
        with open(results_path) as f:
            results = json.load(f)

    print(f"  {len(results)} attribution records loaded.")

    # ------------------------------------------------------------------ #
    # Step 4: Analysis
    # ------------------------------------------------------------------ #
    if _should_run("analysis", args.from_step):
        print("\n=== Step 4: Running analysis and generating figures ===")
        run_analysis(results, config)

    print("\n=== Experiment 0 complete. ===")
    print(f"  Results in: {config.results_dir.resolve()}")


if __name__ == "__main__":
    main()
