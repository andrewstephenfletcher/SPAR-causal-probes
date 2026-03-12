"""
run_pipeline.py — Run the full DCT probe experiment pipeline.

Steps:
  1. data_preparation.py  — Prepare datasets, prompts, calibration texts
  2. dct_find_vectors.py  — Find and save DCT steering vectors
  3. judge_vectors.py     — Steer model, judge completions via LLM

Usage:
  python run_pipeline.py --experiment qwen-1.5-7b

After this completes, open the analysis notebook and set EXPERIMENT_NAME
at the top to match the experiment you ran.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def run_step(name, script_path, extra_args=None, skip_if_exists=None):
    if skip_if_exists and all(p.exists() for p in skip_if_exists):
        files = ", ".join(str(p) for p in skip_if_exists)
        print(f"\n  SKIP: {name} (outputs already exist: {files})")
        return

    print(f"\n{'='*60}")
    print(f"  STEP: {name}")
    print(f"  Script: {script_path}")
    print(f"{'='*60}\n")

    cmd = [sys.executable, str(script_path.resolve())] + (extra_args or [])
    start = time.time()
    result = subprocess.run(cmd, cwd=str(script_path.parent.resolve()))
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n  FAILED after {elapsed:.1f}s (exit code {result.returncode})")
        sys.exit(result.returncode)

    print(f"\n  Completed in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, help="Experiment name from dct_params.json")
    args = parser.parse_args()
    experiment = args.experiment

    base = Path(__file__).parent
    exp_dir = base / "experiments" / experiment

    params_path = base / "dct_params.json"
    with open(params_path) as f:
        all_params = json.load(f)
    if experiment not in all_params:
        print(f"ERROR: Unknown experiment '{experiment}'. Available: {list(all_params)}")
        sys.exit(1)
    params = all_params[experiment]
    judge_prompt = params.get("JUDGE_PROMPT")

    print("DCT PROBE EXPERIMENT PIPELINE")
    print(f"Experiment: {experiment}")
    print(f"Working directory: {base.resolve()}")
    if judge_prompt is not None:
        print(f"Judge prompt: custom (from JUDGE_PROMPT param)")
    else:
        print(f"Judge prompt: default (JUDGE_SYSTEM)")

    exp_args = ["--experiment", experiment]

    steps = [
        ("Data preparation",    base / "data_preparation.py",  None,       [base / "data" / "steering_prompts.jsonl"]),
        ("DCT vector discovery", base / "dct_find_vectors.py", exp_args,   [exp_dir / "vectors" / "dct_vectors.pt"]),
        ("Steering + LLM judge", base / "judge_vectors.py",    exp_args,   [exp_dir / "results" / "judge_results.jsonl"]),
    ]

    for name, path, _, __ in steps:
        if not path.exists():
            print(f"ERROR: {path} not found")
            sys.exit(1)

    total_start = time.time()
    for name, path, extra_args, skip_if_exists in steps:
        run_step(name, path, extra_args=extra_args, skip_if_exists=skip_if_exists)

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE — {total_elapsed:.1f}s total")
    print(f"{'='*60}")
    print(f"\nOutputs:")
    print(f"  Vectors: {exp_dir / 'vectors' / 'dct_vectors.pt'}")
    print(f"  Results: {exp_dir / 'results' / 'judge_results.jsonl'}")
    print(f"\nNext: open the analysis notebook and set EXPERIMENT_NAME = \"{experiment}\"")


if __name__ == "__main__":
    main()
