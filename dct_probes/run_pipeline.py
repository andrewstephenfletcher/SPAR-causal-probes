"""
run_pipeline.py — Run the full DCT probe experiment pipeline.

Steps:
  1. data_preparation.py  — Prepare datasets, prompts, calibration texts
  2. dct_find_vectors.py  — Find and save DCT steering vectors
  3. judge_vectors.py     — Steer model, judge completions via LLM

After this completes, open the analysis notebook to train probes
and compare DCT vectors against MM/LR baselines.
"""

import subprocess
import sys
import time
from pathlib import Path


def run_step(name, script_path):
    print(f"\n{'='*60}")
    print(f"  STEP: {name}")
    print(f"  Script: {script_path}")
    print(f"{'='*60}\n")

    start = time.time()
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(script_path.parent),
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n  FAILED after {elapsed:.1f}s (exit code {result.returncode})")
        sys.exit(result.returncode)

    print(f"\n  Completed in {elapsed:.1f}s")


def main():
    base = Path("dct_probes")

    print("DCT PROBE EXPERIMENT PIPELINE")
    print(f"Working directory: {base.resolve()}")

    steps = [
        ("Data preparation", base / "data_preparation.py"),
        ("DCT vector discovery", base / "dct_find_vectors.py"),
        ("Steering + LLM judge", base / "judge_vectors.py"),
    ]

    # Check all scripts exist before starting
    for name, path in steps:
        if not path.exists():
            print(f"ERROR: {path} not found")
            sys.exit(1)

    total_start = time.time()
    for name, path in steps:
        run_step(name, path)

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE — {total_elapsed:.1f}s total")
    print(f"{'='*60}")
    print(f"\nOutputs:")
    print(f"  Vectors: {base / 'vectors' / 'dct_vectors.pt'}")
    print(f"  Results: {base / 'results' / 'judge_results.jsonl'}")
    print(f"\nNext: open the analysis notebook")


if __name__ == "__main__":
    main()
