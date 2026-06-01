"""
Experiment 11: Unified Prefill-Awareness Data Collection — orchestrator.

Run from the prefill_awareness/ directory:

    # Run everything (all models, all datasets):
    python -m prefill_probe.run_experiment11

    # Large-model pod (llama70b): generate small-model responses + extract llama70b activations
    python -m prefill_probe.run_experiment11 --target-model llama70b

    # Large-model pod (gemma31b): skip generate (copy responses first), extract only
    python -m prefill_probe.run_experiment11 --target-model gemma31b --from-step extract

    # Large-model pod (qwen32b): extract only
    python -m prefill_probe.run_experiment11 --target-model qwen32b --from-step extract

    # After downloading all activations, run validation locally:
    python -m prefill_probe.run_experiment11 --from-step validate

    # Re-run everything from scratch:
    python -m prefill_probe.run_experiment11 --force

Steps:
  1. generate  — generate responses from all 6 models (3 large + 3 small)
  2. extract   — extract full-depth + token-position activations for target model(s)
  3. validate  — completeness checks, suffix normalisation audit, collection summary

--target-model controls which large model is processed in the extract step.
The generate step always runs small source models (needed for all cross conditions);
on a multi-pod workflow, run generate once on the llama pod and copy outputs across.

Prerequisites:
  - HF_TOKEN set in environment or .env file; model licences accepted on HuggingFace.
"""

import os
os.environ["HF_HOME"] = "/root/.cache/huggingface"

import argparse
import json
import platform
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent.parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

import torch
import transformers

from prefill_probe.config import Experiment11Config
from prefill_probe.extract_ex11 import (
    extract_all_activations_gemma31b,
    extract_all_activations_llama70b,
    extract_all_activations_qwen32b,
)
from prefill_probe.generate_ex11 import generate_all_responses, load_all_responses
from prefill_probe.utils import get_device


_STEP_ORDER = ["generate", "extract", "validate"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 11: Unified Data Collection")
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
        "--target-model",
        choices=["llama70b", "gemma31b", "qwen32b", "all"],
        default="all",
        help=(
            "Which large target model to run extract for. "
            "'all': all three (default). "
            "Generate step always includes all small source models regardless."
        ),
    )
    return parser.parse_args()


def _should_run(step: str, from_step: str | None) -> bool:
    if from_step is None:
        return True
    return _STEP_ORDER.index(step) >= _STEP_ORDER.index(from_step)


def _validate(config: Experiment11Config) -> None:
    """
    Completeness and quality checks:
      1. Response file counts (6 models × 3 datasets = 18 files)
      2. Suffix normalisation: all response_normalized fields end with '.'
      3. Activation file counts (3 targets × 4 conditions × 3 datasets × 2 types = 72 files)
      4. Collection summary JSON
    """
    print("\n=== Step 3: Validation ===")
    issues = []
    all_model_ids = dict(config.target_models + config.source_models)
    target_keys = [k for k, _ in config.target_models]
    # Per-target conditions: each target has 5 cross conditions (all 5 other models)
    target_conditions = {
        "llama70b": ["self", "cross_llama8b", "cross_gemma4b", "cross_qwen7b",
                     "cross_gemma31b", "cross_qwen32b"],
        "gemma31b": ["self", "cross_llama8b", "cross_gemma4b", "cross_qwen7b",
                     "cross_llama70b", "cross_qwen32b"],
        "qwen32b":  ["self", "cross_llama8b", "cross_gemma4b", "cross_qwen7b",
                     "cross_llama70b", "cross_gemma31b"],
    }

    # 1. Response files
    total_responses = 0
    suffix_violations = 0
    for model_key in all_model_ids:
        norm_field = f"response_{model_key}_normalized"
        for ds in config.datasets:
            path = config.generations_dir / f"{model_key}_{ds}_responses.json"
            if not path.exists():
                issues.append(f"Missing response file: {path}")
                continue
            with open(path) as f:
                records = json.load(f)
            total_responses += len(records)
            for r in records:
                if norm_field in r and not r[norm_field].endswith("."):
                    suffix_violations += 1

    print(f"  Response records found: {total_responses} "
          f"(expected ~{len(all_model_ids) * len(config.datasets) * config.n_prompts_per_dataset})")
    if suffix_violations:
        issues.append(f"Suffix normalisation violations: {suffix_violations} records don't end with '.'")
    else:
        print("  Suffix normalisation: all response_normalized fields end with '.'  OK")

    # 2. Activation files
    total_act_files = 0
    expected_act = 0
    for target_key in target_keys:
        act_dir = config.activations_dir_for(target_key)
        t_conditions = target_conditions.get(target_key, [])
        expected_act += len(t_conditions) * len(config.datasets) * 2
        for cond in t_conditions:
            cond_dir = act_dir / cond
            for ds in config.datasets:
                for suffix in ["", "_token_positions"]:
                    path = cond_dir / f"{ds}{suffix}.pt"
                    if not path.exists():
                        issues.append(f"Missing activation file: {path}")
                    else:
                        try:
                            data = torch.load(path, weights_only=False)
                            if isinstance(data, list) and len(data) > 0:
                                total_act_files += 1
                            else:
                                issues.append(f"Empty activation file: {path}")
                        except Exception as e:
                            issues.append(f"Cannot load {path}: {e}")
    print(f"  Activation files found: {total_act_files} / {expected_act}")

    # 3. Summary
    summary = {
        "total_response_records": total_responses,
        "suffix_violations": suffix_violations,
        "total_activation_files": total_act_files,
        "expected_activation_files": expected_act,
        "issues": issues,
    }
    summary_path = config.metadata_dir / "collection_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary → {summary_path}")

    if issues:
        print(f"\n  ISSUES ({len(issues)}):")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("  Validation passed with no issues.")


def main() -> None:
    args = parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    config = Experiment11Config()

    print(f"\nExperiment 11: Unified Prefill-Awareness Data Collection")
    print(f"  Device:           {get_device()}")
    print(f"  Output directory: {config.output_dir.resolve()}")
    print(f"  Target model(s):  {args.target_model}")

    # Log environment
    env = {
        "device": get_device(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    if torch.cuda.is_available():
        env["gpu_count"] = torch.cuda.device_count()
        env["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(env["gpu_count"])]
    env_path = config.metadata_dir / "environment.json"
    with open(env_path, "w") as f:
        json.dump(env, f, indent=2)
    print(f"  Environment logged → {env_path}")

    # ------------------------------------------------------------------ #
    # Step 1: Generate
    # ------------------------------------------------------------------ #
    if _should_run("generate", args.from_step):
        print("\n=== Step 1: Generating responses ===")
        # On a single-target pod, only run small source models + that target's own responses.
        gen_target = args.target_model
        all_responses = generate_all_responses(config, force=args.force, target_model=gen_target)
    else:
        print("\n[Skipping Step 1] Loading existing responses...")
        try:
            all_responses = load_all_responses(config)
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
            print("  Cannot skip generate step — missing files. Exiting.")
            sys.exit(1)

    n_total = sum(
        len(records)
        for per_ds in all_responses.values()
        for records in per_ds.values()
    )
    print(f"  Working with {n_total} total response records.")

    # ------------------------------------------------------------------ #
    # Step 2: Extract
    # ------------------------------------------------------------------ #
    if _should_run("extract", args.from_step):
        print("\n=== Step 2: Extracting activations ===")

        run_llama70b = args.target_model in ("llama70b", "all")
        run_gemma31b = args.target_model in ("gemma31b", "all")
        run_qwen32b  = args.target_model in ("qwen32b",  "all")

        if run_llama70b:
            print("\n  --- Llama 3.3 70B ---")
            extract_all_activations_llama70b(all_responses, config)

        if run_gemma31b:
            print("\n  --- Gemma 4 31B ---")
            extract_all_activations_gemma31b(all_responses, config)

        if run_qwen32b:
            print("\n  --- Qwen 32B ---")
            extract_all_activations_qwen32b(all_responses, config)
    else:
        print("\n[Skipping Step 2] Using existing activation files.")

    # ------------------------------------------------------------------ #
    # Step 3: Validate
    # ------------------------------------------------------------------ #
    if _should_run("validate", args.from_step):
        _validate(config)
    else:
        print("\n[Skipping Step 3] Validation skipped.")

    print(f"\n=== Experiment 11 complete. ===")
    print(f"  Results in: {config.output_dir.resolve()}")


if __name__ == "__main__":
    main()
