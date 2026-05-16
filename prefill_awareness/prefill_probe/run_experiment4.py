"""
Experiment 4: Scaling Analysis — end-to-end orchestration.

Run from the prefill_awareness/ directory:

    # Run everything (single-pod, both models):
    python -m prefill_probe.run_experiment4

    # Two-pod workflow — Step A on the Llama pod (2× A100 80GB):
    python -m prefill_probe.run_experiment4 --target-model llama70b

    # Two-pod workflow — Step B on the Gemma pod (1× A100 80GB), after copying
    # responses.json from the Llama pod to the same relative path:
    python -m prefill_probe.run_experiment4 --target-model gemma31b --from-step extract

    # After downloading all activations, run probe + analysis locally (CPU):
    python -m prefill_probe.run_experiment4 --from-step probe

    # Resume a specific step:
    python -m prefill_probe.run_experiment4 --from-step extract
    python -m prefill_probe.run_experiment4 --force   # re-run everything

Steps:
  1. generate   — generate Llama 70B and Gemma 31B responses; loads Llama 8B
                  and Gemma 9B responses from Experiment 1
  2. extract    — extract residual-stream activations at every layer, all conditions
  3. perplexity — compute per-token log-probs under each target model
  4. probe      — train LinearProbes at every layer, all conditions; save JSON
  5. analysis   — Figures 1-4 and summary table

--target-model controls which model is processed in the extract and perplexity
steps.  The generate step always runs both (it needs VRAM for both models
sequentially; run it on the Llama pod where 2× A100 80GB is available).

Prerequisites:
  - Experiment 1 must have been run: outputs/experiment1/{generations,activations}/
    must contain responses.json, self_prefill.pt, cross_gemma_prefill.pt,
    and perplexity.json.
  - HF_TOKEN set and model licences accepted on HuggingFace for both models.
"""

import os
os.environ["HF_HOME"] = "/root/.cache/huggingface"

import argparse
import json
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent.parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from prefill_probe.analysis_ex4 import generate_all_figures, load_probe_results_if_present
from prefill_probe.config import Experiment4Config
from prefill_probe.extract_ex4 import (
    extract_all_activations_gemma31b,
    extract_all_activations_gemma4b,
    extract_all_activations_llama70b,
    extract_all_activations_qwen7b,
    extract_all_activations_qwen32b,
    run_all_extractions,
)
from prefill_probe.generate_ex4 import generate_all_responses
from prefill_probe.perplexity_ex4 import (
    _compute_perplexity_for_model,
    compute_all_perplexity,
)
from prefill_probe.probe_ex4 import train_all_probes
from prefill_probe.utils import get_device

_STEP_ORDER = ["generate", "extract", "perplexity", "probe", "analysis"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 4: Scaling Analysis")
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
        "--target-model",
        choices=["llama70b", "gemma31b", "gemma4b", "gemma-all",
                 "qwen7b", "qwen32b", "mistral-all", "both"],
        default="both",
        help=(
            "Which model(s) to run generate/extract/perplexity for.  "
            "'llama70b': 2× A100 pod.  "
            "'gemma31b'/'gemma4b'/'gemma-all': Gemma models on a Gemma machine.  "
            "'qwen7b'/'qwen32b'/'mistral-all': Mistral models on a Mistral machine.  "
            "'both': all models (default).  "
            "Probe and analysis always run for all models with available results."
        ),
    )
    return parser.parse_args()


def _should_run(step: str, from_step: str | None) -> bool:
    if from_step is None:
        return True
    return _STEP_ORDER.index(step) >= _STEP_ORDER.index(from_step)


def main() -> None:
    args = parse_args()

    # Load HF_TOKEN (and any other env vars) from .env if present
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # python-dotenv not installed; rely on env vars being set externally

    config = Experiment4Config()

    run_llama70b  = args.target_model in ("llama70b", "both")
    run_gemma31b  = args.target_model in ("gemma31b",  "gemma-all",   "both")
    run_gemma4b   = args.target_model in ("gemma4b",   "gemma-all",   "both")
    run_qwen7b  = args.target_model in ("qwen7b",  "mistral-all", "both")
    run_qwen32b = args.target_model in ("qwen32b", "mistral-all", "both")

    print(f"\nExperiment 4: Scaling Analysis")
    print(f"  Device: {get_device()}")
    print(f"  Output directory: {config.output_dir_ex4.resolve()}")
    print(f"  Target model(s): {args.target_model}")

    # Save environment info
    import platform
    import torch
    import transformers
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
    env_path = config.output_dir_ex4 / "environment.json"
    with open(env_path, "w") as f:
        json.dump(env, f, indent=2)
    print(f"  Environment logged → {env_path}")

    # ------------------------------------------------------------------ #
    # Step 1: Generate
    # ------------------------------------------------------------------ #
    if _should_run("generate", args.from_step):
        print("\n=== Step 1: Generating responses ===")
        responses = generate_all_responses(config, force=args.force)
    else:
        resp_path = config.generations_dir_ex4 / "responses.json"
        print(f"\n[Skipping Step 1] Loading responses from {resp_path}")
        with open(resp_path) as f:
            responses = json.load(f)

    print(f"  Working with {len(responses)} prompts.")

    # ------------------------------------------------------------------ #
    # Step 2: Extract
    # ------------------------------------------------------------------ #
    if _should_run("extract", args.from_step):
        print("\n=== Step 2: Extracting activations ===")
        if run_llama70b:
            print("  --- Llama 3.3 70B ---")
            extract_all_activations_llama70b(responses, config)
        if run_gemma31b:
            print("  --- Gemma 4 31B ---")
            extract_all_activations_gemma31b(responses, config)
        if run_gemma4b:
            print("  --- Gemma 4 4B ---")
            extract_all_activations_gemma4b(responses, config)
        if run_qwen7b:
            print("  --- Qwen 7B ---")
            extract_all_activations_qwen7b(responses, config)
        if run_qwen32b:
            print("  --- Qwen 32B ---")
            extract_all_activations_qwen32b(responses, config)
    else:
        print("\n[Skipping Step 2] Using existing activation files.")

    # ------------------------------------------------------------------ #
    # Step 3: Perplexity
    # ------------------------------------------------------------------ #
    if _should_run("perplexity", args.from_step):
        print("\n=== Step 3: Computing perplexity ===")
        if run_llama70b:
            _compute_perplexity_for_model(
                target_model_id=config.llama70b_model_id,
                conditions={
                    "self_prefill":  "response_llama70b",
                    "cross_gemma9b": "response_gemma9b",
                    "cross_llama8b": "response_llama8b",
                },
                responses=responses,
                output_path=config.activations_dir_llama70b / "perplexity.json",
                force=args.force,
            )
        if run_gemma31b:
            _compute_perplexity_for_model(
                target_model_id=config.gemma31b_model_id,
                conditions={
                    "self_prefill":  "response_gemma31b",
                    "cross_llama8b": "response_llama8b",
                    "cross_gemma9b": "response_gemma9b",
                },
                responses=responses,
                output_path=config.activations_dir_gemma31b / "perplexity.json",
                force=args.force,
            )
        if run_gemma4b:
            _compute_perplexity_for_model(
                target_model_id=config.gemma4b_model_id,
                conditions={
                    "self_prefill":  "response_gemma4b",
                    "cross_llama8b": "response_llama8b",
                    "cross_gemma9b": "response_gemma9b",
                },
                responses=responses,
                output_path=config.activations_dir_gemma4b / "perplexity.json",
                force=args.force,
            )
        if run_qwen7b:
            _compute_perplexity_for_model(
                target_model_id=config.qwen7b_model_id,
                conditions={
                    "self_prefill":  "response_qwen7b",
                    "cross_llama8b": "response_llama8b",
                    "cross_gemma9b": "response_gemma9b",
                },
                responses=responses,
                output_path=config.activations_dir_qwen7b / "perplexity.json",
                force=args.force,
            )
        if run_qwen32b:
            _compute_perplexity_for_model(
                target_model_id=config.qwen32b_model_id,
                conditions={
                    "self_prefill":  "response_qwen32b",
                    "cross_llama8b": "response_llama8b",
                    "cross_gemma9b": "response_gemma9b",
                },
                responses=responses,
                output_path=config.activations_dir_qwen32b / "perplexity.json",
                force=args.force,
            )
    else:
        print("\n[Skipping Step 3] Using existing perplexity files.")

    # ------------------------------------------------------------------ #
    # Step 4: Probe
    # ------------------------------------------------------------------ #
    if _should_run("probe", args.from_step):
        print("\n=== Step 4: Training probes ===")
        all_results = train_all_probes(responses, config, force=args.force)
    else:
        print("\n[Skipping Step 4] Loading existing probe results.")
        all_results = load_probe_results_if_present(config)

    # ------------------------------------------------------------------ #
    # Step 5: Analysis
    # ------------------------------------------------------------------ #
    if _should_run("analysis", args.from_step) and all_results:
        print("\n=== Step 5: Generating figures and summary ===")
        generate_all_figures(all_results, config)
    elif not all_results:
        print("\n[Skipping Step 5] No probe results available.")

    print(f"\n=== Experiment 4 complete. ===")
    print(f"  Results in: {config.results_dir_ex4.resolve()}")


if __name__ == "__main__":
    main()
