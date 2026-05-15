"""
Experiment 10: Probe Generalisation — end-to-end orchestration.

Run from the prefill_awareness/ directory:

    # Full run (requires large-model VRAM):
    python -m prefill_probe.run_experiment10

    # Skip to a specific step (earlier outputs must already exist):
    python -m prefill_probe.run_experiment10 --from-step extract
    python -m prefill_probe.run_experiment10 --from-step probe
    python -m prefill_probe.run_experiment10 --from-step analysis

    # Re-run everything:
    python -m prefill_probe.run_experiment10 --force

Steps:
  1. generate  — generate responses from Llama 70B, Llama 8B, Gemma 31B,
                 Mistral 24B for BigCodeBench / OASST1 / GPQA
  2. extract   — extract Llama 70B residual-stream activations at layer 60
                 for all (dataset, condition) pairs
  3. probe     — train 9 probes; evaluate in 9×9 transfer matrix
  4. analysis  — three heatmap figures + summary table

Prerequisites:
  - HF_TOKEN set and model licences accepted on HuggingFace
  - GPQA requires explicit access grant at huggingface.co/datasets/Idavidrein/gpqa
"""

import argparse
import json
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent.parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from prefill_probe.analysis_ex10 import generate_all_figures
from prefill_probe.config import Experiment10Config
from prefill_probe.extract_ex10 import extract_all_activations, load_all_activations
from prefill_probe.generate_ex10 import generate_all_responses, load_all_dataset_responses
from prefill_probe.probe_ex10 import train_and_evaluate
from prefill_probe.utils import get_device

_STEP_ORDER = ["generate", "extract", "probe", "analysis"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 10: Probe Generalisation")
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
    return parser.parse_args()


def _should_run(step: str, from_step: str | None) -> bool:
    if from_step is None:
        return True
    return _STEP_ORDER.index(step) >= _STEP_ORDER.index(from_step)


def main() -> None:
    args = parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    config = Experiment10Config()

    print(f"\nExperiment 10: Probe Generalisation")
    print(f"  Device:           {get_device()}")
    print(f"  Target model:     {config.target_model_id}")
    print(f"  Probe layer:      {config.probe_layer}")
    print(f"  Datasets:         {config.datasets}")
    print(f"  Cross sources:    {config.cross_sources}")
    print(f"  Prompts/dataset:  {config.n_prompts_per_dataset}")
    print(f"  Max new tokens:   {config.max_new_tokens}")
    print(f"  Output directory: {config.output_dir.resolve()}")

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
    env_path = config.output_dir / "environment.json"
    with open(env_path, "w") as f:
        json.dump(env, f, indent=2)

    # ------------------------------------------------------------------ #
    # Step 1: Generate
    # ------------------------------------------------------------------ #
    if _should_run("generate", args.from_step):
        print("\n=== Step 1: Generating responses ===")
        all_responses = generate_all_responses(config, force=args.force)
    else:
        print("\n[Skipping Step 1] Loading existing responses...")
        all_responses = load_all_dataset_responses(config)

    for ds, recs in all_responses.items():
        print(f"  {ds}: {len(recs)} prompts")

    # ------------------------------------------------------------------ #
    # Step 2: Extract
    # ------------------------------------------------------------------ #
    if _should_run("extract", args.from_step):
        print("\n=== Step 2: Extracting activations ===")
        extract_all_activations(all_responses, config)
    else:
        print("\n[Skipping Step 2] Using existing activation files.")

    # ------------------------------------------------------------------ #
    # Step 3: Probe
    # ------------------------------------------------------------------ #
    if _should_run("probe", args.from_step):
        print("\n=== Step 3: Training probes and building transfer matrix ===")
        all_activations = load_all_activations(config)
        matrix_data = train_and_evaluate(all_activations, config, force=args.force)
    else:
        print("\n[Skipping Step 3] Loading existing transfer matrix.")
        matrix_path = config.results_dir / "transfer_matrix.json"
        with open(matrix_path) as f:
            matrix_data = json.load(f)

    # ------------------------------------------------------------------ #
    # Step 4: Analysis
    # ------------------------------------------------------------------ #
    if _should_run("analysis", args.from_step):
        print("\n=== Step 4: Generating figures and summary ===")
        generate_all_figures(matrix_data, config)

    print(f"\n=== Experiment 10 complete. ===")
    print(f"  Results in: {config.results_dir.resolve()}")
    print(f"  Figures in: {config.figures_dir.resolve()}")


if __name__ == "__main__":
    main()
