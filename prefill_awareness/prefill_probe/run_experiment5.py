"""
Experiment 5: Causal Steering — end-to-end orchestration.

Run from the prefill_awareness/ directory:

    # Full run (all steps):
    python -m prefill_probe.run_experiment5

    # Resume from a specific step:
    python -m prefill_probe.run_experiment5 --from-step 5a
    python -m prefill_probe.run_experiment5 --from-step analysis

    # Skip GPU steps (run analysis/judging locally after downloading results):
    python -m prefill_probe.run_experiment5 --from-step judge
    python -m prefill_probe.run_experiment5 --from-step analysis

    # Force re-run a step:
    python -m prefill_probe.run_experiment5 --from-step calibrate --force

Steps:
  1. direction   — retrain probe at steering_layer to extract direction vector
  2. calibrate   — measure residual-stream norm, propose alpha values
  3. sweep       — quick 10-prompt layer sweep to confirm steering_layer choice
  4. 5a          — attribution steering (900 generations)
  5. 5b          — sentiment steering evaluation (300 generations)
  6. judge       — LLM judge for 5B responses (OpenRouter API)
  7. analysis    — generate all four figures and summary tables

Prerequisites:
  - Experiment 4 must have been run; activations must exist at
    outputs/experiment4/activations/llama70b/{self_prefill,cross_gemma9b,cross_llama8b}.pt
  - HF_TOKEN set in environment for model downloads.
  - OPENROUTER_API_KEY set for the judge step.
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_here = Path(__file__).resolve().parent.parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from prefill_probe.analysis_ex5 import generate_all_figures
from prefill_probe.config import Experiment5Config
from prefill_probe.generate_5a import (
    run_experiment_5a,
    run_layer_sweep,
    run_random_vector_control,
)
from prefill_probe.generate_5b import run_experiment_5b
from prefill_probe.judge_5b import run_all_judging
from prefill_probe.steer_ex5 import (
    calibrate_alpha,
    extract_probe_direction,
    random_steering_direction,
)
from prefill_probe.utils import clear_device_cache, get_device, get_device_map

_STEP_ORDER = ["direction", "calibrate", "sweep", "5a", "5b", "judge", "analysis"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 5: Causal Steering")
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run the current step, overwriting existing outputs.",
    )
    parser.add_argument(
        "--from-step",
        choices=_STEP_ORDER,
        default=None,
        metavar="STEP",
        help=(
            f"Skip steps before STEP (earlier outputs must exist). "
            f"One of: {', '.join(_STEP_ORDER)}"
        ),
    )
    parser.add_argument(
        "--skip-5b", action="store_true",
        help="Run only 5A (attribution steering), skip 5B sentiment experiment.",
    )
    parser.add_argument(
        "--skip-sweep", action="store_true",
        help="Skip the layer sweep (use config.steering_layer directly).",
    )
    return parser.parse_args()


def _should_run(step: str, from_step: str | None) -> bool:
    if from_step is None:
        return True
    return _STEP_ORDER.index(step) >= _STEP_ORDER.index(from_step)


def _load_responses(config: Experiment5Config) -> list[dict]:
    resp_path = config.ex4_generations_dir / "responses.json"
    if not resp_path.exists():
        raise FileNotFoundError(
            f"Experiment 4 responses not found at {resp_path}. "
            "Run Experiment 4 first."
        )
    with open(resp_path) as f:
        return json.load(f)


def _load_model(config: Experiment5Config):
    device_str = get_device()
    device_map = get_device_map()
    print(f"  Loading {config.llama70b_model_id} (fp16) on {device_str}...")

    if device_map is not None:
        model = AutoModelForCausalLM.from_pretrained(
            config.llama70b_model_id,
            torch_dtype=torch.float16,
            device_map=device_map,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.llama70b_model_id,
            torch_dtype=torch.float16,
        ).to(device_str)

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(config.llama70b_model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def _unload_model(model) -> None:
    del model
    gc.collect()
    clear_device_cache()


def main() -> None:
    args = parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    config = Experiment5Config()

    print("\nExperiment 5: Causal Steering")
    print(f"  Device: {get_device()}")
    print(f"  Output directory: {config.output_dir_ex5.resolve()}")
    print(f"  Steering layer: {config.steering_layer}")
    print(f"  Steering condition: {config.steering_condition}")

    # Log environment
    import platform, transformers
    env = {
        "device": get_device(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "python_version": platform.python_version(),
    }
    if torch.cuda.is_available():
        env["gpu_count"] = torch.cuda.device_count()
        env["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(env["gpu_count"])]
    with open(config.output_dir_ex5 / "environment.json", "w") as f:
        json.dump(env, f, indent=2)

    responses = _load_responses(config)
    print(f"  Loaded {len(responses)} prompts from Experiment 4.")

    # ------------------------------------------------------------------ #
    # Step 1: Direction extraction (CPU, uses saved activations)
    # ------------------------------------------------------------------ #
    if _should_run("direction", args.from_step):
        print("\n=== Step 1: Extracting probe direction ===")
        steering_vec_np = extract_probe_direction(
            responses, config,
            layer_idx=config.steering_layer,
            condition=config.steering_condition,
            force=args.force,
        )
    else:
        direction_path = (
            config.results_dir_ex5
            / f"probe_direction_layer{config.steering_layer}_{config.steering_condition}.npy"
        )
        print(f"\n[Skipping Step 1] Loading direction from {direction_path}")
        steering_vec_np = np.load(direction_path).astype(np.float32)

    steering_vec = torch.tensor(steering_vec_np, dtype=torch.float32)
    print(f"  Steering vector: shape={steering_vec.shape}, norm={steering_vec.norm():.4f}")

    # ------------------------------------------------------------------ #
    # Step 2: Alpha calibration (requires model)
    # ------------------------------------------------------------------ #
    if _should_run("calibrate", args.from_step):
        print("\n=== Step 2: Alpha calibration ===")
        model, tokenizer = _load_model(config)
        calibrate_alpha(
            model, tokenizer, config.llama70b_model_id,
            responses, config, force=args.force,
        )
        _unload_model(model)
    else:
        print("\n[Skipping Step 2] Using existing calibration.")

    # Load alphas for downstream steps
    if not config.load_alphas_from_calibration():
        raise RuntimeError(
            "Alpha calibration not found. Run with --from-step calibrate."
        )
    print(f"  Alphas: conservative={config.alpha_conservative:.4f}, "
          f"moderate={config.alpha_moderate:.4f}, "
          f"aggressive={config.alpha_aggressive:.4f}")

    # ------------------------------------------------------------------ #
    # Step 3: Layer sweep (requires model, 10 prompts)
    # ------------------------------------------------------------------ #
    if _should_run("sweep", args.from_step) and not args.skip_sweep:
        print("\n=== Step 3: Layer sweep ===")
        model, tokenizer = _load_model(config)
        sweep_results = run_layer_sweep(
            model, tokenizer, responses, steering_vec, config, force=args.force,
        )
        _unload_model(model)

        best_layer = max(
            sweep_results, key=lambda k: sweep_results[k]["not_me_rate"]
        )
        print(f"  Best layer from sweep: {best_layer} "
              f"(not_me_rate={sweep_results[best_layer]['not_me_rate']:.2%})")
        print(f"  Using configured steering_layer={config.steering_layer}.")
    elif args.skip_sweep:
        print("\n[Skipping Step 3] Layer sweep skipped (--skip-sweep).")
    else:
        print("\n[Skipping Step 3] Using existing layer sweep.")

    # ------------------------------------------------------------------ #
    # Step 4: Experiment 5A (requires model, ~1.5 hours)
    # ------------------------------------------------------------------ #
    if _should_run("5a", args.from_step):
        print("\n=== Step 4: Experiment 5A — Attribution Steering ===")
        model, tokenizer = _load_model(config)

        # Random-vector control first (fast, 10 prompts × 3 alphas = 30 gens)
        print("  Running random-vector control...")
        random_vec = torch.tensor(
            random_steering_direction(steering_vec_np.shape[0]), dtype=torch.float32
        )
        run_random_vector_control(
            model, tokenizer, responses, random_vec, config, force=args.force,
        )

        # Main 5A experiment
        results_5a = run_experiment_5a(
            model, tokenizer, responses, steering_vec, config, force=args.force,
        )
        _unload_model(model)
        print(f"  5A complete: {len(results_5a)} attribution records.")
    else:
        print("\n[Skipping Step 4] Loading existing 5A results.")
        att_path = config.generations_dir_ex5 / "attribution_5a.json"
        with open(att_path) as f:
            results_5a = json.load(f)

    # ------------------------------------------------------------------ #
    # Step 5: Experiment 5B (requires model, ~2 hours)
    # ------------------------------------------------------------------ #
    results_5b: list[dict] = []
    if _should_run("5b", args.from_step) and not args.skip_5b:
        print("\n=== Step 5: Experiment 5B — Sentiment Steering ===")
        model, tokenizer = _load_model(config)
        results_5b = run_experiment_5b(
            model, tokenizer, responses, steering_vec, config, force=args.force,
        )
        _unload_model(model)
        print(f"  5B complete: {len(results_5b)} evaluation records.")
    elif args.skip_5b:
        print("\n[Skipping Step 5] --skip-5b flag set.")
    else:
        print("\n[Skipping Step 5] Loading existing 5B results.")
        sent_path = config.generations_dir_ex5 / "sentiment_5b.json"
        if sent_path.exists():
            with open(sent_path) as f:
                results_5b = json.load(f)

    # ------------------------------------------------------------------ #
    # Step 6: LLM judging of 5B (API calls, no GPU needed)
    # ------------------------------------------------------------------ #
    judged_5b: list[dict] = []
    if _should_run("judge", args.from_step) and results_5b and not args.skip_5b:
        print("\n=== Step 6: LLM judging for 5B ===")
        judged_5b = run_all_judging(results_5b, config, force=args.force)
    else:
        judged_path = config.results_dir_ex5 / "sentiment_judged_5b.json"
        if judged_path.exists():
            with open(judged_path) as f:
                judged_5b = json.load(f)
        if not args.skip_5b:
            print("\n[Skipping Step 6] Using existing judge results.")

    # ------------------------------------------------------------------ #
    # Step 7: Analysis (CPU)
    # ------------------------------------------------------------------ #
    if _should_run("analysis", args.from_step):
        print("\n=== Step 7: Generating figures and summary tables ===")
        generate_all_figures(results_5a, judged_5b, config)
    else:
        print("\n[Skipping Step 7] Analysis skipped.")

    print(f"\n=== Experiment 5 complete. ===")
    print(f"  Results in: {config.results_dir_ex5.resolve()}")


if __name__ == "__main__":
    main()
