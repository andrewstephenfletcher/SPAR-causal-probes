"""
Experiment 6: Steering Control Analysis — end-to-end orchestration.

Run from the prefill_awareness/ directory:

    # Full run:
    python -m prefill_probe.run_experiment6

    # Resume from a specific step:
    python -m prefill_probe.run_experiment6 --from-step analysis_a
    python -m prefill_probe.run_experiment6 --from-step analysis

    # Force re-run a step:
    python -m prefill_probe.run_experiment6 --from-step norms --force

Steps:
  1. directions  — extract probe direction at all 8 steering layers
  2. norms       — compute per-layer mean residual-stream norm
  3. analysis_a  — 10 random vectors at layer 24 (1 100-prompt × 12 conditions)
  4. analysis_b  — alpha sweep for probe vs. random at layer 24 (21 × 100)
  5. analysis_c  — layer sweep at all 8 layers (17 × 100)
  6. analysis_d  — magnitude sweep at best layer from C (6 × 100)
  7. analysis    — generate all five figures, summary table, checklist

Prerequisites:
  - Experiment 4 activations at outputs/experiment4/activations/llama70b/
  - Experiment 4 responses at outputs/experiment4/generations/responses.json
  - HF_TOKEN set for model downloads.
"""

from __future__ import annotations

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

from prefill_probe.analysis_ex6 import (
    compute_analysis_c,
    find_best_layer,
    generate_all_figures,
)
from prefill_probe.config import Experiment6Config
from prefill_probe.generate_ex6 import (
    make_conditions_analysis_a,
    make_conditions_analysis_b,
    make_conditions_analysis_c,
    make_conditions_analysis_d,
    prepare_prompts,
    run_conditions,
)
from prefill_probe.steer_ex6 import (
    compute_layer_norms,
    extract_probe_directions,
    generate_random_steering_vectors,
    load_layer_norms,
    save_layer_norms,
)
from prefill_probe.utils import clear_device_cache, get_device, get_device_map

_STEP_ORDER = [
    "directions", "norms", "analysis_a", "analysis_b", "analysis_c", "analysis_d", "analysis"
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 6: Steering Control Analysis")
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
        "--best-layer", type=int, default=None,
        help="Override best layer for Analysis D (default: determined from Analysis C).",
    )
    return parser.parse_args()


def _should_run(step: str, from_step: str | None) -> bool:
    if from_step is None:
        return True
    return _STEP_ORDER.index(step) >= _STEP_ORDER.index(from_step)


def _load_responses(config: Experiment6Config) -> list[dict]:
    resp_path = config.ex4_generations_dir / "responses.json"
    if not resp_path.exists():
        raise FileNotFoundError(
            f"Experiment 4 responses not found at {resp_path}. Run Experiment 4 first."
        )
    with open(resp_path) as f:
        return json.load(f)


def _load_model(config: Experiment6Config):
    device_str = get_device()
    device_map = get_device_map()
    print(f"  Loading {config.model_id} (fp16) on {device_str}...")

    if device_map is not None:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_id, torch_dtype=torch.float16, device_map=device_map
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_id, torch_dtype=torch.float16
        ).to(device_str)

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def _unload_model(model) -> None:
    del model
    gc.collect()
    clear_device_cache()


def _load_probe_directions(config: Experiment6Config) -> dict[int, np.ndarray]:
    """Load all probe directions from cache (must exist)."""
    directions: dict[int, np.ndarray] = {}
    for layer in config.steering_layers:
        p = config.results_dir_ex6 / f"probe_direction_layer{layer}_{config.probe_condition}.npy"
        if not p.exists():
            raise FileNotFoundError(
                f"Probe direction missing at {p}. Run with --from-step directions."
            )
        directions[layer] = np.load(p).astype(np.float32)
    return directions


def main() -> None:
    args = parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    config = Experiment6Config()

    print("\nExperiment 6: Steering Control Analysis")
    print(f"  Device:           {get_device()}")
    print(f"  Output directory: {config.output_dir_ex6.resolve()}")
    print(f"  Steering layers:  {config.steering_layers}")
    print(f"  Alpha fractions:  {config.alpha_fractions}")
    print(f"  Random vectors:   {config.n_random_vectors}")
    print(f"  Prompts:          {config.n_prompts}")

    # Log environment
    import platform, transformers as _tf
    env = {
        "device": get_device(),
        "torch_version": torch.__version__,
        "transformers_version": _tf.__version__,
        "python_version": platform.python_version(),
    }
    if torch.cuda.is_available():
        env["gpu_count"] = torch.cuda.device_count()
        env["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(env["gpu_count"])]
    with open(config.output_dir_ex6 / "environment.json", "w") as f:
        json.dump(env, f, indent=2)

    responses = _load_responses(config)
    print(f"  Loaded {len(responses)} responses from Experiment 4.")

    prompts = prepare_prompts(responses, config.n_prompts)

    # ------------------------------------------------------------------ #
    # Step 1: Probe directions at all 8 layers  (CPU, uses saved Ex4 acts)
    # ------------------------------------------------------------------ #
    if _should_run("directions", args.from_step):
        print("\n=== Step 1: Extracting probe directions at all steering layers ===")
        probe_dirs = extract_probe_directions(responses, config, force=args.force)
    else:
        print("\n[Skipping Step 1] Loading cached probe directions...")
        probe_dirs = _load_probe_directions(config)

    print(f"  Probe directions ready: {list(probe_dirs.keys())}")
    probe_vec_24 = torch.tensor(probe_dirs[24], dtype=torch.float16)

    # ------------------------------------------------------------------ #
    # Step 2: Per-layer residual norms  (requires model)
    # ------------------------------------------------------------------ #
    if _should_run("norms", args.from_step):
        existing = load_layer_norms(config)
        if existing is not None and not args.force:
            print(f"\n[Skipping Step 2] Loaded layer norms from cache.")
            layer_norms = existing
        else:
            print("\n=== Step 2: Computing per-layer residual norms ===")
            model, tokenizer = _load_model(config)
            layer_norms = compute_layer_norms(
                model, tokenizer, config.model_id,
                responses, config.steering_layers, config.n_prompts_norm,
            )
            save_layer_norms(layer_norms, config)
            _unload_model(model)
    else:
        print("\n[Skipping Step 2] Loading layer norms from cache...")
        layer_norms = load_layer_norms(config)
        if layer_norms is None:
            raise RuntimeError("Layer norms not found. Run with --from-step norms.")

    print(f"  Layer norms: { {k: round(v, 1) for k, v in layer_norms.items()} }")

    # Generate random vectors (seeded, same across all analyses)
    hidden_dim   = probe_dirs[24].shape[0]
    random_vecs  = generate_random_steering_vectors(
        hidden_dim, config.n_random_vectors, seed=config.random_seed
    )
    random_vec_0 = random_vecs[0]  # first vector, used as the single random for B/C/D

    # Determine the alpha for Analysis A (largest fraction, same magnitude as Exp 5 moderate)
    exp5_alpha = config.load_exp5_alpha()
    if exp5_alpha is not None:
        alpha_frac_a = exp5_alpha / (layer_norms[24] / 100.0)
        print(f"  Analysis A alpha: loaded from Exp5 calibration "
              f"(moderate={exp5_alpha:.4f} → fraction={alpha_frac_a:.2f}×)")
    else:
        alpha_frac_a = config.alpha_fractions[-1]  # 1.5× as specified
        print(f"  Analysis A alpha: Exp5 calibration not found, using {alpha_frac_a:.2f}×")

    # ------------------------------------------------------------------ #
    # Step 3: Analysis A — 10 random vectors at layer 24
    # ------------------------------------------------------------------ #
    if _should_run("analysis_a", args.from_step):
        print("\n=== Step 3: Analysis A — multiple random vectors at layer 24 ===")
        model, tokenizer = _load_model(config)
        conditions_a = make_conditions_analysis_a(
            probe_vec_24, random_vecs, layer_norms[24], alpha_frac=alpha_frac_a,
        )
        results_a = run_conditions(
            model, tokenizer, prompts, conditions_a, config,
            analysis_name="analysis_a", force=args.force,
        )
        _unload_model(model)
    else:
        print("\n[Skipping Step 3] Loading Analysis A results...")
        a_path = config.generations_dir_ex6 / "analysis_a.json"
        with open(a_path) as f:
            results_a = json.load(f)

    # ------------------------------------------------------------------ #
    # Step 4: Analysis B — alpha sweep at layer 24
    # ------------------------------------------------------------------ #
    if _should_run("analysis_b", args.from_step):
        print("\n=== Step 4: Analysis B — alpha sweep at layer 24 ===")
        model, tokenizer = _load_model(config)
        conditions_b = make_conditions_analysis_b(
            probe_vec_24, random_vec_0, layer_norms[24], config.alpha_fractions,
        )
        results_b = run_conditions(
            model, tokenizer, prompts, conditions_b, config,
            analysis_name="analysis_b", force=args.force,
        )
        _unload_model(model)
    else:
        print("\n[Skipping Step 4] Loading Analysis B results...")
        b_path = config.generations_dir_ex6 / "analysis_b.json"
        with open(b_path) as f:
            results_b = json.load(f)

    # ------------------------------------------------------------------ #
    # Step 5: Analysis C — layer sweep across 8 layers
    # ------------------------------------------------------------------ #
    if _should_run("analysis_c", args.from_step):
        print("\n=== Step 5: Analysis C — layer sweep ===")
        model, tokenizer = _load_model(config)
        conditions_c = make_conditions_analysis_c(
            probe_dirs, random_vec_0, layer_norms, config.steering_layers,
            alpha_frac=1.0,
        )
        results_c = run_conditions(
            model, tokenizer, prompts, conditions_c, config,
            analysis_name="analysis_c", force=args.force,
        )
        _unload_model(model)
    else:
        print("\n[Skipping Step 5] Loading Analysis C results...")
        c_path = config.generations_dir_ex6 / "analysis_c.json"
        with open(c_path) as f:
            results_c = json.load(f)

    # Determine the best layer for Analysis D
    if args.best_layer is not None:
        best_layer = args.best_layer
        print(f"\n  Best layer (user override): {best_layer}")
    else:
        c_df       = compute_analysis_c(results_c, config.steering_layers)
        best_layer = find_best_layer(c_df)
        print(f"\n  Best layer (max specificity from Analysis C): {best_layer}")

    # Save best_layer for the analysis step
    with open(config.results_dir_ex6 / "best_layer.json", "w") as f:
        json.dump({"best_layer": best_layer}, f)

    # ------------------------------------------------------------------ #
    # Step 6: Analysis D — magnitude sweep at best layer
    # ------------------------------------------------------------------ #
    if _should_run("analysis_d", args.from_step):
        print(f"\n=== Step 6: Analysis D — magnitude sweep at layer {best_layer} ===")
        model, tokenizer = _load_model(config)
        probe_vec_best = torch.tensor(probe_dirs[best_layer], dtype=torch.float16)
        conditions_d = make_conditions_analysis_d(
            probe_vec_best, random_vec_0,
            layer_norms[best_layer], best_layer,
            config.alpha_fractions_d, alpha_frac_probe=1.0,
        )
        results_d = run_conditions(
            model, tokenizer, prompts, conditions_d, config,
            analysis_name="analysis_d", force=args.force,
        )
        _unload_model(model)
    else:
        print("\n[Skipping Step 6] Loading Analysis D results...")
        d_path = config.generations_dir_ex6 / "analysis_d.json"
        with open(d_path) as f:
            results_d = json.load(f)

    # ------------------------------------------------------------------ #
    # Step 7: Analysis — figures, tables, checklist
    # ------------------------------------------------------------------ #
    if _should_run("analysis", args.from_step):
        print("\n=== Step 7: Generating figures and summary tables ===")

        # Load best_layer if we skipped analysis_c/analysis_d steps
        bl_path = config.results_dir_ex6 / "best_layer.json"
        if bl_path.exists():
            with open(bl_path) as f:
                best_layer = json.load(f)["best_layer"]

        # Load Exp5 probe effect for Figure 5 comparison, if available
        exp5_probe_effect: float | None = None
        exp5_summary = config.ex5_results_dir / "summary_5a_effects.csv"
        if exp5_summary.exists():
            try:
                import pandas as pd
                effects_df = pd.read_csv(exp5_summary)
                self_row = effects_df[effects_df["prefill_source"] == "self"]
                if len(self_row) > 0:
                    exp5_probe_effect = float(self_row["effect"].iloc[0])
                    print(f"  Exp5 probe effect (self prefill): {exp5_probe_effect:+.3f}")
            except Exception:
                pass

        generate_all_figures(
            results_a, results_b, results_c, results_d, config,
            best_layer=best_layer, exp5_probe_effect=exp5_probe_effect,
        )
    else:
        print("\n[Skipping Step 7] Analysis skipped.")

    print(f"\n=== Experiment 6 complete. ===")
    print(f"  Results in: {config.results_dir_ex6.resolve()}")


if __name__ == "__main__":
    main()
