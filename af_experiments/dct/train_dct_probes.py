#!/usr/bin/env python3
# Enable CPU fallback for MPS-unsupported ops (e.g. linalg_qr, linalg_svd).
# Must be set before importing torch.
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

"""
Train DCT features and save them as a probe collection.

Wraps the existing dct.py module (ExponentialDCT, SlicedModel,
DeltaActivations, SteeringCalibrator) to:
  1. Load a model
  2. Compute source/target activations on training prompt(s)
  3. Calibrate R
  4. Run ExponentialDCT.fit() to discover steering vectors
  5. Rank features by importance
  6. Save everything as a DCTProbeCollection

Usage (from af_experiments/dct/):
    python train_dct_probes.py \
        --model google/gemma-2-9b-it \
        --source_layer 9 \
        --target_layer 19 \
        --width 512 \
        --prompts "Tell me a story" \
        --output probes/gemma9b_s9_t19/

    # Multiple prompts from a file
    python train_dct_probes.py \
        --model google/gemma-2-9b-it \
        --source_layer 9 \
        --target_layer 19 \
        --width 512 \
        --prompt_file prompts.txt \
        --output probes/gemma9b_s9_t19/
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Import dct.py from the same directory ──
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from dct import (
    SlicedModel,
    DeltaActivations,
    SteeringCalibrator,
    ExponentialDCT,
)

# ── Import probe format from pipeline/ ──
from pipeline.probes.dct_probe import DCTProbe, DCTProbeCollection


def get_source_target_activations(
    model,
    tokenizer,
    prompts: list[str],
    source_layer: int,
    target_layer: int,
    sliced_model: SlicedModel,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run model on prompts and extract activations for DCT training.

    X is the residual stream entering the source layer (input to the slice).
    Y is the output of the sliced model on X (baseline target activations
    with no perturbation). This matches how DeltaActivations.forward()
    computes delta = sliced_model(x + theta) - y.
    """
    all_X = []
    all_Y = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states
        # hidden_states[source_layer] = residual stream entering layer s
        x = hidden_states[source_layer]  # (1, seq_len, d_model)

        # Y = sliced_model(X) with no perturbation — the baseline
        with torch.no_grad():
            y = sliced_model(x)  # (1, seq_len, d_model)

        all_X.append(x.detach())
        all_Y.append(y.detach())

    # Pad to same seq_len and stack
    max_len = max(x.shape[1] for x in all_X)
    d_model = all_X[0].shape[2]

    h_dtype = all_X[0].dtype
    X = torch.zeros(len(prompts), max_len, d_model, device=device, dtype=h_dtype)
    Y = torch.zeros(len(prompts), max_len, d_model, device=device, dtype=h_dtype)

    for i, (x, y) in enumerate(zip(all_X, all_Y)):
        seq_len = x.shape[1]
        X[i, :seq_len] = x[0]
        Y[i, :seq_len] = y[0]

    return X, Y


def main():
    parser = argparse.ArgumentParser(
        description="Train DCT features and save as probe collection"
    )

    # Model
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--device", default=None,
                        help="Device to use (cuda/mps/cpu). Auto-detected if not set.")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])

    # DCT architecture
    parser.add_argument("--source_layer", type=int, required=True, help="Source layer s")
    parser.add_argument("--target_layer", type=int, required=True, help="Target layer t")
    parser.add_argument("--width", type=int, default=512, help="Number of DCT features")
    parser.add_argument("--measure_positions", default="-3:",
                        help="Slice for measurement token positions (default: last 3)")

    # DCT training
    parser.add_argument("--iterations", type=int, default=10, help="SOGI iterations (tau)")
    parser.add_argument("--init", default="jacobian", choices=["random", "jacobian"])
    parser.add_argument("--d_proj", type=int, default=32, help="Jacobian projection dim")
    parser.add_argument("--beta", type=float, default=1.0, help="Update momentum")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--factor_batch_size", type=int, default=16)

    # R calibration
    parser.add_argument("--lambda_target", type=float, default=0.5)
    parser.add_argument("--R_override", type=float, default=None)

    # Training prompts
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompts", nargs="+")
    prompt_group.add_argument("--prompt_file")

    # Output
    parser.add_argument("--output", required=True, help="Output directory for probe collection")

    args = parser.parse_args()

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}

    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    print(f"Device:          {args.device}")

    # Parse measure_positions slice
    parts = args.measure_positions.split(":")
    start = int(parts[0]) if parts[0] else None
    stop = int(parts[1]) if len(parts) > 1 and parts[1] else None
    measure_slice = slice(start, stop)

    # Load prompts
    if args.prompts:
        prompts = args.prompts
    else:
        with open(args.prompt_file) as f:
            prompts = [line.strip() for line in f if line.strip()]

    print(f"{'='*60}")
    print(f"DCT Probe Training")
    print(f"{'='*60}")
    print(f"Model:           {args.model}")
    print(f"Layers:          {args.source_layer} -> {args.target_layer} "
          f"(depth {args.target_layer - args.source_layer})")
    print(f"Width:           {args.width}")
    print(f"Iterations:      {args.iterations}")
    print(f"Init:            {args.init}")
    print(f"Prompts ({len(prompts)}):")
    for p in prompts[:5]:
        print(f"  - {p[:80]}{'...' if len(p) > 80 else ''}")
    print()

    # ── Load model ──
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype_map[args.dtype], device_map=args.device,
    )
    model.eval()

    # ── Build DCT components from dct.py ──
    print("Building SlicedModel and DeltaActivations...")
    sliced_model = SlicedModel(model, args.source_layer, args.target_layer)
    delta_acts = DeltaActivations(sliced_model, target_position_indices=measure_slice)

    # ── Get source/target activations ──
    print("Extracting source/target activations on training prompts...")
    X, Y = get_source_target_activations(
        model, tokenizer, prompts,
        args.source_layer, args.target_layer,
        sliced_model=sliced_model,
        device=args.device,
    )
    print(f"  X shape: {X.shape}  (n_prompts, seq_len, d_model)")
    print(f"  Y shape: {Y.shape}")

    # ── Calibrate R ──
    if args.R_override is not None:
        R_cal = args.R_override
        print(f"\nUsing provided R = {R_cal:.4f}")
    else:
        print(f"\nCalibrating R (lambda_target={args.lambda_target})...")
        calibrator = SteeringCalibrator(target_ratio=args.lambda_target)
        R_cal = calibrator.calibrate(
            delta_acts, X, Y,
            batch_size=args.batch_size,
            factor_batch_size=args.factor_batch_size,
        )
        print(f"  R_cal = {R_cal:.4f}")

    # ── Train ExponentialDCT ──
    print(f"\nTraining ExponentialDCT (width={args.width}, "
          f"iters={args.iterations}, init={args.init})...")
    t0 = time.time()

    dct = ExponentialDCT(num_factors=args.width)
    U, V = dct.fit(
        delta_acts, X, Y,
        batch_size=args.batch_size,
        factor_batch_size=args.factor_batch_size,
        init=args.init,
        d_proj=args.d_proj,
        input_scale=R_cal,
        max_iters=args.iterations,
        beta=args.beta,
    )

    elapsed = time.time() - t0
    print(f"  Training complete in {elapsed:.1f}s")

    # ── Rank features ──
    print("\nRanking features by importance...")
    scores, indices = dct.rank(
        delta_acts, X, Y,
        batch_size=args.batch_size,
        factor_batch_size=args.factor_batch_size,
    )
    alphas = dct.alphas.detach().cpu()
    scores = scores.detach().cpu()
    indices = indices.detach().cpu()

    print(f"  Top 10 feature scores (alpha^2): {scores[:10].tolist()}")

    # U, V from .fit() are (d_model, num_factors) — columns are directions
    # Transpose to (num_factors, d_model) and reorder by importance
    V_all = V.detach().cpu().T
    U_all = U.detach().cpu().T

    V_sorted = V_all[indices]
    U_sorted = U_all[indices]
    alphas_sorted = alphas[indices]

    # ── Save ──
    print(f"\n{'='*60}")
    print("Saving probe collection...")
    print(f"{'='*60}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    probes = []
    for i in range(args.width):
        probes.append(DCTProbe(
            v=V_sorted[i], u=U_sorted[i],
            alpha=float(alphas_sorted[i]),
            source_layer=args.source_layer,
            target_layer=args.target_layer,
            model_name=args.model,
            feature_index=i,
            dct_width=args.width,
            dct_iterations=args.iterations,
            R_cal=R_cal,
            training_prompts=prompts,
            judge_label=None,
            judge_confidence=None,
            extra={
                "init": args.init, "beta": args.beta, "d_proj": args.d_proj,
                "lambda_target": args.lambda_target,
                "original_index": int(indices[i]),
                "score": float(scores[i]),
            },
        ))

    collection = DCTProbeCollection(
        probes=probes, model_name=args.model,
        source_layer=args.source_layer, target_layer=args.target_layer,
        dct_width=args.width, dct_iterations=args.iterations,
        R_cal=R_cal, training_prompts=prompts,
    )

    collection.save(output_dir)
    print(f"Saved {len(probes)} probes to {output_dir}")

    # Raw outputs for debugging
    torch.save({
        "V": V_sorted, "U": U_sorted, "alphas": alphas_sorted,
        "scores": scores, "indices": indices, "R_cal": R_cal,
        "objective_values": dct.objective_values,
        "training_prompts": prompts,
        "config": {
            "model": args.model, "source_layer": args.source_layer,
            "target_layer": args.target_layer, "width": args.width,
            "iterations": args.iterations, "init": args.init,
            "d_proj": args.d_proj, "beta": args.beta,
            "lambda_target": args.lambda_target,
            "measure_positions": args.measure_positions,
        },
    }, output_dir / "dct_raw_results.pt")

    # ── Summary ──
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"  Features trained:  {args.width}")
    print(f"  R_cal:             {R_cal:.4f}")
    print(f"  Top alpha:         {float(alphas_sorted[0]):.4f}")
    print(f"  Objective trace:   {dct.objective_values}")
    print()
    print("Next steps:")
    print(f"  1. Extract activations on LIARS' BENCH:")
    print(f"     python scripts/extract_activations.py \\")
    print(f"       --model {args.model} \\")
    print(f"       --layers {args.source_layer} {args.target_layer} \\")
    print(f"       --output_dir data/activations")
    print()
    print(f"  2. Evaluate probes:")
    print(f"     python scripts/evaluate_probe.py \\")
    print(f"       --probe_collection {output_dir} \\")
    print(f"       --activations_dir data/activations \\")
    print(f"       --token_strategy last_prompt mean_response \\")
    print(f"       --probed_layer source target")


if __name__ == "__main__":
    main()
