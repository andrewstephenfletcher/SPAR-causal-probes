#!/usr/bin/env python3
"""
Convert DCT training outputs to the probe format used by this pipeline.

Bridges between your DCT experiment notebook outputs and the
standardized DCTProbe / DCTProbeCollection format.

Usage:
    cd af_experiments/dct
    python scripts/convert_dct_to_probes.py \
        --dct_output path/to/dct_results/ \
        --model google/gemma-2-9b-it \
        --source_layer 9 \
        --target_layer 19 \
        --output probes/dct_run_001/
"""

import argparse
import json
import sys
from pathlib import Path

import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_DCT_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_DCT_DIR))

from pipeline.probes.dct_probe import DCTProbe, DCTProbeCollection


def load_dct_outputs(dct_output_dir: Path) -> dict:
    """Load raw DCT training outputs.

    *** ADAPT THIS FUNCTION TO MATCH YOUR NOTEBOOK'S OUTPUT FORMAT ***
    """
    dct_output_dir = Path(dct_output_dir)

    # Option 1: dct_raw_results.pt from train_dct_probes.py
    for name in ["dct_raw_results.pt", "dct_results.pt"]:
        pt_file = dct_output_dir / name
        if pt_file.exists():
            return torch.load(pt_file, map_location="cpu", weights_only=False)

    # Option 2: Separate files
    V = torch.load(dct_output_dir / "V.pt", map_location="cpu")
    U = torch.load(dct_output_dir / "U.pt", map_location="cpu")
    alphas = torch.load(dct_output_dir / "alphas.pt", map_location="cpu")

    meta = {}
    meta_file = dct_output_dir / "metadata.json"
    if meta_file.exists():
        with open(meta_file) as f:
            meta = json.load(f)

    return {
        "V": V, "U": U, "alphas": alphas,
        "R_cal": meta.get("R_cal", 1.0),
        "training_prompts": meta.get("training_prompts", []),
        "dct_iterations": meta.get("dct_iterations", 10),
    }


def main():
    parser = argparse.ArgumentParser(description="Convert DCT outputs to probe format")
    parser.add_argument("--dct_output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source_layer", type=int, required=True)
    parser.add_argument("--target_layer", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--judge_labels", default=None, help="JSON file with LLM judge labels")
    parser.add_argument("--normalize", action="store_true", default=True)
    args = parser.parse_args()

    print(f"Loading DCT outputs from: {args.dct_output}")
    dct_data = load_dct_outputs(Path(args.dct_output))

    V = dct_data["V"]
    U = dct_data["U"]
    alphas = dct_data["alphas"]
    n_features = V.shape[0]

    print(f"Found {n_features} features, d_model={V.shape[1]}")

    if args.normalize:
        V = V / V.norm(dim=1, keepdim=True).clamp(min=1e-8)
        U = U / U.norm(dim=1, keepdim=True).clamp(min=1e-8)

    judge_labels = {}
    if args.judge_labels:
        with open(args.judge_labels) as f:
            raw = json.load(f)
        judge_labels = {int(k): v for k, v in raw.items()}
        print(f"Loaded judge labels for {len(judge_labels)} features")

    probes = []
    for i in range(n_features):
        jl = judge_labels.get(i, {})
        probes.append(DCTProbe(
            v=V[i], u=U[i], alpha=float(alphas[i]),
            source_layer=args.source_layer, target_layer=args.target_layer,
            model_name=args.model, feature_index=i,
            dct_width=n_features,
            dct_iterations=dct_data.get("dct_iterations", 10),
            R_cal=dct_data.get("R_cal", 1.0),
            training_prompts=dct_data.get("training_prompts", []),
            judge_label=jl.get("label"),
            judge_confidence=jl.get("confidence"),
        ))

    collection = DCTProbeCollection(
        probes=probes, model_name=args.model,
        source_layer=args.source_layer, target_layer=args.target_layer,
        dct_width=n_features,
        dct_iterations=dct_data.get("dct_iterations", 10),
        R_cal=dct_data.get("R_cal", 1.0),
        training_prompts=dct_data.get("training_prompts", []),
    )

    collection.save(Path(args.output))
    print(f"Saved {n_features} probes to {args.output}")


if __name__ == "__main__":
    main()
