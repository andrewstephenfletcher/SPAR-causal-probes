#!/usr/bin/env python3
"""
Evaluate a DCT probe on LIARS' BENCH using cached activations.

Usage (from repo root):
    python -m af_experiments.dct.scripts.evaluate_probe \
        --probe_collection af_experiments/dct/probes/gemma9b_s9_t19/ \
        --activations_dir af_experiments/dct/data/activations \
        --token_strategy last_prompt mean_response \
        --probed_layer source target

Or directly:
    cd af_experiments/dct
    python scripts/evaluate_probe.py \
        --probe_collection probes/gemma9b_s9_t19/ \
        --activations_dir data/activations \
        --token_strategy last_prompt \
        --probed_layer source
"""

import argparse
import sys
from pathlib import Path
from itertools import product

import numpy as np
import torch
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_DCT_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_DCT_DIR))

from pipeline.probes.dct_probe import DCTProbe, DCTProbeCollection
from pipeline.data.activation_store import ActivationStore
from pipeline.eval.token_strategies import TokenStrategy, select_tokens
from pipeline.eval.metrics import compute_metrics, compute_metrics_per_dataset


def evaluate_single_probe(
    probe, store, token_strategy, probed_layer,
    control_dataset="alpaca", target_fpr=0.01, datasets=None,
):
    """Evaluate one probe across all available datasets."""
    layer_idx = probe.source_layer if probed_layer == "source" else probe.target_layer
    direction = probe.v if probed_layer == "source" else probe.u

    model_name = probe.model_name
    available = store.list_datasets(model_name)

    if datasets:
        eval_datasets = [d for d in datasets if d in available]
    else:
        eval_datasets = [d for d in available if d != control_dataset]

    # Threshold calibration on control data
    control_scores = None
    if control_dataset in available:
        hidden_states, meta = store.load(model_name, control_dataset, layers=[layer_idx])
        states = hidden_states[layer_idx]

        scores_list = []
        for i, ex_meta in enumerate(meta):
            h = states[i, : ex_meta["seq_length"]]
            selected = select_tokens(h, token_strategy, ex_meta["prompt_length"], direction=direction)
            score = float(selected @ direction.to(selected.dtype))
            scores_list.append(score)
        control_scores = np.array(scores_list)

    # Evaluate per dataset
    results = {}
    all_scores = []
    all_labels = []
    per_example = []

    for dataset_name in tqdm(eval_datasets, desc="Evaluating datasets"):
        try:
            hidden_states, meta = store.load(model_name, dataset_name, layers=[layer_idx])
        except (KeyError, FileNotFoundError):
            print(f"  Skipping {dataset_name}: activations not found")
            continue

        states = hidden_states[layer_idx]
        ds_scores = []
        ds_labels = []

        for i, ex_meta in enumerate(meta):
            h = states[i, : ex_meta["seq_length"]]
            selected = select_tokens(h, token_strategy, ex_meta["prompt_length"], direction=direction)
            score = float(selected @ direction.to(selected.dtype))

            ds_scores.append(score)
            ds_labels.append(ex_meta["label"])

            per_example.append({
                "example_id": ex_meta.get("example_id", str(i)),
                "dataset": dataset_name,
                "label": ex_meta["label"],
                "score": score,
                "probed_layer": probed_layer,
                "token_strategy": token_strategy,
            })

        results[dataset_name] = {"labels": np.array(ds_labels), "scores": np.array(ds_scores)}
        all_scores.extend(ds_scores)
        all_labels.extend(ds_labels)

    metrics = compute_metrics_per_dataset(results, control_scores=control_scores, target_fpr=target_fpr)

    return {
        "metrics": metrics,
        "all_scores": np.array(all_scores),
        "all_labels": np.array(all_labels),
        "per_example": per_example,
        "control_scores": control_scores,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate DCT probes on LIARS' BENCH")

    probe_group = parser.add_mutually_exclusive_group(required=True)
    probe_group.add_argument("--probe", help="Path to a single .pt probe file")
    probe_group.add_argument("--probe_collection", help="Path to a probe collection directory")

    parser.add_argument("--filter_label", default=None)
    parser.add_argument("--max_probes", type=int, default=None)
    parser.add_argument("--activations_dir", required=True)
    parser.add_argument("--token_strategy", nargs="+", default=["last_prompt"],
                        choices=[s.value for s in TokenStrategy])
    parser.add_argument("--probed_layer", nargs="+", default=["source"],
                        choices=["source", "target"])
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--control_dataset", default="alpaca")
    parser.add_argument("--target_fpr", type=float, default=0.01)

    parser.add_argument("--wandb_project", default="dct-probe-liars-bench")
    parser.add_argument("--wandb_tags", nargs="*", default=[])
    parser.add_argument("--wandb_notes", default=None)
    parser.add_argument("--no_wandb", action="store_true")

    args = parser.parse_args()

    store = ActivationStore(args.activations_dir)

    if args.probe:
        probes = [DCTProbe.load(args.probe)]
    else:
        collection = DCTProbeCollection.load(args.probe_collection)
        if args.filter_label:
            probes = collection.filter_by_label(args.filter_label)
            print(f"Filtered to {len(probes)} probes with label '{args.filter_label}'")
        else:
            probes = collection.probes
        if args.max_probes:
            probes = probes[: args.max_probes]

    print(f"Evaluating {len(probes)} probe(s)")
    print(f"Token strategies: {args.token_strategy}")
    print(f"Probed layers: {args.probed_layer}")

    # Collect all results, then log to a single W&B run.
    summary_rows = []   # one row per (probe, strategy, layer, dataset)
    all_per_example = []
    best = {"auroc": -1.0, "result": None, "label": ""}

    for probe in probes:
        for token_strategy, probed_layer in product(args.token_strategy, args.probed_layer):
            config_label = f"feat{probe.feature_index}_{probed_layer}_{token_strategy}"
            print(f"\n{'='*60}")
            print(f"Run: {config_label}")
            print(f"  Probe: feature {probe.feature_index}, "
                  f"alpha={probe.alpha:.4f}, judge={probe.judge_label}")
            print(f"  Strategy: {token_strategy} @ {probed_layer} layer")

            result = evaluate_single_probe(
                probe=probe, store=store,
                token_strategy=token_strategy, probed_layer=probed_layer,
                control_dataset=args.control_dataset, target_fpr=args.target_fpr,
                datasets=args.datasets,
            )

            for ds_name, m in result["metrics"].items():
                print(f"  {ds_name:>12}: bal_acc={m.balanced_accuracy:.3f}  "
                      f"auroc={m.auroc:.3f}  recall@1%fpr={m.recall_at_1pct_fpr:.3f}  "
                      f"(n_lie={m.n_lies}, n_honest={m.n_honest})")
                summary_rows.append({
                    "feature_index": probe.feature_index,
                    "alpha": round(probe.alpha, 4),
                    "judge_label": probe.judge_label or "",
                    "token_strategy": token_strategy,
                    "probed_layer": probed_layer,
                    "dataset": ds_name,
                    "bal_acc": round(m.balanced_accuracy, 4),
                    "auroc": round(m.auroc, 4),
                    "recall_at_1pct_fpr": round(m.recall_at_1pct_fpr, 4),
                    "n_lie": m.n_lies,
                    "n_honest": m.n_honest,
                })
                # Track best result (by auroc, excluding aggregate row)
                if ds_name != "average" and m.auroc > best["auroc"] and len(result["all_scores"]) > 0:
                    best = {"auroc": m.auroc, "result": result, "label": config_label}

            all_per_example.extend(result["per_example"])

    if not args.no_wandb:
        from pipeline.logging_utils.wandb_logger import (
            init_wandb_run, log_summary_table, log_score_distributions,
            log_roc_curve, log_per_example_table,
        )

        probe_collection_name = Path(args.probe_collection).name if args.probe_collection else "single"
        run = init_wandb_run(
            config={
                "probe_collection": probe_collection_name,
                "n_probes": len(probes),
                "token_strategies": args.token_strategy,
                "probed_layers": args.probed_layer,
                "control_dataset": args.control_dataset,
                "target_fpr": args.target_fpr,
            },
            project=args.wandb_project,
            name=probe_collection_name,
            tags=args.wandb_tags,
            notes=args.wandb_notes,
        )
        log_summary_table(run, summary_rows)
        log_per_example_table(run, all_per_example)
        if best["result"] is not None:
            r = best["result"]
            log_score_distributions(run, r["all_labels"], r["all_scores"], name=best["label"])
            log_roc_curve(run, r["all_labels"], r["all_scores"], name=best["label"])
        run.finish()
        print(f"\nW&B run complete.")

    print("\nAll evaluations complete!")


if __name__ == "__main__":
    main()
