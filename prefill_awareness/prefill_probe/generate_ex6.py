"""
Generation loops for Experiment 6 (Steering Control Analysis).

A single run_conditions() function handles all four analyses with checkpointing
and resume support.  Helper functions build the condition lists for each analysis.

A "condition" is a dict with:
  condition_id  — unique string key for checkpointing
  label         — human-readable: "probe", "random_0", "baseline", etc.
  vector        — torch.Tensor (unit norm), ignored when alpha=0
  alpha         — float, actual perturbation magnitude added to residual stream
  layer_idx     — int, transformer layer index for the hook
  alpha_fraction — float, alpha / (layer_norm/100), stored for plotting
"""

from __future__ import annotations

import json

import numpy as np
import torch
from tqdm import tqdm

from .config import Experiment6Config
from .generate_5a import build_attribution_input, generate_steered, parse_attribution


# ---------------------------------------------------------------------------
# Prompt selection
# ---------------------------------------------------------------------------

def prepare_prompts(responses: list[dict], n: int) -> list[dict]:
    """Select up to n prompts: test split first, then val."""
    test = [r for r in responses if r["split"] == "test"]
    val  = [r for r in responses if r["split"] == "val"]
    pool = (test + val)[:n]
    print(f"  Using {len(pool)} prompts ({len(test)} test + {len(val)} val available).")
    return pool


# ---------------------------------------------------------------------------
# Core generation loop
# ---------------------------------------------------------------------------

def run_conditions(
    model,
    tokenizer,
    prompts: list[dict],
    conditions: list[dict],
    config: Experiment6Config,
    analysis_name: str,
    force: bool = False,
) -> list[dict]:
    """
    Run the attribution question for every (condition, prompt) pair.
    Saves results incrementally and supports resume from checkpoint.

    Returns a list of result records, one per (condition, prompt).
    """
    out_path     = config.generations_dir_ex6 / f"{analysis_name}.json"
    partial_path = config.generations_dir_ex6 / f"{analysis_name}_partial.json"

    if out_path.exists() and not force:
        with open(out_path) as f:
            results = json.load(f)
        print(f"  Loaded {analysis_name} ({len(results)} records) from {out_path}")
        return results

    results: list[dict] = []
    completed: set[tuple[int, str]] = set()

    if partial_path.exists() and not force:
        with open(partial_path) as f:
            results = json.load(f)
        completed = {(r["prompt_id"], r["condition_id"]) for r in results}
        print(f"  Resuming {analysis_name} from {len(completed)} completed generations.")

    total     = len(conditions) * len(prompts)
    remaining = total - len(completed)
    print(f"  {remaining} generations remaining ({total} total).")

    with tqdm(total=remaining, desc=analysis_name) as pbar:
        for cond in conditions:
            cid            = cond["condition_id"]
            vec            = cond["vector"]
            alpha          = cond["alpha"]
            layer_idx      = cond["layer_idx"]
            label          = cond["label"]
            alpha_fraction = cond.get("alpha_fraction", None)

            for r in prompts:
                pid = r["prompt_id"]
                if (pid, cid) in completed:
                    continue

                input_ids = build_attribution_input(
                    tokenizer, r["instruction"], r["response_llama70b"]
                )
                raw    = generate_steered(
                    model, tokenizer, input_ids, vec,
                    alpha=alpha, layer_idx=layer_idx,
                    max_new_tokens=config.max_new_tokens, seed=config.seed,
                )
                parsed = parse_attribution(raw)

                rec: dict = {
                    "condition_id":   cid,
                    "label":          label,
                    "layer_idx":      layer_idx,
                    "alpha":          alpha,
                    "prompt_id":      pid,
                    "raw_response":   raw,
                    "parsed":         parsed,
                }
                if alpha_fraction is not None:
                    rec["alpha_fraction"] = alpha_fraction

                results.append(rec)
                completed.add((pid, cid))
                pbar.update(1)

                if len(results) % config.checkpoint_interval == 0:
                    with open(partial_path, "w") as f:
                        json.dump(results, f)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    if partial_path.exists():
        partial_path.unlink()

    print(f"  Saved {len(results)} records → {out_path}")
    return results


# ---------------------------------------------------------------------------
# Condition builders — one per analysis
# ---------------------------------------------------------------------------

def make_conditions_analysis_a(
    probe_vec: torch.Tensor,
    random_vecs: list[torch.Tensor],
    layer_norm_24: float,
    alpha_frac: float = 1.5,
) -> list[dict]:
    """
    Analysis A — 1 probe + n random vectors + 1 baseline, all at layer 24.
    Alpha is the Experiment-5 moderate magnitude (alpha_frac × layer_norm_24 / 100).
    """
    layer_idx = 24
    alpha = alpha_frac * layer_norm_24 / 100.0

    conditions: list[dict] = [
        {
            "condition_id":   "baseline",
            "label":          "baseline",
            "vector":         probe_vec,   # alpha=0 => vector irrelevant
            "alpha":          0.0,
            "layer_idx":      layer_idx,
            "alpha_fraction": 0.0,
        },
        {
            "condition_id":   f"probe_a{alpha_frac:.2f}",
            "label":          "probe",
            "vector":         probe_vec,
            "alpha":          alpha,
            "layer_idx":      layer_idx,
            "alpha_fraction": alpha_frac,
        },
    ]

    for i, rvec in enumerate(random_vecs):
        conditions.append({
            "condition_id":   f"random_{i}_a{alpha_frac:.2f}",
            "label":          f"random_{i}",
            "vector":         rvec,
            "alpha":          alpha,
            "layer_idx":      layer_idx,
            "alpha_fraction": alpha_frac,
        })

    return conditions


def make_conditions_analysis_b(
    probe_vec: torch.Tensor,
    random_vec_0: torch.Tensor,
    layer_norm_24: float,
    alpha_fracs: list[float],
) -> list[dict]:
    """
    Analysis B — alpha sweep (positive + negative) for probe and one random vector
    at layer 24, plus a shared baseline.

    Produces (2 × len(alpha_fracs) × 2 vectors) + 1 baseline = 21 conditions
    for default 5-element alpha_fracs list.
    """
    layer_idx = 24
    scale = layer_norm_24 / 100.0

    conditions: list[dict] = [
        {
            "condition_id":   "baseline",
            "label":          "baseline",
            "vector":         probe_vec,
            "alpha":          0.0,
            "layer_idx":      layer_idx,
            "alpha_fraction": 0.0,
        }
    ]

    for frac in alpha_fracs:
        alpha = frac * scale
        for sign, sign_str in [(1, "pos"), (-1, "neg")]:
            conditions.extend([
                {
                    "condition_id":   f"probe_{sign_str}_a{frac:.2f}",
                    "label":          f"probe_{sign_str}",
                    "vector":         probe_vec,
                    "alpha":          sign * alpha,
                    "layer_idx":      layer_idx,
                    "alpha_fraction": sign * frac,
                },
                {
                    "condition_id":   f"random_{sign_str}_a{frac:.2f}",
                    "label":          f"random_{sign_str}",
                    "vector":         random_vec_0,
                    "alpha":          sign * alpha,
                    "layer_idx":      layer_idx,
                    "alpha_fraction": sign * frac,
                },
            ])

    return conditions


def make_conditions_analysis_c(
    probe_vecs: dict[int, np.ndarray],
    random_vec_0: torch.Tensor,
    layer_norms: dict[int, float],
    steering_layers: list[int],
    alpha_frac: float = 1.0,
) -> list[dict]:
    """
    Analysis C — probe + random at each of 8 layers, scaled to each layer's norm.
    One shared baseline (alpha=0) avoids running 8 identical forward passes.
    """
    conditions: list[dict] = [
        {
            "condition_id":   "baseline",
            "label":          "baseline",
            "vector":         random_vec_0,   # alpha=0 => vector irrelevant
            "alpha":          0.0,
            "layer_idx":      steering_layers[0],
            "alpha_fraction": 0.0,
        }
    ]

    for layer in steering_layers:
        alpha = alpha_frac * layer_norms[layer] / 100.0
        probe_t = torch.tensor(probe_vecs[layer], dtype=torch.float16)

        conditions.append({
            "condition_id":   f"probe_l{layer}",
            "label":          "probe",
            "vector":         probe_t,
            "alpha":          alpha,
            "layer_idx":      layer,
            "alpha_fraction": alpha_frac,
        })
        conditions.append({
            "condition_id":   f"random_l{layer}",
            "label":          "random",
            "vector":         random_vec_0,
            "alpha":          alpha,
            "layer_idx":      layer,
            "alpha_fraction": alpha_frac,
        })

    return conditions


def make_conditions_analysis_d(
    probe_vec: torch.Tensor,
    random_vec_0: torch.Tensor,
    layer_norm: float,
    best_layer: int,
    alpha_fracs_random: list[float],
    alpha_frac_probe: float = 1.0,
) -> list[dict]:
    """
    Analysis D — probe at alpha_frac=1.0 vs random vector at escalating magnitudes
    at best_layer (determined from Analysis C).
    """
    scale = layer_norm / 100.0

    conditions: list[dict] = [
        {
            "condition_id":   "baseline",
            "label":          "baseline",
            "vector":         probe_vec,
            "alpha":          0.0,
            "layer_idx":      best_layer,
            "alpha_fraction": 0.0,
        },
        {
            "condition_id":   f"probe_a{alpha_frac_probe:.2f}",
            "label":          "probe",
            "vector":         probe_vec,
            "alpha":          alpha_frac_probe * scale,
            "layer_idx":      best_layer,
            "alpha_fraction": alpha_frac_probe,
        },
    ]

    for frac in alpha_fracs_random:
        conditions.append({
            "condition_id":   f"random_a{frac:.2f}",
            "label":          f"random_a{frac:.2f}",
            "vector":         random_vec_0,
            "alpha":          frac * scale,
            "layer_idx":      best_layer,
            "alpha_fraction": frac,
        })

    return conditions
