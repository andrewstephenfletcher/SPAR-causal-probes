"""
Probe training and geometric analysis for Experiment 9.

Three analysis blocks, all CPU-only (works from saved activations):

  9.1  Truth probe comparison
       Train logistic-regression probes on Geometry of Truth data at layers
       [16, 24, 30], load the Experiment 1 prefill probe direction, and
       compute cosine similarities + cross-application AUROCs.

  9.2  Eval/deploy direction comparison
       Load the saved CAA vector and build a pairwise cosine-similarity matrix
       across all named directions plus random baselines.

  9.3  Per-model "person vector" analysis
       Re-train probes from Experiment 3 activation files (self vs. each cross-
       model condition) to extract the per-model probe directions.  Decompose
       them into a shared "not-self" PC1 and model-specific residuals.

All intermediate artefacts (directions, probe objects) are cached to disk so
that re-running without --force skips completed steps.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .config import Experiment3Config, Experiment9Config
from .probe import (
    apply_normaliser,
    build_dataset_for_layer,
    evaluate_probe,
    fit_normaliser,
    LinearProbe,
    train_probe,
)
from .probe_ex3 import build_dataset, load_all_activations, CROSS_CONDITIONS


# ---------------------------------------------------------------------------
# Utility: probe direction extraction (shared with steer_ex6.py approach)
# ---------------------------------------------------------------------------

@dataclass
class ProbeBundle:
    """Probe + normalisation + direction, the triple needed for cross-domain eval."""
    direction: np.ndarray     # unit-norm float32 in original activation space
    weight_norm: np.ndarray   # weight in the normalised space (probe.linear.weight[0])
    mean: np.ndarray          # training-set mean (float32)
    std: np.ndarray           # training-set std (float32)
    test_auroc: float
    test_acc: float
    best_wd: float
    n_train: int
    n_test: int
    probe: LinearProbe        # trained probe object (for cross-domain logits)


def _extract_direction(probe: LinearProbe, std: np.ndarray) -> np.ndarray:
    """Convert normalised-space probe weight to unit-norm original-space direction."""
    w = probe.linear.weight.data[0].numpy().astype(np.float32)
    w_orig = w / std.astype(np.float32)
    return w_orig / np.linalg.norm(w_orig)


def _train_and_bundle(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    wd_grid: list[float],
) -> Optional[ProbeBundle]:
    if (
        len(X_train) < 4 or len(X_val) < 2 or len(X_test) < 2
        or len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2
    ):
        return None

    probe, mean, std, best_wd, _ = train_probe(X_train, y_train, X_val, y_val, wd_grid)
    test_acc, test_auroc, _ = evaluate_probe(probe, mean, std, X_test, y_test)
    direction = _extract_direction(probe, std)

    return ProbeBundle(
        direction=direction,
        weight_norm=probe.linear.weight.data[0].numpy().astype(np.float32),
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        test_auroc=float(test_auroc),
        test_acc=float(test_acc),
        best_wd=float(best_wd),
        n_train=int(len(X_train)),
        n_test=int(len(X_test)),
        probe=probe,
    )


# ---------------------------------------------------------------------------
# 9.1 — Prefill probe direction (from Experiment 1)
# ---------------------------------------------------------------------------

def get_prefill_probe_direction(
    config: Experiment9Config,
    force: bool = False,
) -> ProbeBundle:
    """
    Re-train (or load cached) the Experiment 1 prefill probe at layer 30.
    Uses Llama 8B activations: self_prefill vs. cross_gemma_prefill.
    """
    cache_npy  = config.results_dir_ex9 / "prefill_direction_layer30.npy"
    cache_json = config.results_dir_ex9 / "prefill_direction_layer30.json"
    cache_mean = config.results_dir_ex9 / "prefill_mean_layer30.npy"
    cache_std  = config.results_dir_ex9 / "prefill_std_layer30.npy"
    cache_wt   = config.results_dir_ex9 / "prefill_weightnorm_layer30.npy"

    if cache_npy.exists() and not force:
        print("  Prefill probe direction: loading from cache.")
        direction   = np.load(cache_npy).astype(np.float32)
        mean        = np.load(cache_mean).astype(np.float32)
        std         = np.load(cache_std).astype(np.float32)
        weight_norm = np.load(cache_wt).astype(np.float32)
        with open(cache_json) as f:
            meta = json.load(f)

        # Rebuild probe from cached weight (for cross-domain evaluation)
        probe = LinearProbe(direction.shape[0])
        with torch.no_grad():
            probe.linear.weight.data[0] = torch.tensor(weight_norm)
            probe.linear.bias.data.fill_(0.0)

        return ProbeBundle(
            direction=direction,
            weight_norm=weight_norm,
            mean=mean, std=std,
            test_auroc=meta["test_auroc"],
            test_acc=meta["test_acc"],
            best_wd=meta["best_wd"],
            n_train=meta["n_train"],
            n_test=meta["n_test"],
            probe=probe,
        )

    self_path  = config.ex1_activations_dir / "self_prefill.pt"
    cross_path = config.ex1_activations_dir / "cross_gemma_prefill.pt"
    resp_path  = config.ex1_generations_dir / "responses.json"

    if not self_path.exists() or not cross_path.exists():
        raise FileNotFoundError(
            f"Experiment 1 activations not found at {config.ex1_activations_dir}. "
            "Run Experiment 1 first."
        )

    self_data  = torch.load(self_path,  weights_only=False)
    cross_data = torch.load(cross_path, weights_only=False)

    with open(resp_path) as f:
        responses = json.load(f)
    split_map = {r["prompt_id"]: r["split"] for r in responses}

    dataset = build_dataset_for_layer(self_data, cross_data, split_map, layer_idx=30)
    bundle = _train_and_bundle(
        dataset["train"]["X"], dataset["train"]["y"],
        dataset["val"]["X"],   dataset["val"]["y"],
        dataset["test"]["X"],  dataset["test"]["y"],
        config.probe_regularisation_grid,
    )
    if bundle is None:
        raise RuntimeError("Insufficient Experiment 1 data to train prefill probe at layer 30.")

    np.save(cache_npy,  bundle.direction)
    np.save(cache_mean, bundle.mean)
    np.save(cache_std,  bundle.std)
    np.save(cache_wt,   bundle.weight_norm)
    with open(cache_json, "w") as f:
        json.dump({
            "test_auroc": bundle.test_auroc,
            "test_acc":   bundle.test_acc,
            "best_wd":    bundle.best_wd,
            "n_train":    bundle.n_train,
            "n_test":     bundle.n_test,
        }, f, indent=2)

    print(f"  Prefill direction saved → {cache_npy}  "
          f"AUROC={bundle.test_auroc:.4f}")
    return bundle


# ---------------------------------------------------------------------------
# 9.1 — Truth probes (Geometry of Truth datasets)
# ---------------------------------------------------------------------------

def _build_truth_dataset_at_layer(
    records: list[dict],
    layer_idx: int,
    split: str,
) -> tuple[np.ndarray, np.ndarray]:
    X_list, y_list = [], []
    for rec in records:
        if rec.get("split") != split:
            continue
        act = rec["layer_activations"].get(layer_idx)
        if act is None:
            continue
        X_list.append(act.astype(np.float32))
        y_list.append(rec["label"])
    if not X_list:
        return np.empty((0, 4096), dtype=np.float32), np.array([], dtype=int)
    return np.stack(X_list), np.array(y_list, dtype=int)


def get_truth_probe_directions(
    all_truth_acts: dict[str, list[dict]],
    config: Experiment9Config,
    force: bool = False,
) -> dict[str, dict[int, ProbeBundle]]:
    """
    Train truth probes for each (dataset, layer) pair.

    Returns: {dataset_name: {layer_idx: ProbeBundle}}
    """
    results: dict[str, dict[int, ProbeBundle]] = {}

    for dataset_name, records in all_truth_acts.items():
        results[dataset_name] = {}
        for layer_idx in config.truth_layers:
            cache_npy  = config.results_dir_ex9 / f"truth_direction_{dataset_name}_layer{layer_idx}.npy"
            cache_json = config.results_dir_ex9 / f"truth_direction_{dataset_name}_layer{layer_idx}.json"
            cache_mean = config.results_dir_ex9 / f"truth_mean_{dataset_name}_layer{layer_idx}.npy"
            cache_std  = config.results_dir_ex9 / f"truth_std_{dataset_name}_layer{layer_idx}.npy"
            cache_wt   = config.results_dir_ex9 / f"truth_weightnorm_{dataset_name}_layer{layer_idx}.npy"

            if cache_npy.exists() and not force:
                direction   = np.load(cache_npy).astype(np.float32)
                mean        = np.load(cache_mean).astype(np.float32)
                std         = np.load(cache_std).astype(np.float32)
                weight_norm = np.load(cache_wt).astype(np.float32)
                with open(cache_json) as f:
                    meta = json.load(f)
                probe = LinearProbe(direction.shape[0])
                with torch.no_grad():
                    probe.linear.weight.data[0] = torch.tensor(weight_norm)
                    probe.linear.bias.data.fill_(0.0)
                results[dataset_name][layer_idx] = ProbeBundle(
                    direction=direction, weight_norm=weight_norm,
                    mean=mean, std=std,
                    test_auroc=meta["test_auroc"], test_acc=meta["test_acc"],
                    best_wd=meta["best_wd"], n_train=meta["n_train"],
                    n_test=meta["n_test"], probe=probe,
                )
                print(f"  Truth probe {dataset_name}/layer{layer_idx}: "
                      f"loaded (AUROC={meta['test_auroc']:.4f})")
                continue

            # Use 70% of records for train, remainder for an internal val (15% of train),
            # and the rest for test.  Since we only have train/test splits in the data,
            # carve out a val set from train.
            X_train_all, y_train_all = _build_truth_dataset_at_layer(records, layer_idx, "train")
            X_test,      y_test      = _build_truth_dataset_at_layer(records, layer_idx, "test")

            if len(X_train_all) < 6:
                print(f"  WARNING: not enough data for truth probe {dataset_name}/layer{layer_idx}")
                continue

            # Carve out last 20% of train as val
            n_val  = max(2, len(X_train_all) // 5)
            X_val,   y_val   = X_train_all[-n_val:], y_train_all[-n_val:]
            X_train, y_train = X_train_all[:-n_val], y_train_all[:-n_val]

            print(f"  Training truth probe {dataset_name}/layer{layer_idx} "
                  f"(n_train={len(X_train)}, n_test={len(X_test)})...")

            bundle = _train_and_bundle(
                X_train, y_train, X_val, y_val, X_test, y_test,
                config.probe_regularisation_grid,
            )
            if bundle is None:
                print(f"  WARNING: could not train truth probe {dataset_name}/layer{layer_idx}")
                continue

            np.save(cache_npy,  bundle.direction)
            np.save(cache_mean, bundle.mean)
            np.save(cache_std,  bundle.std)
            np.save(cache_wt,   bundle.weight_norm)
            with open(cache_json, "w") as f:
                json.dump({
                    "test_auroc": bundle.test_auroc, "test_acc": bundle.test_acc,
                    "best_wd": bundle.best_wd, "n_train": bundle.n_train,
                    "n_test": bundle.n_test, "layer": layer_idx, "dataset": dataset_name,
                }, f, indent=2)

            results[dataset_name][layer_idx] = bundle
            print(f"  Truth probe {dataset_name}/layer{layer_idx}: "
                  f"AUROC={bundle.test_auroc:.4f}")

    return results


# ---------------------------------------------------------------------------
# 9.3 — Per-model "person vector" directions (from Experiment 3)
# ---------------------------------------------------------------------------

def get_per_model_directions(
    config: Experiment9Config,
    force: bool = False,
) -> dict[str, ProbeBundle]:
    """
    Re-train probes from Experiment 3 activations (self vs. each cross-model
    condition) to extract per-model probe directions at layer 30.

    Returns: {"gemma": ProbeBundle, "mistral": ProbeBundle, ...}
    """
    ex3_config = Experiment3Config()

    # Check Ex3 activations exist
    self_path = ex3_config.activations_dir_ex3 / "activations_self.pt"
    if not self_path.exists():
        raise FileNotFoundError(
            f"Experiment 3 activations not found at {ex3_config.activations_dir_ex3}. "
            "Run Experiment 3 first."
        )

    all_acts = load_all_activations(ex3_config)

    # Load Ex3 responses to get correct split labels
    resp_path = ex3_config.generations_dir_ex3 / "responses_all.json"
    if resp_path.exists():
        with open(resp_path) as f:
            responses = json.load(f)
        override_map = {r["prompt_id"]: r for r in responses}
        for cond, acts in all_acts.items():
            for d in acts:
                r = override_map.get(d["prompt_id"])
                if r:
                    d["split"]   = r["split"]
                    d["dataset"] = r["dataset"]

    results: dict[str, ProbeBundle] = {}
    wd_grid = config.probe_regularisation_grid

    for cond in CROSS_CONDITIONS:
        cache_npy  = config.results_dir_ex9 / f"permodel_direction_{cond}.npy"
        cache_json = config.results_dir_ex9 / f"permodel_direction_{cond}.json"
        cache_mean = config.results_dir_ex9 / f"permodel_mean_{cond}.npy"
        cache_std  = config.results_dir_ex9 / f"permodel_std_{cond}.npy"
        cache_wt   = config.results_dir_ex9 / f"permodel_weightnorm_{cond}.npy"

        if cache_npy.exists() and not force:
            direction   = np.load(cache_npy).astype(np.float32)
            mean        = np.load(cache_mean).astype(np.float32)
            std         = np.load(cache_std).astype(np.float32)
            weight_norm = np.load(cache_wt).astype(np.float32)
            with open(cache_json) as f:
                meta = json.load(f)
            probe = LinearProbe(direction.shape[0])
            with torch.no_grad():
                probe.linear.weight.data[0] = torch.tensor(weight_norm)
                probe.linear.bias.data.fill_(0.0)
            results[cond] = ProbeBundle(
                direction=direction, weight_norm=weight_norm,
                mean=mean, std=std,
                test_auroc=meta["test_auroc"], test_acc=meta["test_acc"],
                best_wd=meta["best_wd"], n_train=meta["n_train"],
                n_test=meta["n_test"], probe=probe,
            )
            print(f"  Per-model direction '{cond}': loaded (AUROC={meta['test_auroc']:.4f})")
            continue

        if cond not in all_acts or not all_acts[cond]:
            print(f"  WARNING: no activations for condition '{cond}' in Experiment 3.")
            continue

        X_tr, y_tr = build_dataset(all_acts["self"], all_acts[cond], split="train")
        X_v,  y_v  = build_dataset(all_acts["self"], all_acts[cond], split="val")
        X_te, y_te = build_dataset(all_acts["self"], all_acts[cond], split="test")

        print(f"  Training per-model probe '{cond}' "
              f"(n_train={len(X_tr)}, n_test={len(X_te)})...")

        bundle = _train_and_bundle(X_tr, y_tr, X_v, y_v, X_te, y_te, wd_grid)
        if bundle is None:
            print(f"  WARNING: could not train per-model probe for '{cond}'.")
            continue

        np.save(cache_npy,  bundle.direction)
        np.save(cache_mean, bundle.mean)
        np.save(cache_std,  bundle.std)
        np.save(cache_wt,   bundle.weight_norm)
        with open(cache_json, "w") as f:
            json.dump({
                "test_auroc": bundle.test_auroc, "test_acc": bundle.test_acc,
                "best_wd": bundle.best_wd, "n_train": bundle.n_train,
                "n_test": bundle.n_test, "condition": cond,
            }, f, indent=2)

        results[cond] = bundle
        print(f"  Per-model direction '{cond}': AUROC={bundle.test_auroc:.4f}")

    return results


# ---------------------------------------------------------------------------
# Cross-application AUROC helpers
# ---------------------------------------------------------------------------

def cross_apply_auroc(
    probe: LinearProbe,
    mean: np.ndarray,
    std: np.ndarray,
    X_other: np.ndarray,
    y_other: np.ndarray,
) -> float:
    """
    Apply a probe trained on domain A to data from domain B.
    Uses domain A's normalisation statistics.
    """
    if len(X_other) < 2 or len(np.unique(y_other)) < 2:
        return float("nan")
    X_norm = apply_normaliser(X_other, mean, std)
    X_t = torch.tensor(X_norm, dtype=torch.float32)
    probe.eval()
    with torch.no_grad():
        logits = probe(X_t).cpu().numpy()
    return float(roc_auc_score(y_other, logits))


def direction_auroc(
    direction: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
) -> float:
    """AUROC using raw dot-product scores (for CAA vectors without a probe)."""
    if len(X) < 2 or len(np.unique(y)) < 2:
        return float("nan")
    scores = X @ direction.astype(np.float32)
    return float(roc_auc_score(y, scores))


# ---------------------------------------------------------------------------
# Cosine similarity utilities
# ---------------------------------------------------------------------------

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two arrays (unit-norm assumed; use dot product)."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def random_unit_vectors(hidden_dim: int, n: int, seed: int = 42) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    vecs = []
    for _ in range(n):
        v = rng.standard_normal(hidden_dim).astype(np.float32)
        vecs.append(v / np.linalg.norm(v))
    return vecs


# ---------------------------------------------------------------------------
# 9.3 — Person vector SVD analysis
# ---------------------------------------------------------------------------

def person_vector_analysis(
    per_model_bundles: dict[str, ProbeBundle],
) -> dict:
    """
    Decompose per-model probe directions into a shared 'not-self' PC1 and
    model-specific residuals.

    Returns a dict with:
      pairwise_cosines  — n×n matrix (symmetric)
      labels            — condition names in matrix order
      shared_direction  — PC1 (shape [hidden_dim])
      pc1_variance_frac — {cond: float} fraction of each direction's variance on PC1
      residuals         — {cond: unit-norm residual after removing PC1}
      residual_cosines  — n×n matrix of residual cosine similarities
      singular_values   — first 4 singular values (variance profile)
    """
    conds = list(per_model_bundles.keys())
    n = len(conds)
    if n == 0:
        return {}

    dirs = np.stack([per_model_bundles[c].direction for c in conds])  # (n, d)

    # Pairwise cosine matrix
    pairwise = np.array([
        [cosine_sim(dirs[i], dirs[j]) for j in range(n)]
        for i in range(n)
    ])

    # SVD to find shared PC1
    U, S, Vt = np.linalg.svd(dirs, full_matrices=False)
    shared_direction = Vt[0].astype(np.float32)  # first principal component

    # Variance explained by PC1 for each direction
    pc1_var_frac = {}
    for i, cond in enumerate(conds):
        proj = float(np.dot(dirs[i], shared_direction))
        pc1_var_frac[cond] = float(proj ** 2)  # fraction of unit-norm direction on PC1

    # Residuals (person vectors): project out shared direction, then unit-normalise
    residuals: dict[str, np.ndarray] = {}
    for i, cond in enumerate(conds):
        proj = np.dot(dirs[i], shared_direction)
        resid = dirs[i] - proj * shared_direction
        norm = np.linalg.norm(resid)
        if norm > 1e-8:
            residuals[cond] = (resid / norm).astype(np.float32)
        else:
            residuals[cond] = resid.astype(np.float32)

    # Residual cosine matrix
    resid_dirs = [residuals[c] for c in conds]
    resid_cosines = np.array([
        [cosine_sim(resid_dirs[i], resid_dirs[j]) for j in range(n)]
        for i in range(n)
    ])

    return {
        "labels": conds,
        "pairwise_cosines": pairwise.tolist(),
        "shared_direction": shared_direction,
        "singular_values": S.tolist(),
        "pc1_variance_frac": pc1_var_frac,
        "residuals": residuals,
        "residual_cosines": resid_cosines.tolist(),
    }


# ---------------------------------------------------------------------------
# Full analysis pipeline
# ---------------------------------------------------------------------------

def run_all_analyses(
    all_truth_acts: dict[str, list[dict]],
    prefill_bundle: ProbeBundle,
    truth_bundles: dict[str, dict[int, ProbeBundle]],
    eval_deploy_direction: np.ndarray,
    per_model_bundles: dict[str, ProbeBundle],
    config: Experiment9Config,
) -> dict:
    """
    Compute all similarity metrics and AUROC cross-application results.
    Returns the full results dict (serialisable to JSON except numpy arrays).
    """
    results: dict = {}
    layer = config.primary_layer
    wd_grid = config.probe_regularisation_grid

    # ------------------------------------------------------------------
    # 9.1 — Truth probe AUROCs (already computed, just report)
    # ------------------------------------------------------------------
    truth_aurocs: dict = {}
    for ds, layer_map in truth_bundles.items():
        truth_aurocs[ds] = {
            str(l): round(b.test_auroc, 5) for l, b in layer_map.items()
        }
    results["truth_probe_aurocs"] = truth_aurocs

    # ------------------------------------------------------------------
    # 9.1 — Cosine similarity: prefill vs. truth at layer 30
    # ------------------------------------------------------------------
    cosine_prefill_truth: dict = {}
    for ds in config.got_datasets:
        if ds in truth_bundles and layer in truth_bundles[ds]:
            sim = cosine_sim(prefill_bundle.direction, truth_bundles[ds][layer].direction)
            cosine_prefill_truth[ds] = round(sim, 5)
            print(f"  Cosine(prefill, truth_{ds}) at layer {layer}: {sim:+.4f}")
    results["cosine_prefill_truth"] = cosine_prefill_truth

    # ------------------------------------------------------------------
    # 9.1 — Random baseline cosines
    # ------------------------------------------------------------------
    hidden_dim = prefill_bundle.direction.shape[0]
    rand_vecs  = random_unit_vectors(hidden_dim, config.n_random_vectors, config.random_seed)
    rand_vs_prefill = [cosine_sim(v, prefill_bundle.direction) for v in rand_vecs]
    rand_vs_truth = {ds: [] for ds in config.got_datasets}
    for v in rand_vecs:
        for ds in config.got_datasets:
            if ds in truth_bundles and layer in truth_bundles[ds]:
                rand_vs_truth[ds].append(cosine_sim(v, truth_bundles[ds][layer].direction))

    results["random_baseline"] = {
        "n": config.n_random_vectors,
        "vs_prefill": {
            "mean": round(float(np.mean(rand_vs_prefill)), 5),
            "std":  round(float(np.std(rand_vs_prefill)),  5),
            "values": [round(x, 5) for x in rand_vs_prefill],
        },
        "vs_truth": {
            ds: {
                "mean": round(float(np.mean(vs)), 5),
                "std":  round(float(np.std(vs)),  5),
                "values": [round(x, 5) for x in vs],
            }
            for ds, vs in rand_vs_truth.items() if vs
        },
        "expected_magnitude_4096d": round(1.0 / np.sqrt(hidden_dim), 4),
    }
    print(f"  Random baseline vs prefill: mean={np.mean(rand_vs_prefill):+.4f} "
          f"(expected ~{1.0/np.sqrt(hidden_dim):.4f})")

    # ------------------------------------------------------------------
    # 9.1 — Cross-application AUROC
    # ------------------------------------------------------------------
    cross_auroc: dict = {}

    # Prefill probe on truth data (both datasets, at layer 30)
    for ds in config.got_datasets:
        if ds not in all_truth_acts:
            continue
        truth_X, truth_y = _build_test_acts_from_records(all_truth_acts[ds], layer)
        auroc = cross_apply_auroc(
            prefill_bundle.probe, prefill_bundle.mean, prefill_bundle.std,
            truth_X, truth_y,
        )
        cross_auroc[f"prefill_on_{ds}"] = round(auroc, 5)
        print(f"  Cross-app AUROC: prefill probe on {ds} truth data: {auroc:.4f}")

    # Truth probe on prefill data (test split of Ex1)
    prefill_test_X, prefill_test_y = _build_prefill_test_acts(config)
    for ds in config.got_datasets:
        if ds in truth_bundles and layer in truth_bundles[ds]:
            tb = truth_bundles[ds][layer]
            auroc = cross_apply_auroc(
                tb.probe, tb.mean, tb.std,
                prefill_test_X, prefill_test_y,
            )
            cross_auroc[f"{ds}_on_prefill"] = round(auroc, 5)
            print(f"  Cross-app AUROC: truth_{ds} probe on prefill data: {auroc:.4f}")

    # Eval/deploy direction (dot product) on prefill data
    auroc_ed = direction_auroc(eval_deploy_direction, prefill_test_X, prefill_test_y)
    cross_auroc["eval_deploy_on_prefill"] = round(auroc_ed, 5)
    print(f"  Cross-app AUROC: eval/deploy direction on prefill data: {auroc_ed:.4f}")

    results["cross_application_auroc"] = cross_auroc

    # ------------------------------------------------------------------
    # 9.2 — Full pairwise cosine similarity matrix
    # ------------------------------------------------------------------
    named_dirs: dict[str, np.ndarray] = {"prefill": prefill_bundle.direction}
    for ds in config.got_datasets:
        if ds in truth_bundles and layer in truth_bundles[ds]:
            named_dirs[f"truth_{ds}"] = truth_bundles[ds][layer].direction
    named_dirs["eval_deploy"] = eval_deploy_direction
    for cond, bundle in per_model_bundles.items():
        named_dirs[f"permodel_{cond}"] = bundle.direction

    dir_labels = list(named_dirs.keys())
    dir_vecs   = [named_dirs[k] for k in dir_labels]
    n_dirs = len(dir_labels)

    cosine_matrix = np.array([
        [cosine_sim(dir_vecs[i], dir_vecs[j]) for j in range(n_dirs)]
        for i in range(n_dirs)
    ])

    # Random baseline row/column: mean cosine with each named direction
    rand_vs_named = [
        [cosine_sim(v, d) for d in dir_vecs]
        for v in rand_vecs
    ]
    rand_mean_row = np.mean(rand_vs_named, axis=0).tolist()

    results["cosine_matrix"] = {
        "labels": dir_labels,
        "values": cosine_matrix.tolist(),
        "random_mean_row": [round(x, 5) for x in rand_mean_row],
    }

    # ------------------------------------------------------------------
    # 9.2 — Key pairwise cosines (summary table)
    # ------------------------------------------------------------------
    key_pairs = [
        ("prefill", "eval_deploy"),
    ]
    for ds in config.got_datasets:
        key_pairs += [
            (f"truth_{ds}", "eval_deploy"),
            ("prefill", f"truth_{ds}"),
        ]

    key_cosines = {}
    for a, b in key_pairs:
        if a in named_dirs and b in named_dirs:
            key_cosines[f"{a}_vs_{b}"] = round(cosine_sim(named_dirs[a], named_dirs[b]), 5)

    results["key_cosines"] = key_cosines
    print("\n  Key cosine similarities:")
    for k, v in key_cosines.items():
        print(f"    {k}: {v:+.4f}")

    # ------------------------------------------------------------------
    # 9.3 — Person vector analysis
    # ------------------------------------------------------------------
    if per_model_bundles:
        pva = person_vector_analysis(per_model_bundles)
        results["person_vector_analysis"] = {
            "labels": pva["labels"],
            "pairwise_cosines": pva["pairwise_cosines"],
            "singular_values": pva["singular_values"],
            "pc1_variance_frac": pva["pc1_variance_frac"],
            "residual_cosines": pva["residual_cosines"],
        }
        # Save shared direction as numpy
        np.save(config.results_dir_ex9 / "shared_notself_direction.npy",
                pva["shared_direction"])
        for cond, resid in pva["residuals"].items():
            np.save(config.results_dir_ex9 / f"person_vector_{cond}.npy", resid)

        print("\n  Person vector analysis:")
        print(f"    Singular values: {[round(s, 3) for s in pva['singular_values']]}")
        print(f"    PC1 variance fractions: "
              f"{ {k: round(v, 3) for k, v in pva['pc1_variance_frac'].items()} }")
    else:
        results["person_vector_analysis"] = {}
        print("  Person vector analysis: skipped (no per-model bundles).")

    return results


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _build_test_acts_from_records(
    records: list[dict],
    layer_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, y) from truth activation records for the test split."""
    X_list, y_list = [], []
    for rec in records:
        if rec.get("split") != "test":
            continue
        act = rec["layer_activations"].get(layer_idx)
        if act is None:
            continue
        X_list.append(act.astype(np.float32))
        y_list.append(rec["label"])
    if not X_list:
        return np.empty((0, 4096), dtype=np.float32), np.array([], dtype=int)
    return np.stack(X_list), np.array(y_list, dtype=int)


def _build_prefill_test_acts(
    config: Experiment9Config,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load Ex1 self+cross activations at layer 30 for the test split.
    Returns (X, y) with label 0=self, 1=cross-model.
    """
    self_data  = torch.load(config.ex1_activations_dir / "self_prefill.pt",  weights_only=False)
    cross_data = torch.load(config.ex1_activations_dir / "cross_gemma_prefill.pt", weights_only=False)

    with open(config.ex1_generations_dir / "responses.json") as f:
        responses = json.load(f)
    split_map = {r["prompt_id"]: r["split"] for r in responses}

    dataset = build_dataset_for_layer(self_data, cross_data, split_map, layer_idx=30)
    return dataset["test"]["X"], dataset["test"]["y"]
