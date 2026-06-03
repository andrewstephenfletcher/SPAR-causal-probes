"""
Experiment 0k: Iterative Comparative Rewrite.

For each variant × task, runs an iterative loop:
  Iter 0  : Raw Gemini response from 0f (baseline score also from 0f).
  Iter 1+ : Gemini rewrites addressing Opus's specific criticism, with Opus's
            reference response visible. Opus re-scores. Repeat up to max_iterations.

Each variant constrains what Gemini may change:
  style_only  — formatting, prose, variable names, comments only
  logic_only  — algorithm and code implementation only
  both        — unconstrained

Output: results_dir/{variant}_trajectories.json
  List of per-task trajectory dicts, each containing an 'iterations' list.
"""

import json
import time
from pathlib import Path

from tqdm import tqdm

from .config import Experiment0kConfig
from .helpers import CostTracker, call_gemini, detect_one, make_client


def load_tasks(config: Experiment0kConfig) -> list[dict]:
    """Top-N Gemini BCB tasks from 0f with both responses attached."""
    det_path = config.experiment0f_dir / "results" / "detection_results.json"
    gemini_path = config.experiment0f_dir / "generations" / "responses_gemini_pro.json"
    opus_path = config.experiment0f_dir / "generations" / "responses_opus_45.json"

    detections = json.loads(det_path.read_text())
    gemini_responses = json.loads(gemini_path.read_text())
    opus_responses = json.loads(opus_path.read_text())

    candidates = [
        r for r in detections
        if r.get("source") == "gemini_pro"
        and r.get("dataset") == "bigcodebench"
        and r.get("tamper_prob") is not None
        and r["task_id"] in gemini_responses
        and r["task_id"] in opus_responses
    ]
    top_n = sorted(candidates, key=lambda r: r["tamper_prob"], reverse=True)[: config.n_tasks]

    prompt_map = _load_prompt_map()
    return [
        {
            "task_id":              r["task_id"],
            "prompt":               prompt_map.get(r["task_id"], f"[Task: {r['task_id']}]"),
            "gemini_response":      gemini_responses[r["task_id"]],
            "opus_response":        opus_responses[r["task_id"]],
            "baseline_tamper_prob": r["tamper_prob"],
            "baseline_reason":      r.get("reason", ""),
        }
        for r in top_n
    ]


def _load_prompt_map() -> dict[str, str]:
    from experiment0g.benchmarks import load_tasks as _load
    all_tasks, _ = _load(100, 0)
    return {t["task_id"]: t["prompt"] for t in all_tasks}


def run_iterative(
    tasks: list[dict],
    config: Experiment0kConfig,
    cost_tracker: CostTracker,
    force: bool = False,
) -> dict[str, list[dict]]:
    """
    Returns {variant_name: [trajectory_dict, ...]} for all variants.
    Each trajectory has: task_id, iterations, converged, final_score, n_iters.
    """
    client = make_client(config)
    all_trajectories: dict[str, list[dict]] = {}

    for variant_name, constraint in config.variants.items():
        out_path = config.results_dir / f"{variant_name}_trajectories.json"

        existing: dict[str, dict] = {}
        if out_path.exists() and not force:
            saved = json.loads(out_path.read_text())
            existing = {t["task_id"]: t for t in saved}
            print(f"  [{variant_name}] {len(existing)} trajectories cached.")

        trajectories = []
        with tqdm(total=len(tasks), desc=variant_name) as pbar:
            for task in tasks:
                if cost_tracker.exceeded():
                    print("\n  Cost cap reached.")
                    break
                tid = task["task_id"]
                if tid in existing:
                    trajectories.append(existing[tid])
                    pbar.update(1)
                    continue

                traj = _run_task_loop(task, variant_name, constraint, config, client, cost_tracker)
                trajectories.append(traj)
                _save(trajectories, out_path)
                pbar.update(1)

        _save(trajectories, out_path)
        all_trajectories[variant_name] = trajectories

        n_conv = sum(1 for t in trajectories if t["converged"])
        print(f"  [{variant_name}] {n_conv}/{len(trajectories)} converged  {cost_tracker.report()}")

    return all_trajectories


def _run_task_loop(
    task: dict,
    variant_name: str,
    constraint: str,
    config: Experiment0kConfig,
    client,
    cost_tracker: CostTracker,
) -> dict:
    tid = task["task_id"]

    # Iter 0: raw Gemini response + 0f baseline score (no API call)
    iterations = [
        {
            "iter":        0,
            "response":    task["gemini_response"],
            "tamper_prob": task["baseline_tamper_prob"],
            "reason":      task["baseline_reason"],
            "source":      "0f",
        }
    ]

    for i in range(1, config.max_iterations + 1):
        if cost_tracker.exceeded():
            break

        prev = iterations[-1]
        if prev["tamper_prob"] is not None and prev["tamper_prob"] < config.convergence_threshold:
            break

        rewrite_prompt = config.iterative_prompt_template.format(
            opus_reason=prev["reason"] or "(no reason given)",
            constraint=constraint,
            task_prompt=task["prompt"],
            previous_response=prev["response"],
            opus_response=task["opus_response"],
        )

        rewrite = call_gemini(rewrite_prompt, config, client, cost_tracker)
        time.sleep(config.get_delay(config.gemini_id))

        if rewrite is None:
            print(f"  [{variant_name}/{tid}] iter {i}: rewrite failed, stopping.")
            break

        score, reason = detect_one(task["prompt"], rewrite, config, client, cost_tracker)
        time.sleep(config.get_delay(config.evaluator_id))

        iterations.append({
            "iter":        i,
            "response":    rewrite,
            "tamper_prob": score,
            "reason":      reason,
        })

        # Plateau check
        if (
            prev["tamper_prob"] is not None
            and score is not None
            and abs(score - prev["tamper_prob"]) < config.plateau_delta
            and score >= config.convergence_threshold
        ):
            break

    final_score = iterations[-1]["tamper_prob"]
    converged = final_score is not None and final_score < config.convergence_threshold
    conv_iter = next(
        (it["iter"] for it in iterations
         if (it["tamper_prob"] or 999) < config.convergence_threshold),
        None,
    )

    return {
        "task_id":          tid,
        "variant":          variant_name,
        "iterations":       iterations,
        "converged":        converged,
        "convergence_iter": conv_iter,
        "final_score":      final_score,
        "n_iters":          len(iterations),
    }


def _save(trajectories: list[dict], path: Path) -> None:
    path.write_text(json.dumps(trajectories, indent=2))
