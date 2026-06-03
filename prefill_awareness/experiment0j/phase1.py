"""
Phase 1: Iterative Refinement Loop for Experiment 0j.

Takes the 20 Gemini Pro BigCodeBench tasks where Opus 4.5 gave the highest
tamper_prob in experiment 0f. For each, runs up to 5 refinement iterations:

  Iter 0  : Raw Gemini response from 0f (with existing detection score/reason).
  Iter 1+: Gemini rewrites addressing Opus's specific criticism; Opus re-scores.

Stops early if tamper_prob < convergence_threshold or score change < 5 points.

Output: results_dir/phase1_trajectories.json
"""

import json
import time
from pathlib import Path

from tqdm import tqdm

from .config import Experiment0jConfig
from .run_detection import (
    CostTracker,
    call_gemini_rewrite,
    detect_one,
    make_client,
)


def load_phase1_tasks(config: Experiment0jConfig) -> tuple[list[dict], list[dict]]:
    """
    Returns (tasks, baseline_records) for the top-N Gemini tasks from 0f.

    tasks          — list of {task_id, prompt} dicts (prompts injected from 0f generations)
    baseline_records — list of iter-0 records: {task_id, response, tamper_prob, reason}
    """
    det_path = config.experiment0f_dir / "results" / "detection_results.json"
    gen_path  = config.experiment0f_dir / "generations" / "responses_gemini_pro.json"

    detections = json.loads(det_path.read_text())
    gemini_responses = json.loads(gen_path.read_text())

    bcb_gemini = [
        r for r in detections
        if r["source"] == "gemini_pro"
        and r.get("dataset") == "bigcodebench"
        and r.get("tamper_prob") is not None
        and r["task_id"] in gemini_responses
    ]
    top_n = sorted(bcb_gemini, key=lambda r: r["tamper_prob"], reverse=True)[
        : config.n_phase1_tasks
    ]

    # Load dataset once for all prompt lookups
    prompt_map = _load_prompt_map()

    tasks = []
    baseline_records = []
    for r in top_n:
        tid = r["task_id"]
        tasks.append({"task_id": tid, "prompt": prompt_map.get(tid, f"[Task: {tid}]")})
        baseline_records.append({
            "task_id":    tid,
            "response":   gemini_responses[tid],
            "tamper_prob": r["tamper_prob"],
            "reason":     r.get("reason", ""),
        })

    return tasks, baseline_records


def _load_prompt_map() -> dict[str, str]:
    """Load BigCodeBench prompts for all 100 test tasks (same shuffle as 0f/0g)."""
    from experiment0g.benchmarks import load_tasks
    all_tasks, _ = load_tasks(100, 0)
    return {t["task_id"]: t["prompt"] for t in all_tasks}


def run_phase1(
    tasks: list[dict],
    baseline_records: list[dict],
    config: Experiment0jConfig,
    cost_tracker: CostTracker,
    force: bool = False,
) -> list[dict]:
    out_path = config.results_dir / "phase1_trajectories.json"

    existing: dict[str, dict] = {}
    if out_path.exists() and not force:
        saved = json.loads(out_path.read_text())
        existing = {t["task_id"]: t for t in saved}
        print(f"  [phase1] {len(existing)} trajectories cached.")

    baseline_map = {r["task_id"]: r for r in baseline_records}
    trajectories = []

    client = make_client(config)

    with tqdm(total=len(tasks), desc="Phase 1") as pbar:
        for task in tasks:
            if cost_tracker.exceeded():
                print("\n  Cost cap reached.")
                break
            tid = task["task_id"]
            if tid in existing:
                trajectories.append(existing[tid])
                pbar.update(1)
                continue

            bl = baseline_map[tid]
            traj = _run_task_loop(task, bl, config, client, cost_tracker)
            trajectories.append(traj)
            _save(trajectories, out_path)
            pbar.update(1)

    _save(trajectories, out_path)
    return trajectories


def _run_task_loop(
    task: dict,
    baseline: dict,
    config: Experiment0jConfig,
    client,
    cost_tracker: CostTracker,
) -> dict:
    tid = task["task_id"]
    iterations = [
        {
            "iter":       0,
            "response":   baseline["response"],
            "tamper_prob": baseline["tamper_prob"],
            "reason":     baseline["reason"],
            "source":     "0f",
        }
    ]

    for i in range(1, config.max_iterations + 1):
        if cost_tracker.exceeded():
            break

        prev = iterations[-1]

        if prev["tamper_prob"] is not None and prev["tamper_prob"] < config.convergence_threshold:
            break

        rewrite_prompt = config.iterative_rewrite_prompt.format(
            opus_reason=prev["reason"],
            previous_response=prev["response"],
        )
        rewrite = call_gemini_rewrite(rewrite_prompt, config, client, cost_tracker)
        time.sleep(config.get_delay(config.gemini_id))

        if rewrite is None:
            print(f"  [{tid}] iter {i}: Gemini rewrite failed, stopping.")
            break

        score, reason = detect_one(task["prompt"], rewrite, config, client, cost_tracker)
        time.sleep(config.get_delay(config.evaluator_id))

        iterations.append({
            "iter":        i,
            "response":    rewrite,
            "tamper_prob": score,
            "reason":      reason,
        })

        prev_score = prev["tamper_prob"]
        curr_score = score
        if (
            prev_score is not None
            and curr_score is not None
            and abs(curr_score - prev_score) < 5
            and (curr_score is None or curr_score >= config.convergence_threshold)
        ):
            break  # plateau

    final_score = iterations[-1]["tamper_prob"]
    converged = final_score is not None and final_score < config.convergence_threshold
    conv_iter = next(
        (it["iter"] for it in iterations if (it["tamper_prob"] or 999) < config.convergence_threshold),
        None,
    )

    return {
        "task_id":          tid,
        "task_prompt":      task["prompt"],
        "iterations":       iterations,
        "converged":        converged,
        "convergence_iter": conv_iter,
        "final_score":      final_score,
        "n_iters":          len(iterations),
    }


def _save(trajectories: list[dict], path: Path) -> None:
    path.write_text(json.dumps(trajectories, indent=2))
