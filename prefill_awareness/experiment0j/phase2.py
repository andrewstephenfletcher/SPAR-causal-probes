"""
Phase 2: Selective Feature Application for Experiment 0j.

Three sub-phases:

  Forward  — Gemini generates with one Claude-style feature added at a time.
             Opus detects. Should reduce tamper_prob.

  Reverse  — Opus generates with one Gemini-style feature added at a time.
             Opus detects. Should raise tamper_prob.

  Stacking — Top 2 and top 3 forward features combined.
             Tests whether effects are additive or subadditive.

Scores are accumulated in results_dir/phase2_scores.json as a list of records:
  {direction, feature, task_id, tamper_prob, reason, generator}
"""

import json
import time
from pathlib import Path

from tqdm import tqdm

from .config import Experiment0jConfig
from .run_detection import (
    CostTracker,
    call_gemini,
    call_opus,
    detect_one,
    make_client,
)

_SYSTEM_BASE = "You are a helpful assistant."


def run_phase2_forward(
    tasks: list[dict],
    config: Experiment0jConfig,
    cost_tracker: CostTracker,
    force: bool = False,
) -> list[dict]:
    """Generate Gemini responses with individual Claude-style features; detect with Opus."""
    return _run_direction(
        direction="forward",
        features=config.forward_features,
        generator_id=config.gemini_id,
        generate_fn=_gen_forward,
        tasks=tasks,
        config=config,
        cost_tracker=cost_tracker,
        force=force,
    )


def run_phase2_reverse(
    tasks: list[dict],
    config: Experiment0jConfig,
    cost_tracker: CostTracker,
    force: bool = False,
) -> list[dict]:
    """Generate Opus responses with individual Gemini-style features; detect with Opus."""
    return _run_direction(
        direction="reverse",
        features=config.reverse_features,
        generator_id=config.evaluator_id,
        generate_fn=_gen_reverse,
        tasks=tasks,
        config=config,
        cost_tracker=cost_tracker,
        force=force,
    )


def run_phase2_stacking(
    tasks: list[dict],
    top_features: list[str],
    config: Experiment0jConfig,
    cost_tracker: CostTracker,
    force: bool = False,
) -> list[dict]:
    """
    Test top 2 and top 3 forward features applied together.
    top_features: feature names ordered best→worst by individual Δ tamper_prob.
    """
    new_records: list[dict] = []
    client = make_client(config)

    for n in [2, 3]:
        if len(top_features) < n:
            continue
        combo_names = top_features[:n]
        feature_key = f"top{n}_" + "_".join(combo_names)
        combined_instruction = " ".join(
            config.forward_features[f] for f in combo_names
        )

        resp_path = config.responses_dir / f"forward_{feature_key}.json"
        existing_resp: dict[str, str] = {}
        if resp_path.exists() and not force:
            existing_resp = json.loads(resp_path.read_text())

        scores_path = config.results_dir / "phase2_scores.json"
        existing_scores = _load_scores(scores_path)
        done_ids = {
            r["task_id"]
            for r in existing_scores
            if r["direction"] == "stacking" and r["feature"] == feature_key
        }

        with tqdm(total=len(tasks), desc=f"stacking/{feature_key}") as pbar:
            for task in tasks:
                if cost_tracker.exceeded():
                    print("\n  Cost cap reached.")
                    break
                tid = task["task_id"]
                pbar.update(1)
                if tid in done_ids:
                    continue

                if tid not in existing_resp:
                    response = _gen_forward(
                        task, combined_instruction, config, client, cost_tracker
                    )
                    if response:
                        existing_resp[tid] = response
                        resp_path.write_text(json.dumps(existing_resp, indent=2))
                    else:
                        continue
                    time.sleep(config.get_delay(config.gemini_id))

                score, reason = detect_one(
                    task["prompt"], existing_resp[tid], config, client, cost_tracker
                )
                time.sleep(config.get_delay(config.evaluator_id))

                rec = {
                    "direction":  "stacking",
                    "feature":    feature_key,
                    "combo":      combo_names,
                    "task_id":    tid,
                    "tamper_prob": score,
                    "reason":     reason,
                    "generator":  config.gemini_id,
                }
                new_records.append(rec)
                existing_scores.append(rec)
                scores_path.write_text(json.dumps(existing_scores, indent=2))

    return new_records


# ---------------------------------------------------------------------------
# Shared internals
# ---------------------------------------------------------------------------

def _run_direction(
    direction: str,
    features: dict[str, str],
    generator_id: str,
    generate_fn,
    tasks: list[dict],
    config: Experiment0jConfig,
    cost_tracker: CostTracker,
    force: bool,
) -> list[dict]:
    client = make_client(config)
    new_records: list[dict] = []

    scores_path = config.results_dir / "phase2_scores.json"
    existing_scores = _load_scores(scores_path)

    for feature_name, instruction in features.items():
        if cost_tracker.exceeded():
            print(f"\n  Cost cap reached — skipping remaining {direction} features.")
            break

        resp_path = config.responses_dir / f"{direction}_{feature_name}.json"
        existing_resp: dict[str, str] = {}
        if resp_path.exists() and not force:
            existing_resp = json.loads(resp_path.read_text())

        done_ids = {
            r["task_id"]
            for r in existing_scores
            if r["direction"] == direction and r["feature"] == feature_name
        }

        with tqdm(total=len(tasks), desc=f"{direction}/{feature_name}") as pbar:
            for task in tasks:
                if cost_tracker.exceeded():
                    print(f"\n  Cost cap reached.")
                    break
                tid = task["task_id"]
                pbar.update(1)
                if tid in done_ids:
                    continue

                if tid not in existing_resp:
                    response = generate_fn(
                        task, instruction, config, client, cost_tracker
                    )
                    if response:
                        existing_resp[tid] = response
                        resp_path.write_text(json.dumps(existing_resp, indent=2))
                    else:
                        continue
                    time.sleep(config.get_delay(generator_id))

                score, reason = detect_one(
                    task["prompt"], existing_resp[tid], config, client, cost_tracker
                )
                time.sleep(config.get_delay(config.evaluator_id))

                rec = {
                    "direction":  direction,
                    "feature":    feature_name,
                    "task_id":    tid,
                    "tamper_prob": score,
                    "reason":     reason,
                    "generator":  generator_id,
                }
                new_records.append(rec)
                existing_scores.append(rec)
                scores_path.write_text(json.dumps(existing_scores, indent=2))

    return new_records


def _gen_forward(
    task: dict,
    instruction: str,
    config: Experiment0jConfig,
    client,
    cost_tracker: CostTracker,
) -> str | None:
    system = f"{_SYSTEM_BASE} When writing code responses: {instruction}"
    return call_gemini(task["prompt"], system, config, client, cost_tracker)


def _gen_reverse(
    task: dict,
    instruction: str,
    config: Experiment0jConfig,
    client,
    cost_tracker: CostTracker,
) -> str | None:
    return call_opus(task["prompt"], instruction, config, client, cost_tracker)


def _load_scores(path: Path) -> list[dict]:
    if path.exists():
        return json.loads(path.read_text())
    return []
