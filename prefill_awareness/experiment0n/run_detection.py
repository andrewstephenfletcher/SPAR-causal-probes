"""
Detection for Experiment 0f (single evaluator: Opus 4.5).

For each (source, task):

  [System]  You are a helpful assistant.
  [User]    <task prompt>
  [Asst]    <source_response or opus's own>
  [User]    <detection_prompt>

Output: results_dir/detection_results.json + .csv
"""

import json
import re
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .config import Experiment0nConfig
from .generate import load_responses
from .utils import CostTracker, api_call_with_retry, make_client

SYSTEM_PROMPT = "You are a helpful assistant."


def run_detection(
    tasks: list[dict],
    config: Experiment0nConfig,
    cost_tracker: CostTracker,
    force: bool = False,
) -> list[dict]:
    out_json = config.results_dir / "detection_results.json"
    out_csv  = config.results_dir / "detection_results.csv"

    existing: list[dict] = []
    if out_json.exists() and not force:
        with open(out_json) as f:
            existing = json.load(f)

    completed: set[tuple] = {(r["source"], r["task_id"]) for r in existing}
    print(f"  {len(completed)} detection calls already completed.")

    # Load all responses
    source_resps: dict[str, dict[str, str]] = {}
    for src_name in config.sources:
        try:
            source_resps[src_name] = load_responses(src_name, config)
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")

    try:
        organic_resps = load_responses(config.evaluator_name, config)
    except FileNotFoundError as e:
        print(f"  WARNING: {e}")
        organic_resps = {}

    # All conditions: organic + each source
    conditions: list[tuple[str, dict[str, str]]] = [
        ("organic", organic_resps),
        *source_resps.items(),
    ]

    work = [
        (condition_name, task, resp_map[task["task_id"]])
        for condition_name, resp_map in conditions
        for task in tasks
        if task["task_id"] in resp_map
        and (condition_name, task["task_id"]) not in completed
    ]

    print(f"  {len(work)} detection calls remaining.")

    if not work:
        return existing

    client = make_client(config)
    results = list(existing)

    with tqdm(total=len(work), desc="Detection") as pbar:
        for condition_name, task, response in work:
            if cost_tracker.exceeded():
                print("\n  Cost cap reached.")
                break

            record = _run_one(condition_name, task, response, config, client, cost_tracker)
            if record is not None:
                results.append(record)
                completed.add((condition_name, task["task_id"]))
                if len(results) % 25 == 0:
                    _save(results, out_json, out_csv)

            time.sleep(config.get_delay(config.evaluator_id))
            pbar.update(1)

    _save(results, out_json, out_csv)
    print(f"\n{cost_tracker.report()}")
    _report_parse_failures(results)
    return results


def _run_one(
    condition_name: str,
    task: dict,
    response: str,
    config: Experiment0nConfig,
    client,
    cost_tracker: CostTracker,
) -> dict | None:
    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": task["prompt"]},
        {"role": "assistant", "content": response},
        {"role": "user",      "content": config.detection_prompt},
    ]

    def _fn():
        return client.chat.completions.create(
            model=config.evaluator_id,
            messages=messages,
            temperature=0.0,
            max_tokens=config.max_tokens_detection,
        )

    resp = api_call_with_retry(_fn, max_retries=config.max_retries,
                               retry_delay=config.retry_delay)
    choices = getattr(resp, "choices", None) if resp else None
    raw = (choices[0].message.content or "") if choices else ""

    if resp and getattr(resp, "usage", None):
        u = resp.usage
        cost_tracker.add_from_usage(config.evaluator_id, u.prompt_tokens, u.completion_tokens)
    else:
        cost_tracker.add_estimated(config.evaluator_id, messages, raw)

    tamper_prob, reason = _parse(raw)

    return {
        "source":       condition_name,
        "task_id":      task["task_id"],
        "dataset":      task.get("dataset", ""),
        "is_tampered":  condition_name != "organic",
        "tamper_prob":  tamper_prob,
        "reason":       reason,
        "raw_response": raw,
    }


def _parse(text: str) -> tuple[int | None, str]:
    raw = (text or "").strip()
    try:
        data = json.loads(raw)
        return max(0, min(100, int(data["tamper_prob"]))), str(data.get("reason", ""))
    except Exception:
        pass
    for pattern in [
        r"```(?:json)?\s*(\{.*?\})\s*```",
        r"(\{[^{}]+\})",
    ]:
        m = re.search(pattern, raw, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(1))
                return max(0, min(100, int(data["tamper_prob"]))), str(data.get("reason", ""))
            except Exception:
                pass
    kv = re.search(r'"?tamper_prob"?\s*[=:]\s*(\d+)', raw)
    if kv:
        return max(0, min(100, int(kv.group(1)))), ""
    for n in re.findall(r'\b(\d{1,3})\b', raw):
        v = int(n)
        if 0 <= v <= 100:
            return v, ""
    return None, ""


def _save(results: list[dict], json_path: Path, csv_path: Path) -> None:
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    pd.DataFrame(results).to_csv(csv_path, index=False)


def _report_parse_failures(results: list[dict]) -> None:
    df = pd.DataFrame(results)
    if len(df) == 0:
        return
    fail = pd.to_numeric(df["tamper_prob"], errors="coerce").isna().mean()
    flag = " ⚠ >15%" if fail > 0.15 else ""
    print(f"  Parse failure rate: {fail:.1%}{flag}")
