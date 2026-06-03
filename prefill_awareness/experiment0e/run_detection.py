"""
Detection for Experiment 0e (single-turn).

For each (evaluator, source, task):

  [System]  You are a helpful assistant.
  [User]    <task prompt>
  [Asst]    <source_response or evaluator's own>
  [User]    <detection_prompt>

evaluator rates tamper_prob 0-100.

Output: results_dir/detection_results.json + .csv
"""

import json
import re
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .config import Experiment0eConfig
from .generate import _build_messages, load_responses
from .utils import CostTracker, api_call_with_retry, make_client


def run_all_detection(
    tasks: list[dict],
    evaluator_names: list[str],
    config: Experiment0eConfig,
    cost_tracker: CostTracker,
    force: bool = False,
) -> list[dict]:
    out_json = config.results_dir / "detection_results.json"
    out_csv  = config.results_dir / "detection_results.csv"

    existing: list[dict] = []
    if out_json.exists() and not force:
        with open(out_json) as f:
            existing = json.load(f)

    completed: set[tuple] = {
        (r["evaluator"], r["source"], r["task_id"]) for r in existing
    }
    print(f"  {len(completed)} detection calls already completed.")

    # Pre-load all source responses and evaluator self-responses
    source_resps: dict[str, dict[str, str]] = {}
    for src_name in config.sources:
        try:
            source_resps[src_name] = load_responses(src_name, config)
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")

    eval_resps: dict[str, dict[str, str]] = {}
    for ev_name in evaluator_names:
        # cheap models that are also sources — reuse
        src_name = ev_name if ev_name in config.sources else None
        if src_name:
            eval_resps[ev_name] = source_resps.get(src_name, {})
        else:
            try:
                eval_resps[ev_name] = load_responses(ev_name, config)
            except FileNotFoundError as e:
                print(f"  WARNING: {e}")
                eval_resps[ev_name] = {}

    client = make_client(config)
    results = list(existing)

    for ev_name in evaluator_names:
        if cost_tracker.exceeded():
            print(f"\n  Cost cap reached. Stopping before {ev_name}.")
            break

        ev_id = config.evaluators.get(ev_name)
        if not ev_id:
            continue

        # Organic condition + each source condition
        conditions: list[tuple[str, dict[str, str]]] = [
            ("organic", eval_resps.get(ev_name, {})),
        ]
        for src_name, src_resp in source_resps.items():
            conditions.append((src_name, src_resp))

        work = []
        for condition_name, resp_map in conditions:
            for task in tasks:
                tid = task["task_id"]
                key = (ev_name, condition_name, tid)
                if key in completed:
                    continue
                response = resp_map.get(tid)
                if not response:
                    continue
                work.append((condition_name, task, response))

        if not work:
            print(f"  [{ev_name}] all detection calls cached.")
            continue

        print(f"  [{ev_name}] {len(work)} detection calls remaining...")

        with tqdm(total=len(work), desc=ev_name) as pbar:
            for condition_name, task, response in work:
                if cost_tracker.exceeded():
                    print("\n  Cost cap reached.")
                    break

                record = _run_one(
                    ev_name, ev_id, condition_name, task, response, config, client, cost_tracker
                )
                if record is not None:
                    results.append(record)
                    completed.add((ev_name, condition_name, task["task_id"]))
                    if len(results) % 25 == 0:
                        _save(results, out_json, out_csv)

                time.sleep(config.get_delay(ev_id))
                pbar.update(1)

        _save(results, out_json, out_csv)
        print(f"  [{ev_name}] done. {cost_tracker.report()}")

    _save(results, out_json, out_csv)
    _report_parse_failures(results, config, evaluator_names)
    return results


def _run_one(
    ev_name: str,
    ev_id: str,
    condition_name: str,
    task: dict,
    response: str,
    config: Experiment0eConfig,
    client,
    cost_tracker: CostTracker,
) -> dict | None:
    # Build the 3-message context + detection query
    system = "You are a helpful assistant."
    task_prompt = task["prompt"]

    # Truncate SWE-bench prompts for cost control (§10 item 3)
    if task.get("dataset") == "swebench":
        raw = task.get("_raw_problem", "")
        if len(raw) > config.swebench_max_problem_chars:
            truncated = raw[:config.swebench_max_problem_chars] + "\n[...truncated for detection]"
            task_prompt = task_prompt.replace(raw, truncated)

    if "gemini" in ev_id.lower():
        messages = [
            {"role": "user",      "content": f"[Instructions: {system}]\n\n{task_prompt}"},
            {"role": "assistant", "content": response},
            {"role": "user",      "content": config.detection_prompt},
        ]
    else:
        messages = [
            {"role": "system",    "content": system},
            {"role": "user",      "content": task_prompt},
            {"role": "assistant", "content": response},
            {"role": "user",      "content": config.detection_prompt},
        ]

    extra = {"thinking": {"type": "disabled"}} if config.is_thinking_model(ev_id) else {}

    def _fn():
        return client.chat.completions.create(
            model=ev_id,
            messages=messages,
            temperature=0.0,
            max_tokens=config.get_max_tokens_detection(ev_id),
            extra_body=extra,
        )

    resp = api_call_with_retry(_fn, max_retries=config.max_retries,
                               retry_delay=config.retry_delay)
    choices = getattr(resp, "choices", None) if resp else None
    raw = (choices[0].message.content or "") if choices else ""

    if resp and getattr(resp, "usage", None):
        u = resp.usage
        cost_tracker.add_from_usage(ev_id, u.prompt_tokens, u.completion_tokens)
    else:
        cost_tracker.add_estimated(ev_id, messages, raw)

    tamper_prob, reason = _parse(raw)
    is_organic = (condition_name == "organic")

    return {
        "evaluator":    ev_name,
        "source":       condition_name,
        "task_id":      task["task_id"],
        "dataset":      task.get("dataset", ""),
        "is_tampered":  not is_organic,
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


def _report_parse_failures(
    results: list[dict],
    config: Experiment0eConfig,
    evaluator_names: list[str],
) -> None:
    df = pd.DataFrame(results)
    if len(df) == 0:
        return
    print("\n  Parse failure rate by evaluator:")
    for ev in evaluator_names:
        sub = df[df["evaluator"] == ev]
        if len(sub) == 0:
            continue
        fail = pd.to_numeric(sub["tamper_prob"], errors="coerce").isna().mean()
        flag = " ⚠ >20%" if fail > 0.20 else (" ⚠ >15%" if fail > 0.15 else "")
        print(f"    {ev}: {fail:.1%}{flag}")
