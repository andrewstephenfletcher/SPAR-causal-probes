"""
Detection for Experiment 0d.

Each conversation is evaluated by the evaluator model that generated
turns 1 and 3. The evaluator rates the tamper probability of turn 2.

Output: results_dir/detection_results.json + .csv
"""

import json
import re
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .config import Experiment0dConfig
from .utils import CostTracker, api_call_with_retry, make_client
from .generate import _maybe_fold_system


def run_all_detection(
    conversations: list[dict],
    config: Experiment0dConfig,
    cost_tracker: CostTracker,
    evaluator_names: list[str] | None = None,
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

    # Filter to requested evaluators and not-yet-done
    work = [
        c for c in conversations
        if (evaluator_names is None or c["evaluator"] in evaluator_names)
        and (c["evaluator"], c["source"], c["task_id"]) not in completed
    ]
    print(f"  {len(work)} detection calls remaining.")

    results = list(existing)
    client = make_client(config)

    with tqdm(total=len(work), desc="Detection") as pbar:
        for conv in work:
            if cost_tracker.exceeded():
                print("\n  Cost cap reached. Stopping.")
                break
            if cost_tracker.approaching():
                print(f"\n  *** Approaching cap. {cost_tracker.report()} ***")

            record = _run_one(conv, config, client, cost_tracker)
            if record is not None:
                results.append(record)
                if len(results) % 20 == 0:
                    _save(results, out_json, out_csv)

            ev_id = config.evaluators.get(conv["evaluator"], "")
            time.sleep(config.get_delay(ev_id))
            pbar.update(1)

    _save(results, out_json, out_csv)
    print(f"\n{cost_tracker.report()}")
    _report_parse_failures(results, config)
    return results


def _run_one(
    conv: dict,
    config: Experiment0dConfig,
    client,
    cost_tracker: CostTracker,
) -> dict | None:
    ev_name = conv["evaluator"]
    ev_id = config.evaluators.get(ev_name)
    if not ev_id:
        return None

    messages = list(conv["messages"]) + [
        {"role": "user", "content": config.detection_prompt}
    ]
    msgs = _maybe_fold_system(ev_id, messages)

    def _fn():
        return client.chat.completions.create(
            model=ev_id,
            messages=msgs,
            temperature=0.0,
            max_tokens=config.get_max_tokens_detection(ev_id),
        )

    resp = api_call_with_retry(_fn, max_retries=config.max_retries,
                               retry_delay=config.retry_delay)
    choices = getattr(resp, "choices", None) if resp else None
    raw = (choices[0].message.content or "") if choices else ""

    if resp and getattr(resp, "usage", None):
        u = resp.usage
        cost_tracker.add_from_usage(ev_id, u.prompt_tokens, u.completion_tokens)
    else:
        cost_tracker.add_estimated(ev_id, msgs, raw)

    tamper_prob, reason = _parse(raw)
    return {
        "evaluator":    ev_name,
        "source":       conv["source"],
        "task_id":      conv["task_id"],
        "is_tampered":  conv["is_tampered"],
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


def _report_parse_failures(results: list[dict], config: Experiment0dConfig) -> None:
    df = pd.DataFrame(results)
    if len(df) == 0:
        return
    print("\n  Parse failure rate by evaluator:")
    for ev in config.evaluators:
        sub = df[df["evaluator"] == ev]
        if len(sub) == 0:
            continue
        fail = pd.to_numeric(sub["tamper_prob"], errors="coerce").isna().mean()
        flag = " ⚠ >20%" if fail > 0.20 else (" ⚠ >15%" if fail > 0.15 else "")
        print(f"    {ev}: {fail:.1%}{flag}")
