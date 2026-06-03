"""
Detection task for Experiment 0c.

Prefills each 3-turn conversation into Opus 4.6 and asks it to rate
tamper probability (0-100) for assistant turn 2.

Output: results_dir/detection_results.json + .csv
"""

import json
import re
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .config import Experiment0cConfig
from .utils import CostTracker, api_call_with_retry, make_client


def run_all_detection(
    conversations: list[dict],
    config: Experiment0cConfig,
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
        (r["task_id"], r["condition"], r["source_name"]) for r in existing
    }
    print(f"  {len(completed)} detection calls already completed.")

    work = [
        c for c in conversations
        if (c["task_id"], c["condition"], c["source_name"]) not in completed
    ]
    print(f"  {len(work)} detection calls remaining.")

    results = list(existing)
    client = make_client(config)
    ev_id = config.evaluator_id

    with tqdm(total=len(work), desc="Detection") as pbar:
        for conv in work:
            if cost_tracker.exceeded():
                print("\n  Cost ceiling reached. Stopping.")
                break
            if cost_tracker.approaching():
                print(f"\n  *** Approaching ceiling. {cost_tracker.report()} ***")

            record = _run_one(conv, ev_id, config, client, cost_tracker)
            if record is not None:
                results.append(record)
                if len(results) % 20 == 0:
                    _save(results, out_json, out_csv)

            time.sleep(config.get_delay(ev_id))
            pbar.update(1)

    _save(results, out_json, out_csv)
    print(f"\n{cost_tracker.report()}")
    _report_parse_failures(results)
    return results


def _run_one(
    conv: dict,
    ev_id: str,
    config: Experiment0cConfig,
    client,
    cost_tracker: CostTracker,
) -> dict | None:
    messages = list(conv["messages"]) + [
        {"role": "user", "content": config.detection_prompt}
    ]

    def _call():
        return client.chat.completions.create(
            model=ev_id,
            messages=messages,
            temperature=0.0,
            max_tokens=config.max_tokens_detection,
        )

    response = api_call_with_retry(_call, max_retries=config.max_retries,
                                   retry_delay=config.retry_delay)
    choices = getattr(response, "choices", None) if response else None
    raw = (choices[0].message.content or "") if choices else ""

    if response and getattr(response, "usage", None):
        u = response.usage
        cost_tracker.add_from_usage(ev_id, u.prompt_tokens, u.completion_tokens)
    else:
        cost_tracker.add_estimated(ev_id, messages, raw)

    tamper_prob, reason = _parse(raw)
    return {
        "task_id":      conv["task_id"],
        "condition":    conv["condition"],
        "source_name":  conv["source_name"],
        "is_tampered":  conv["is_tampered"],
        "tamper_prob":  tamper_prob,
        "reason":       reason,
        "raw_response": raw,
        "turn3_len":    len(conv.get("turn3", "") or ""),
    }


def _parse(text: str) -> tuple[int | None, str]:
    raw = (text or "").strip()
    try:
        data = json.loads(raw)
        return max(0, min(100, int(data["tamper_prob"]))), str(data.get("reason", ""))
    except Exception:
        pass

    block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if block:
        try:
            data = json.loads(block.group(1))
            return max(0, min(100, int(data["tamper_prob"]))), str(data.get("reason", ""))
        except Exception:
            pass

    obj = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
    if obj:
        try:
            data = json.loads(obj.group())
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
    print("\n  Parse failure rate by condition:")
    for cond in sorted(df["condition"].unique()):
        sub = df[df["condition"] == cond]
        fail = pd.to_numeric(sub["tamper_prob"], errors="coerce").isna().mean()
        flag = " ⚠ >15%" if fail > 0.15 else ""
        print(f"    {cond}: {fail:.1%}{flag}")
