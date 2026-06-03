"""
Detection and binary attribution tasks for Experiment 0b.

Primary task (detection):
  For each (evaluator, conversation):
    - Organic condition:  all assistant turns from evaluator → ask tamper prob → expect low
    - Tampered condition: middle turn replaced by source → ask tamper prob → expect high

Secondary task (binary, agentic only):
  Same conversation pairs but using the "me / not me" prompt.

Both tasks share the same result file with a 'task_type' field.

Output: results_dir/detection_results.json + .csv
"""

import json
import re
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .config import Experiment0bConfig
from .utils import CostTracker, api_call_with_retry, make_client


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_all_detection(
    conversations: list[dict],
    trajectories: dict[str, dict[str, list[str]]],
    oasst1_replacements: dict[str, dict[str, str]],
    config: Experiment0bConfig,
    cost_tracker: CostTracker,
    force: bool = False,
) -> list[dict]:
    """
    Run detection (and optionally binary) tasks for all evaluators.

    Args:
        conversations:       from data.load_all_conversations()
        trajectories:        {model_name: {task_id: [t1, t2, t3]}} from generate_trajectories
        oasst1_replacements: {conv_id: {model_name: text}} from data.load_oasst1_replacements()
        config:              experiment config
        cost_tracker:        shared cost tracker
        force:               overwrite existing results
    """
    out_json = config.results_dir / "detection_results.json"
    out_csv  = config.results_dir / "detection_results.csv"

    existing: list[dict] = []
    if out_json.exists() and not force:
        with open(out_json) as f:
            existing = json.load(f)

    completed: set[tuple] = {
        (r["evaluator"], r["source"], r["conv_id"], r["task_type"])
        for r in existing
    }
    print(f"  {len(completed)} detection calls already completed.")

    results = list(existing)
    client = make_client(config)

    agentic = [c for c in conversations if c["dataset"] == "agentic"]
    oasst1  = [c for c in conversations if c["dataset"] == "oasst1"]

    # Determine task types to run
    task_types = ["detection"]
    if config.run_binary_task:
        task_types.append("binary")

    # Build work list
    work = _build_work_list(
        agentic, oasst1, task_types, config, completed,
    )
    print(f"  {len(work)} detection calls remaining.")

    with tqdm(total=len(work), desc="Detection") as pbar:
        for item in work:
            if cost_tracker.hard_exceeded():
                print("\n  Hard cost ceiling reached. Stopping.")
                break
            if cost_tracker.phase_exceeded():
                print(f"\n  Phase cap reached. Stopping. {cost_tracker.report()}")
                break
            if cost_tracker.approaching_cap():
                print(f"\n  *** Approaching cap. {cost_tracker.report()} ***")

            record = _run_one(
                item, trajectories, oasst1_replacements, config, client, cost_tracker
            )
            if record is not None:
                results.append(record)
                if len(results) % 20 == 0:
                    _save(results, out_json, out_csv)

            time.sleep(config.get_delay(config.evaluators.get(item["eval_name"], "")))
            pbar.update(1)

    _save(results, out_json, out_csv)
    print(f"\n{cost_tracker.report()}")
    _report_parse_failures(results, config)
    return results


# ---------------------------------------------------------------------------
# Work list construction
# ---------------------------------------------------------------------------

def _build_work_list(
    agentic: list[dict],
    oasst1: list[dict],
    task_types: list[str],
    config: Experiment0bConfig,
    completed: set[tuple],
) -> list[dict]:
    work = []
    sources_with_self = ["self"] + list(config.sources.keys())

    for eval_name in config.evaluators:
        # Agentic: both task types
        for conv in agentic:
            for source in sources_with_self:
                for tt in task_types:
                    key = (eval_name, source, conv["conv_id"], tt)
                    if key not in completed:
                        work.append({
                            "eval_name": eval_name,
                            "source":    source,
                            "conv":      conv,
                            "task_type": tt,
                        })

        # OASST1: detection only
        for conv in oasst1:
            for source in sources_with_self:
                key = (eval_name, source, conv["conv_id"], "detection")
                if key not in completed:
                    work.append({
                        "eval_name": eval_name,
                        "source":    source,
                        "conv":      conv,
                        "task_type": "detection",
                    })

    return work


# ---------------------------------------------------------------------------
# Single call
# ---------------------------------------------------------------------------

def _run_one(
    item: dict,
    trajectories: dict,
    oasst1_replacements: dict,
    config: Experiment0bConfig,
    client,
    cost_tracker: CostTracker,
) -> dict | None:
    eval_name = item["eval_name"]
    eval_id   = config.evaluators[eval_name]
    source    = item["source"]
    conv      = item["conv"]
    task_type = item["task_type"]
    is_tampered = (source != "self")

    messages = _build_messages(
        conv, eval_name, source, task_type, trajectories, oasst1_replacements, config
    )
    if messages is None:
        return None

    max_tokens = (
        config.max_tokens_detection if task_type == "detection"
        else config.max_tokens_binary
    )

    def _call():
        return client.chat.completions.create(
            model=eval_id,
            messages=messages,
            temperature=0.0,
            max_tokens=max_tokens,
        )

    response = api_call_with_retry(
        _call,
        max_retries=config.max_retries,
        retry_delay=config.retry_delay,
    )

    choices = getattr(response, "choices", None) if response else None
    raw = (choices[0].message.content or "") if choices else ""

    if response and getattr(response, "usage", None):
        u = response.usage
        cost_tracker.add_from_usage(eval_id, u.prompt_tokens, u.completion_tokens)
    else:
        cost_tracker.add_estimated(eval_id, messages, raw)

    if task_type == "detection":
        tamper_prob, reason = parse_detection_response(raw)
        return {
            "conv_id":            conv["conv_id"],
            "dataset":            conv["dataset"],
            "evaluator":          eval_name,
            "source":             source,
            "is_tampered":        is_tampered,
            "task_type":          "detection",
            "tamper_prob":        tamper_prob,
            "reason":             reason,
            "raw_response":       raw,
            "target_turn_number": conv["target_turn_number"],
        }
    else:  # binary
        parsed, explanation = parse_binary_response(raw)
        return {
            "conv_id":            conv["conv_id"],
            "dataset":            conv["dataset"],
            "evaluator":          eval_name,
            "source":             source,
            "is_tampered":        is_tampered,
            "task_type":          "binary",
            "parsed":             parsed,
            "explanation":        explanation,
            "raw_response":       raw,
            "target_turn_number": conv["target_turn_number"],
        }


# ---------------------------------------------------------------------------
# Message construction
# ---------------------------------------------------------------------------

def _build_messages(
    conv: dict,
    eval_name: str,
    source: str,
    task_type: str,
    trajectories: dict,
    oasst1_replacements: dict,
    config: Experiment0bConfig,
) -> list[dict] | None:
    is_agentic = conv["dataset"] == "agentic"

    if is_agentic:
        return _build_agentic_messages(
            conv, eval_name, source, task_type, trajectories, config
        )
    else:
        return _build_oasst1_messages(
            conv, eval_name, source, task_type, oasst1_replacements, config
        )


def _build_agentic_messages(
    conv: dict,
    eval_name: str,
    source: str,
    task_type: str,
    trajectories: dict,
    config: Experiment0bConfig,
) -> list[dict] | None:
    """
    For agentic: use evaluator's trajectory for organic turns 1 and 3.
    For tampered: substitute the source model's turn 2 at target_msg_idx.
    """
    task_id = conv["task_id"]
    eval_turns = trajectories.get(eval_name, {}).get(task_id)
    if not eval_turns or len(eval_turns) < 3:
        return None

    # For tampered condition, get source model's middle turn
    if source != "self":
        src_turns = trajectories.get(source, {}).get(task_id)
        if not src_turns or len(src_turns) < 3:
            return None
        replacement_turn2 = src_turns[1]  # index 1 = middle turn
    else:
        replacement_turn2 = None

    # Build messages: [sys, user, asst1, user:tool0, asst2, user:tool1, asst3, user:prompt]
    tool_outs = conv["tool_outputs"]
    turn2 = replacement_turn2 if replacement_turn2 is not None else eval_turns[1]

    messages = [
        {"role": "system",    "content": "You are a helpful assistant."},
        {"role": "user",      "content": conv["messages"][0]["content"]},  # task desc
        {"role": "assistant", "content": eval_turns[0]},                   # turn 1 (always eval)
        {"role": "user",      "content": tool_outs[0]},
        {"role": "assistant", "content": turn2},                           # turn 2 (eval or src)
        {"role": "user",      "content": tool_outs[1]},
        {"role": "assistant", "content": eval_turns[2]},                   # turn 3 (always eval)
    ]

    _append_prompt(messages, task_type, conv["target_turn_number"], config)
    return messages


def _build_oasst1_messages(
    conv: dict,
    eval_name: str,
    source: str,
    task_type: str,
    oasst1_replacements: dict,
    config: Experiment0bConfig,
) -> list[dict] | None:
    """
    For OASST1: organic uses the original messages.
    Tampered replaces the last assistant turn with the source's replacement.
    """
    msgs = list(conv["messages"])
    target_msg_idx = conv["target_msg_idx"]

    if source != "self":
        replacement = oasst1_replacements.get(conv["conv_id"], {}).get(source)
        if not replacement:
            return None
        msgs = list(msgs)  # copy
        msgs[target_msg_idx] = {"role": "assistant", "content": replacement}

    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    messages.extend(msgs)
    _append_prompt(messages, task_type, conv["target_turn_number"], config)
    return messages


def _append_prompt(
    messages: list[dict],
    task_type: str,
    target_turn_number: int,
    config: Experiment0bConfig,
) -> None:
    if task_type == "detection":
        prompt = config.detection_prompt_template.format(turn_number=target_turn_number)
    else:
        prompt = config.binary_prompt
    messages.append({"role": "user", "content": prompt})


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_detection_response(text: str) -> tuple[int | None, str]:
    """
    Parse JSON response: {"tamper_prob": <int>, "reason": "<str>"}
    Returns (tamper_prob or None, reason).
    """
    raw = (text or "").strip()

    # Direct JSON parse
    try:
        data = json.loads(raw)
        prob = int(data["tamper_prob"])
        return max(0, min(100, prob)), str(data.get("reason", ""))
    except Exception:
        pass

    # JSON inside markdown code block
    block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if block:
        try:
            data = json.loads(block.group(1))
            prob = int(data["tamper_prob"])
            return max(0, min(100, prob)), str(data.get("reason", ""))
        except Exception:
            pass

    # Any JSON-like object in the text
    obj_match = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
    if obj_match:
        try:
            data = json.loads(obj_match.group())
            prob = int(data["tamper_prob"])
            return max(0, min(100, prob)), str(data.get("reason", ""))
        except Exception:
            pass

    # tamper_prob key without full JSON
    kv = re.search(r'"?tamper_prob"?\s*[=:]\s*(\d+)', raw)
    if kv:
        return max(0, min(100, int(kv.group(1)))), ""

    # Last resort: standalone 0-100 number
    nums = re.findall(r'\b(\d{1,3})\b', raw)
    for n in nums:
        v = int(n)
        if 0 <= v <= 100:
            return v, ""

    return None, ""


def parse_binary_response(text: str) -> tuple[str, str]:
    """Parse <answer>me/not me</answer> format. Returns (label, explanation)."""
    t = (text or "").strip().lower()
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", t)
    if m:
        ans = m.group(1).strip()
        label = "not_me" if "not me" in ans else ("me" if "me" in ans else "unparseable")
    elif "not me" in t:
        label = "not_me"
    elif re.match(r"^me\b", t):
        label = "me"
    else:
        label = "unparseable"

    expl_m = re.search(r"<explanation>\s*(.*?)\s*</explanation>", text, re.DOTALL)
    explanation = expl_m.group(1).strip() if expl_m else ""
    return label, explanation


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _save(results: list[dict], json_path: Path, csv_path: Path) -> None:
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    pd.DataFrame(results).to_csv(csv_path, index=False)


def _report_parse_failures(results: list[dict], config: Experiment0bConfig) -> None:
    df = pd.DataFrame(results)
    det = df[df["task_type"] == "detection"]
    if len(det) == 0:
        return
    print("\n  Parse failure rate (detection task) by evaluator:")
    for ev in config.evaluators:
        sub = det[det["evaluator"] == ev]
        if len(sub) == 0:
            continue
        fail_rate = sub["tamper_prob"].isna().mean()
        flag = " ⚠ >15%" if fail_rate > 0.15 else ""
        print(f"    {ev}: {fail_rate:.1%}{flag}")
