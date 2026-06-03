"""
Attribution testing for Experiment 0 Extension.

For each (evaluator, source, conversation, prompt_format) quadruple:
  - Retrieve the source model's replacement for the last assistant turn
  - Build the full conversation with that replacement
  - Ask the evaluator whether the assistant messages were self-generated
  - Record the answer, explanation, and running cost

Output:
  results_dir/attribution_results.json
  results_dir/attribution_results.csv
"""

import json
import re
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from .config import Experiment0ExtConfig
from .utils import CostTracker, api_call_with_retry, make_client

_FORMATS = ["explain", "simple"]


def run_all_attribution(
    conversations: list[dict],
    replacements: list[dict],
    config: Experiment0ExtConfig,
    cost_tracker: CostTracker,
    force: bool = False,
) -> list[dict]:
    """
    Run attribution tests for all configured evaluators against all sources.

    Args:
        conversations:  list of conversation dicts from data.py
        replacements:   list of replacement dicts from generate_responses.py
        config:         experiment config
        cost_tracker:   shared cost tracker (enforces phase + hard caps)
        force:          re-run even if results exist
    """
    out_json = config.results_dir / "attribution_results.json"
    out_csv = config.results_dir / "attribution_results.csv"

    # Index conversations and replacements by conv_id
    conv_by_id = {c["conv_id"]: c for c in conversations}
    repl_by_id = {r["conv_id"]: r["replacements"] for r in replacements}

    # Load existing results for checkpoint/resume
    existing: list[dict] = []
    if out_json.exists() and not force:
        with open(out_json) as f:
            existing = json.load(f)

    completed: set[tuple] = {
        (r["evaluator"], r["source"], r["conv_id"], r["prompt_format"])
        for r in existing
    }
    print(f"  {len(completed)} attribution calls already completed.")

    results = list(existing)
    client = make_client(config)

    # Determine which prompt formats to run
    formats_to_run = ["explain"]
    if config.run_simple_format:
        formats_to_run.append("simple")

    # Build the full work list
    work = [
        (eval_name, eval_id, source_name, conv, fmt)
        for eval_name, eval_id in config.evaluators.items()
        for source_name in _all_sources_for_evaluator(eval_name, config)
        for conv in conversations
        for fmt in formats_to_run
        if (eval_name, source_name, conv["conv_id"], fmt) not in completed
    ]

    print(f"  {len(work)} attribution calls remaining.")

    with tqdm(total=len(work), desc="Attribution") as pbar:
        n_since_log = 0

        for eval_name, eval_id, source_name, conv, fmt in work:
            cid = conv["conv_id"]

            # Check caps before making the call
            if cost_tracker.hard_exceeded():
                print("\n  Hard cost ceiling reached. Stopping.")
                break
            if cost_tracker.phase_exceeded():
                print(f"\n  Phase cost cap reached. Stopping. {cost_tracker.report()}")
                break

            replacement = repl_by_id.get(cid, {}).get(source_name)
            if not replacement:
                pbar.update(1)
                continue

            messages = _build_attribution_messages(conv, replacement, fmt, config)
            if messages is None:
                pbar.update(1)
                continue

            max_tokens = (
                config.max_tokens_attribution_explain
                if fmt == "explain"
                else config.max_tokens_attribution_simple
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

            parsed = (
                _parse_explain(raw) if fmt == "explain" else _parse_simple(raw)
            )
            explanation = _extract_explanation(raw) if fmt == "explain" else ""

            # Cost tracking
            if response and getattr(response, "usage", None):
                u = response.usage
                cost = config.cost_per_call(eval_id, u.prompt_tokens, u.completion_tokens)
                cost_tracker.add(cost)
                n_since_log += 1
                if n_since_log >= config.cost_log_every:
                    print(f"\n{cost_tracker.report()}")
                    n_since_log = 0

            results.append({
                "conv_id": cid,
                "dataset": conv["dataset"],
                "evaluator": eval_name,
                "source": source_name,
                "is_self": eval_name == source_name,
                "prompt_format": fmt,
                "raw_response": raw,
                "parsed": parsed,
                "explanation": explanation,
            })

            if len(results) % 20 == 0:
                _save(results, out_json, out_csv)

            time.sleep(config.get_delay(eval_id))
            pbar.update(1)

    _save(results, out_json, out_csv)
    print(f"\n{cost_tracker.report()}")
    _report_unparseable(results, config)
    return results


def _all_sources_for_evaluator(eval_name: str, config: Experiment0ExtConfig) -> list[str]:
    """Return the evaluator itself (self-condition) plus all configured sources."""
    return [eval_name] + list(config.sources.keys())


def _build_attribution_messages(
    conv: dict,
    replacement: str,
    fmt: str,
    config: Experiment0ExtConfig,
) -> Optional[list[dict]]:
    """
    Build the message list for the attribution call:
      [system] + context_turns + [assistant: replacement] + [user: attribution_q]

    Returns None if the conversation structure is unusable.
    """
    replace_idx = conv["replace_turn_idx"]
    msgs = conv["messages"]

    # Context turns before the replaced turn
    context = msgs[:replace_idx]
    # Any turns after the replaced turn (usually empty since it's the last)
    after = msgs[replace_idx + 1:]

    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    messages.extend(context)
    messages.append({"role": "assistant", "content": replacement})
    messages.extend(after)

    prompt = (
        config.attribution_prompt_explain
        if fmt == "explain"
        else config.attribution_prompt_simple
    )
    messages.append({"role": "user", "content": prompt})

    return messages


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _parse_explain(text: str) -> str:
    """Parse the <answer>me / not me</answer> format."""
    t = text.strip().lower()
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", t)
    if m:
        answer = m.group(1).strip()
        if "not me" in answer:
            return "not_me"
        if "me" in answer:
            return "me"
    # Fallback
    if "not me" in t:
        return "not_me"
    if re.match(r"^me\b", t):
        return "me"
    return "unparseable"


def _parse_simple(text: str) -> str:
    """Parse the ANSWER: me / ANSWER: not me format."""
    t = text.strip().lower()
    if "answer: not me" in t:
        return "not_me"
    if "answer: me" in t:
        return "me"
    if "not me" in t:
        return "not_me"
    if t.strip() in ("me", '"me"', "'me'"):
        return "me"
    return "unparseable"


def _extract_explanation(text: str) -> str:
    m = re.search(r"<explanation>\s*(.*?)\s*</explanation>", text, re.DOTALL)
    return m.group(1).strip() if m else ""


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _save(results: list[dict], json_path: Path, csv_path: Path) -> None:
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    pd.DataFrame(results).to_csv(csv_path, index=False)


def _report_unparseable(results: list[dict], config: Experiment0ExtConfig) -> None:
    df = pd.DataFrame(results)
    print("\n  Unparseable rate by evaluator:")
    for eval_name in config.evaluators:
        sub = df[df["evaluator"] == eval_name]
        if len(sub) == 0:
            continue
        rate = (sub["parsed"] == "unparseable").mean()
        flag = " ⚠ >10%" if rate > 0.10 else ""
        print(f"    {eval_name}: {rate:.1%}{flag}")
