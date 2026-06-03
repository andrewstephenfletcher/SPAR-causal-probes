"""
Single-turn generation for Experiment 0f.

Generates responses for 3 source models + Opus 4.5 self-responses.
No role persona in system prompt. max_tokens set high to avoid truncation.

Saved outputs:
  generations_dir/responses_{model_name}.json  — {task_id: response_text}
"""

import json
import time
from pathlib import Path

from tqdm import tqdm

from .config import Experiment0fConfig
from .utils import CostTracker, api_call_with_retry, make_client

SYSTEM_PROMPT = "You are a helpful assistant."


def generate_all(
    tasks: list[dict],
    config: Experiment0fConfig,
    cost_tracker: CostTracker,
    force: bool = False,
) -> None:
    client = make_client(config)

    # Sources first
    for src_name, src_id in config.sources.items():
        _generate_model_responses(src_name, src_id, tasks, config, client, cost_tracker, force)

    # Evaluator self-responses
    _generate_model_responses(
        config.evaluator_name, config.evaluator_id,
        tasks, config, client, cost_tracker, force,
    )


def load_responses(model_name: str, config: Experiment0fConfig) -> dict[str, str]:
    path = config.generations_dir / f"responses_{model_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"No responses for {model_name} at {path}")
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Per-model generation
# ---------------------------------------------------------------------------

def _generate_model_responses(
    model_name: str,
    model_id: str,
    tasks: list[dict],
    config: Experiment0fConfig,
    client,
    cost_tracker: CostTracker,
    force: bool,
) -> dict[str, str]:
    path = config.generations_dir / f"responses_{model_name}.json"
    existing: dict[str, str] = {}
    if path.exists() and not force:
        with open(path) as f:
            existing = json.load(f)

    missing = [t for t in tasks if t["task_id"] not in existing]
    if not missing:
        print(f"  [{model_name}] all {len(tasks)} responses cached.")
        return existing

    print(f"  [{model_name}] generating {len(missing)} responses...")

    for task in tqdm(missing, desc=model_name):
        if cost_tracker.exceeded():
            print(f"\n  Cost cap reached — stopping {model_name}.")
            break

        messages = _build_messages(model_id, task["prompt"])
        text = _call(model_id, messages, config, client, cost_tracker)
        if text:
            existing[task["task_id"]] = text
            if len(existing) % 20 == 0:
                _save(existing, path)

        time.sleep(config.get_delay(model_id))

    _save(existing, path)
    print(f"  [{model_name}] {len(existing)}/{len(tasks)} responses on disk.")
    return existing


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_messages(model_id: str, user_prompt: str) -> list[dict]:
    if "gemini" in model_id.lower():
        return [{"role": "user", "content": f"[Instructions: {SYSTEM_PROMPT}]\n\n{user_prompt}"}]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]


def _call(
    model_id: str,
    messages: list[dict],
    config: Experiment0fConfig,
    client,
    cost_tracker: CostTracker,
) -> str | None:
    is_gemini = "gemini" in model_id.lower()
    extra = {"thinking": {"type": "disabled"}} if is_gemini else {}

    def _fn():
        return client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=config.temperature,
            max_tokens=config.max_tokens_generation,
            extra_body=extra,
        )

    resp = api_call_with_retry(_fn, max_retries=config.max_retries,
                               retry_delay=config.retry_delay)
    choices = getattr(resp, "choices", None) if resp else None
    text = (choices[0].message.content or "") if choices else ""

    if resp and getattr(resp, "usage", None):
        u = resp.usage
        cost_tracker.add_from_usage(model_id, u.prompt_tokens, u.completion_tokens)
    else:
        cost_tracker.add_estimated(model_id, messages, text)

    if not text or len(text) < 20:
        return None
    return text


def _save(data: dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
