"""
Single-turn generation for Experiment 0e.

For each (model, task): one call → response text.

Source responses depend only on (source_model, task) and are shared across all
evaluators.  Evaluator self-responses are generated once per (evaluator, task).

Cheap evaluators (sonnet_45, gpt_4o_mini, gemini_flash) are also sources,
so their self-responses ARE their source responses — no extra generation needed.

Saved outputs:
  generations_dir/responses_{model_name}.json  — {task_id: response_text}
"""

import json
import time
from pathlib import Path

from tqdm import tqdm

from .config import Experiment0eConfig
from .utils import CostTracker, api_call_with_retry, make_client

SYSTEM_PROMPT = "You are a helpful assistant."


def generate_all(
    tasks: list[dict],
    evaluator_names: list[str],
    config: Experiment0eConfig,
    cost_tracker: CostTracker,
    force: bool = False,
) -> None:
    """
    Generate source responses (cheap models only) + evaluator self-responses.
    Results saved to generations_dir/responses_{name}.json.
    """
    client = make_client(config)

    # Sources first (cheap, needed by everyone)
    for src_name in config.sources:
        _generate_model_responses(src_name, config.sources[src_name],
                                  tasks, config, client, cost_tracker, force)

    # Evaluators — skip if already generated as source
    for ev_name in evaluator_names:
        if ev_name in config.sources:
            print(f"  [{ev_name}] self-responses = source responses (reusing)")
            continue
        ev_id = config.evaluators.get(ev_name)
        if ev_id is None:
            continue
        _generate_model_responses(ev_name, ev_id,
                                  tasks, config, client, cost_tracker, force)


def load_responses(model_name: str, config: Experiment0eConfig) -> dict[str, str]:
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
    config: Experiment0eConfig,
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

        messages = _build_messages(model_id, SYSTEM_PROMPT, task["prompt"])
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
# Low-level helpers
# ---------------------------------------------------------------------------

def _build_messages(model_id: str, system: str, user: str) -> list[dict]:
    if "gemini" in model_id.lower():
        return [{"role": "user", "content": f"[Instructions: {system}]\n\n{user}"}]
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]


def _call(
    model_id: str,
    messages: list[dict],
    config: Experiment0eConfig,
    client,
    cost_tracker: CostTracker,
) -> str | None:
    extra = {"thinking": {"type": "disabled"}} if config.is_thinking_model(model_id) else {}

    def _fn():
        return client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=config.temperature,
            max_tokens=config.get_max_tokens_generation(model_id),
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
