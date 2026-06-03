"""
Interleaved conversation generation for Experiment 0d.

For each (evaluator, source, task):
  Turn 1: evaluator (sets the trajectory) — cached per (evaluator, task)
  Turn 2: source    (conditioned on evaluator's turn 1)
  Turn 3: evaluator (conditioned on source's turn 2)

For the organic condition (source = evaluator):
  All three turns come from the evaluator.

Saved outputs:
  generations_dir/turn1s_{evaluator}.json   — {task_id: text}
  generations_dir/conversations.json         — list of conversation records
"""

import json
import time
from pathlib import Path

from tqdm import tqdm

from .config import Experiment0dConfig
from .utils import CostTracker, api_call_with_retry, make_client


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_for_evaluators(
    tasks: list[dict],
    evaluator_names: list[str],
    config: Experiment0dConfig,
    cost_tracker: CostTracker,
    force: bool = False,
) -> list[dict]:
    """
    Generate conversations for the given evaluators × all sources + organic.

    Returns full conversations list (all evaluators on disk).
    """
    client = make_client(config)
    tasks = tasks[: config.n_tasks]

    conv_path = config.generations_dir / "conversations.json"
    existing: list[dict] = []
    if conv_path.exists() and not force:
        with open(conv_path) as f:
            existing = json.load(f)

    done: dict[tuple, dict] = {
        (r["evaluator"], r["source"], r["task_id"]): r for r in existing
    }
    print(f"  {len(done)} conversations already on disk.")

    for ev_name in evaluator_names:
        if cost_tracker.exceeded():
            print(f"\n  Cost cap reached. Stopping after {ev_name}.")
            break

        ev_id = config.evaluators.get(ev_name)
        if ev_id is None:
            print(f"  [{ev_name}] not in config.evaluators — skipping.")
            continue

        # ---- Turn 1 (shared across all source conditions for this evaluator)
        turn1s = _load_or_generate_turn1s(ev_name, ev_id, tasks, config, client, cost_tracker, force)

        # ---- Conversations: organic + each source
        sources_to_run = {"self": ev_id, **config.sources}

        for src_name, src_id in sources_to_run.items():
            missing = [
                t for t in tasks
                if (ev_name, src_name, t["task_id"]) not in done
            ]
            if not missing:
                print(f"  [{ev_name} / {src_name}] all cached.")
                continue

            print(f"  [{ev_name} / {src_name}] generating {len(missing)} conversations...")

            for task in tqdm(missing, desc=f"{ev_name}/{src_name}"):
                if cost_tracker.exceeded():
                    break
                if cost_tracker.approaching():
                    print(f"\n  *** Approaching cap. {cost_tracker.report()} ***")

                conv = _generate_one(
                    task, ev_name, ev_id, src_name, src_id,
                    turn1s, config, client, cost_tracker,
                )
                if conv is not None:
                    key = (ev_name, src_name, task["task_id"])
                    done[key] = conv
                    if len(done) % 15 == 0:
                        _save(list(done.values()), conv_path)

                time.sleep(config.get_delay(src_id))

            _save(list(done.values()), conv_path)

    return list(done.values())


def load_conversations(config: Experiment0dConfig) -> list[dict]:
    path = config.generations_dir / "conversations.json"
    if not path.exists():
        raise FileNotFoundError(f"No conversations at {path}. Run generate step first.")
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Turn 1 per evaluator
# ---------------------------------------------------------------------------

def _load_or_generate_turn1s(
    ev_name: str,
    ev_id: str,
    tasks: list[dict],
    config: Experiment0dConfig,
    client,
    cost_tracker: CostTracker,
    force: bool,
) -> dict[str, str]:
    path = config.generations_dir / f"turn1s_{ev_name}.json"

    # Try to reuse from 0c (only Opus 4.6 shares the same generation system prompt)
    if not path.exists() and ev_name == "opus_46" and not force:
        prev = config.prev_experiment_dir / "generations" / "turn1s.json"
        if prev.exists():
            print(f"  [{ev_name}] reusing turn 1s from {prev}")
            with open(prev) as f:
                cached = json.load(f)
            # Save under 0d path for consistency
            with open(path, "w") as f:
                json.dump(cached, f, indent=2)

    existing: dict[str, str] = {}
    if path.exists() and not force:
        with open(path) as f:
            existing = json.load(f)

    missing = [t for t in tasks if t["task_id"] not in existing]
    if not missing:
        print(f"  [{ev_name}] all {len(tasks)} turn 1s cached.")
        return existing

    print(f"  [{ev_name}] generating {len(missing)} turn 1s...")
    for task in tqdm(missing, desc=f"Turn1/{ev_name}"):
        messages = [
            {"role": "system", "content": config.generation_system_prompt},
            {"role": "user",   "content": task["description"]},
        ]
        text = _call(ev_id, messages, config, client, cost_tracker)
        if text:
            existing[task["task_id"]] = text
            with open(path, "w") as f:
                json.dump(existing, f, indent=2)
        time.sleep(config.get_delay(ev_id))

    print(f"  [{ev_name}] turn 1s: {len(existing)}/{len(tasks)}")
    return existing


# ---------------------------------------------------------------------------
# Single conversation
# ---------------------------------------------------------------------------

def _generate_one(
    task: dict,
    ev_name: str,
    ev_id: str,
    src_name: str,
    src_id: str,
    turn1s: dict[str, str],
    config: Experiment0dConfig,
    client,
    cost_tracker: CostTracker,
) -> dict | None:
    task_id    = task["task_id"]
    tool_outs  = task["tool_outputs"]
    is_organic = (src_name == "self")
    is_tampered = not is_organic

    turn1 = turn1s.get(task_id)
    if not turn1:
        print(f"  [{ev_name}] missing turn 1 for {task_id}.")
        return None

    ctx_for_turn2 = [
        {"role": "system",    "content": config.generation_system_prompt},
        {"role": "user",      "content": task["description"]},
        {"role": "assistant", "content": turn1},
        {"role": "user",      "content": tool_outs[0]},
    ]

    # Turn 2: source (or evaluator for organic)
    turn2_model = ev_id if is_organic else src_id
    turn2 = _call(turn2_model, ctx_for_turn2, config, client, cost_tracker)
    if not turn2:
        return None

    time.sleep(config.get_delay(turn2_model))

    # Turn 3: evaluator conditioned on source's turn 2
    ctx_for_turn3 = ctx_for_turn2 + [
        {"role": "assistant", "content": turn2},
        {"role": "user",      "content": tool_outs[1]},
    ]
    turn3 = _call(ev_id, ctx_for_turn3, config, client, cost_tracker)
    if not turn3:
        return None

    messages = [
        {"role": "system",    "content": config.generation_system_prompt},
        {"role": "user",      "content": task["description"]},
        {"role": "assistant", "content": turn1},
        {"role": "user",      "content": tool_outs[0]},
        {"role": "assistant", "content": turn2},
        {"role": "user",      "content": tool_outs[1]},
        {"role": "assistant", "content": turn3},
    ]

    tokens_est = sum(len(m.get("content", "") or "") for m in messages) // 4
    if tokens_est > config.max_conversation_tokens:
        print(f"  [{ev_name}/{src_name}/{task_id}] skipped: ~{tokens_est} tokens")
        return None

    return {
        "evaluator":    ev_name,
        "source":       src_name,
        "task_id":      task_id,
        "is_tampered":  is_tampered,
        "messages":     messages,
        "turn1":        turn1,
        "turn2":        turn2,
        "turn3":        turn3,
        "tokens_est":   tokens_est,
    }


# ---------------------------------------------------------------------------
# Low-level API call
# ---------------------------------------------------------------------------

def _call(
    model_id: str,
    messages: list[dict],
    config: Experiment0dConfig,
    client,
    cost_tracker: CostTracker,
) -> str | None:
    # Gemini fallback: fold system message into first user message if needed
    msgs = _maybe_fold_system(model_id, messages)

    def _fn():
        return client.chat.completions.create(
            model=model_id,
            messages=msgs,
            temperature=config.temperature,
            max_tokens=config.get_max_tokens_generation(model_id),
        )

    resp = api_call_with_retry(_fn, max_retries=config.max_retries,
                               retry_delay=config.retry_delay)
    choices = getattr(resp, "choices", None) if resp else None
    text = (choices[0].message.content or "") if choices else ""

    if resp and getattr(resp, "usage", None):
        u = resp.usage
        cost_tracker.add_from_usage(model_id, u.prompt_tokens, u.completion_tokens)
    else:
        cost_tracker.add_estimated(model_id, msgs, text)

    if not text or len(text) < 20:
        return None
    if any(p in text.lower() for p in ["i can't help", "i cannot help", "i'm not able to"]):
        print(f"  [{model_id}] refusal — skipping.")
        return None
    return text


def _maybe_fold_system(model_id: str, messages: list[dict]) -> list[dict]:
    """
    Some models (Gemini via OpenRouter) may reject system-role messages.
    If the model is Gemini, fold the system message into the first user turn.
    """
    if "gemini" not in model_id.lower():
        return messages
    if not messages or messages[0]["role"] != "system":
        return messages
    sys_content = messages[0]["content"]
    rest = messages[1:]
    if rest and rest[0]["role"] == "user":
        merged = f"[Instructions: {sys_content}]\n\n{rest[0]['content']}"
        return [{"role": "user", "content": merged}] + rest[1:]
    return rest


def _save(conversations: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        json.dump(conversations, f, indent=2)
