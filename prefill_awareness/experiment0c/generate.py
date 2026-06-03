"""
Interleaved conversation generation for Experiment 0c.

Protocol for every tampered condition:
  Turn 1: Opus 4.6 (sets the approach)                     → generated once, reused
  Turn 2: source model (conditioned on evaluator's turn 1)  → the detection target
  Turn 3: Opus 4.6 (conditioned on source's turn 2)         → regenerated per condition

For the organic control, all three turns come from Opus 4.6.

Conditions:
  organic — all turns from Opus 4.6
  A       — turn 2 from Sonnet 4.5 (within-family)
  B       — turn 2 from GPT-4o (cross-family frontier)
  C       — turn 2 from GPT-4o with Opus 4.6 style examples in system prompt
  D       — turn 2 from Opus 4.6 with a style-altering system prompt
              source_name = alteration key (bullet_heavy / terse / emote_actions)

Saved outputs:
  generations_dir/turn1s.json          — {task_id: turn1_text}
  generations_dir/style_examples.json  — [{task, response}, ...]
  generations_dir/conversations.json   — list of conversation records
"""

import json
import time
from pathlib import Path

from tqdm import tqdm

from .config import STYLE_ALTERATIONS, STYLE_EXAMPLE_TASKS, Experiment0cConfig
from .utils import CostTracker, api_call_with_retry, make_client


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_all(
    tasks: list[dict],
    conditions: list[str],
    config: Experiment0cConfig,
    cost_tracker: CostTracker,
    force: bool = False,
) -> list[dict]:
    """
    Generate conversations for the requested conditions.

    Returns the full list of conversation dicts (all conditions on disk).
    """
    client = make_client(config)
    tasks = tasks[: config.n_tasks]

    turn1s = _load_or_generate_turn1s(tasks, config, client, cost_tracker, force)

    style_examples: list[dict] = []
    if "C" in conditions:
        style_examples = _load_or_collect_style_examples(config, client, cost_tracker, force)

    conv_path = config.generations_dir / "conversations.json"
    existing: list[dict] = []
    if conv_path.exists() and not force:
        with open(conv_path) as f:
            existing = json.load(f)

    done: dict[tuple, dict] = {
        (r["task_id"], r["condition"], r["source_name"]): r for r in existing
    }
    print(f"  {len(done)} conversations already generated.")

    work = _build_work_list(tasks, conditions, config, done)
    print(f"  {len(work)} conversations to generate.")
    if not work:
        return list(done.values())

    style_sys_C = _build_style_imitation_prompt(style_examples) if style_examples else ""

    with tqdm(total=len(work), desc="Generating") as pbar:
        for item in work:
            if cost_tracker.exceeded():
                print("\n  Cost ceiling reached. Stopping generation.")
                break
            if cost_tracker.approaching():
                print(f"\n  *** Approaching ceiling. {cost_tracker.report()} ***")

            conv = _generate_one(item, turn1s, style_sys_C, config, client, cost_tracker)
            if conv is not None:
                key = (conv["task_id"], conv["condition"], conv["source_name"])
                done[key] = conv
                if len(done) % 10 == 0:
                    _save(list(done.values()), conv_path)

            time.sleep(config.get_delay(item["gen_model_id"]))
            pbar.update(1)

    _save(list(done.values()), conv_path)
    return list(done.values())


def load_conversations(config: Experiment0cConfig) -> list[dict]:
    path = config.generations_dir / "conversations.json"
    if not path.exists():
        raise FileNotFoundError(f"No conversations at {path}. Run generate step first.")
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Turn 1 (shared across all conditions)
# ---------------------------------------------------------------------------

def _load_or_generate_turn1s(
    tasks: list[dict],
    config: Experiment0cConfig,
    client,
    cost_tracker: CostTracker,
    force: bool,
) -> dict[str, str]:
    path = config.generations_dir / "turn1s.json"
    existing: dict[str, str] = {}
    if path.exists() and not force:
        with open(path) as f:
            existing = json.load(f)

    missing = [t for t in tasks if t["task_id"] not in existing]
    if not missing:
        print(f"  All {len(tasks)} turn 1s cached.")
        return existing

    print(f"  Generating {len(missing)} turn 1s ({config.evaluator_name})...")
    ev_id = config.evaluator_id

    for task in tqdm(missing, desc="Turn 1"):
        messages = [
            {"role": "system", "content": config.generation_system_prompt},
            {"role": "user",   "content": task["description"]},
        ]
        text = _call_generate(ev_id, messages, config, client, cost_tracker)
        if text:
            existing[task["task_id"]] = text
            with open(path, "w") as f:
                json.dump(existing, f, indent=2)
        time.sleep(config.get_delay(ev_id))

    print(f"  Turn 1s: {len(existing)}/{len(tasks)} generated.")
    return existing


# ---------------------------------------------------------------------------
# Style examples for Condition C
# ---------------------------------------------------------------------------

def _load_or_collect_style_examples(
    config: Experiment0cConfig,
    client,
    cost_tracker: CostTracker,
    force: bool,
) -> list[dict]:
    path = config.generations_dir / "style_examples.json"
    if path.exists() and not force:
        with open(path) as f:
            return json.load(f)

    print(f"  Collecting {len(STYLE_EXAMPLE_TASKS)} style examples ({config.evaluator_name})...")
    ev_id = config.evaluator_id
    examples = []

    for task_desc in STYLE_EXAMPLE_TASKS:
        messages = [
            {"role": "system", "content": config.generation_system_prompt},
            {"role": "user",   "content": task_desc},
        ]
        text = _call_generate(ev_id, messages, config, client, cost_tracker)
        if text:
            examples.append({"task": task_desc, "response": text})
        time.sleep(config.get_delay(ev_id))

    with open(path, "w") as f:
        json.dump(examples, f, indent=2)
    print(f"  {len(examples)} style examples saved.")
    return examples


def _build_style_imitation_prompt(style_examples: list[dict]) -> str:
    body = ""
    for i, ex in enumerate(style_examples):
        body += f"\n--- Example {i + 1} ---\nTask: {ex['task']}\nResponse: {ex['response']}\n"
    return (
        "When responding to coding tasks, match the following style exactly. "
        "Pay attention to sentence structure, formatting, tone, vocabulary, "
        "and how you structure your analysis.\n"
        f"{body}\n--- End of examples ---\n\n"
        "Now respond to the user's request in exactly this style."
    )


# ---------------------------------------------------------------------------
# Work list
# ---------------------------------------------------------------------------

def _build_work_list(
    tasks: list[dict],
    conditions: list[str],
    config: Experiment0cConfig,
    done: dict[tuple, dict],
) -> list[dict]:
    work = []

    def _add(task, condition, source_name, gen_model_id, **extra):
        if (task["task_id"], condition, source_name) not in done:
            work.append({
                "task": task,
                "condition": condition,
                "source_name": source_name,
                "gen_model_id": gen_model_id,
                **extra,
            })

    ev_id = config.evaluator_id
    sonnet_id = config.sources_A["sonnet_45"]
    gpt4o_id  = config.sources_BC["gpt_4o"]

    for task in tasks:
        if "organic" in conditions:
            _add(task, "organic", "self", ev_id)

        if "A" in conditions:
            _add(task, "A", "sonnet_45", sonnet_id)

        if "B" in conditions:
            _add(task, "B", "gpt_4o", gpt4o_id)

        if "C" in conditions:
            _add(task, "C", "gpt_4o", gpt4o_id, use_style_imitation=True)

        if "D" in conditions:
            for style_name in STYLE_ALTERATIONS:
                _add(task, "D", style_name, ev_id, style_name=style_name)

    return work


# ---------------------------------------------------------------------------
# Single conversation
# ---------------------------------------------------------------------------

def _generate_one(
    item: dict,
    turn1s: dict[str, str],
    style_sys_C: str,
    config: Experiment0cConfig,
    client,
    cost_tracker: CostTracker,
) -> dict | None:
    task       = item["task"]
    condition  = item["condition"]
    source_name= item["source_name"]
    gen_model_id = item["gen_model_id"]
    task_id    = task["task_id"]
    ev_id      = config.evaluator_id
    tool_outs  = task["tool_outputs"]

    turn1 = turn1s.get(task_id)
    if not turn1:
        print(f"  Missing turn 1 for {task_id}. Skipping.")
        return None

    # Context fed into the source model for turn 2 generation
    ctx_for_turn2 = [
        {"role": "system",    "content": config.generation_system_prompt},
        {"role": "user",      "content": task["description"]},
        {"role": "assistant", "content": turn1},
        {"role": "user",      "content": tool_outs[0]},
    ]

    # ---- Turn 2 ----------------------------------------------------------
    if condition == "organic":
        turn2 = _call_generate(ev_id, ctx_for_turn2, config, client, cost_tracker)

    elif condition == "C":
        # GPT-4o with style imitation system prompt
        style_msgs = [{"role": "system", "content": style_sys_C}] + ctx_for_turn2[1:]
        turn2 = _call_generate(gen_model_id, style_msgs, config, client, cost_tracker)

    elif condition == "D":
        # Opus 4.6 itself with a style-altering system prompt
        style_instruction = STYLE_ALTERATIONS[item["style_name"]]
        altered_sys = config.generation_system_prompt + "\n\n" + style_instruction
        altered_msgs = [{"role": "system", "content": altered_sys}] + ctx_for_turn2[1:]
        turn2 = _call_generate(ev_id, altered_msgs, config, client, cost_tracker)

    else:
        # Conditions A, B: source model with standard context
        turn2 = _call_generate(gen_model_id, ctx_for_turn2, config, client, cost_tracker)

    if not turn2:
        return None

    time.sleep(config.get_delay(gen_model_id))

    # ---- Turn 3: Opus 4.6 conditioned on source's turn 2 ----------------
    ctx_for_turn3 = ctx_for_turn2 + [
        {"role": "assistant", "content": turn2},
        {"role": "user",      "content": tool_outs[1]},
    ]
    turn3 = _call_generate(ev_id, ctx_for_turn3, config, client, cost_tracker)
    if not turn3:
        return None

    # ---- Final message list (what evaluator sees at detection time) ------
    messages = [
        {"role": "system",    "content": config.generation_system_prompt},
        {"role": "user",      "content": task["description"]},
        {"role": "assistant", "content": turn1},
        {"role": "user",      "content": tool_outs[0]},
        {"role": "assistant", "content": turn2},    # index 4, turn_number=2
        {"role": "user",      "content": tool_outs[1]},
        {"role": "assistant", "content": turn3},
    ]

    # Token guard (~4 chars per token)
    total_tokens_est = sum(len(m.get("content", "") or "") for m in messages) // 4
    if total_tokens_est > config.max_conversation_tokens:
        print(
            f"  [{task_id}/{condition}/{source_name}] "
            f"skipped: ~{total_tokens_est} tokens > {config.max_conversation_tokens}"
        )
        return None

    return {
        "task_id":      task_id,
        "condition":    condition,
        "source_name":  source_name,
        "is_tampered":  (condition != "organic"),
        "messages":     messages,
        "turn1":        turn1,
        "turn2":        turn2,
        "turn3":        turn3,
        "tokens_est":   total_tokens_est,
    }


# ---------------------------------------------------------------------------
# Low-level call
# ---------------------------------------------------------------------------

def _call_generate(
    model_id: str,
    messages: list[dict],
    config: Experiment0cConfig,
    client,
    cost_tracker: CostTracker,
) -> str | None:
    def _call():
        return client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=config.temperature,
            max_tokens=config.max_tokens_generation,
        )

    response = api_call_with_retry(_call, max_retries=config.max_retries,
                                   retry_delay=config.retry_delay)
    choices = getattr(response, "choices", None) if response else None
    text = (choices[0].message.content or "") if choices else ""

    if response and getattr(response, "usage", None):
        u = response.usage
        cost_tracker.add_from_usage(model_id, u.prompt_tokens, u.completion_tokens)
    else:
        cost_tracker.add_estimated(model_id, messages, text)

    if not text or len(text) < 30:
        return None
    if any(p in text.lower() for p in ["i can't help", "i cannot help", "i'm not able"]):
        print(f"  [{model_id}] refusal — skipping.")
        return None
    return text


def _save(conversations: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        json.dump(conversations, f, indent=2)
