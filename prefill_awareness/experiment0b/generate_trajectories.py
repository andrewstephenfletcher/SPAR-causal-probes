"""
Generate agentic coding trajectories for Experiment 0b.

For each (model, task) pair, we run a 3-turn simulated agentic coding session:
  1. Model reads the task description and starts exploring
  2. Model receives tool_output[0] and analyses the issue
  3. Model receives tool_output[1] and proposes/implements a fix

The tool outputs are identical across all models, so the only variation
is in the assistant turns — giving a clean stylistic fingerprint.

Output: generations_dir/trajectories.json
[
    {
        "task_id":    "agentic_001",
        "model_name": "opus_45",
        "turns":      ["turn1 text", "turn2 text", "turn3 text"]
    },
    ...
]
"""

import json
import time
from pathlib import Path

from tqdm import tqdm

from .config import Experiment0bConfig
from .task_scenarios import AGENTIC_TASKS
from .utils import api_call_with_retry, make_client


def generate_all_trajectories(
    config: Experiment0bConfig,
    force: bool = False,
) -> dict[str, dict[str, list[str]]]:
    """
    Generate 3-turn trajectories for all models × all tasks.

    Returns: {model_name: {task_id: [turn1, turn2, turn3]}}
    """
    out_path = config.generations_dir / "trajectories.json"
    existing: list[dict] = []
    if out_path.exists() and not force:
        with open(out_path) as f:
            existing = json.load(f)

    # Build lookup: (model_name, task_id) → turns
    done: dict[tuple, list[str]] = {
        (r["model_name"], r["task_id"]): r["turns"]
        for r in existing
    }

    client = make_client(config)
    tasks = AGENTIC_TASKS[: config.n_agentic_tasks]

    for model_name, model_id in config.all_generation_models.items():
        missing = [(t, model_id) for t in tasks if (model_name, t["task_id"]) not in done]
        if not missing:
            print(f"  [{model_name}] all trajectories cached.")
            continue

        print(f"\n  [{model_name}] {model_id} — generating {len(missing)} trajectories...")
        delay = config.get_delay(model_id)

        for task, mid in tqdm(missing, desc=model_name):
            turns = _generate_trajectory(task, mid, config, client)
            if turns:
                done[(model_name, task["task_id"])] = turns

            time.sleep(delay)

        # Save checkpoint after each model
        _save(done, out_path)
        print(f"  [{model_name}] done. Checkpoint saved.")

    return _build_lookup(done)


def _generate_trajectory(
    task: dict,
    model_id: str,
    config: Experiment0bConfig,
    client,
) -> list[str] | None:
    """
    Runs the 3-turn exchange for one (model, task) pair.
    Returns [turn1, turn2, turn3] or None if generation failed.
    """
    system = config.generation_system_prompt
    tool_outs = task["tool_outputs"]
    turns = []

    # Build up the conversation incrementally
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": task["description"]},
    ]

    for turn_idx in range(3):
        def _call():
            return client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=config.temperature,
                max_tokens=config.max_tokens_generation,
            )

        response = api_call_with_retry(
            _call,
            max_retries=config.max_retries,
            retry_delay=config.retry_delay,
        )

        choices = getattr(response, "choices", None) if response else None
        text = (choices[0].message.content or "") if choices else ""
        if not text:
            print(f"    Empty response at turn {turn_idx + 1} for {task['task_id']}")
            return None

        turns.append(text)
        messages.append({"role": "assistant", "content": text})

        # After turns 1 and 2, inject the corresponding tool output
        if turn_idx < 2:
            messages.append({"role": "user", "content": tool_outs[turn_idx]})

    return turns


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _save(done: dict[tuple, list[str]], path: Path) -> None:
    records = [
        {"model_name": model_name, "task_id": task_id, "turns": turns}
        for (model_name, task_id), turns in done.items()
    ]
    with open(path, "w") as f:
        json.dump(records, f, indent=2)


def _build_lookup(done: dict[tuple, list[str]]) -> dict[str, dict[str, list[str]]]:
    """Returns {model_name: {task_id: [turn1, turn2, turn3]}}."""
    result: dict[str, dict[str, list[str]]] = {}
    for (model_name, task_id), turns in done.items():
        result.setdefault(model_name, {})[task_id] = turns
    return result


def load_trajectories(config: Experiment0bConfig) -> dict[str, dict[str, list[str]]]:
    """Load pre-generated trajectories from disk."""
    path = config.generations_dir / "trajectories.json"
    if not path.exists():
        raise FileNotFoundError(f"Trajectories not found at {path}. Run generate step first.")
    with open(path) as f:
        records = json.load(f)
    return _build_lookup({(r["model_name"], r["task_id"]): r["turns"] for r in records})
