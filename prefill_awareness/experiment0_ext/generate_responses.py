"""
Generate replacement assistant turns for Experiment 0 Extension.

For each conversation, each model in config.all_generation_models generates
a replacement for the last assistant turn (replace_turn_idx).

Output: generations_dir/replacements.json
[
    {
        "conv_id":      str,
        "dataset":      str,
        "replacements": {
            "opus_45":   "...",
            "llama_8b":  "...",
            ...
        }
    },
    ...
]
"""

import json
import time
from pathlib import Path

from tqdm import tqdm

from .config import Experiment0ExtConfig
from .utils import api_call_with_retry, make_client


def generate_all_replacements(
    conversations: list[dict],
    config: Experiment0ExtConfig,
    force: bool = False,
) -> list[dict]:
    """
    Generate replacement turns from all models in config.all_generation_models.
    Checkpoints after every model finishes its batch.
    """
    out_path = config.generations_dir / "replacements.json"

    # Load existing replacements keyed by conv_id
    existing: dict[str, dict] = {}
    if out_path.exists() and not force:
        with open(out_path) as f:
            for rec in json.load(f):
                existing[rec["conv_id"]] = rec.get("replacements", {})

    client = make_client(config)

    for model_name, model_id in config.all_generation_models.items():
        missing = [
            c for c in conversations
            if model_name not in existing.get(c["conv_id"], {})
        ]
        if not missing:
            print(f"  [{model_name}] already complete, skipping.")
            continue

        print(f"\n  [{model_name}] {model_id} — generating {len(missing)} replacements...")
        delay = config.get_delay(model_id)

        for i, conv in enumerate(tqdm(missing, desc=model_name)):
            replacement = _generate_one(conv, model_id, config, client)
            cid = conv["conv_id"]
            if cid not in existing:
                existing[cid] = {}
            existing[cid][model_name] = replacement or ""

            time.sleep(delay)

            if (i + 1) % 10 == 0:
                _save(conversations, existing, out_path)

        _save(conversations, existing, out_path)
        print(f"  [{model_name}] done.")

    return _merge(conversations, existing)


def _generate_one(
    conv: dict,
    model_id: str,
    config: Experiment0ExtConfig,
    client,
) -> str | None:
    """
    Send the conversation context up to replace_turn_idx and get a response.
    The last message sent must be a user message.
    """
    replace_idx = conv["replace_turn_idx"]
    context_msgs = conv["messages"][:replace_idx]

    # Validate: context must end with a user turn
    if not context_msgs or context_msgs[-1]["role"] != "user":
        # Walk back to find the last user turn
        last_user = next(
            (i for i in range(len(context_msgs) - 1, -1, -1)
             if context_msgs[i]["role"] == "user"),
            None,
        )
        if last_user is None:
            return None
        context_msgs = context_msgs[: last_user + 1]

    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    messages.extend(context_msgs)

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
    if response is None:
        return None
    choices = getattr(response, "choices", None)
    return choices[0].message.content if choices else None


def _save(
    conversations: list[dict],
    existing: dict[str, dict],
    path: Path,
) -> None:
    records = [
        {
            "conv_id": c["conv_id"],
            "dataset": c["dataset"],
            "replacements": existing.get(c["conv_id"], {}),
        }
        for c in conversations
    ]
    with open(path, "w") as f:
        json.dump(records, f, indent=2)


def _merge(conversations: list[dict], existing: dict[str, dict]) -> list[dict]:
    return [
        {
            "conv_id": c["conv_id"],
            "dataset": c["dataset"],
            "replacements": existing.get(c["conv_id"], {}),
        }
        for c in conversations
    ]
