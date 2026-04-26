"""
Generate responses from all configured models for each prompt.

Saves incrementally: after each model finishes all prompts, the
partial responses are merged into responses.json so a crash only
loses the current model's work.

Output: generations_dir/responses.json
[
    {
        "prompt_id": "alpaca_001",
        "dataset": "alpaca",
        "instruction": "...",
        "responses": {
            "llama_8b": "...",
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

from .config import Experiment0Config
from .utils import api_call_with_retry, make_client


def generate_all_responses(
    prompts: list[dict],
    config: Experiment0Config,
    force: bool = False,
) -> list[dict]:
    out_path = config.generations_dir / "responses.json"

    # Build a lookup of already-completed responses
    existing: dict[str, dict] = {}
    if out_path.exists() and not force:
        with open(out_path) as f:
            for rec in json.load(f):
                existing[rec["prompt_id"]] = rec.get("responses", {})

    client = make_client(config)

    # Work model by model to minimise switching overhead
    for model_key, model_id in config.models.items():
        # Check how many prompts still need this model
        missing = [
            p for p in prompts
            if model_key not in existing.get(p["prompt_id"], {})
        ]
        if not missing:
            print(f"  [{model_key}] already complete, skipping.")
            continue

        print(f"\n  [{model_key}] {model_id} — generating {len(missing)} responses...")
        delay = config.get_delay(model_key)

        for i, prompt in enumerate(tqdm(missing, desc=model_key)):
            def _call():
                return client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt["instruction"]},
                    ],
                    temperature=config.temperature,
                    max_tokens=config.max_tokens_generation,
                )

            response = api_call_with_retry(
                _call,
                max_retries=config.max_retries,
                retry_delay=config.retry_delay,
            )

            choices = getattr(response, "choices", None) if response else None
            text = choices[0].message.content if choices else ""
            pid = prompt["prompt_id"]
            if pid not in existing:
                existing[pid] = {}
            existing[pid][model_key] = text

            time.sleep(delay)

            # Save every 20 prompts so crashes don't lose much work
            if (i + 1) % 20 == 0:
                _save(prompts, existing, out_path)

        _save(prompts, existing, out_path)
        print(f"  [{model_key}] done. Saved checkpoint.")

    return _merge(prompts, existing)


def _save(prompts: list[dict], existing: dict[str, dict], path: Path) -> None:
    records = [
        {
            "prompt_id": p["prompt_id"],
            "dataset": p["dataset"],
            "instruction": p["instruction"],
            "responses": existing.get(p["prompt_id"], {}),
        }
        for p in prompts
    ]
    with open(path, "w") as f:
        json.dump(records, f, indent=2)


def _merge(prompts: list[dict], existing: dict[str, dict]) -> list[dict]:
    return [
        {
            "prompt_id": p["prompt_id"],
            "dataset": p["dataset"],
            "instruction": p["instruction"],
            "responses": existing.get(p["prompt_id"], {}),
        }
        for p in prompts
    ]
