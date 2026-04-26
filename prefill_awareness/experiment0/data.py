"""
Prompt loading for Experiment 0.

Preferentially loads from Experiment 3 outputs (prompts_all.json), subsampling
50 per dataset. Falls back to loading directly from HuggingFace if not found.
"""

import json
import random
from pathlib import Path

from .config import Experiment0Config

_EX3_PROMPTS = Path("outputs/experiment3/generations/prompts_all.json")


def load_prompts(config: Experiment0Config) -> list[dict]:
    """Load 50 prompts per dataset, save to generations_dir/prompts.json."""
    out_path = config.generations_dir / "prompts.json"
    if out_path.exists():
        print(f"  Loading existing prompts from {out_path}")
        with open(out_path) as f:
            return json.load(f)

    if _EX3_PROMPTS.exists():
        prompts = _load_from_ex3(config)
    else:
        prompts = _load_from_hf(config)

    with open(out_path, "w") as f:
        json.dump(prompts, f, indent=2)
    print(f"  Saved {len(prompts)} prompts → {out_path}")
    return prompts


def _load_from_ex3(config: Experiment0Config) -> list[dict]:
    print(f"  Loading prompts from Experiment 3: {_EX3_PROMPTS}")
    with open(_EX3_PROMPTS) as f:
        all_prompts = json.load(f)

    rng = random.Random(42)
    result = []
    for ds in config.datasets:
        pool = [p for p in all_prompts if p["dataset"] == ds]
        sample = rng.sample(pool, min(config.n_prompts_per_dataset, len(pool)))
        # Reassign sequential IDs for Experiment 0
        for i, p in enumerate(sample):
            result.append({
                "prompt_id": f"{ds}_{i+1:03d}",
                "dataset": ds,
                "instruction": p["instruction"],
            })
        print(f"    {ds}: {len(sample)} prompts")

    return result


def _load_from_hf(config: Experiment0Config) -> list[dict]:
    from datasets import load_dataset

    print("  Experiment 3 outputs not found; loading from HuggingFace.")
    result = []

    for ds in config.datasets:
        if ds == "alpaca":
            prompts = _load_alpaca(config.n_prompts_per_dataset)
        elif ds == "oasst1":
            prompts = _load_oasst1(config.n_prompts_per_dataset)
        elif ds == "mmlu":
            prompts = _load_mmlu(config.n_prompts_per_dataset)
        else:
            raise ValueError(f"Unknown dataset: {ds}")
        result.extend(prompts)
        print(f"    {ds}: {len(prompts)} prompts")

    return result


def _load_alpaca(n: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    pool = [r for r in ds if not r["input"] and len(r["instruction"]) > 20]
    random.Random(42).shuffle(pool)
    return [
        {"prompt_id": f"alpaca_{i+1:03d}", "dataset": "alpaca", "instruction": r["instruction"]}
        for i, r in enumerate(pool[:n])
    ]


def _load_oasst1(n: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("OpenAssistant/oasst1", split="train")
    pool = [
        r for r in ds
        if r["parent_id"] is None
        and r["role"] == "prompter"
        and r.get("lang", "") == "en"
        and 20 < len(r["text"]) < 500
    ]
    random.Random(42).shuffle(pool)
    return [
        {"prompt_id": f"oasst1_{i+1:03d}", "dataset": "oasst1", "instruction": r["text"]}
        for i, r in enumerate(pool[:n])
    ]


def _load_mmlu(n: int) -> list[dict]:
    from datasets import load_dataset
    subjects = [
        "high_school_biology", "computer_science", "philosophy",
        "us_history", "college_mathematics", "high_school_physics",
    ]
    per_subject = max(1, n // len(subjects))
    result = []
    rng = random.Random(42)
    for subj in subjects:
        ds = load_dataset("cais/mmlu", subj, split="test")
        pool = list(ds)
        rng.shuffle(pool)
        for r in pool[:per_subject]:
            choices = r["choices"]
            instr = (
                f"Answer the following question with a brief explanation:\n"
                f"Question: {r['question']}\n"
                f"A) {choices[0]}  B) {choices[1]}  C) {choices[2]}  D) {choices[3]}"
            )
            result.append({
                "prompt_id": f"mmlu_{subj}_{len(result)+1:03d}",
                "dataset": "mmlu",
                "instruction": instr,
            })
    return result[:n]
