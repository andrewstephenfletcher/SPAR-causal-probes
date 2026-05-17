"""
Prompt loading for Experiment 11 (Unified Prefill-Awareness Data Collection).

Each loader returns a list of dicts:
    prompt_id   — unique integer within the dataset
    dataset     — "bigcodebench" | "oasst1" | "gpqa"
    instruction — plain-text prompt string
    split       — "train" | "val" | "test"

GPQA is loaded as free-form (question text only, no MCQ choices) unlike Exp 10.
Split assignment is deterministic hash-based, identical to Exp 10.
"""

import hashlib
from typing import Literal

from .config import Experiment11Config


Split = Literal["train", "val", "test"]


def _assign_split(prompt_id: int, train_frac: float, val_frac: float) -> Split:
    """Deterministic, hash-based split assignment (identical to Exp 10)."""
    h = int(hashlib.md5(str(prompt_id).encode()).hexdigest(), 16) % 1000
    if h < train_frac * 1000:
        return "train"
    if h < (train_frac + val_frac) * 1000:
        return "val"
    return "test"


def load_bigcodebench(config: Experiment11Config) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("bigcode/bigcodebench", split="v0.1.2")
    prompts = []
    for i, row in enumerate(ds):
        if i >= config.n_prompts_per_dataset:
            break
        prompts.append({
            "prompt_id": i,
            "dataset": "bigcodebench",
            "instruction": row["instruct_prompt"].strip(),
            "split": _assign_split(i, config.train_frac, config.val_frac),
        })
    return prompts


def load_oasst1(config: Experiment11Config) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("OpenAssistant/oasst1", split="train")
    roots = [
        r for r in ds
        if r["role"] == "prompter"
        and r["parent_id"] is None
        and r.get("lang", "en") == "en"
        and not r.get("deleted", False)
    ]
    prompts = []
    for i, row in enumerate(roots):
        if i >= config.n_prompts_per_dataset:
            break
        prompts.append({
            "prompt_id": i,
            "dataset": "oasst1",
            "instruction": row["text"].strip(),
            "split": _assign_split(i, config.train_frac, config.val_frac),
        })
    return prompts


def load_gpqa(config: Experiment11Config) -> list[dict]:
    """
    Loads GPQA as free-form questions (question text only, no MCQ choices).
    Requires HF_TOKEN and dataset access at https://huggingface.co/datasets/Idavidrein/gpqa
    """
    from datasets import load_dataset
    ds = load_dataset("Idavidrein/gpqa", "gpqa_main", split="train")
    prompts = []
    for i, row in enumerate(ds):
        if i >= config.n_prompts_per_dataset:
            break
        prompts.append({
            "prompt_id": i,
            "dataset": "gpqa",
            "instruction": row["Question"].strip(),
            "split": _assign_split(i, config.train_frac, config.val_frac),
        })
    return prompts


_LOADERS = {
    "bigcodebench": load_bigcodebench,
    "oasst1":       load_oasst1,
    "gpqa":         load_gpqa,
}


def load_all_prompts(config: Experiment11Config) -> dict[str, list[dict]]:
    """Return {dataset_name: [prompt_dicts]} for all configured datasets."""
    result = {}
    for name in config.datasets:
        print(f"  Loading {name}...")
        result[name] = _LOADERS[name](config)
        splits = {r["split"] for r in result[name]}
        counts = {s: sum(1 for r in result[name] if r["split"] == s) for s in splits}
        print(f"    {len(result[name])} prompts  {counts}")
    return result
