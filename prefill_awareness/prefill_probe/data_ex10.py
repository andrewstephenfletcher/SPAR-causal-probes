"""
Prompt loading for Experiment 10 (Probe Generalisation).

Each loader returns a list of dicts with keys:
    prompt_id  — unique integer within the dataset
    dataset    — "bigcodebench" | "oasst1" | "gpqa"
    instruction — the plain-text prompt string
    split      — "train" | "val" | "test"

The split is assigned deterministically by prompt_id so that the same prompt
always lands in the same split regardless of how many prompts are loaded.
"""

import hashlib
from typing import Literal

from .config import Experiment10Config


Split = Literal["train", "val", "test"]


def _assign_split(prompt_id: int, train_frac: float, val_frac: float) -> Split:
    """Deterministic, hash-based split assignment."""
    h = int(hashlib.md5(str(prompt_id).encode()).hexdigest(), 16) % 1000
    if h < train_frac * 1000:
        return "train"
    if h < (train_frac + val_frac) * 1000:
        return "val"
    return "test"


def load_bigcodebench(config: Experiment10Config) -> list[dict]:
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


def load_oasst1(config: Experiment10Config) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("OpenAssistant/oasst1", split="train")
    # Root-level English prompter messages only
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


def load_gpqa(config: Experiment10Config) -> list[dict]:
    """
    Loads GPQA (graduate-level questions).  Requires HF_TOKEN and dataset
    access to be granted at https://huggingface.co/datasets/Idavidrein/gpqa
    """
    from datasets import load_dataset
    ds = load_dataset("Idavidrein/gpqa", "gpqa_main", split="train")
    prompts = []
    for i, row in enumerate(ds):
        if i >= config.n_prompts_per_dataset:
            break
        # Format as a four-choice question without revealing the correct answer
        q = row["Question"].strip()
        choices = [
            row["Correct Answer"],
            row["Incorrect Answer 1"],
            row["Incorrect Answer 2"],
            row["Incorrect Answer 3"],
        ]
        # Shuffle choices deterministically by prompt_id so they're not always
        # in the same order (correct answer is always listed first in the raw data)
        import random
        rng = random.Random(i)
        rng.shuffle(choices)
        instruction = (
            f"{q}\n\n"
            f"A) {choices[0]}\n"
            f"B) {choices[1]}\n"
            f"C) {choices[2]}\n"
            f"D) {choices[3]}\n\n"
            "Answer with the letter of the correct choice."
        )
        prompts.append({
            "prompt_id": i,
            "dataset": "gpqa",
            "instruction": instruction,
            "split": _assign_split(i, config.train_frac, config.val_frac),
        })
    return prompts


_LOADERS = {
    "bigcodebench": load_bigcodebench,
    "oasst1":       load_oasst1,
    "gpqa":         load_gpqa,
}


def load_all_prompts(config: Experiment10Config) -> dict[str, list[dict]]:
    """Return {dataset_name: [prompt_dicts]} for all configured datasets."""
    result = {}
    for name in config.datasets:
        print(f"  Loading {name}...")
        result[name] = _LOADERS[name](config)
        splits = {r["split"] for r in result[name]}
        counts = {s: sum(1 for r in result[name] if r["split"] == s) for s in splits}
        print(f"    {len(result[name])} prompts  {counts}")
    return result
