"""
Dataset preparation for Experiment 3.

Three datasets (300 prompts each):
  - Alpaca: reuse the 300 prompts from Experiment 1
  - OASST1: 300 English first-turn user messages
  - MMLU:   300 multiple-choice questions across 6 subjects

All prompts share the same dict schema:
  {
    "prompt_id": str,       # "alpaca_0001", "oasst1_0001", "mmlu_0001"
    "dataset":   str,       # "alpaca", "oasst1", "mmlu"
    "instruction": str,
    "split":     str,       # "train", "val", "test"
    "ex1_prompt_id": int,   # only present for alpaca (for Ex1 response reuse)
  }

Combined master file: generations_dir_ex3 / "prompts_all.json"
"""

import json
import random

from .config import Config, Experiment3Config


# ---------------------------------------------------------------------------
# Alpaca (reuse Experiment 1)
# ---------------------------------------------------------------------------

def load_alpaca_prompts(
    ex3_config: Experiment3Config,
    ex1_config: Config,
) -> list[dict]:
    """
    Return the 300 Alpaca prompts from Experiment 1, re-keyed with Ex3 prompt IDs.
    """
    prompts_path = ex1_config.generations_dir / "prompts.json"
    if not prompts_path.exists():
        raise FileNotFoundError(
            f"Experiment 1 prompts not found at {prompts_path}. "
            "Run run_experiment1.py (through the 'data' step) first."
        )
    with open(prompts_path) as f:
        ex1_prompts = json.load(f)

    prompts = []
    for p in ex1_prompts:
        prompts.append({
            "prompt_id": f"alpaca_{p['prompt_id']:04d}",
            "dataset": "alpaca",
            "instruction": p["instruction"],
            "split": p["split"],
            "ex1_prompt_id": p["prompt_id"],
        })
    return prompts


# ---------------------------------------------------------------------------
# OASST1
# ---------------------------------------------------------------------------

def load_oasst1_prompts(
    ex3_config: Experiment3Config,
    n: int = 300,
) -> list[dict]:
    """
    Load n English first-turn prompter messages from OpenAssistant/oasst1.
    Saves to / loads from generations_dir_ex3/prompts_oasst1.json.
    """
    out_path = ex3_config.generations_dir_ex3 / "prompts_oasst1.json"
    if out_path.exists():
        print(f"    Found existing OASST1 prompts at {out_path}, loading...")
        with open(out_path) as f:
            return json.load(f)

    from datasets import load_dataset
    print("    Downloading OpenAssistant/oasst1...")
    ds = load_dataset("OpenAssistant/oasst1", split="train")

    # First-turn prompter messages: role="prompter", parent_id=None, lang="en"
    seen: set[str] = set()
    candidates: list[str] = []
    for row in ds:
        if row.get("role") != "prompter":
            continue
        if row.get("parent_id") is not None:
            continue
        if row.get("lang", "en") != "en":
            continue
        text = row["text"].strip()
        if not (20 < len(text) < 500):
            continue
        if text in seen:
            continue
        seen.add(text)
        candidates.append(text)

    random.seed(42)
    random.shuffle(candidates)
    selected = candidates[:n]

    prompts: list[dict] = []
    n_train = int(n * 0.70)
    n_val   = int(n * 0.85)
    for i, instruction in enumerate(selected):
        split = "train" if i < n_train else ("val" if i < n_val else "test")
        prompts.append({
            "prompt_id": f"oasst1_{i + 1:04d}",
            "dataset": "oasst1",
            "instruction": instruction,
            "split": split,
        })

    with open(out_path, "w") as f:
        json.dump(prompts, f, indent=2)
    print(f"    Saved {len(prompts)} OASST1 prompts → {out_path}")
    return prompts


# ---------------------------------------------------------------------------
# MMLU
# ---------------------------------------------------------------------------

_MMLU_SUBJECTS = [
    "high_school_biology",
    "computer_science",
    "philosophy",
    "us_history",
    "college_mathematics",
    "high_school_physics",
]


def load_mmlu_prompts(
    ex3_config: Experiment3Config,
    n: int = 300,
) -> list[dict]:
    """
    Load n MMLU questions stratified across 6 subjects (up to 50 per subject).
    Saves to / loads from generations_dir_ex3/prompts_mmlu.json.
    """
    out_path = ex3_config.generations_dir_ex3 / "prompts_mmlu.json"
    if out_path.exists():
        print(f"    Found existing MMLU prompts at {out_path}, loading...")
        with open(out_path) as f:
            return json.load(f)

    from datasets import load_dataset

    per_subject = n // len(_MMLU_SUBJECTS)  # 50 each for 6 subjects
    raw: list[dict] = []

    for subject in _MMLU_SUBJECTS:
        print(f"    Downloading MMLU subject: {subject}...")
        try:
            ds = load_dataset("cais/mmlu", subject, split="test")
        except Exception as e:
            print(f"    WARNING: could not load {subject}: {e}. Skipping.")
            continue
        rows = list(ds)
        random.seed(42)
        random.shuffle(rows)
        selected = rows[:per_subject]
        for row in selected:
            choices = row["choices"]
            instruction = (
                "Answer the following multiple choice question with a brief explanation:\n"
                f"Question: {row['question']}\n"
                f"A) {choices[0]}\n"
                f"B) {choices[1]}\n"
                f"C) {choices[2]}\n"
                f"D) {choices[3]}"
            )
            raw.append({"instruction": instruction})

    random.seed(42)
    random.shuffle(raw)
    raw = raw[:n]

    prompts: list[dict] = []
    n_train = int(n * 0.70)
    n_val   = int(n * 0.85)
    for i, item in enumerate(raw):
        split = "train" if i < n_train else ("val" if i < n_val else "test")
        prompts.append({
            "prompt_id": f"mmlu_{i + 1:04d}",
            "dataset": "mmlu",
            "instruction": item["instruction"],
            "split": split,
        })

    with open(out_path, "w") as f:
        json.dump(prompts, f, indent=2)
    print(f"    Saved {len(prompts)} MMLU prompts → {out_path}")
    return prompts


# ---------------------------------------------------------------------------
# Combined loader
# ---------------------------------------------------------------------------

def load_all_prompts(
    ex3_config: Experiment3Config,
    ex1_config: Config,
) -> list[dict]:
    """
    Load all prompts from Alpaca, OASST1, and MMLU.
    Returns combined list and saves to prompts_all.json.
    """
    master_path = ex3_config.generations_dir_ex3 / "prompts_all.json"
    if master_path.exists():
        print(f"  Found existing combined prompts at {master_path}, loading...")
        with open(master_path) as f:
            return json.load(f)

    print("  Loading Alpaca prompts (reusing Experiment 1)...")
    alpaca = load_alpaca_prompts(ex3_config, ex1_config)
    print(f"    Alpaca: {len(alpaca)} prompts")

    print("  Loading OASST1 prompts...")
    oasst1 = load_oasst1_prompts(ex3_config)
    print(f"    OASST1: {len(oasst1)} prompts")

    print("  Loading MMLU prompts...")
    mmlu = load_mmlu_prompts(ex3_config)
    print(f"    MMLU: {len(mmlu)} prompts")

    all_prompts = alpaca + oasst1 + mmlu

    with open(master_path, "w") as f:
        json.dump(all_prompts, f, indent=2)

    print(
        f"  Combined: {len(alpaca)} Alpaca + {len(oasst1)} OASST1 + "
        f"{len(mmlu)} MMLU = {len(all_prompts)} total → {master_path}"
    )
    return all_prompts
