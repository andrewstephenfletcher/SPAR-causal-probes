"""
Dataset loading for Experiment 0m.

Datasets: OASST1 (replaces SWE-bench), BigCodeBench, GPQA.
Identical to experiment0f/benchmarks.py except SWE-bench is omitted.
"""

import hashlib
from typing import List


def load_all_tasks(n_per_dataset: int = 100) -> List[dict]:
    tasks = []
    tasks.extend(_load_oasst1(n_per_dataset))
    tasks.extend(_load_bigcodebench(n_per_dataset))
    tasks.extend(_load_gpqa(n_per_dataset))
    print(f"  Loaded {len(tasks)} tasks "
          f"({sum(1 for t in tasks if t['dataset']=='oasst1')} oasst, "
          f"{sum(1 for t in tasks if t['dataset']=='bigcodebench')} bcb, "
          f"{sum(1 for t in tasks if t['dataset']=='gpqa')} gpqa)")
    return tasks


# ---------------------------------------------------------------------------
# OASST1
# ---------------------------------------------------------------------------

def _load_oasst1(n: int) -> List[dict]:
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("OpenAssistant/oasst1", split="train")
    rows = list(ds)

    roots = [
        r for r in rows
        if r.get("role") == "prompter"
        and r.get("parent_id") is None
        and r.get("lang", "en") == "en"
        and not r.get("deleted", False)
        and 50 < len(r.get("text", "")) < 2000
    ]

    import random
    rng = random.Random(42)
    rng.shuffle(roots)

    tasks = []
    for i, row in enumerate(roots[:n]):
        text = row.get("text", "")
        mid = row.get("message_id") or f"oasst_{i}"
        tasks.append({
            "task_id": f"oasst_{str(mid)[:12]}",
            "dataset": "oasst1",
            "prompt":  text,
        })
    print(f"  [OASST1] {len(tasks)} tasks loaded")
    return tasks


# ---------------------------------------------------------------------------
# BigCodeBench
# ---------------------------------------------------------------------------

_BCB_PROMPT = (
    "Implement the following function and explain your approach.\n\n"
    "{instruct_prompt}"
)


def _load_bigcodebench(n: int) -> List[dict]:
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("bigcode/bigcodebench", split="v0.1.4")
    rows = list(ds)

    import random
    rng = random.Random(42)
    rng.shuffle(rows)

    tasks = []
    for row in rows[:n]:
        ip = row.get("instruct_prompt", "")
        tid = row.get("task_id") or _hash(ip)
        tasks.append({
            "task_id": f"bcb_{tid}",
            "dataset": "bigcodebench",
            "prompt":  _BCB_PROMPT.format(instruct_prompt=ip),
        })
    print(f"  [BigCodeBench] {len(tasks)} tasks loaded")
    return tasks


# ---------------------------------------------------------------------------
# GPQA
# ---------------------------------------------------------------------------

_GPQA_PROMPT = (
    "Answer the following question with a detailed explanation of your reasoning.\n\n"
    "{question}"
)

_GPQA_CHOICE_PATTERNS = [
    r'\n\s*[A-D]\.\s',
    r'\n\s*\([A-D]\)\s',
    r'\n\s*\([a-d]\)\s',
]


def _strip_choices(text: str) -> str:
    import re
    for pat in _GPQA_CHOICE_PATTERNS:
        m = re.search(pat, text)
        if m:
            return text[:m.start()].strip()
    return text.strip()


def _load_gpqa(n: int) -> List[dict]:
    from datasets import load_dataset  # type: ignore

    ds = None
    for hub_id, cfg in [
        ("Wanfq/gpqa", "gpqa_extended"),
        ("Idavidrein/gpqa", "gpqa_extended"),
    ]:
        try:
            ds = load_dataset(hub_id, cfg, split="train")
            print(f"  [GPQA] loaded from {hub_id}")
            break
        except Exception as e:
            print(f"  [GPQA] {hub_id} failed: {e}")

    if ds is None:
        print("  [GPQA] falling back to MMLU-Pro hard science")
        return _load_mmlu_pro_fallback(n)

    rows = list(ds)
    import random
    rng = random.Random(42)
    rng.shuffle(rows)

    tasks = []
    for i, row in enumerate(rows[:n]):
        q = _strip_choices(row.get("Question", row.get("question", "")))
        tid = row.get("id") or row.get("record_id") or f"gpqa_{i}"
        tasks.append({
            "task_id": f"gpqa_{tid}",
            "dataset": "gpqa",
            "prompt":  _GPQA_PROMPT.format(question=q),
        })
    print(f"  [GPQA] {len(tasks)} tasks loaded")
    return tasks


def _load_mmlu_pro_fallback(n: int) -> List[dict]:
    from datasets import load_dataset  # type: ignore

    target_subjects = {"physics", "chemistry", "biology"}
    try:
        ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
        rows = [r for r in ds
                if any(s in r.get("category", "").lower() for s in target_subjects)]
    except Exception as e:
        print(f"  [MMLU-Pro] failed: {e}")
        return []

    import random
    rng = random.Random(42)
    rng.shuffle(rows)

    tasks = []
    for i, row in enumerate(rows[:n]):
        q = row.get("question", "")
        tasks.append({
            "task_id": f"gpqa_mmlu_{i}",
            "dataset": "gpqa",
            "prompt":  _GPQA_PROMPT.format(question=q),
        })
    print(f"  [MMLU-Pro fallback] {len(tasks)} tasks loaded")
    return tasks


def _hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:12]
