"""
BigCodeBench loader for Experiment 0g.

Returns two disjoint sets:
  - test pool:    tasks 0-99   (same 100 as used in 0f)
  - fewshot pool: tasks 100-119 (20 tasks for Sonnet fewshot examples)

Both use the same seed=42 shuffle as 0f so task_ids are consistent.
"""

import hashlib
from typing import List

_BCB_PROMPT = (
    "Implement the following function and explain your approach.\n\n"
    "{instruct_prompt}"
)


def load_tasks(n_test: int = 100, n_fewshot: int = 20) -> tuple[List[dict], List[dict]]:
    from datasets import load_dataset  # type: ignore
    import random

    ds = load_dataset("bigcode/bigcodebench", split="v0.1.4")
    rows = list(ds)

    rng = random.Random(42)
    rng.shuffle(rows)

    def _make_task(row: dict) -> dict:
        ip = row.get("instruct_prompt", "")
        tid = row.get("task_id") or _hash(ip)
        return {
            "task_id": f"bcb_{tid}",
            "dataset": "bigcodebench",
            "prompt":  _BCB_PROMPT.format(instruct_prompt=ip),
        }

    test_tasks    = [_make_task(r) for r in rows[:n_test]]
    fewshot_tasks = [_make_task(r) for r in rows[n_test: n_test + n_fewshot]]

    print(f"  [BigCodeBench] {len(test_tasks)} test tasks, "
          f"{len(fewshot_tasks)} fewshot tasks")
    return test_tasks, fewshot_tasks


def _hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:12]
