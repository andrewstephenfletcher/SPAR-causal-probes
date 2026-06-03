"""
Dataset loading for Experiment 0e.

Each task dict has keys:
  task_id   : str  — unique identifier
  dataset   : str  — "swebench" | "bigcodebench" | "gpqa"
  prompt    : str  — full user message to send to source/evaluator models
"""

import hashlib
from typing import List


def load_all_tasks(n_per_dataset: int = 100) -> List[dict]:
    tasks = []
    tasks.extend(_load_swebench(n_per_dataset))
    tasks.extend(_load_bigcodebench(n_per_dataset))
    tasks.extend(_load_gpqa(n_per_dataset))
    print(f"  Loaded {len(tasks)} tasks "
          f"({sum(1 for t in tasks if t['dataset']=='swebench')} swe, "
          f"{sum(1 for t in tasks if t['dataset']=='bigcodebench')} bcb, "
          f"{sum(1 for t in tasks if t['dataset']=='gpqa')} gpqa)")
    return tasks


# ---------------------------------------------------------------------------
# SWE-bench
# ---------------------------------------------------------------------------

_SWE_PROMPT = (
    "You are a software engineer. Read the following GitHub issue and propose a fix.\n\n"
    "Repository: {repo}\n\n"
    "Issue:\n{problem_statement}\n\n"
    "Analyze the issue, identify the root cause, and describe your proposed fix "
    "including any code changes."
)


def _load_swebench(n: int) -> List[dict]:
    from datasets import load_dataset  # type: ignore

    try:
        ds = load_dataset("princeton-nlp/SWE-bench", split="dev")
        rows = list(ds)
    except Exception as e:
        print(f"  [SWE-bench] dev split failed ({e}), trying test split")
        ds = load_dataset("princeton-nlp/SWE-bench", split="test")
        rows = list(ds)

    filtered = [r for r in rows if 100 < len(r.get("problem_statement", "")) < 8000]

    # Deterministic shuffle
    import random
    rng = random.Random(42)
    rng.shuffle(filtered)

    # Supplement from test if dev was too small
    if len(filtered) < n:
        print(f"  [SWE-bench] only {len(filtered)} after filtering — supplementing from test")
        try:
            test_ds = load_dataset("princeton-nlp/SWE-bench", split="test")
            test_rows = [r for r in test_ds
                         if 100 < len(r.get("problem_statement", "")) < 8000]
            rng2 = random.Random(43)
            rng2.shuffle(test_rows)
            seen_ids = {r.get("instance_id") for r in filtered}
            for r in test_rows:
                if r.get("instance_id") not in seen_ids:
                    filtered.append(r)
                    seen_ids.add(r.get("instance_id"))
                if len(filtered) >= n:
                    break
        except Exception as e2:
            print(f"  [SWE-bench] test supplement failed: {e2}")

    selected = filtered[:n]
    tasks = []
    for row in selected:
        ps = row.get("problem_statement", "")
        repo = row.get("repo", "")
        iid = row.get("instance_id") or _hash(ps)
        tasks.append({
            "task_id": f"swe_{iid}",
            "dataset": "swebench",
            "prompt":  _SWE_PROMPT.format(repo=repo, problem_statement=ps),
            "_raw_problem": ps,
        })
    print(f"  [SWE-bench] {len(tasks)} tasks loaded")
    return tasks


# ---------------------------------------------------------------------------
# BigCodeBench
# ---------------------------------------------------------------------------

_BCB_PROMPT = (
    "You are a Python developer. Implement the following function and explain your approach.\n\n"
    "{instruct_prompt}"
)


def _load_bigcodebench(n: int) -> List[dict]:
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("bigcode/bigcodebench", split="v0.1.4")
    rows = list(ds)

    import random
    rng = random.Random(42)
    rng.shuffle(rows)

    selected = rows[:n]
    tasks = []
    for row in selected:
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
    "You are a scientist with expertise in physics, chemistry, and biology.\n"
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

    selected = rows[:n]
    tasks = []
    for i, row in enumerate(selected):
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
