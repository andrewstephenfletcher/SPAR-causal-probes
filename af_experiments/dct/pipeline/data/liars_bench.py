"""
LIARS' BENCH dataset loading utilities.

Loads the benchmark from HuggingFace (Cadenza-Labs/liars-bench) and provides
structured access to examples grouped by sub-dataset and model.

Note: You need to accept the dataset terms on HuggingFace before use:
  https://huggingface.co/datasets/Cadenza-Labs/liars-bench
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# The seven sub-datasets in LIARS' BENCH
DATASET_NAMES = [
    "HP-C",   # Harmful Prompt - Compliance
    "HP-KR",  # Harmful Prompt - Knowledge Refusal
    "CG",     # Character Game
    "ID",     # Instructed Deception
    "IT",     # Instructed Truthfulness (control-ish)
    "GS",     # Gradient Subversion (fine-tuned)
    "ST",     # Sleeper Trigger (fine-tuned)
]

MODEL_NAMES = [
    "mistral-small-3.1-24b-instruct",
    "llama-v3.3-70b-instruct",
    "qwen-2.5-72b-instruct",
    "gemma-3-27b-it",
]

CONTROL_DATASET = "alpaca"


@dataclass
class LiarsBenchExample:
    """A single example from LIARS' BENCH."""

    example_id: str
    dataset_name: str
    model_name: str
    prompt: str
    response: str
    label: int  # 1 = lie, 0 = honest
    messages: list[dict] | None = None


def load_liars_bench(
    dataset_name: Optional[str] = None,
    model_name: Optional[str] = None,
    split: str = "test",
    cache_dir: Optional[str] = None,
) -> list[LiarsBenchExample]:
    """Load LIARS' BENCH examples from HuggingFace."""
    import re
    import pandas as pd
    from datasets import load_dataset

    kwargs = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    # Known configs for Cadenza-Labs/liars-bench (used as fallback)
    _KNOWN_CONFIGS = [
        "alpaca",
        "convincing-game",
        "gender-secret",
        "harm-pressure-choice",
        "harm-pressure-knowledge-report",
        "insider-trading",
        "instructed-deception",
        "soft-trigger",
    ]

    def _load_one_config(config: str) -> pd.DataFrame:
        """Load a single config and normalise schema."""
        ds = load_dataset("Cadenza-Labs/liars-bench", config, split=split, **kwargs)
        cdf = ds.to_pandas()
        # Some configs omit the 'dataset' column — use the config name.
        if "dataset" not in cdf.columns:
            cdf["dataset"] = config
        return cdf

    try:
        # Try loading without specifying a config (works when there is only one).
        ds = load_dataset("Cadenza-Labs/liars-bench", split=split, **kwargs)
        df = ds.to_pandas()
        if "dataset" not in df.columns:
            df["dataset"] = "unknown"
    except ValueError as e:
        # Multiple configs present — parse names from error message, fall back to known list.
        match = re.search(r"\[([^\]]+)\]", str(e))
        configs = re.findall(r"'([^']+)'", match.group(1)) if match else []
        if not configs:
            configs = _KNOWN_CONFIGS
        df = pd.concat([_load_one_config(c) for c in configs], ignore_index=True)

    if dataset_name is not None:
        df = df[df["dataset"] == dataset_name]
    if model_name is not None:
        df = df[df["model"] == model_name]

    examples = []
    for _, row in df.iterrows():
        # messages may arrive as a numpy array of dicts from pandas — normalise to list or None.
        raw_msgs = row.get("messages")
        if raw_msgs is None or (isinstance(raw_msgs, float)):
            messages = None
        else:
            messages = list(raw_msgs)
        examples.append(
            LiarsBenchExample(
                example_id=str(row.get("index", row.name)),
                dataset_name=row["dataset"],
                model_name=row["model"],
                prompt=str(row.get("prompt", "")),
                response=str(row.get("response", "")),
                label=int(row["deceptive"]),  # bool True/False -> 1/0
                messages=messages,
            )
        )

    return examples


def load_liars_bench_grouped(
    model_name: Optional[str] = None,
    split: str = "test",
    cache_dir: Optional[str] = None,
) -> dict[str, list[LiarsBenchExample]]:
    """Load LIARS' BENCH grouped by sub-dataset name."""
    examples = load_liars_bench(
        model_name=model_name, split=split, cache_dir=cache_dir
    )

    grouped = {}
    for ex in examples:
        if ex.dataset_name not in grouped:
            grouped[ex.dataset_name] = []
        grouped[ex.dataset_name].append(ex)

    return grouped
