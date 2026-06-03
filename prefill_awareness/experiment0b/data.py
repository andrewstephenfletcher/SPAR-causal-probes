"""
Data loading for Experiment 0b.

Returns two datasets:
  1. agentic  — 20 synthetic coding tasks (from task_scenarios.py)
  2. oasst1   — 15 multi-turn conversations reused from Experiment 0 Extension

Each returned conversation dict:
{
    "conv_id":              str,
    "dataset":              "agentic" | "oasst1",
    "messages":             list[dict],          # user/asst turns (no system)
    "assistant_turn_indices": list[int],          # indices into messages
    "target_turn_number":   int,                 # 1-indexed asst turn to ask about
    "target_msg_idx":       int,                 # absolute index in messages
    # agentic only:
    "task_id":              str,
    "tool_outputs":         list[str],
}
"""

import json
import random
from pathlib import Path
from typing import Optional

from .config import Experiment0bConfig
from .task_scenarios import AGENTIC_TASKS


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_all_conversations(config: Experiment0bConfig) -> list[dict]:
    """
    Return agentic + OASST1 conversations.
    Saves to generations_dir/conversations.json for reproducibility.
    """
    out_path = config.generations_dir / "conversations.json"
    if out_path.exists():
        print(f"  Loading existing conversations from {out_path}")
        with open(out_path) as f:
            return json.load(f)

    agentic = _build_agentic_conversations(config)
    oasst1 = _load_oasst1_from_first_run(config)
    all_convos = agentic + oasst1

    with open(out_path, "w") as f:
        json.dump(all_convos, f, indent=2)
    print(
        f"  Saved {len(agentic)} agentic + {len(oasst1)} OASST1 "
        f"= {len(all_convos)} conversations → {out_path}"
    )
    return all_convos


# ---------------------------------------------------------------------------
# Agentic conversations (from task_scenarios.py)
# ---------------------------------------------------------------------------

def _build_agentic_conversations(config: Experiment0bConfig) -> list[dict]:
    """
    Build conversation stubs for each agentic task.

    The messages list encodes the fixed conversation skeleton:
      user: task description
      asst: PLACEHOLDER (filled in from trajectories during detection)
      user: tool_output_0
      asst: PLACEHOLDER
      user: tool_output_1
      asst: PLACEHOLDER

    The actual assistant content is stored in the trajectory files —
    these stubs carry the metadata and tool output text.
    """
    tasks = AGENTIC_TASKS[: config.n_agentic_tasks]
    convos = []
    for task in tasks:
        tool_outs = task["tool_outputs"]
        # Build skeleton: user, asst, user, asst, user, asst  (3-turn agentic)
        skeleton = [
            {"role": "user",      "content": task["description"]},
            {"role": "assistant", "content": "__TURN_1__"},
            {"role": "user",      "content": tool_outs[0]},
            {"role": "assistant", "content": "__TURN_2__"},   # ← target
            {"role": "user",      "content": tool_outs[1]},
            {"role": "assistant", "content": "__TURN_3__"},
        ]
        asst_idxs = [i for i, m in enumerate(skeleton) if m["role"] == "assistant"]
        # Target: middle assistant turn (index 1 in asst_idxs, message index 3)
        target_asst_pos = 1          # 0-indexed position in asst_idxs
        target_msg_idx = asst_idxs[target_asst_pos]
        target_turn_number = target_asst_pos + 1   # 1-indexed

        convos.append({
            "conv_id": task["task_id"],
            "dataset": "agentic",
            "task_id": task["task_id"],
            "tool_outputs": tool_outs,
            "messages": skeleton,
            "assistant_turn_indices": asst_idxs,
            "target_turn_number": target_turn_number,
            "target_msg_idx": target_msg_idx,
        })
    return convos


# ---------------------------------------------------------------------------
# OASST1 conversations reused from Experiment 0 Extension
# ---------------------------------------------------------------------------

def _load_oasst1_from_first_run(config: Experiment0bConfig) -> list[dict]:
    """
    Load OASST1 multi-turn conversations from the first run's outputs.
    Falls back to loading fresh from HuggingFace if first run outputs are absent.
    """
    first_convos_path = config.first_run_dir / "generations" / "conversations.json"

    if not first_convos_path.exists():
        print(
            f"  First run conversations not found at {first_convos_path}. "
            "Loading OASST1 fresh from HuggingFace..."
        )
        return _load_oasst1_fresh(config)

    with open(first_convos_path) as f:
        all_convos = json.load(f)

    oasst1 = [c for c in all_convos if c.get("dataset") == "oasst1_multiturn"]
    rng = random.Random(42)
    rng.shuffle(oasst1)
    oasst1 = oasst1[: config.n_oasst1_convos]

    # Re-compute target turn for detection: the last assistant turn
    result = []
    for conv in oasst1:
        msgs = conv["messages"]
        asst_idxs = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
        if len(asst_idxs) < 2:
            continue
        # Use last assistant turn as detection target (same turn that was replaced in run 0ext)
        target_msg_idx = asst_idxs[-1]
        target_turn_number = len(asst_idxs)  # 1-indexed (it's the last one)
        result.append({
            "conv_id": conv["conv_id"],
            "dataset": "oasst1",
            "messages": msgs,
            "assistant_turn_indices": asst_idxs,
            "target_turn_number": target_turn_number,
            "target_msg_idx": target_msg_idx,
            "replace_turn_idx": conv.get("replace_turn_idx", target_msg_idx),
        })

    print(f"  Loaded {len(result)} OASST1 conversations from first run.")
    return result


def _load_oasst1_fresh(config: Experiment0bConfig) -> list[dict]:
    """Load OASST1 multi-turn conversations directly from HuggingFace."""
    from datasets import load_dataset

    ds_splits = load_dataset("OpenAssistant/oasst1")
    messages: dict[str, dict] = {}
    children: dict[str, list[str]] = {}

    for split in ("train", "validation"):
        for row in ds_splits[split]:
            mid = row["message_id"]
            messages[mid] = row
            pid = row.get("parent_id")
            if pid:
                children.setdefault(pid, []).append(mid)

    roots = [
        mid for mid, msg in messages.items()
        if msg.get("parent_id") is None
        and msg.get("role") == "prompter"
        and msg.get("lang", "") == "en"
    ]

    convos = []
    for root_id in roots:
        path = []
        current: Optional[str] = root_id
        while current:
            msg = messages[current]
            role = "user" if msg["role"] == "prompter" else "assistant"
            path.append({"role": role, "content": msg["text"]})
            kids = children.get(current, [])
            if not kids:
                break
            kids_sorted = sorted(
                kids,
                key=lambda k: (messages[k].get("rank") is None, messages[k].get("rank", 0)),
            )
            current = kids_sorted[0]

        asst_idxs = [i for i, m in enumerate(path) if m["role"] == "assistant"]
        if len(path) < 4 or len(asst_idxs) < 2:
            continue

        convos.append({
            "conv_id": root_id,
            "dataset": "oasst1",
            "messages": path,
            "assistant_turn_indices": asst_idxs,
            "target_turn_number": len(asst_idxs),
            "target_msg_idx": asst_idxs[-1],
            "replace_turn_idx": asst_idxs[-1],
        })

    rng = random.Random(42)
    rng.shuffle(convos)
    return convos[: config.n_oasst1_convos]


# ---------------------------------------------------------------------------
# Replacement lookup helpers
# ---------------------------------------------------------------------------

def load_oasst1_replacements(config: Experiment0bConfig) -> dict[str, dict[str, str]]:
    """
    Load replacement turns from Experiment 0 Extension for OASST1 conversations.

    Returns: {conv_id: {model_name: replacement_text}}
    """
    repl_path = config.first_run_dir / "generations" / "replacements.json"
    if not repl_path.exists():
        print(f"  No first-run replacements found at {repl_path}.")
        return {}

    with open(repl_path) as f:
        records = json.load(f)

    return {r["conv_id"]: r.get("replacements", {}) for r in records}
