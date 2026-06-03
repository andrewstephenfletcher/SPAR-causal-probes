"""
Multi-turn conversation loading for Experiment 0 Extension.

Loading priority:
  1. SWE-bench trajectories from HuggingFace (Option A)
  2. OASST1 multi-turn + HH-RLHF (Option C — faithful Africa et al. replication)
  3. Raises if neither is available

Each returned conversation:
{
    "conv_id":          str,
    "dataset":          str,   # "oasst1_multiturn" | "hh_rlhf" | "swebench"
    "messages":         [{"role": "user"|"assistant", "content": str}, ...],
    "replace_turn_idx": int,   # index of the last assistant turn to replace
}
"""

import json
import random
from pathlib import Path
from typing import Optional

from .config import Experiment0ExtConfig


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_conversations(config: Experiment0ExtConfig) -> list[dict]:
    """
    Load multi-turn conversations, trying SWE-bench first.
    Falls back to OASST1 + HH-RLHF.
    Saves to generations_dir/conversations.json for reproducibility.
    """
    out_path = config.generations_dir / "conversations.json"
    if out_path.exists():
        print(f"  Loading existing conversations from {out_path}")
        with open(out_path) as f:
            return json.load(f)

    swe = _try_load_swebench(config.n_conversations)
    if swe:
        print(f"  Using SWE-bench trajectories: {len(swe)} conversations")
        conversations = swe
    else:
        print("  SWE-bench trajectories unavailable. Using OASST1 + HH-RLHF.")
        n_each = config.n_conversations // 2
        remainder = config.n_conversations - 2 * n_each
        oasst = _load_oasst1_multiturn(n_each + remainder, seed=42)
        hh = _load_hh_rlhf(n_each, seed=42)
        conversations = oasst + hh
        print(
            f"  Loaded {len(oasst)} OASST1 + {len(hh)} HH-RLHF "
            f"= {len(conversations)} total"
        )

    conversations = _truncate_all(conversations, config.max_context_chars)

    with open(out_path, "w") as f:
        json.dump(conversations, f, indent=2)
    print(f"  Saved {len(conversations)} conversations → {out_path}")
    return conversations


# ---------------------------------------------------------------------------
# Option A: SWE-bench trajectories
# ---------------------------------------------------------------------------

def _try_load_swebench(n: int) -> Optional[list[dict]]:
    """
    Attempt to load SWE-bench trajectory data from HuggingFace.
    Returns None if no usable data is found.
    """
    from datasets import load_dataset

    candidates = [
        # Known SWE-bench Verified trajectory datasets — update if new ones appear
        ("princeton-nlp/SWE-bench_Verified_trajectories", None),
        ("SWE-bench/SWE-bench_trajectories", None),
    ]

    for dataset_name, config_name in candidates:
        try:
            kwargs = {"split": "test", "trust_remote_code": True}
            if config_name:
                kwargs["name"] = config_name
            ds = load_dataset(dataset_name, **kwargs)
            cols = ds.column_names
            traj_cols = [
                c for c in cols
                if any(kw in c.lower() for kw in ["trajectory", "messages", "conversation", "log", "history"])
            ]
            if not traj_cols:
                print(f"  [{dataset_name}] found but no trajectory columns: {cols}")
                continue

            print(f"  [{dataset_name}] trajectory columns: {traj_cols}")
            convos = _parse_swebench_trajectories(ds, traj_cols[0], n)
            if convos:
                return convos
        except Exception as e:
            print(f"  [{dataset_name}] unavailable: {e}")
            continue

    return None


def _parse_swebench_trajectories(ds, col: str, n: int) -> list[dict]:
    """Parse a HuggingFace SWE-bench dataset into our conversation format."""
    convos = []
    for i, row in enumerate(ds):
        raw = row[col]
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except Exception:
                continue
        if not isinstance(raw, list):
            continue

        # Normalise message dicts
        msgs = []
        for m in raw:
            role = m.get("role", "")
            content = m.get("content", "")
            if role in ("user", "human"):
                msgs.append({"role": "user", "content": str(content)})
            elif role in ("assistant", "agent", "model"):
                msgs.append({"role": "assistant", "content": str(content)})

        asst_idxs = [j for j, m in enumerate(msgs) if m["role"] == "assistant"]
        if len(msgs) < 4 or len(asst_idxs) < 2:
            continue

        convos.append({
            "conv_id": f"swebench_{i}",
            "dataset": "swebench",
            "messages": msgs,
            "replace_turn_idx": asst_idxs[-1],
        })
        if len(convos) >= n:
            break

    return convos


# ---------------------------------------------------------------------------
# Option C: OASST1 multi-turn
# ---------------------------------------------------------------------------

def _load_oasst1_multiturn(n: int, seed: int = 42) -> list[dict]:
    """
    Load multi-turn conversation trees from OASST1.

    Builds trees from the message_id / parent_id graph, follows the
    highest-ranked child at each step to get a linear path, and keeps
    conversations with at least 4 turns and 2 assistant turns.
    """
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

    # Root messages: English, prompter, no parent
    roots = [
        mid
        for mid, msg in messages.items()
        if msg.get("parent_id") is None
        and msg.get("role") == "prompter"
        and msg.get("lang", "") == "en"
    ]

    conversations = []
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
            # Follow highest-ranked (lowest rank number = best)
            kids_sorted = sorted(
                kids,
                key=lambda k: (messages[k].get("rank") is None, messages[k].get("rank", 0)),
            )
            current = kids_sorted[0]

        asst_idxs = [i for i, m in enumerate(path) if m["role"] == "assistant"]
        if len(path) < 4 or len(asst_idxs) < 2:
            continue

        conversations.append({
            "conv_id": root_id,
            "dataset": "oasst1_multiturn",
            "messages": path,
            "replace_turn_idx": asst_idxs[-1],
        })

    rng = random.Random(seed)
    rng.shuffle(conversations)
    return conversations[:n]


# ---------------------------------------------------------------------------
# Option C: HH-RLHF
# ---------------------------------------------------------------------------

def _load_hh_rlhf(n: int, seed: int = 42) -> list[dict]:
    """
    Load multi-turn conversations from Anthropic's HH-RLHF dataset.

    The dataset stores conversations as a single string with
    \\n\\nHuman: / \\n\\nAssistant: markers. We parse these into turn lists.
    """
    from datasets import load_dataset

    ds = load_dataset("Anthropic/hh-rlhf", split="test")

    conversations = []
    for i, row in enumerate(ds):
        text = row.get("chosen", "")
        if not text:
            continue

        turns = _parse_hh_rlhf_text(text)
        if turns is None:
            continue

        asst_idxs = [j for j, m in enumerate(turns) if m["role"] == "assistant"]
        if len(turns) < 4 or len(asst_idxs) < 2:
            continue

        total_chars = sum(len(m["content"]) for m in turns)
        if not (200 < total_chars < 20_000):
            continue

        conversations.append({
            "conv_id": f"hh_rlhf_{i}",
            "dataset": "hh_rlhf",
            "messages": turns,
            "replace_turn_idx": asst_idxs[-1],
        })

    rng = random.Random(seed)
    rng.shuffle(conversations)
    return conversations[:n]


def _parse_hh_rlhf_text(text: str) -> Optional[list[dict]]:
    """
    Parse a HH-RLHF conversation string into a list of role/content dicts.

    Format:
        \\n\\nHuman: <text>\\n\\nAssistant: <text>\\n\\nHuman: <text>...
    """
    # Normalise: strip leading whitespace/newlines then split on Human turns
    text = text.strip()
    if text.startswith("Human: "):
        text = "\n\n" + text

    # Split by human turns
    human_chunks = text.split("\n\nHuman: ")
    turns = []
    for chunk in human_chunks:
        if not chunk.strip():
            continue
        if "\n\nAssistant: " in chunk:
            user_part, asst_part = chunk.split("\n\nAssistant: ", 1)
            user_content = user_part.strip()
            asst_content = asst_part.strip()
            # asst_part might itself contain further Human: turns split later
            if user_content:
                turns.append({"role": "user", "content": user_content})
            if asst_content:
                turns.append({"role": "assistant", "content": asst_content})
        else:
            content = chunk.strip()
            if content:
                turns.append({"role": "user", "content": content})

    # Validate alternating structure
    for j in range(len(turns) - 1):
        if turns[j]["role"] == turns[j + 1]["role"]:
            return None  # malformed

    return turns if turns else None


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------

def _truncate_all(conversations: list[dict], max_chars: int) -> list[dict]:
    """
    Truncate conversation context to at most max_chars total characters.
    Keeps the most recent turns (including replace_turn_idx).
    """
    result = []
    n_truncated = 0
    for conv in conversations:
        msgs = conv["messages"]
        replace_idx = conv["replace_turn_idx"]
        total = sum(len(m["content"]) for m in msgs)

        if total <= max_chars:
            result.append(conv)
            continue

        # Keep messages[:replace_turn_idx] trimmed from the front
        context_msgs = msgs[:replace_idx]
        # Keep as many recent context turns as fit
        chars_used = 0
        keep_from = len(context_msgs)
        for j in range(len(context_msgs) - 1, -1, -1):
            chars_used += len(context_msgs[j]["content"])
            if chars_used > max_chars:
                keep_from = j + 1
                break
            keep_from = j

        truncated_msgs = context_msgs[keep_from:] + [msgs[replace_idx]]
        new_replace_idx = len(truncated_msgs) - 1

        result.append({
            **conv,
            "messages": truncated_msgs,
            "replace_turn_idx": new_replace_idx,
            "truncated": True,
        })
        n_truncated += 1

    if n_truncated:
        print(f"  Truncated {n_truncated}/{len(conversations)} conversations to {max_chars} chars")
    return result
