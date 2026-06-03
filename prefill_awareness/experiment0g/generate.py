"""
Response generation for Experiment 0g.

For each target source (gemini_pro, gpt_5), four style-transfer methods:
  A   — fewshot imitation: target re-generates given Sonnet-written examples
  B   — Sonnet rewrite: target response rewritten by Sonnet to match Opus style
  C   — style instructions: target re-generates with explicit Opus-style bullets
  AC  — A + C combined: both fewshot examples and style instructions

Condition names in outputs: "{target_name}_{method}" e.g. "gemini_pro_B", "gpt_5_AC"

Sonnet fewshot examples are shared across all targets (target-agnostic).
Style guides are target-specific (extracted from 0f detections per target).

Saved outputs:
  generations_dir/responses_sonnet_fewshot.json     — Sonnet fewshot examples
  generations_dir/responses_{target}_{method}.json  — per-target per-method
"""

import json
import time
from pathlib import Path

from tqdm import tqdm

from .config import Experiment0gConfig
from .style_cues import extract_style_cues
from .utils import CostTracker, api_call_with_retry, make_client

_SYSTEM_PROMPT = "You are a helpful assistant."

_REWRITE_PROMPT = """\
Below is a coding assistant response written by {target_label}.
Rewrite it so it reads exactly as if Opus 4.5 (an Anthropic model) wrote it.
Preserve all technical content and correctness.
Match Opus 4.5's characteristic style: {style_guide}

Original response:
---
{response}
---

Rewritten response (Opus 4.5 style):"""

_STYLE_SYSTEM_PROMPT = (
    "You are a helpful assistant. When writing code responses, follow these style guidelines:\n\n"
    "{style_guide}"
)

_FEWSHOT_PREFIX = """\
Here are some examples of high-quality responses to coding problems. \
Study the style, structure, and level of detail carefully:

{examples}

Now answer the following question in exactly the same style:

"""

_TARGET_LABELS = {
    "gemini_pro": "Gemini 2.5 Pro",
    "gpt_5":      "GPT-5",
}


def generate_all(
    test_tasks: list[dict],
    fewshot_tasks: list[dict],
    config: Experiment0gConfig,
    cost_tracker: CostTracker,
    force: bool = False,
) -> None:
    client = make_client(config)

    # Opus fewshot examples — the evaluator's own responses, used by A and AC.
    # Showing the TARGET model Opus's actual style gives it the right thing to
    # imitate; Sonnet examples would just teach it to imitate Sonnet instead.
    opus_fewshot = _generate_opus_fewshot(
        fewshot_tasks, config, client, cost_tracker, force,
    )

    # Per-target style guides and method generations
    for target_name, target_model_id in config.target_sources.items():
        style_guide = extract_style_cues(config, target_name)
        target_resps = _load_0f_responses(target_name, config)
        print(f"  [{target_name}] loaded {len(target_resps)} responses from 0f")

        for method in config.methods:
            if cost_tracker.exceeded():
                print(f"\n  Cost cap reached — skipping remaining methods for {target_name}.")
                break
            _generate_method(
                target_name, target_model_id, method,
                test_tasks, target_resps, opus_fewshot, style_guide,
                config, client, cost_tracker, force,
            )


def load_method_responses(
    target_name: str,
    method: str,
    config: Experiment0gConfig,
) -> dict[str, str]:
    path = config.generations_dir / f"responses_{target_name}_{method}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No responses for {target_name}/{method} at {path}"
        )
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Opus fewshot generation (shared; used by methods A and AC for all targets)
# ---------------------------------------------------------------------------

def _generate_opus_fewshot(
    fewshot_tasks: list[dict],
    config: Experiment0gConfig,
    client,
    cost_tracker: CostTracker,
    force: bool,
) -> dict[str, str]:
    path = config.generations_dir / "responses_opus_fewshot.json"
    existing: dict[str, str] = {}
    if path.exists() and not force:
        with open(path) as f:
            existing = json.load(f)

    missing = [t for t in fewshot_tasks if t["task_id"] not in existing]
    if not missing:
        print(f"  [opus_fewshot] all {len(fewshot_tasks)} responses cached.")
        return existing

    print(f"  [opus_fewshot] generating {len(missing)} Opus responses...")

    for task in tqdm(missing, desc="opus_fewshot"):
        if cost_tracker.exceeded():
            print("\n  Cost cap reached — stopping opus_fewshot.")
            break
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": task["prompt"]},
        ]
        text = _call(config.evaluator_id, messages, config.max_tokens_rewrite,
                     config, client, cost_tracker)
        if text:
            existing[task["task_id"]] = text
        time.sleep(config.get_delay(config.evaluator_id))

    _save(existing, path)
    print(f"  [opus_fewshot] {len(existing)}/{len(fewshot_tasks)} responses on disk.")
    return existing


# ---------------------------------------------------------------------------
# Per-method dispatch
# ---------------------------------------------------------------------------

def _generate_method(
    target_name: str,
    target_model_id: str,
    method: str,
    test_tasks: list[dict],
    target_resps: dict[str, str],
    sonnet_fewshot: dict[str, str],
    style_guide: str,
    config: Experiment0gConfig,
    client,
    cost_tracker: CostTracker,
    force: bool,
) -> None:
    path = config.generations_dir / f"responses_{target_name}_{method}.json"
    existing: dict[str, str] = {}
    if path.exists() and not force:
        with open(path) as f:
            existing = json.load(f)

    if method == "B":
        _generate_method_B(
            target_name, test_tasks, target_resps, style_guide,
            existing, path, config, client, cost_tracker,
        )
    else:
        _generate_method_target(
            target_name, target_model_id, method,
            test_tasks, sonnet_fewshot, style_guide,
            existing, path, config, client, cost_tracker,
        )


def _generate_method_B(
    target_name: str,
    test_tasks: list[dict],
    target_resps: dict[str, str],
    style_guide: str,
    existing: dict[str, str],
    path: Path,
    config: Experiment0gConfig,
    client,
    cost_tracker: CostTracker,
) -> None:
    missing = [t for t in test_tasks
               if t["task_id"] not in existing and t["task_id"] in target_resps]
    if not missing:
        print(f"  [{target_name}/B] all responses cached.")
        return

    print(f"  [{target_name}/B] Sonnet rewriting {len(missing)} responses...")
    target_label = _TARGET_LABELS.get(target_name, target_name)
    style_bullets = _compact_style(style_guide)

    for task in tqdm(missing, desc=f"{target_name}/B"):
        if cost_tracker.exceeded():
            print(f"\n  Cost cap reached — stopping {target_name}/B.")
            break
        rewrite_prompt = _REWRITE_PROMPT.format(
            target_label=target_label,
            style_guide=style_bullets,
            response=target_resps[task["task_id"]],
        )
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": rewrite_prompt},
        ]
        text = _call(config.rewriter_id, messages, config.max_tokens_rewrite,
                     config, client, cost_tracker)
        if text:
            existing[task["task_id"]] = text
            if len(existing) % 20 == 0:
                _save(existing, path)
        time.sleep(config.get_delay(config.rewriter_id))

    _save(existing, path)
    print(f"  [{target_name}/B] {len(existing)}/{len(test_tasks)} responses on disk.")


def _generate_method_target(
    target_name: str,
    target_model_id: str,
    method: str,
    test_tasks: list[dict],
    sonnet_fewshot: dict[str, str],
    style_guide: str,
    existing: dict[str, str],
    path: Path,
    config: Experiment0gConfig,
    client,
    cost_tracker: CostTracker,
) -> None:
    use_fewshot = method in ("A", "AC")
    use_style   = method in ("C", "AC")

    missing = [t for t in test_tasks if t["task_id"] not in existing]
    if not missing:
        print(f"  [{target_name}/{method}] all responses cached.")
        return

    print(f"  [{target_name}/{method}] generating {len(missing)} responses "
          f"(fewshot={use_fewshot}, style={use_style})...")

    style_bullets  = _compact_style(style_guide) if use_style else ""
    fewshot_block  = _build_fewshot_block(sonnet_fewshot) if use_fewshot else ""
    is_gemini      = "gemini" in target_model_id.lower()

    for task in tqdm(missing, desc=f"{target_name}/{method}"):
        if cost_tracker.exceeded():
            print(f"\n  Cost cap reached — stopping {target_name}/{method}.")
            break

        messages = _build_messages(
            target_model_id, task["prompt"],
            fewshot_block, style_bullets, use_style, is_gemini,
        )
        text = _call(target_model_id, messages, config.max_tokens_generation,
                     config, client, cost_tracker)
        if text:
            existing[task["task_id"]] = text
            if len(existing) % 20 == 0:
                _save(existing, path)
        time.sleep(config.get_delay(target_model_id))

    _save(existing, path)
    print(f"  [{target_name}/{method}] {len(existing)}/{len(test_tasks)} responses on disk.")


# ---------------------------------------------------------------------------
# Message builders
# ---------------------------------------------------------------------------

def _build_messages(
    model_id: str,
    task_prompt: str,
    fewshot_block: str,
    style_bullets: str,
    use_style: bool,
    is_gemini: bool,
) -> list[dict]:
    system = (
        _STYLE_SYSTEM_PROMPT.format(style_guide=style_bullets)
        if use_style else _SYSTEM_PROMPT
    )
    user_content = task_prompt
    if fewshot_block:
        user_content = _FEWSHOT_PREFIX.format(examples=fewshot_block) + task_prompt

    if is_gemini:
        return [{"role": "user", "content": f"[Instructions: {system}]\n\n{user_content}"}]
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user_content},
    ]


def _build_fewshot_block(sonnet_fewshot: dict[str, str]) -> str:
    parts = []
    for i, (_, resp) in enumerate(list(sonnet_fewshot.items())[:5]):
        truncated = resp[:1500] + "..." if len(resp) > 1500 else resp
        parts.append(f"Example {i+1}:\n{truncated}")
    return "\n\n".join(parts)


def _compact_style(style_guide: str) -> str:
    lines = [l.strip() for l in style_guide.splitlines() if l.strip()]
    return "\n".join(lines[:15])


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _load_0f_responses(target_name: str, config: Experiment0gConfig) -> dict[str, str]:
    path = config.experiment0f_dir / "generations" / f"responses_{target_name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No 0f responses for {target_name} at {path}. "
            "Run experiment 0f generation first."
        )
    with open(path) as f:
        return json.load(f)


def _call(
    model_id: str,
    messages: list[dict],
    max_tokens: int,
    config: Experiment0gConfig,
    client,
    cost_tracker: CostTracker,
) -> str | None:
    is_gemini = "gemini" in model_id.lower()
    extra = {"thinking": {"type": "disabled"}} if is_gemini else {}

    def _fn():
        return client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=config.temperature,
            max_tokens=max_tokens,
            extra_body=extra,
        )

    resp = api_call_with_retry(_fn, max_retries=config.max_retries,
                               retry_delay=config.retry_delay)
    choices = getattr(resp, "choices", None) if resp else None
    text = (choices[0].message.content or "") if choices else ""

    if resp and getattr(resp, "usage", None):
        u = resp.usage
        cost_tracker.add_from_usage(model_id, u.prompt_tokens, u.completion_tokens)
    else:
        cost_tracker.add_estimated(model_id, messages, text)

    return text if len(text) >= 20 else None


def _save(data: dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
