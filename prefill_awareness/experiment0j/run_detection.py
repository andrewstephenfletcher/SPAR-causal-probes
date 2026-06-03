"""
Detection helpers for Experiment 0j.

Opus 4.5 is the sole evaluator. Messages follow 0g's conversation format:
  [System] You are a helpful assistant.
  [User]   <task prompt>
  [Asst]   <response to evaluate>
  [User]   <detection prompt>
"""

import json
import re
import time
from openai import OpenAI

from .config import Experiment0jConfig


def make_client(config: Experiment0jConfig) -> OpenAI:
    return OpenAI(base_url=config.api_base, api_key=config.api_key)


def detect_one(
    task_prompt: str,
    response: str,
    config: Experiment0jConfig,
    client: OpenAI,
    cost_tracker,
) -> tuple[int | None, str]:
    messages = [
        {"role": "system",    "content": "You are a helpful assistant."},
        {"role": "user",      "content": task_prompt},
        {"role": "assistant", "content": response},
        {"role": "user",      "content": config.detection_prompt},
    ]

    def _fn():
        return client.chat.completions.create(
            model=config.evaluator_id,
            messages=messages,
            temperature=0.0,
            max_tokens=config.max_tokens_detection,
        )

    resp = _call_with_retry(_fn, config)
    choices = getattr(resp, "choices", None) if resp else None
    raw = (choices[0].message.content or "") if choices else ""

    if resp and getattr(resp, "usage", None):
        u = resp.usage
        cost_tracker.add(config.evaluator_id, u.prompt_tokens, u.completion_tokens, config)
    else:
        cost_tracker.add_estimated(config.evaluator_id, messages, raw, config)

    return _parse(raw)


def call_gemini(
    task_prompt: str,
    system_instruction: str,
    config: Experiment0jConfig,
    client: OpenAI,
    cost_tracker,
    max_tokens: int | None = None,
) -> str | None:
    """Call Gemini for generation. Prepends system instruction to user message (no sys role)."""
    if max_tokens is None:
        max_tokens = config.max_tokens_generation
    user_content = f"[Instructions: {system_instruction}]\n\n{task_prompt}"
    messages = [{"role": "user", "content": user_content}]

    def _fn():
        return client.chat.completions.create(
            model=config.gemini_id,
            messages=messages,
            temperature=config.temperature,
            max_tokens=max_tokens,
            extra_body={"thinking": {"type": "disabled"}},
        )

    resp = _call_with_retry(_fn, config)
    choices = getattr(resp, "choices", None) if resp else None
    text = (choices[0].message.content or "") if choices else ""

    if resp and getattr(resp, "usage", None):
        u = resp.usage
        cost_tracker.add(config.gemini_id, u.prompt_tokens, u.completion_tokens, config)
    else:
        cost_tracker.add_estimated(config.gemini_id, messages, text, config)

    return text if len(text) >= 20 else None


def call_gemini_rewrite(
    rewrite_prompt: str,
    config: Experiment0jConfig,
    client: OpenAI,
    cost_tracker,
) -> str | None:
    """Call Gemini for iterative rewrite (no system role, plain user message)."""
    messages = [{"role": "user", "content": rewrite_prompt}]

    def _fn():
        return client.chat.completions.create(
            model=config.gemini_id,
            messages=messages,
            temperature=config.temperature,
            max_tokens=config.max_tokens_generation,
            extra_body={"thinking": {"type": "disabled"}},
        )

    resp = _call_with_retry(_fn, config)
    choices = getattr(resp, "choices", None) if resp else None
    text = (choices[0].message.content or "") if choices else ""

    if resp and getattr(resp, "usage", None):
        u = resp.usage
        cost_tracker.add(config.gemini_id, u.prompt_tokens, u.completion_tokens, config)
    else:
        cost_tracker.add_estimated(config.gemini_id, messages, text, config)

    return text if len(text) >= 20 else None


def call_opus(
    task_prompt: str,
    system_instruction: str,
    config: Experiment0jConfig,
    client: OpenAI,
    cost_tracker,
) -> str | None:
    """Call Opus for reverse-direction generation."""
    messages = [
        {"role": "system", "content": f"You are a helpful assistant. {system_instruction}"},
        {"role": "user",   "content": task_prompt},
    ]

    def _fn():
        return client.chat.completions.create(
            model=config.evaluator_id,
            messages=messages,
            temperature=config.temperature,
            max_tokens=config.max_tokens_reverse,
        )

    resp = _call_with_retry(_fn, config)
    choices = getattr(resp, "choices", None) if resp else None
    text = (choices[0].message.content or "") if choices else ""

    if resp and getattr(resp, "usage", None):
        u = resp.usage
        cost_tracker.add(config.evaluator_id, u.prompt_tokens, u.completion_tokens, config)
    else:
        cost_tracker.add_estimated(config.evaluator_id, messages, text, config)

    return text if len(text) >= 20 else None


def _parse(text: str) -> tuple[int | None, str]:
    raw = (text or "").strip()
    for candidate in [raw, *_extract_json_blocks(raw)]:
        try:
            data = json.loads(candidate)
            v = data.get("tamper_prob")
            if v is not None:
                return max(0, min(100, int(v))), str(data.get("reason", ""))
        except Exception:
            pass
    m = re.search(r'"tamper_prob"\s*[=:]\s*(\d+)', raw)
    if m:
        return max(0, min(100, int(m.group(1)))), ""
    for n in re.findall(r'\b(\d{1,3})\b', raw):
        v = int(n)
        if 0 <= v <= 100:
            return v, ""
    return None, ""


def _extract_json_blocks(text: str) -> list[str]:
    out = []
    for pat in [r"```(?:json)?\s*(\{.*?\})\s*```", r"(\{[^{}]+\})"]:
        for m in re.finditer(pat, text, re.DOTALL):
            out.append(m.group(1))
    return out


def _call_with_retry(fn, config: Experiment0jConfig):
    for attempt in range(config.max_retries):
        try:
            return fn()
        except Exception as e:
            if attempt < config.max_retries - 1:
                wait = config.retry_delay
                if "429" in str(e):
                    try:
                        wait = float(
                            e.response.json()["error"]["metadata"].get(
                                "retry_after_seconds", config.retry_delay
                            )
                        ) + 2
                    except Exception:
                        pass
                print(f"  attempt {attempt + 1} failed, retrying in {wait:.0f}s: {e}")
                time.sleep(wait)
            else:
                print(f"  all {config.max_retries} attempts failed, skipping.")
                return None


class CostTracker:
    def __init__(self, cap: float, log_every: int = 25):
        self._total = 0.0
        self._calls = 0
        self.cap = cap
        self.log_every = log_every

    def add(self, model_id: str, in_tok: int, out_tok: int, config) -> None:
        self._total += config.cost_per_call(model_id, in_tok, out_tok)
        self._calls += 1
        if self._calls % self.log_every == 0:
            print(f"  {self.report()}")

    def add_estimated(self, model_id: str, messages: list, response: str, config) -> None:
        in_tok = sum(len((m.get("content") or "")) for m in messages) // 4
        out_tok = len(response) // 4
        self.add(model_id, in_tok, out_tok, config)

    def exceeded(self) -> bool:
        return self._total >= self.cap

    def report(self) -> str:
        return f"Cost: ${self._total:.4f} / ${self.cap:.2f} ({self._calls} calls)"
