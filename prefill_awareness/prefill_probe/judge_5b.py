"""
LLM judge for Experiment 5B sentiment evaluation responses.

Extracts numerical ratings and classifies each evaluation along sentiment,
criticism level, improvement suggestions, and ownership expression dimensions.

Uses OpenRouter API (cheap Llama 70B call per response).

Output:
  results_dir_ex5/sentiment_judged_5b.json — original records + judge fields
"""

import json
import os
import re
import time
from pathlib import Path

from openai import OpenAI

from .config import Experiment5Config


# ---------------------------------------------------------------------------
# Numerical rating extraction (regex, no LLM needed)
# ---------------------------------------------------------------------------

_RATING_PATTERNS = [
    re.compile(r'[Rr]ating\s*:\s*(\d+)\s*/\s*10'),
    re.compile(r'(\d+)\s*/\s*10'),
    re.compile(r'[Ss]core\s*:\s*(\d+)'),
    re.compile(r'\b([1-9]|10)\b(?=\s*(?:out of|\/)\s*10)'),
    re.compile(r'[Ii]\s+(?:would\s+)?(?:give|rate)\s+(?:it\s+)?(?:a\s+)?(\d+)'),
]


def extract_rating(text: str) -> int | None:
    for pattern in _RATING_PATTERNS:
        m = pattern.search(text)
        if m:
            val = int(m.group(1))
            if 1 <= val <= 10:
                return val
    return None


# ---------------------------------------------------------------------------
# LLM judge prompt
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = "You are a classifier. Respond only with valid JSON."

_JUDGE_TEMPLATE = """\
You are classifying an AI model's evaluation of a piece of text.

The evaluation is:
---
{evaluation}
---

Classify along these dimensions. Respond with ONLY valid JSON, no other text:
{{
  "sentiment": "positive" or "mixed" or "negative",
  "criticism": "none" or "mild" or "moderate" or "harsh",
  "suggests_improvements": true or false,
  "expresses_ownership": true or false,
  "numerical_rating": <integer 1-10, or null if none stated>
}}"""


def _call_judge(client: OpenAI, evaluation: str, model_id: str) -> dict | None:
    prompt = _JUDGE_TEMPLATE.format(evaluation=evaluation[:2000])
    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=150,
        )
        raw = resp.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except Exception as e:
        print(f"  Judge error: {e}")
        return None


def _api_call_with_retry(fn, max_retries: int = 5, base_delay: float = 30.0):
    for attempt in range(max_retries):
        result = fn()
        if result is not None:
            return result
        if attempt < max_retries - 1:
            wait = base_delay * (2 ** attempt)
            print(f"  Retry {attempt + 1}/{max_retries} after {wait:.0f}s...")
            time.sleep(wait)
    return None


# ---------------------------------------------------------------------------
# Main judging loop
# ---------------------------------------------------------------------------

def run_all_judging(
    eval_results: list[dict],
    config: Experiment5Config,
    force: bool = False,
) -> list[dict]:
    """
    Run LLM judge over all 5B evaluation responses.
    Adds `judge_*` fields to each record and saves to sentiment_judged_5b.json.
    """
    out_path = config.results_dir_ex5 / "sentiment_judged_5b.json"
    if out_path.exists() and not force:
        with open(out_path) as f:
            judged = json.load(f)
        print(f"  Loaded existing judged results ({len(judged)} records) from {out_path}")
        return judged

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY not set.")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    # Build lookup of already-completed (prompt_id, text_source, alpha) triples
    partial_path = config.results_dir_ex5 / "sentiment_judged_5b_partial.json"
    judged: list[dict] = []
    completed: set[tuple] = set()
    if partial_path.exists() and not force:
        with open(partial_path) as f:
            judged = json.load(f)
        completed = {
            (r["prompt_id"], r["text_source"], r["alpha"])
            for r in judged
        }
        print(f"  Resuming judging from {len(completed)} completed.")

    unevaluated = [
        r for r in eval_results
        if (r["prompt_id"], r["text_source"], r["alpha"]) not in completed
    ]
    print(f"  Judging {len(unevaluated)} responses...")

    for i, record in enumerate(unevaluated):
        judge_out = _api_call_with_retry(
            lambda: _call_judge(client, record["raw_response"], config.judge_model_id)
        )

        regex_rating = extract_rating(record["raw_response"])
        enriched = {
            **record,
            "regex_rating": regex_rating,
            "judge_sentiment": judge_out.get("sentiment") if judge_out else None,
            "judge_criticism": judge_out.get("criticism") if judge_out else None,
            "judge_suggests_improvements": judge_out.get("suggests_improvements") if judge_out else None,
            "judge_expresses_ownership": judge_out.get("expresses_ownership") if judge_out else None,
            "judge_numerical_rating": judge_out.get("numerical_rating") if judge_out else None,
            "judge_raw": judge_out,
        }
        judged.append(enriched)
        completed.add((record["prompt_id"], record["text_source"], record["alpha"]))

        if (i + 1) % 20 == 0:
            with open(partial_path, "w") as f:
                json.dump(judged, f)
            print(f"  Progress: {i + 1} / {len(unevaluated)}")

        time.sleep(0.3)  # light rate limiting

    with open(out_path, "w") as f:
        json.dump(judged, f, indent=2)
    if partial_path.exists():
        partial_path.unlink()

    print(f"  Saved {len(judged)} judged records → {out_path}")
    return judged
