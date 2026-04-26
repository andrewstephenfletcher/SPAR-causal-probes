import time
from typing import Any, Callable

import requests
from openai import OpenAI

from .config import Experiment0Config


def make_client(config: Experiment0Config) -> OpenAI:
    return OpenAI(base_url=config.api_base, api_key=config.api_key)


def _extract_retry_after(exc: Exception) -> float | None:
    """Pull retry_after_seconds out of an OpenRouter 429 error body, if present."""
    try:
        body = exc.response.json()  # openai.APIStatusError has .response
        metadata = body.get("error", {}).get("metadata", {})
        secs = metadata.get("retry_after_seconds")
        if secs is not None:
            return float(secs) + 2.0  # small buffer
    except Exception:
        pass
    return None


def api_call_with_retry(
    fn: Callable,
    *args,
    max_retries: int = 5,
    retry_delay: float = 60.0,
    **kwargs,
) -> Any:
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            is_rate_limit = "429" in str(e)
            if attempt < max_retries - 1:
                wait = _extract_retry_after(e) if is_rate_limit else retry_delay
                if wait is None:
                    wait = retry_delay
                print(f"  Attempt {attempt + 1} failed ({'rate limit' if is_rate_limit else 'error'}): waiting {wait:.0f}s...")
                time.sleep(wait)
            else:
                print(f"  All {max_retries} attempts failed. Skipping.")
                return None


def verify_model_ids(config: Experiment0Config) -> dict[str, str]:
    """
    Fetch the OpenRouter model list and verify every configured model ID exists.
    Returns a dict of {short_name: verified_model_id}.
    Raises AssertionError if any ID is missing.
    """
    resp = requests.get(
        "https://openrouter.ai/api/v1/models",
        headers={"Authorization": f"Bearer {config.api_key}"},
        timeout=30,
    )
    resp.raise_for_status()
    available = {m["id"] for m in resp.json()["data"]}

    print("\nVerifying model IDs on OpenRouter:")
    verified = {}
    missing = []
    for name, model_id in config.models.items():
        if model_id in available:
            print(f"  [OK] {name}: {model_id}")
            verified[name] = model_id
        else:
            # Try fuzzy match: strip ":free" suffix and look for any variant
            base = model_id.replace(":free", "")
            candidates = sorted(m for m in available if base.split("/")[-1].split(":")[0] in m)
            print(f"  [MISSING] {name}: {model_id}")
            if candidates:
                print(f"    Closest matches: {candidates[:3]}")
            missing.append((name, model_id))

    if missing:
        raise AssertionError(
            f"The following model IDs were not found on OpenRouter:\n"
            + "\n".join(f"  {n}: {m}" for n, m in missing)
            + "\nUpdate config.py with the correct IDs."
        )

    return verified
