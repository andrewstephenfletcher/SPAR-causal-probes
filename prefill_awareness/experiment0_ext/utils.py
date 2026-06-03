import time
from typing import Any, Callable

import requests
from openai import OpenAI

from .config import Experiment0ExtConfig


def make_client(config: Experiment0ExtConfig) -> OpenAI:
    return OpenAI(base_url=config.api_base, api_key=config.api_key)


def _extract_retry_after(exc: Exception) -> float | None:
    """Pull retry_after_seconds from an OpenRouter 429 error body, if present."""
    try:
        body = exc.response.json()
        metadata = body.get("error", {}).get("metadata", {})
        secs = metadata.get("retry_after_seconds")
        if secs is not None:
            return float(secs) + 2.0
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
                print(
                    f"  Attempt {attempt + 1} failed "
                    f"({'rate limit' if is_rate_limit else 'error'}): "
                    f"waiting {wait:.0f}s..."
                )
                time.sleep(wait)
            else:
                print(f"  All {max_retries} attempts failed. Skipping.")
                return None


def verify_model_ids(config: Experiment0ExtConfig) -> None:
    """
    Verify all configured model IDs exist on OpenRouter.
    Raises AssertionError with a helpful message if any are missing.
    """
    resp = requests.get(
        f"{config.api_base}/models",
        headers={"Authorization": f"Bearer {config.api_key}"},
        timeout=30,
    )
    resp.raise_for_status()
    available = {m["id"] for m in resp.json()["data"]}

    print("\nVerifying model IDs on OpenRouter:")
    missing = []
    for name, model_id in config.all_generation_models.items():
        if model_id in available:
            print(f"  [OK] {name}: {model_id}")
        else:
            base_slug = model_id.replace(":free", "").split("/")[-1]
            candidates = sorted(m for m in available if base_slug.split(":")[0] in m)[:3]
            print(f"  [MISSING] {name}: {model_id}")
            if candidates:
                print(f"    Closest: {candidates}")
            missing.append((name, model_id))

    if missing:
        raise AssertionError(
            "The following model IDs were not found on OpenRouter:\n"
            + "\n".join(f"  {n}: {m}" for n, m in missing)
            + "\nUpdate config.py with the correct IDs."
        )


class CostTracker:
    """Tracks cumulative cost and enforces a per-phase cap."""

    def __init__(self, phase_cap: float, hard_ceiling: float, phase_name: str = ""):
        self.phase_cap = phase_cap
        self.hard_ceiling = hard_ceiling
        self.phase_name = phase_name
        self._phase_cost = 0.0
        self._total_cost = 0.0

    def add(self, cost: float) -> None:
        self._phase_cost += cost
        self._total_cost += cost

    def phase_exceeded(self) -> bool:
        return self._phase_cost >= self.phase_cap

    def hard_exceeded(self) -> bool:
        return self._total_cost >= self.hard_ceiling

    def report(self) -> str:
        return (
            f"  Cost — phase '{self.phase_name}': ${self._phase_cost:.4f} "
            f"(cap ${self.phase_cap:.2f}) | total: ${self._total_cost:.4f} "
            f"(ceiling ${self.hard_ceiling:.2f})"
        )
