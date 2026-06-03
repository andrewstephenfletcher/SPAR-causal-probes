import time
from typing import Any, Callable

import requests
from openai import OpenAI

from .config import Experiment0nConfig


def make_client(config: Experiment0nConfig) -> OpenAI:
    return OpenAI(base_url=config.api_base, api_key=config.api_key)


def _extract_retry_after(exc: Exception) -> float | None:
    try:
        body = exc.response.json()
        secs = body.get("error", {}).get("metadata", {}).get("retry_after_seconds")
        if secs is not None:
            return float(secs) + 2.0
    except Exception:
        pass
    return None


def api_call_with_retry(
    fn: Callable,
    *args,
    max_retries: int = 3,
    retry_delay: float = 15.0,
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
                    f"waiting {wait:.0f}s — {e}"
                )
                time.sleep(wait)
            else:
                print(f"  All {max_retries} attempts failed. Skipping.")
                return None


def verify_all_models(config: Experiment0nConfig) -> bool:
    resp = requests.get(
        f"{config.api_base}/models",
        headers={"Authorization": f"Bearer {config.api_key}"},
        timeout=30,
    )
    resp.raise_for_status()
    available = {m["id"] for m in resp.json()["data"]}

    all_models = {config.evaluator_name: config.evaluator_id, **config.sources}
    missing = []
    print("\nVerifying models:")
    for name, mid in all_models.items():
        if mid in available:
            print(f"  [OK]      {name}: {mid}")
        else:
            slug = mid.split("/")[-1]
            candidates = sorted(m for m in available if slug.split("-")[0] in m.lower())[:5]
            print(f"  [MISSING] {name}: {mid}")
            if candidates:
                print(f"    Candidates: {candidates}")
            missing.append((name, mid))

    if missing:
        print(f"\n{len(missing)} model(s) not found.")
        return False
    print("\nAll model IDs verified.")
    return True


class CostTracker:
    def __init__(self, config: Experiment0nConfig, label: str = "experiment0n"):
        self.config = config
        self.label = label
        self._total = 0.0
        self._calls = 0

    def add_from_usage(self, model_id: str, input_tokens: int, output_tokens: int) -> float:
        cost = self.config.cost_per_call(model_id, input_tokens, output_tokens)
        self._total += cost
        self._calls += 1
        if self._calls % self.config.cost_log_every == 0:
            print(f"  {self.report()}")
        return cost

    def add_estimated(self, model_id: str, messages: list[dict], response: str) -> float:
        inp = sum(len(m.get("content", "") or "") for m in messages) // 4
        out = len(response) // 4
        return self.add_from_usage(model_id, inp, out)

    def exceeded(self) -> bool:
        return self._total >= self.config.cost_cap

    def approaching(self, margin: float = 3.0) -> bool:
        return self._total >= self.config.cost_cap - margin

    def report(self) -> str:
        return (
            f"Cost — {self.label}: "
            f"${self._total:.4f} / ${self.config.cost_cap:.2f} cap"
        )
