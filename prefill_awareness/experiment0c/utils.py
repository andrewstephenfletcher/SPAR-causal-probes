import time
from typing import Any, Callable

import requests
from openai import OpenAI

from .config import Experiment0cConfig


def make_client(config: Experiment0cConfig) -> OpenAI:
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
                    f"waiting {wait:.0f}s — {e}"
                )
                time.sleep(wait)
            else:
                print(f"  All {max_retries} attempts failed. Skipping.")
                return None


def verify_all_models(config: Experiment0cConfig) -> bool:
    resp = requests.get(
        f"{config.api_base}/models",
        headers={"Authorization": f"Bearer {config.api_key}"},
        timeout=30,
    )
    resp.raise_for_status()
    available = {m["id"] for m in resp.json()["data"]}

    all_models = {**config.evaluator, **config.all_source_models}
    missing = []
    print("\nVerifying model IDs on OpenRouter:")
    for name, model_id in all_models.items():
        if model_id in available:
            print(f"  [OK]      {name}: {model_id}")
        else:
            base_slug = model_id.split("/")[-1]
            candidates = sorted(m for m in available if base_slug in m)[:3]
            print(f"  [MISSING] {name}: {model_id}")
            if candidates:
                print(f"    Closest: {candidates}")
            missing.append((name, model_id))

    if missing:
        print(
            f"\n{len(missing)} model(s) not found on OpenRouter. "
            "Fix config.py or pass --skip-verify.\n"
            + "\n".join(f"  {n}: {m}" for n, m in missing)
        )
        return False
    return True


class CostTracker:
    def __init__(
        self,
        config: Experiment0cConfig,
        label: str = "experiment0c",
    ):
        self.config = config
        self.label = label
        self._total_cost = 0.0
        self._call_count = 0

    def add_from_usage(self, model_id: str, input_tokens: int, output_tokens: int) -> float:
        cost = self.config.cost_per_call(model_id, input_tokens, output_tokens)
        self._total_cost += cost
        self._call_count += 1
        if self._call_count % self.config.cost_log_every == 0:
            print(f"  {self.report()}")
        return cost

    def add_estimated(self, model_id: str, messages: list[dict], response_text: str) -> float:
        input_tokens = sum(len(m.get("content", "") or "") for m in messages) // 4
        output_tokens = len(response_text) // 4
        return self.add_from_usage(model_id, input_tokens, output_tokens)

    def exceeded(self) -> bool:
        return self._total_cost >= self.config.hard_ceiling

    def approaching(self, margin: float = 1.5) -> bool:
        return self._total_cost >= self.config.hard_ceiling - margin

    def report(self) -> str:
        return (
            f"Cost — {self.label}: "
            f"${self._total_cost:.4f} / ${self.config.hard_ceiling:.2f} ceiling"
        )
