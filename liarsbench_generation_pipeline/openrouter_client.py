import os
import time
from dataclasses import dataclass
from typing import Any

from openai import OpenAI


@dataclass
class OpenRouterConfig:
    api_key_env: str = "OPENROUTER_API_KEY"
    base_url: str = "https://openrouter.ai/api/v1"
    timeout_seconds: float = 120.0
    max_retries: int = 3
    app_name: str | None = None
    app_url: str | None = None


class OpenRouterChatClient:
    def __init__(self, config: OpenRouterConfig):
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise ValueError(
                f"Missing API key in env var '{config.api_key_env}'. "
                "Set it before running the pipeline."
            )

        self._cfg = config
        self._client = OpenAI(api_key=api_key, base_url=config.base_url, timeout=config.timeout_seconds)

    def chat_completion(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        headers: dict[str, str] = {}
        if self._cfg.app_name:
            headers["X-Title"] = self._cfg.app_name
        if self._cfg.app_url:
            headers["HTTP-Referer"] = self._cfg.app_url

        last_err = None
        for attempt in range(self._cfg.max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_headers=headers or None,
                )
                text = (response.choices[0].message.content or "").strip()
                return {
                    "text": text,
                    "usage": response.usage.model_dump() if response.usage else {},
                    "id": response.id,
                }
            except Exception as err:  # noqa: BLE001
                last_err = err
                if attempt < self._cfg.max_retries - 1:
                    time.sleep(1.5 * (attempt + 1))

        raise RuntimeError(f"OpenRouter request failed after retries: {last_err}")
