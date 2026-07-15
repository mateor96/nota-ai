import json
import logging
from typing import AsyncGenerator

import httpx

from . import MAX_OUTPUT_TOKENS
from .exceptions import ProviderAuthError, ProviderModelError, ProviderUnavailableError

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:1234"
DEFAULT_MODEL = ""  # empty = use whatever LM Studio has loaded


def _extract_error_message(body: str) -> str:
    try:
        data = json.loads(body)
        err = data.get("error") if isinstance(data, dict) else None
        if isinstance(err, dict):
            return err.get("message") or str(err)
        if isinstance(err, str):
            return err
    except json.JSONDecodeError:
        pass
    return (body or "no detail").strip()[:300]


class LMStudioService:
    def __init__(self, base_url: str = DEFAULT_BASE_URL, model: str = DEFAULT_MODEL):
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.model = model or DEFAULT_MODEL

    def provider_name(self) -> str:
        return f"lmstudio/{self.model}" if self.model else "lmstudio"

    async def stream_chat(self, prompt: str) -> AsyncGenerator[str, None]:
        payload: dict = {
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "temperature": 0.3,
        }
        if self.model:
            payload["model"] = self.model

        # Read timeout is generous because a reasoning model on a long transcript
        # can take a minute or more to think through before streaming the answer.
        timeout = httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=5.0)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST", f"{self.base_url}/v1/chat/completions", json=payload
                ) as response:
                    if response.status_code >= 400:
                        await response.aread()
                        body = response.text
                        msg = _extract_error_message(body)
                        if response.status_code == 404:
                            raise ProviderModelError(
                                f"Model not found in LM Studio: {msg}",
                                provider=self.provider_name(),
                            )
                        raise ProviderModelError(
                            f"LM Studio rejected the request: {msg}",
                            provider=self.provider_name(),
                        )
                    yielded_content = False
                    saw_reasoning = False
                    finish_reason = None
                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            choice = json.loads(data)["choices"][0]
                            delta = choice.get("delta", {})
                            # Reasoning models stream their chain-of-thought in a
                            # separate field; it isn't the answer, so don't surface
                            # it — but note it to explain an empty result below.
                            if delta.get("reasoning_content") or delta.get("reasoning"):
                                saw_reasoning = True
                            content = delta.get("content") or ""
                            if content:
                                yielded_content = True
                                yield content
                            if choice.get("finish_reason"):
                                finish_reason = choice["finish_reason"]
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue

                    # No answer text came back. Don't return an empty summary
                    # silently — explain why so the user can act.
                    if not yielded_content:
                        if finish_reason == "length" or saw_reasoning:
                            raise ProviderModelError(
                                "The model used its entire output budget thinking and "
                                "never wrote the summary. This happens with reasoning "
                                "(\"thinking\") models on long transcripts — switch to a "
                                "non-reasoning model in settings, or use a shorter input.",
                                provider=self.provider_name(),
                            )
                        raise ProviderModelError(
                            "The model returned an empty response.",
                            provider=self.provider_name(),
                        )
        except httpx.ConnectError:
            raise ProviderUnavailableError(
                f"Cannot connect to LM Studio at {self.base_url}. Is it running?",
                provider=self.provider_name(),
            )
        except httpx.ReadTimeout:
            raise ProviderUnavailableError(
                "LM Studio timed out. The model may still be loading.",
                provider=self.provider_name(),
            )

    async def check_health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/v1/models")
                return resp.status_code == 200
        except Exception:
            return False
