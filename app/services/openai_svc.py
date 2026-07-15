import logging
from typing import AsyncGenerator

from . import MAX_OUTPUT_TOKENS
from .exceptions import ProviderAuthError, ProviderModelError, ProviderUnavailableError

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-5-mini"

# Reasoning-model families reject a custom temperature (only the default is
# allowed) and spend output budget on hidden chain-of-thought.
_REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def _is_reasoning_model(model: str) -> bool:
    return model.startswith(_REASONING_PREFIXES)


class OpenAIService:
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        if not api_key:
            raise ProviderAuthError("OpenAI API key is required.", provider="openai")
        try:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=api_key)
        except ImportError:
            raise ProviderUnavailableError(
                "openai not installed. Run: uv add openai",
                provider="openai",
            )
        self._model = model or DEFAULT_MODEL

    def provider_name(self) -> str:
        return f"openai/{self._model}"

    async def stream_chat(self, prompt: str) -> AsyncGenerator[str, None]:
        import openai
        params: dict = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            # max_completion_tokens is the successor to max_tokens and the only
            # form the reasoning families (gpt-5, o-series) accept.
            "max_completion_tokens": MAX_OUTPUT_TOKENS,
        }
        if not _is_reasoning_model(self._model):
            params["temperature"] = 0.3
        try:
            yielded_content = False
            stream = await self._client.chat.completions.create(**params)
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    yielded_content = True
                    yield delta
            # Don't return a blank summary silently — explain why.
            if not yielded_content:
                if _is_reasoning_model(self._model):
                    raise ProviderModelError(
                        "The model used its entire output budget reasoning and "
                        "never wrote the summary — try a shorter input or a "
                        "non-reasoning model.",
                        provider=self.provider_name(),
                    )
                raise ProviderModelError(
                    "The model returned an empty response.",
                    provider=self.provider_name(),
                )
        except openai.AuthenticationError as e:
            raise ProviderAuthError(f"Invalid OpenAI API key: {e}", provider=self.provider_name())
        except openai.NotFoundError as e:
            raise ProviderModelError(f"Model not found: {e}", provider=self.provider_name())
        except openai.APIConnectionError as e:
            raise ProviderUnavailableError(f"Cannot reach OpenAI API: {e}", provider=self.provider_name())

    async def check_health(self) -> bool:
        import openai
        try:
            resp = await self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": "say ok"}],
                max_completion_tokens=16,
            )
            return bool(resp.choices)
        except (openai.AuthenticationError, openai.NotFoundError):
            return False
        except Exception:
            return False
