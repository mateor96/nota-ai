"""Tests for cloud LLM provider services (Anthropic, OpenAI, Gemini).

We avoid mocking the SDK at the import-system level — too fragile. Instead we
construct the service with a fake key (which lets __init__ succeed) and then
replace `self._client` with a hand-rolled fake. This tests behaviour, not the
specific call signatures inside the SDK.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.anthropic_svc import AnthropicService
from app.services.exceptions import ProviderAuthError
from app.services.gemini import GeminiService
from app.services.openai_svc import OpenAIService


# ── shared helpers ───────────────────────────────────────────────────────────

class _AsyncIter:
    """Async iterator over a fixed list of values."""

    def __init__(self, items):
        self._items = list(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._items:
            raise StopAsyncIteration
        return self._items.pop(0)


# ── Anthropic ────────────────────────────────────────────────────────────────

class TestAnthropic:
    def test_requires_api_key(self):
        with pytest.raises(ProviderAuthError):
            AnthropicService(api_key="")

    def test_constructor_with_key_succeeds(self):
        svc = AnthropicService(api_key="sk-test")
        assert svc._model  # default applied

    def test_default_model_uses_haiku_4_5(self):
        svc = AnthropicService(api_key="sk-test")
        assert svc._model == "claude-haiku-4-5"

    def test_explicit_model_is_used(self):
        svc = AnthropicService(api_key="sk-test", model="claude-opus-4-7")
        assert svc._model == "claude-opus-4-7"

    def test_provider_name_includes_model(self):
        svc = AnthropicService(api_key="sk-test", model="claude-opus-4-7")
        assert svc.provider_name() == "anthropic/claude-opus-4-7"

    async def test_stream_chat_yields_tokens(self):
        svc = AnthropicService(api_key="sk-test")

        # Fake stream context manager — `client.messages.stream(...)` returns
        # an async context manager whose value exposes `text_stream`.
        stream_cm = MagicMock()
        stream_cm.__aenter__ = AsyncMock(
            return_value=SimpleNamespace(text_stream=_AsyncIter(["He", "llo"]))
        )
        stream_cm.__aexit__ = AsyncMock(return_value=None)
        svc._client = MagicMock()
        svc._client.messages.stream = MagicMock(return_value=stream_cm)

        tokens = [t async for t in svc.stream_chat("transcript")]
        assert tokens == ["He", "llo"]

    async def test_empty_stream_raises_model_error(self):
        from app.services.exceptions import ProviderModelError
        svc = AnthropicService(api_key="sk-test")
        stream_cm = MagicMock()
        stream_cm.__aenter__ = AsyncMock(
            return_value=SimpleNamespace(text_stream=_AsyncIter([]))
        )
        stream_cm.__aexit__ = AsyncMock(return_value=None)
        svc._client = MagicMock()
        svc._client.messages.stream = MagicMock(return_value=stream_cm)

        with pytest.raises(ProviderModelError, match="empty response"):
            [_ async for _ in svc.stream_chat("transcript")]

    async def test_stream_chat_maps_auth_error(self):
        import anthropic
        svc = AnthropicService(api_key="sk-test")
        svc._client = MagicMock()
        svc._client.messages.stream = MagicMock(
            side_effect=anthropic.AuthenticationError(
                message="invalid key",
                response=MagicMock(status_code=401, headers={}),
                body=None,
            )
        )
        with pytest.raises(ProviderAuthError):
            [_ async for _ in svc.stream_chat("hello")]


# ── OpenAI ───────────────────────────────────────────────────────────────────

class TestOpenAI:
    def test_requires_api_key(self):
        with pytest.raises(ProviderAuthError):
            OpenAIService(api_key="")

    def test_constructor_with_key_succeeds(self):
        svc = OpenAIService(api_key="sk-test")
        assert svc._model == "gpt-5-mini"

    def test_provider_name_includes_model(self):
        assert OpenAIService(api_key="sk-test", model="gpt-5").provider_name() == "openai/gpt-5"

    async def test_stream_chat_yields_tokens(self):
        svc = OpenAIService(api_key="sk-test")
        # The OpenAI SDK returns an async iterable of chunks; each chunk has
        # choices[0].delta.content
        chunk_a = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="A"))])
        chunk_b = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="B"))])
        chunk_empty = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None))])

        svc._client = MagicMock()
        svc._client.chat.completions.create = AsyncMock(
            return_value=_AsyncIter([chunk_a, chunk_empty, chunk_b])
        )
        tokens = [t async for t in svc.stream_chat("transcript")]
        assert tokens == ["A", "B"]

    async def test_reasoning_model_omits_temperature(self):
        svc = OpenAIService(api_key="sk-test", model="gpt-5-mini")
        chunk = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="ok"))])
        svc._client = MagicMock()
        svc._client.chat.completions.create = AsyncMock(return_value=_AsyncIter([chunk]))

        [_ async for _ in svc.stream_chat("transcript")]

        kwargs = svc._client.chat.completions.create.call_args.kwargs
        assert "temperature" not in kwargs
        assert "max_completion_tokens" in kwargs
        assert "max_tokens" not in kwargs

    async def test_non_reasoning_model_keeps_temperature(self):
        svc = OpenAIService(api_key="sk-test", model="gpt-4o-mini")
        chunk = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="ok"))])
        svc._client = MagicMock()
        svc._client.chat.completions.create = AsyncMock(return_value=_AsyncIter([chunk]))

        [_ async for _ in svc.stream_chat("transcript")]

        assert svc._client.chat.completions.create.call_args.kwargs["temperature"] == 0.3

    async def test_empty_stream_raises_model_error(self):
        from app.services.exceptions import ProviderModelError
        svc = OpenAIService(api_key="sk-test")
        svc._client = MagicMock()
        svc._client.chat.completions.create = AsyncMock(return_value=_AsyncIter([]))

        with pytest.raises(ProviderModelError):
            [_ async for _ in svc.stream_chat("transcript")]

    async def test_stream_chat_maps_auth_error(self):
        import openai
        svc = OpenAIService(api_key="sk-test")
        svc._client = MagicMock()
        svc._client.chat.completions.create = AsyncMock(
            side_effect=openai.AuthenticationError(
                message="bad key",
                response=MagicMock(status_code=401, headers={}, request=MagicMock()),
                body=None,
            )
        )
        with pytest.raises(ProviderAuthError):
            [_ async for _ in svc.stream_chat("hello")]


# ── Gemini ───────────────────────────────────────────────────────────────────

class TestGemini:
    def test_requires_api_key(self):
        with pytest.raises(ProviderAuthError):
            GeminiService(api_key="")

    def test_constructor_with_key_succeeds(self):
        svc = GeminiService(api_key="abc")
        assert svc._model == "gemini-2.5-flash"

    def test_provider_name_includes_model(self):
        assert GeminiService(api_key="abc", model="gemini-2.5-pro").provider_name() == "gemini/gemini-2.5-pro"

    async def test_stream_chat_yields_text_chunks(self):
        svc = GeminiService(api_key="abc")
        chunk_a = SimpleNamespace(text="Hello")
        chunk_b = SimpleNamespace(text=" world")
        chunk_empty = SimpleNamespace(text="")

        svc._client = MagicMock()
        svc._client.aio = MagicMock()
        svc._client.aio.models = MagicMock()
        svc._client.aio.models.generate_content_stream = MagicMock(
            return_value=_AsyncIter([chunk_a, chunk_empty, chunk_b])
        )
        tokens = [t async for t in svc.stream_chat("transcript")]
        assert tokens == ["Hello", " world"]

    async def test_empty_stream_raises_model_error(self):
        from app.services.exceptions import ProviderModelError
        svc = GeminiService(api_key="abc")
        svc._client = MagicMock()
        svc._client.aio = MagicMock()
        svc._client.aio.models = MagicMock()
        svc._client.aio.models.generate_content_stream = MagicMock(
            return_value=_AsyncIter([])
        )
        with pytest.raises(ProviderModelError, match="empty response"):
            [_ async for _ in svc.stream_chat("transcript")]

    async def test_stream_chat_maps_api_key_error(self):
        svc = GeminiService(api_key="abc")
        svc._client = MagicMock()
        svc._client.aio = MagicMock()
        svc._client.aio.models = MagicMock()
        svc._client.aio.models.generate_content_stream = MagicMock(
            side_effect=Exception("API_KEY invalid"),
        )
        with pytest.raises(ProviderAuthError):
            [_ async for _ in svc.stream_chat("hello")]


# ── ImportError fallback (cloud SDK not installed) ────────────────────────────

def test_anthropic_missing_sdk_raises_unavailable(monkeypatch):
    """If `anthropic` isn't importable, constructor should raise ProviderUnavailableError."""
    import sys
    real = sys.modules.pop("anthropic", None)
    monkeypatch.setitem(sys.modules, "anthropic", None)  # forces ImportError on `import anthropic`
    try:
        from app.services.exceptions import ProviderUnavailableError
        with pytest.raises(ProviderUnavailableError, match="anthropic not installed"):
            AnthropicService(api_key="sk-test")
    finally:
        if real is not None:
            sys.modules["anthropic"] = real
        else:
            sys.modules.pop("anthropic", None)
