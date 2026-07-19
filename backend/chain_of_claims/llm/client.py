"""Client facade: selects the configured provider once, lazily."""

from __future__ import annotations

from functools import lru_cache

from ..config import settings
from .provider import LLMProvider


@lru_cache(maxsize=1)
def get_provider() -> LLMProvider:
    if settings.offline:
        from .offline_provider import OfflineProvider

        return OfflineProvider()
    if settings.provider == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider()
    raise RuntimeError(f"Unknown LLM provider: {settings.provider!r}")
