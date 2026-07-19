"""LLM provider interface.

The pipeline never talks to a vendor SDK directly; it calls this interface. That
keeps provider choice (Anthropic today, Bedrock/Vertex later) a config decision
rather than a code change, and lets tests run against a deterministic offline stub.
"""

from __future__ import annotations

import abc
from typing import Callable, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMProvider(abc.ABC):
    """Minimal surface the stages need."""

    @abc.abstractmethod
    def structured_output(
        self,
        *,
        system: str,
        prompt: str,
        schema: Type[T],
        model: str,
        temperature: float = 0.0,
    ) -> T:
        """Return an instance of `schema`, validated. Model is forced to emit it."""

    @abc.abstractmethod
    def tool_loop(
        self,
        *,
        system: str,
        prompt: str,
        tools: list[dict],
        tool_impls: dict[str, Callable[[dict], str]],
        model: str,
        max_turns: int = 6,
    ) -> str:
        """Run an agentic tool-use loop; return the model's final text."""
