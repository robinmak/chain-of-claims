"""Anthropic implementation of LLMProvider (Claude).

Uses the Anthropic Messages API. Structured output is obtained by exposing a single
tool whose input_schema is the target Pydantic model's JSON schema and forcing the
model to call it (tool_choice), which is the robust way to get schema-valid JSON.
"""

from __future__ import annotations

import json
from typing import Callable, Type, TypeVar

from pydantic import BaseModel

from ..config import settings
from .provider import LLMProvider

T = TypeVar("T", bound=BaseModel)


class AnthropicProvider(LLMProvider):
    def __init__(self) -> None:
        # Imported lazily so the package installs/tests without the key present.
        from anthropic import Anthropic

        kwargs: dict = {}
        if settings.anthropic_base_url:
            kwargs["base_url"] = settings.anthropic_base_url
        if settings.anthropic_api_key:
            kwargs["api_key"] = settings.anthropic_api_key
        elif settings.anthropic_auth_token:
            # Proxy/gateway auth: bearer token instead of an API key.
            kwargs["auth_token"] = settings.anthropic_auth_token
        else:
            raise RuntimeError(
                "No Anthropic credentials. Set ANTHROPIC_API_KEY, or "
                "ANTHROPIC_AUTH_TOKEN (+ ANTHROPIC_BASE_URL for a proxy), "
                "or run with COC_OFFLINE=1."
            )
        self._client = Anthropic(**kwargs)

    def structured_output(
        self,
        *,
        system: str,
        prompt: str,
        schema: Type[T],
        model: str,
        temperature: float = 0.0,
    ) -> T:
        tool_name = "emit_" + schema.__name__.lower()
        tool = {
            "name": tool_name,
            "description": f"Return the result as a {schema.__name__} object.",
            "input_schema": schema.model_json_schema(),
        }
        msg = self._client.messages.create(
            model=model,
            max_tokens=4096,
            system=system,
            tools=[tool],
            tool_choice={"type": "tool", "name": tool_name},
            messages=[{"role": "user", "content": prompt}],
        )
        for block in msg.content:
            if getattr(block, "type", None) == "tool_use" and block.name == tool_name:
                return schema.model_validate(block.input)
        raise RuntimeError(f"Model did not return a {tool_name} tool call.")

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
        messages: list[dict] = [{"role": "user", "content": prompt}]
        for _ in range(max_turns):
            msg = self._client.messages.create(
                model=model,
                max_tokens=4096,
                system=system,
                tools=tools,
                messages=messages,
            )
            messages.append({"role": "assistant", "content": msg.content})
            tool_uses = [b for b in msg.content if getattr(b, "type", None) == "tool_use"]
            if not tool_uses:
                # No more tool calls: return concatenated text.
                return "".join(
                    getattr(b, "text", "") for b in msg.content
                    if getattr(b, "type", None) == "text"
                )
            results = []
            for tu in tool_uses:
                impl = tool_impls.get(tu.name)
                out = impl(tu.input) if impl else f"error: unknown tool {tu.name}"
                results.append(
                    {"type": "tool_result", "tool_use_id": tu.id, "content": out}
                )
            messages.append({"role": "user", "content": results})
        return "max_turns_exhausted"
