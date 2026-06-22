"""Build a raw LangChain chat model for tool-calling agents (bind_tools).

Distinct from the StructuredLLM port (which does single structured generations):
the tool-calling engine needs a model it can bind tools to and that returns
tool_calls. Used only by tool-capable providers.
"""
from __future__ import annotations

from typing import Any


def build_chat_model(provider: str, model: str, api_key: str, base_url: str | None = None) -> Any:
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, api_key=api_key, max_tokens=2048)
    if provider in ("openai", "custom"):
        from langchain_openai import ChatOpenAI

        kwargs: dict[str, Any] = {"model": model, "api_key": api_key or "not-needed"}
        if base_url:
            kwargs["base_url"] = base_url
        return ChatOpenAI(**kwargs)
    raise ValueError(f"Unknown chat provider: {provider!r}")
