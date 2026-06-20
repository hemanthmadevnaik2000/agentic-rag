from __future__ import annotations

from app.agent.llm.base import LangChainStructuredLLM, StructuredLLM


def build_anthropic(model: str, api_key: str) -> StructuredLLM:
    from langchain_anthropic import ChatAnthropic

    # Current Claude models: adaptive thinking, no prefill. with_structured_output
    # maps to the native structured-output path under the hood.
    chat = ChatAnthropic(model=model, api_key=api_key, max_tokens=2048)
    return LangChainStructuredLLM(chat)
