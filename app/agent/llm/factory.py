from __future__ import annotations

from app.agent.llm.base import StructuredLLM


def build_structured_llm(
    provider: str, model: str, api_key: str, base_url: str | None = None
) -> StructuredLLM:
    if provider == "anthropic":
        from app.agent.llm.anthropic import build_anthropic

        return build_anthropic(model, api_key)
    if provider == "openai":
        from app.agent.llm.openai import build_openai

        return build_openai(model, api_key)
    if provider == "custom":
        from app.agent.llm.slm import build_slm

        if not base_url:
            raise ValueError("custom provider requires base_url")
        return build_slm(model, api_key, base_url)
    raise ValueError(f"Unknown LLM provider: {provider!r}")
