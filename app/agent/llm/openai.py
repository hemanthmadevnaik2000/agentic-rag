from __future__ import annotations

from app.agent.llm.base import LangChainStructuredLLM, StructuredLLM


def build_openai(model: str, api_key: str) -> StructuredLLM:
    from langchain_openai import ChatOpenAI

    chat = ChatOpenAI(model=model, api_key=api_key)
    return LangChainStructuredLLM(chat)
