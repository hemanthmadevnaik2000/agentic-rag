from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel


class StructuredLLM(ABC):
    """Provider-agnostic structured generation.

    generate() returns a validated instance of the requested pydantic schema.
    """

    @abstractmethod
    async def generate(self, system: str, user: str, schema: type[BaseModel]) -> BaseModel:
        ...


class LangChainStructuredLLM(StructuredLLM):
    """Adapter over a LangChain chat model using native structured output
    (with_structured_output). Works for OpenAI and Anthropic.
    """

    def __init__(self, chat_model) -> None:
        self._model = chat_model

    async def generate(self, system: str, user: str, schema: type[BaseModel]) -> BaseModel:
        runnable = self._model.with_structured_output(schema)
        return await runnable.ainvoke([("system", system), ("user", user)])
