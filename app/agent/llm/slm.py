"""Self-hosted / custom SLM adapter.

Many small/self-hosted models lack native structured output, so we prompt for a
single JSON object and validate it against the schema, with one repair attempt.
"""
from __future__ import annotations

import json
import re

from pydantic import BaseModel, ValidationError

from app.agent.llm.base import StructuredLLM

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> str:
    match = _JSON_RE.search(text)
    return match.group(0) if match else text


class SLMStructuredLLM(StructuredLLM):
    def __init__(self, chat_model) -> None:
        self._model = chat_model

    async def _invoke(self, system: str, user: str) -> str:
        msg = await self._model.ainvoke([("system", system), ("user", user)])
        content = getattr(msg, "content", msg)
        return content if isinstance(content, str) else str(content)

    async def generate(self, system: str, user: str, schema: type[BaseModel]) -> BaseModel:
        schema_json = json.dumps(schema.model_json_schema())
        prompt = (
            f"{user}\n\nReturn ONLY a single JSON object conforming to this JSON Schema. "
            f"No prose, no markdown fences.\nSchema: {schema_json}"
        )
        text = await self._invoke(system, prompt)
        try:
            return schema.model_validate_json(_extract_json(text))
        except (ValidationError, ValueError):
            repair = (
                "Your previous output was not valid for the schema. Fix it and return "
                f"ONLY the JSON object.\nSchema: {schema_json}\nPrevious output:\n{text}"
            )
            text2 = await self._invoke(system, repair)
            return schema.model_validate_json(_extract_json(text2))


def build_slm(model: str, api_key: str | None, base_url: str) -> StructuredLLM:
    from langchain_openai import ChatOpenAI

    chat = ChatOpenAI(model=model, api_key=api_key or "not-needed", base_url=base_url)
    return SLMStructuredLLM(chat)
