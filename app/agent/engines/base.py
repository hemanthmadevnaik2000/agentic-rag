"""Agent engine interface + shared context.

An AgentEngine turns a registered agent + KB target into a compiled LangGraph with a
uniform state contract (input {question}; output state with answer/sources/cached/
rejected/history) so the websocket layer is engine-agnostic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from app.kb.search.retriever import RetrievalTarget


@dataclass
class EngineContext:
    llm_row: dict[str, Any]
    api_key: str
    agent: dict[str, Any]
    target: RetrievalTarget
    name_to_kb_id: dict[str, str]
    checkpointer: Any
    cache_scope: str | None
    cache_kb_ids: list[str]


class AgentEngine(ABC):
    name: str

    @abstractmethod
    def build_graph(self, ctx: EngineContext) -> Any:
        """Return a compiled LangGraph for this engine."""
