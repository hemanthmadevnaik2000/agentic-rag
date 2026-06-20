"""query_knowledge_base capability.

kb_ids are bound into the target server-side from the websocket session; the model
never supplies them, which enforces the KB allowlist (a hallucinated or injected
kb_id can never widen access).
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable

from app.kb.search.retriever import RetrievalTarget, hybrid_retrieve

QueryFn = Callable[[str], Awaitable[list[dict[str, Any]]]]


def make_query_knowledge_base(target: RetrievalTarget) -> QueryFn:
    async def query_knowledge_base(query: str) -> list[dict[str, Any]]:
        return await hybrid_retrieve(query, target)

    return query_knowledge_base
