"""Agent tool surface (for the tool-calling engine).

QueryKnowledgeBase and SubmitAnswer are the model-facing tool schemas (bound via
bind_tools). kb_name is resolved to a kb_id from the agent allowlist server-side,
so the model can never widen KB access. run_query_kb executes a search against the
single selected KB and returns both display text (for the model) and the raw chunks
(for source attribution).
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from pydantic import BaseModel, Field

from app.kb.search.retriever import RetrievalTarget, hybrid_retrieve


class QueryKnowledgeBase(BaseModel):
    """Search one knowledge base for relevant context. Call it multiple times
    (in parallel is fine) to decompose a question or to query different KBs."""

    query: str = Field(description="The search query.")
    kb_name: str = Field(description="Name of the knowledge base to search.")


class SubmitAnswer(BaseModel):
    """Submit the final grounded answer once you have enough context. Cite the
    chunk_id values you used and give a calibrated confidence."""

    answer: str = Field(description="The answer, grounded only in retrieved context.")
    references: list[str] = Field(
        default_factory=list, description="chunk_id values that support the answer."
    )
    confidence: float = Field(description="Self-assessed confidence from 0 to 1.")


async def run_query_kb(
    query: str,
    kb_name: str,
    target: RetrievalTarget,
    name_to_kb_id: dict[str, str],
) -> tuple[str, list[dict[str, Any]]]:
    kb_id = name_to_kb_id.get(kb_name)
    if kb_id is None:
        valid = ", ".join(sorted(name_to_kb_id)) or "(none)"
        return f"Unknown kb_name {kb_name}. Available knowledge bases: {valid}", []
    single = replace(target, kb_ids=(kb_id,))
    chunks = await hybrid_retrieve(query, single)
    lines = []
    for c in chunks:
        cid = c.get("chunk_id")
        fname = c.get("filename")
        snippet = (c.get("text") or "")[:500]
        lines.append(f"[chunk_id={cid} file={fname}] {snippet}")
    text = "\n".join(lines) if lines else "No results found in this knowledge base."
    return text, chunks
