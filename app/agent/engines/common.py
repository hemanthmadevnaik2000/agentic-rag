"""Shared building blocks for agent engines: helpers + reusable graph nodes.

Both engines (pipeline, tool_calling) share the same front (rewrite -> cache_lookup),
tail (cache_write -> respond), source-building, and groundedness constants, so they
live here and are composed by each engine.
"""
from __future__ import annotations

from typing import Any

from app.agent import cache as cache_mod
from app.agent.rewrite import rewrite_query
from app.agent.schemas import Answer

DEFAULT_SYSTEM = (
    "You are a retrieval-grounded assistant. Answer ONLY using context retrieved from "
    "the knowledge bases. If the context does not contain the answer, say so and set "
    "confidence low. Cite the chunk_id values you actually used. Never invent facts or "
    "cite chunk_ids that are not in the retrieved context."
)

FALLBACK_ANSWER = (
    "I could not find a confident, grounded answer in the attached knowledge bases."
)

_MAX_HISTORY_TURNS = 12


def build_sources(
    references: list[str], retrieved: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Group cited chunk_ids by their source document (filename, version)."""
    by_id = {c.get("chunk_id"): c for c in retrieved}
    grouped: dict[tuple[Any, Any], dict[str, Any]] = {}
    for chunk_id in references:
        chunk = by_id.get(chunk_id)
        if chunk is None:
            continue
        key = (chunk.get("filename"), chunk.get("version"))
        entry = grouped.setdefault(
            key,
            {"filename": chunk.get("filename"), "version": chunk.get("version"), "chunk_ids": []},
        )
        entry["chunk_ids"].append(chunk_id)
    return list(grouped.values())


def format_context(chunks: list[dict[str, Any]]) -> str:
    blocks = []
    for c in chunks:
        cid = c.get("chunk_id")
        fname = c.get("filename")
        text = c.get("text", "")
        blocks.append(f"[chunk_id={cid} file={fname}]\n{text}")
    return "\n\n".join(blocks) if blocks else "(no context retrieved)"


def format_history(history: list[dict[str, Any]]) -> str:
    if not history:
        return ""
    recent = history[-_MAX_HISTORY_TURNS:]
    lines = []
    for h in recent:
        role = str(h.get("role", "user")).capitalize()
        content = h.get("content", "")
        lines.append(f"{role}: {content}")
    return "Conversation so far:\n" + "\n".join(lines) + "\n\n"


def make_rewrite_node(structured_llm: Any):
    async def rewrite_node(state: dict[str, Any]) -> dict[str, Any]:
        rewritten = await rewrite_query(
            structured_llm, state["question"], state.get("history", [])
        )
        return {"rewritten": rewritten}

    return rewrite_node


def make_cache_lookup_node(cache_scope: str | None):
    async def cache_lookup_node(state: dict[str, Any]) -> dict[str, Any]:
        if not cache_scope or not cache_mod.cache_enabled():
            return {"cache_decision": "miss", "cached": False}
        query = state.get("rewritten") or state["question"]
        hit = await cache_mod.lookup(cache_scope, query)
        if hit:
            answer = Answer(
                answer=hit.get("answer", ""),
                references=hit.get("references", []),
                confidence=hit.get("confidence", 0.0),
            )
            return {
                "answer": answer,
                "sources": hit.get("sources", []),
                "cached": True,
                "cache_decision": "hit",
            }
        return {"cache_decision": "miss", "cached": False}

    return cache_lookup_node


def make_cache_write_node(cache_scope: str | None, cache_kb_ids: list[str]):
    async def cache_write_node(state: dict[str, Any]) -> dict[str, Any]:
        answer = state["answer"]
        sources = build_sources(answer.references, state.get("retrieved", []) or [])
        if cache_scope and cache_mod.cache_enabled():
            payload = {
                "answer": answer.answer,
                "references": answer.references,
                "sources": sources,
                "confidence": answer.confidence,
            }
            query = state.get("rewritten") or state["question"]
            await cache_mod.store(cache_scope, cache_kb_ids, query, payload)
        return {"sources": sources}

    return cache_write_node


def make_respond_node():
    async def respond_node(state: dict[str, Any]) -> dict[str, Any]:
        answer = state.get("answer")
        turn = [
            {"role": "user", "content": state["question"]},
            {"role": "assistant", "content": answer.answer if answer else ""},
        ]
        return {"history": turn}

    return respond_node
