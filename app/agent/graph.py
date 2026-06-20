"""LangGraph agent with semantic caching + per-session memory.

Flow:
  rewrite -> cache_lookup --hit--> respond
                         --miss--> retrieve -> rerank -> generate -> validate
                                   validate --retry--> generate
                                            --ok-----> cache_write -> respond
                                            --reject-> respond   (not cached)

rewrite produces a standalone query used for retrieval AND as the cache key.
cache_lookup short-circuits on a semantically similar prior query (scoped per KB set).
validate enforces the groundedness gate; only accepted answers are cached.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.agent import cache as cache_mod
from app.agent.llm.base import StructuredLLM
from app.agent.rewrite import rewrite_query
from app.agent.schemas import Answer
from app.kb.search.retriever import RetrievalTarget, fuse_candidates, rerank_candidates

DEFAULT_SYSTEM = (
    "You are a retrieval-grounded assistant. Answer ONLY using the provided context "
    "chunks. If the context does not contain the answer, say so and set confidence low. "
    "Populate references with the chunk_id values you actually used. Never invent facts "
    "or cite chunk_ids that are not in the context."
)

FALLBACK_ANSWER = (
    "I could not find a confident, grounded answer in the attached knowledge bases."
)

_MAX_HISTORY_TURNS = 12


class AgentState(TypedDict, total=False):
    question: str
    rewritten: str
    fused: list[dict[str, Any]]
    retrieved: list[dict[str, Any]]
    answer: Answer
    sources: list[dict[str, Any]]
    cached: bool
    attempts: int
    feedback: str
    rejected: bool
    cache_decision: str
    decision: str
    history: Annotated[list[dict[str, Any]], operator.add]


def _format_context(chunks: list[dict[str, Any]]) -> str:
    blocks = []
    for c in chunks:
        cid = c.get("chunk_id")
        fname = c.get("filename")
        text = c.get("text", "")
        blocks.append(f"[chunk_id={cid} file={fname}]\n{text}")
    return "\n\n".join(blocks) if blocks else "(no context retrieved)"


def _format_history(history: list[dict[str, Any]]) -> str:
    if not history:
        return ""
    recent = history[-_MAX_HISTORY_TURNS:]
    lines = []
    for h in recent:
        role = str(h.get("role", "user")).capitalize()
        content = h.get("content", "")
        lines.append(f"{role}: {content}")
    return "Conversation so far:\n" + "\n".join(lines) + "\n\n"


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


def build_agent_graph(
    llm: StructuredLLM,
    target: RetrievalTarget,
    *,
    confidence_threshold: float,
    max_retries: int,
    system_prompt: str | None,
    checkpointer: Any = None,
    cache_scope: str | None = None,
    cache_kb_ids: list[str] | None = None,
):
    system = system_prompt or DEFAULT_SYSTEM
    cache_kb_ids = cache_kb_ids or []

    def _query(state: AgentState) -> str:
        return state.get("rewritten") or state["question"]

    async def rewrite_node(state: AgentState) -> AgentState:
        rewritten = await rewrite_query(llm, state["question"], state.get("history", []))
        return {"rewritten": rewritten}

    async def cache_lookup_node(state: AgentState) -> AgentState:
        if not cache_scope or not cache_mod.cache_enabled():
            return {"cache_decision": "miss", "cached": False}
        hit = await cache_mod.lookup(cache_scope, _query(state))
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

    async def retrieve_node(state: AgentState) -> AgentState:
        fused = await fuse_candidates(_query(state), target)
        return {"fused": fused}

    async def rerank_node(state: AgentState) -> AgentState:
        reranked = await rerank_candidates(_query(state), state.get("fused", []))
        return {"retrieved": reranked}

    async def generate_node(state: AgentState) -> AgentState:
        context = _format_context(state.get("retrieved", []))
        history = _format_history(state.get("history", []))
        feedback = state.get("feedback", "")
        user = (
            f"{history}"
            f"Question: {_query(state)}\n\n"
            f"Context chunks:\n{context}\n\n"
            f"{feedback}\n"
            "Answer using only the context above and set references to the chunk_id "
            "values you used."
        )
        answer = await llm.generate(system, user, Answer)
        return {"answer": answer}

    async def validate_node(state: AgentState) -> AgentState:
        answer = state.get("answer")
        attempts = state.get("attempts", 0)
        grounded = (
            answer is not None
            and answer.confidence >= confidence_threshold
            and len(answer.references) > 0
        )
        if grounded:
            return {"decision": "cache_write", "rejected": False}
        if attempts < max_retries:
            return {
                "decision": "generate",
                "attempts": attempts + 1,
                "feedback": (
                    "Your previous answer was rejected (low confidence or no "
                    "references). Re-examine the context; only answer if supported, "
                    "otherwise keep confidence low."
                ),
            }
        return {
            "decision": "respond",
            "rejected": True,
            "answer": Answer(answer=FALLBACK_ANSWER, references=[], confidence=0.0),
            "sources": [],
        }

    async def cache_write_node(state: AgentState) -> AgentState:
        answer = state["answer"]
        sources = build_sources(answer.references, state.get("retrieved", []) or [])
        if cache_scope and cache_mod.cache_enabled():
            payload = {
                "answer": answer.answer,
                "references": answer.references,
                "sources": sources,
                "confidence": answer.confidence,
            }
            await cache_mod.store(cache_scope, cache_kb_ids, _query(state), payload)
        return {"sources": sources}

    async def respond_node(state: AgentState) -> AgentState:
        answer = state.get("answer")
        turn = [
            {"role": "user", "content": state["question"]},
            {"role": "assistant", "content": answer.answer if answer else ""},
        ]
        return {"history": turn}

    def route_after_cache(state: AgentState) -> str:
        return "respond" if state.get("cache_decision") == "hit" else "retrieve"

    def route_after_validate(state: AgentState) -> str:
        return state.get("decision", "respond")

    graph = StateGraph(AgentState)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("cache_lookup", cache_lookup_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("generate", generate_node)
    graph.add_node("validate", validate_node)
    graph.add_node("cache_write", cache_write_node)
    graph.add_node("respond", respond_node)

    graph.add_edge(START, "rewrite")
    graph.add_edge("rewrite", "cache_lookup")
    graph.add_conditional_edges(
        "cache_lookup", route_after_cache, {"respond": "respond", "retrieve": "retrieve"}
    )
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "generate")
    graph.add_edge("generate", "validate")
    graph.add_conditional_edges(
        "validate",
        route_after_validate,
        {"generate": "generate", "cache_write": "cache_write", "respond": "respond"},
    )
    graph.add_edge("cache_write", "respond")
    graph.add_edge("respond", END)
    return graph.compile(checkpointer=checkpointer)
