"""PipelineEngine: fixed retrieve -> rerank -> generate -> validate flow.

For models without tool calling (SLMs). Uses the StructuredLLM port (prompt + JSON
repair on SLMs). Shares rewrite / cache / respond nodes with the tool-calling engine.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.agent.engines.base import AgentEngine, EngineContext
from app.agent.engines.common import (
    DEFAULT_SYSTEM,
    FALLBACK_ANSWER,
    format_context,
    format_history,
    make_cache_lookup_node,
    make_cache_write_node,
    make_respond_node,
    make_rewrite_node,
)
from app.agent.llm.factory import build_structured_llm
from app.agent.schemas import Answer
from app.kb.search.retriever import fuse_candidates, rerank_candidates


class PipelineState(TypedDict, total=False):
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


class PipelineEngine(AgentEngine):
    name = "pipeline"

    def build_graph(self, ctx: EngineContext) -> Any:
        llm = build_structured_llm(
            ctx.llm_row["provider"], ctx.llm_row["model"], ctx.api_key, ctx.llm_row["base_url"]
        )
        system = ctx.agent.get("system_prompt") or DEFAULT_SYSTEM
        threshold = ctx.agent["confidence_threshold"]
        max_retries = ctx.agent["max_retries"]
        target = ctx.target

        def _query(state: PipelineState) -> str:
            return state.get("rewritten") or state["question"]

        async def retrieve_node(state: PipelineState) -> PipelineState:
            return {"fused": await fuse_candidates(_query(state), target)}

        async def rerank_node(state: PipelineState) -> PipelineState:
            return {"retrieved": await rerank_candidates(_query(state), state.get("fused", []))}

        async def generate_node(state: PipelineState) -> PipelineState:
            context = format_context(state.get("retrieved", []))
            history = format_history(state.get("history", []))
            feedback = state.get("feedback", "")
            user = (
                f"{history}Question: {_query(state)}\n\n"
                f"Context chunks:\n{context}\n\n{feedback}\n"
                "Answer using only the context above and set references to the chunk_id "
                "values you used."
            )
            return {"answer": await llm.generate(system, user, Answer)}

        async def validate_node(state: PipelineState) -> PipelineState:
            answer = state.get("answer")
            attempts = state.get("attempts", 0)
            grounded = (
                answer is not None
                and answer.confidence >= threshold
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

        def route_after_cache(state: PipelineState) -> str:
            return "respond" if state.get("cache_decision") == "hit" else "retrieve"

        def route_after_validate(state: PipelineState) -> str:
            return state.get("decision", "respond")

        g = StateGraph(PipelineState)
        g.add_node("rewrite", make_rewrite_node(llm))
        g.add_node("cache_lookup", make_cache_lookup_node(ctx.cache_scope))
        g.add_node("retrieve", retrieve_node)
        g.add_node("rerank", rerank_node)
        g.add_node("generate", generate_node)
        g.add_node("validate", validate_node)
        g.add_node("cache_write", make_cache_write_node(ctx.cache_scope, ctx.cache_kb_ids))
        g.add_node("respond", make_respond_node())

        g.add_edge(START, "rewrite")
        g.add_edge("rewrite", "cache_lookup")
        g.add_conditional_edges(
            "cache_lookup", route_after_cache, {"respond": "respond", "retrieve": "retrieve"}
        )
        g.add_edge("retrieve", "rerank")
        g.add_edge("rerank", "generate")
        g.add_edge("generate", "validate")
        g.add_conditional_edges(
            "validate",
            route_after_validate,
            {"generate": "generate", "cache_write": "cache_write", "respond": "respond"},
        )
        g.add_edge("cache_write", "respond")
        g.add_edge("respond", END)
        return g.compile(checkpointer=ctx.checkpointer)
