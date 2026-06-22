"""ToolCallingEngine: a model-driven ReAct-style loop.

agent_node (LLM bound with tools) decides whether to search and how many times;
tool_node executes QueryKnowledgeBase calls (parallel) and feeds results back. The
loop ends when the model calls SubmitAnswer, whose args are the structured, gated
answer. For tool-capable models (OpenAI / Anthropic / tool-capable SLMs).
"""
from __future__ import annotations

import asyncio
import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.agent.engines.base import AgentEngine, EngineContext
from app.agent.engines.common import (
    DEFAULT_SYSTEM,
    FALLBACK_ANSWER,
    make_cache_lookup_node,
    make_cache_write_node,
    make_respond_node,
    make_rewrite_node,
)
from app.agent.llm.chat import build_chat_model
from app.agent.llm.factory import build_structured_llm
from app.agent.schemas import Answer
from app.agent.tools import QueryKnowledgeBase, SubmitAnswer, run_query_kb

_MAX_ITERATIONS = 6
_MAX_HISTORY_TURNS = 12


class ToolState(TypedDict, total=False):
    question: str
    rewritten: str
    messages: list[Any]
    retrieved: list[dict[str, Any]]
    answer: Answer
    sources: list[dict[str, Any]]
    cached: bool
    rejected: bool
    iterations: int
    cache_decision: str
    decision: str
    history: Annotated[list[dict[str, Any]], operator.add]


class ToolCallingEngine(AgentEngine):
    name = "tool_calling"

    def build_graph(self, ctx: EngineContext) -> Any:
        structured_llm = build_structured_llm(
            ctx.llm_row["provider"], ctx.llm_row["model"], ctx.api_key, ctx.llm_row["base_url"]
        )
        chat = build_chat_model(
            ctx.llm_row["provider"], ctx.llm_row["model"], ctx.api_key, ctx.llm_row["base_url"]
        )
        chat_with_tools = chat.bind_tools([QueryKnowledgeBase, SubmitAnswer], tool_choice="any")

        target = ctx.target
        name_to_kb_id = ctx.name_to_kb_id
        threshold = ctx.agent["confidence_threshold"]
        kb_names = ", ".join(sorted(name_to_kb_id)) or "(none)"
        base_system = ctx.agent.get("system_prompt") or DEFAULT_SYSTEM
        system = (
            f"{base_system}\n\nUse the QueryKnowledgeBase tool to gather context; you may "
            "call it multiple times, including in parallel, to cover sub-questions or "
            "different knowledge bases. When you have enough grounded context, call "
            "SubmitAnswer exactly once with your answer, the chunk_id values you used as "
            f"references, and a calibrated confidence. Available knowledge bases: {kb_names}."
        )

        async def seed_node(state: ToolState) -> ToolState:
            from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

            messages: list[Any] = [SystemMessage(content=system)]
            for h in state.get("history", [])[-_MAX_HISTORY_TURNS:]:
                content = h.get("content", "")
                if h.get("role") == "user":
                    messages.append(HumanMessage(content=content))
                else:
                    messages.append(AIMessage(content=content))
            messages.append(HumanMessage(content=state["question"]))
            return {"messages": messages, "retrieved": [], "iterations": 0}

        async def agent_node(state: ToolState) -> ToolState:
            ai = await chat_with_tools.ainvoke(state["messages"])
            return {
                "messages": state["messages"] + [ai],
                "iterations": state.get("iterations", 0) + 1,
            }

        async def tool_node(state: ToolState) -> ToolState:
            from langchain_core.messages import ToolMessage

            last = state["messages"][-1]
            calls = [
                c for c in (getattr(last, "tool_calls", None) or [])
                if c.get("name") == "QueryKnowledgeBase"
            ]

            async def run_one(call: dict[str, Any]):
                args = call.get("args") or {}
                text, chunks = await run_query_kb(
                    args.get("query", ""), args.get("kb_name", ""), target, name_to_kb_id
                )
                return ToolMessage(content=text, tool_call_id=call.get("id")), chunks

            results = await asyncio.gather(*[run_one(c) for c in calls])
            new_messages = [r[0] for r in results]
            new_chunks: list[dict[str, Any]] = []
            for _, chunks in results:
                new_chunks.extend(chunks)
            return {
                "messages": state["messages"] + new_messages,
                "retrieved": (state.get("retrieved", []) or []) + new_chunks,
            }

        async def finalize_node(state: ToolState) -> ToolState:
            last = state["messages"][-1]
            calls = getattr(last, "tool_calls", None) or []
            submit = next((c for c in calls if c.get("name") == "SubmitAnswer"), None)
            if submit:
                args = submit.get("args") or {}
                answer = Answer(
                    answer=args.get("answer", ""),
                    references=args.get("references", []) or [],
                    confidence=float(args.get("confidence", 0.0) or 0.0),
                )
            else:
                content = getattr(last, "content", "") or ""
                answer = Answer(
                    answer=content if isinstance(content, str) else str(content),
                    references=[],
                    confidence=0.0,
                )
            grounded = answer.confidence >= threshold and len(answer.references) > 0
            if grounded:
                return {"answer": answer, "decision": "cache_write", "rejected": False}
            return {
                "answer": Answer(answer=FALLBACK_ANSWER, references=[], confidence=0.0),
                "decision": "respond",
                "rejected": True,
                "sources": [],
            }

        def route_after_cache(state: ToolState) -> str:
            return "respond" if state.get("cache_decision") == "hit" else "seed"

        def route_after_agent(state: ToolState) -> str:
            last = state["messages"][-1]
            names = [c.get("name") for c in (getattr(last, "tool_calls", None) or [])]
            if "SubmitAnswer" in names:
                return "finalize"
            if "QueryKnowledgeBase" in names and state.get("iterations", 0) < _MAX_ITERATIONS:
                return "tools"
            return "finalize"

        def route_after_finalize(state: ToolState) -> str:
            return state.get("decision", "respond")

        g = StateGraph(ToolState)
        g.add_node("rewrite", make_rewrite_node(structured_llm))
        g.add_node("cache_lookup", make_cache_lookup_node(ctx.cache_scope))
        g.add_node("seed", seed_node)
        g.add_node("agent", agent_node)
        g.add_node("tools", tool_node)
        g.add_node("finalize", finalize_node)
        g.add_node("cache_write", make_cache_write_node(ctx.cache_scope, ctx.cache_kb_ids))
        g.add_node("respond", make_respond_node())

        g.add_edge(START, "rewrite")
        g.add_edge("rewrite", "cache_lookup")
        g.add_conditional_edges(
            "cache_lookup", route_after_cache, {"respond": "respond", "seed": "seed"}
        )
        g.add_edge("seed", "agent")
        g.add_conditional_edges(
            "agent", route_after_agent, {"tools": "tools", "finalize": "finalize"}
        )
        g.add_edge("tools", "agent")
        g.add_conditional_edges(
            "finalize", route_after_finalize, {"cache_write": "cache_write", "respond": "respond"}
        )
        g.add_edge("cache_write", "respond")
        g.add_edge("respond", END)
        return g.compile(checkpointer=ctx.checkpointer)
