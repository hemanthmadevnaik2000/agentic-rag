"""LangGraph agent: retrieve -> rerank -> generate (structured) -> validate -> respond.

validate enforces the groundedness gate (self-reported confidence + non-empty
references). On failure it retries generation with feedback up to max_retries,
then returns a safe fallback. Node names map to websocket status events.
"""
from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.agent.llm.base import StructuredLLM
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


class AgentState(TypedDict, total=False):
    question: str
    fused: list[dict[str, Any]]
    retrieved: list[dict[str, Any]]
    answer: Answer
    attempts: int
    feedback: str
    rejected: bool
    decision: str


def _format_context(chunks: list[dict[str, Any]]) -> str:
    blocks = []
    for c in chunks:
        blocks.append(
            f"[chunk_id={c.get('chunk_id')} file={c.get('filename')}]\n{c.get('text', '')}"
        )
    return "\n\n".join(blocks) if blocks else "(no context retrieved)"


def build_agent_graph(
    llm: StructuredLLM,
    target: RetrievalTarget,
    *,
    confidence_threshold: float,
    max_retries: int,
    system_prompt: str | None,
):
    system = system_prompt or DEFAULT_SYSTEM

    async def retrieve_node(state: AgentState) -> AgentState:
        fused = await fuse_candidates(state["question"], target)
        return {"fused": fused}

    async def rerank_node(state: AgentState) -> AgentState:
        reranked = await rerank_candidates(state["question"], state.get("fused", []))
        return {"retrieved": reranked}

    async def generate_node(state: AgentState) -> AgentState:
        context = _format_context(state.get("retrieved", []))
        feedback = state.get("feedback", "")
        user = (
            f"Question: {state['question']}\n\n"
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
            return {"decision": "respond", "rejected": False}
        if attempts < max_retries:
            return {
                "decision": "generate",
                "attempts": attempts + 1,
                "feedback": (
                    "Your previous answer was rejected (low confidence or no "
                    "references). Re-examine the context; only answer if it is "
                    "supported, otherwise keep confidence low."
                ),
            }
        return {
            "decision": "respond",
            "rejected": True,
            "answer": Answer(answer=FALLBACK_ANSWER, references=[], confidence=0.0),
        }

    async def respond_node(state: AgentState) -> AgentState:
        return {}

    def route_after_validate(state: AgentState) -> str:
        return state.get("decision", "respond")

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("generate", generate_node)
    graph.add_node("validate", validate_node)
    graph.add_node("respond", respond_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "generate")
    graph.add_edge("generate", "validate")
    graph.add_conditional_edges(
        "validate", route_after_validate, {"generate": "generate", "respond": "respond"}
    )
    graph.add_edge("respond", END)
    return graph.compile()
