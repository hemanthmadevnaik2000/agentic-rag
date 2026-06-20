"""Assemble a runnable agent graph from a registered agent + requested KBs.

Enforces the KB allowlist: requested kb_ids must be a subset of the agent
configured kb_ids. All KBs in play must share one embedding space and destination
(one Milvus collection), so a multi-KB query is a single filtered search.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from app import crypto
from app.agent.graph import build_agent_graph
from app.agent.llm.factory import build_structured_llm
from app.config import get_settings
from app.db import queries
from app.kb.search.retriever import RetrievalTarget


class RuntimeError_(ValueError):
    """Raised when an agent runtime cannot be built (bad config / mixed KBs)."""


@dataclass
class AgentRuntime:
    graph: Any
    agent: dict[str, Any]
    kb_ids: list[str]


async def build_runtime(
    agent_id: uuid.UUID, requested_kb_ids: list[uuid.UUID] | None = None
) -> AgentRuntime:
    settings = get_settings()

    agent = await queries.get_agent(agent_id)
    if agent is None:
        raise RuntimeError_("Agent not found")

    configured = set(agent["kb_ids"])
    effective = set(requested_kb_ids) if requested_kb_ids else configured
    if not effective:
        raise RuntimeError_("No knowledge bases attached to this agent.")
    if not effective.issubset(configured):
        raise RuntimeError_("Requested kb_ids are not all attached to this agent.")

    llm_row = await queries.get_llm(agent["llm_id"])
    if llm_row is None:
        raise RuntimeError_("Agent LLM registration not found.")
    api_key = (
        crypto.decrypt(llm_row["api_key_enc"]) if llm_row["api_key_enc"] else ""
    )
    llm = build_structured_llm(
        llm_row["provider"], llm_row["model"], api_key, llm_row["base_url"]
    )

    kbs = await queries.get_kbs_by_ids(list(effective))
    if not kbs:
        raise RuntimeError_("Knowledge bases not found.")

    spaces = {(k["embedding_model"], k["embedding_dim"]) for k in kbs}
    if len(spaces) != 1:
        raise RuntimeError_("Attached KBs use different embedding spaces.")
    destinations = {k["destination_id"] for k in kbs}
    if len(destinations) != 1:
        raise RuntimeError_("Attached KBs use different destinations.")

    dest = await queries.get_destination(next(iter(destinations)))
    if dest is None:
        raise RuntimeError_("Destination not found.")
    token = crypto.decrypt(dest["secret_enc"]) if dest["secret_enc"] else ""
    uri = f"http://{dest['host']}:{dest['port']}"

    model, dim = next(iter(spaces))
    provider = (kbs[0].get("metadata") or {}).get(
        "_embedding_provider"
    ) or settings.embedding_provider

    target = RetrievalTarget(
        kb_ids=tuple(str(k) for k in effective),
        embedding_provider=provider,
        embedding_model=model,
        embedding_dim=dim,
        milvus_uri=uri,
        milvus_token=token,
        enable_bm25=settings.enable_bm25,
    )
    graph = build_agent_graph(
        llm,
        target,
        confidence_threshold=agent["confidence_threshold"],
        max_retries=agent["max_retries"],
        system_prompt=agent["system_prompt"],
    )
    return AgentRuntime(graph=graph, agent=agent, kb_ids=[str(k) for k in effective])
