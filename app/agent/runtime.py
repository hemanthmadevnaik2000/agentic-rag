"""Assemble a runnable agent graph from a registered agent + requested KBs.

Resolves the LLM, KBs, destination, cache scope, and a name->kb_id map (the tool
allowlist), then hands an EngineContext to the engine selected for this LLM
(tool-calling vs pipeline). KB allowlist: requested kb_ids must be a subset of the
agent configured kb_ids, and all KBs must share one embedding space + destination.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from app import crypto
from app.agent import cache as cache_mod
from app.agent.checkpointer import get_checkpointer
from app.agent.engines.base import EngineContext
from app.agent.engines.factory import select_engine
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
    api_key = crypto.decrypt(llm_row["api_key_enc"]) if llm_row["api_key_enc"] else ""

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
    host = dest["host"]
    port = dest["port"]
    uri = f"http://{host}:{port}"

    model, dim = next(iter(spaces))
    provider = (kbs[0].get("metadata") or {}).get(
        "_embedding_provider"
    ) or settings.embedding_provider

    kb_id_strings = [str(k) for k in effective]
    name_to_kb_id = {k["name"]: str(k["id"]) for k in kbs}

    target = RetrievalTarget(
        kb_ids=tuple(kb_id_strings),
        embedding_provider=provider,
        embedding_model=model,
        embedding_dim=dim,
        milvus_uri=uri,
        milvus_token=token,
        enable_bm25=settings.enable_bm25,
    )

    try:
        checkpointer = get_checkpointer()
    except RuntimeError:
        checkpointer = None

    ctx = EngineContext(
        llm_row=llm_row,
        api_key=api_key,
        agent=agent,
        target=target,
        name_to_kb_id=name_to_kb_id,
        checkpointer=checkpointer,
        cache_scope=cache_mod.make_scope(kb_id_strings),
        cache_kb_ids=kb_id_strings,
    )
    engine = select_engine(llm_row)
    graph = engine.build_graph(ctx)
    return AgentRuntime(graph=graph, agent=agent, kb_ids=kb_id_strings)
