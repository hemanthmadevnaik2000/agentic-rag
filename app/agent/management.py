"""Agent registration CRUD (configuration), separate from the runtime engine."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel, Field

from app.db import queries


class AgentCreate(BaseModel):
    name: str
    llm_id: uuid.UUID
    kb_ids: list[uuid.UUID] = Field(default_factory=list)
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_retries: int = Field(default=1, ge=0, le=5)
    system_prompt: str | None = None


class AgentOut(BaseModel):
    id: uuid.UUID
    name: str
    llm_id: uuid.UUID
    kb_ids: list[uuid.UUID]
    confidence_threshold: float
    max_retries: int
    system_prompt: str | None
    created_at: datetime


def _to_out(row: dict[str, Any]) -> AgentOut:
    return AgentOut(
        id=row["id"],
        name=row["name"],
        llm_id=row["llm_id"],
        kb_ids=list(row["kb_ids"]),
        confidence_threshold=row["confidence_threshold"],
        max_retries=row["max_retries"],
        system_prompt=row["system_prompt"],
        created_at=row["created_at"],
    )


async def create_agent(payload: AgentCreate) -> AgentOut:
    if await queries.get_llm(payload.llm_id) is None:
        raise HTTPException(status_code=400, detail="llm_id not found.")
    row = await queries.create_agent(
        name=payload.name,
        llm_id=payload.llm_id,
        kb_ids=payload.kb_ids,
        confidence_threshold=payload.confidence_threshold,
        max_retries=payload.max_retries,
        system_prompt=payload.system_prompt,
    )
    return _to_out(row)


async def list_agents() -> list[AgentOut]:
    return [_to_out(r) for r in await queries.list_agents()]


async def get_agent(agent_id: uuid.UUID) -> AgentOut:
    row = await queries.get_agent(agent_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    return _to_out(row)


async def delete_agent(agent_id: uuid.UUID) -> None:
    if not await queries.delete_agent(agent_id):
        raise HTTPException(status_code=404, detail="Agent not found")
