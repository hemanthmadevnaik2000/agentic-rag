from __future__ import annotations

import uuid

from fastapi import APIRouter, status

from app.agent import management
from app.agent.management import AgentCreate, AgentOut

router = APIRouter(prefix="/agents", tags=["agents"])


@router.post("", response_model=AgentOut, status_code=status.HTTP_201_CREATED)
async def create_agent(payload: AgentCreate) -> AgentOut:
    return await management.create_agent(payload)


@router.get("", response_model=list[AgentOut])
async def list_agents() -> list[AgentOut]:
    return await management.list_agents()


@router.get("/{agent_id}", response_model=AgentOut)
async def get_agent(agent_id: uuid.UUID) -> AgentOut:
    return await management.get_agent(agent_id)


@router.delete("/{agent_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_agent(agent_id: uuid.UUID) -> None:
    await management.delete_agent(agent_id)
