from __future__ import annotations

import uuid

from fastapi import APIRouter, status

from app.llms import service
from app.llms.schemas import LLMCreate, LLMOut

router = APIRouter(prefix="/llms", tags=["llms"])


@router.post("", response_model=LLMOut, status_code=status.HTTP_201_CREATED)
async def create_llm(payload: LLMCreate) -> LLMOut:
    return await service.create_llm(payload)


@router.get("", response_model=list[LLMOut])
async def list_llms() -> list[LLMOut]:
    return await service.list_llms()


@router.get("/{llm_id}", response_model=LLMOut)
async def get_llm(llm_id: uuid.UUID) -> LLMOut:
    return await service.get_llm(llm_id)


@router.delete("/{llm_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_llm(llm_id: uuid.UUID) -> None:
    await service.delete_llm(llm_id)
