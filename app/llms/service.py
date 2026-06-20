from __future__ import annotations

import uuid
from typing import Any

from fastapi import HTTPException

from app import crypto
from app.db import queries
from app.llms.schemas import LLMCreate, LLMOut


def _to_out(row: dict[str, Any]) -> LLMOut:
    return LLMOut(
        id=row["id"],
        name=row["name"],
        provider=row["provider"],
        model=row["model"],
        base_url=row["base_url"],
        secret_last4=row["secret_last4"],
        created_at=row["created_at"],
    )


def _validate(payload: LLMCreate) -> None:
    if payload.provider == "custom" and not payload.base_url:
        raise HTTPException(
            status_code=400, detail="base_url is required for provider 'custom'."
        )
    if payload.provider in ("openai", "anthropic") and not payload.api_key:
        raise HTTPException(
            status_code=400, detail=f"api_key is required for provider '{payload.provider}'."
        )


async def create_llm(payload: LLMCreate) -> LLMOut:
    _validate(payload)
    api_key_enc = crypto.encrypt(payload.api_key) if payload.api_key else None
    last4 = crypto.last4(payload.api_key) if payload.api_key else None

    row = await queries.create_llm(
        name=payload.name,
        provider=payload.provider,
        base_url=payload.base_url,
        model=payload.model,
        api_key_enc=api_key_enc,
        key_version=crypto.KEY_VERSION,
        secret_last4=last4,
    )
    return _to_out(row)


async def list_llms() -> list[LLMOut]:
    return [_to_out(r) for r in await queries.list_llms()]


async def get_llm(llm_id: uuid.UUID) -> LLMOut:
    row = await queries.get_llm(llm_id)
    if row is None:
        raise HTTPException(status_code=404, detail="LLM registration not found")
    return _to_out(row)


async def delete_llm(llm_id: uuid.UUID) -> None:
    if not await queries.delete_llm(llm_id):
        raise HTTPException(status_code=404, detail="LLM registration not found")
