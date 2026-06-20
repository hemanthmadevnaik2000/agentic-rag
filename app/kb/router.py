from __future__ import annotations

import json
import uuid
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.kb import service
from app.kb.schemas import KBOut

router = APIRouter(prefix="/kb", tags=["kb"])


def _parse_metadata(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"metadata is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=400, detail="metadata must be a JSON object.")
    return parsed


@router.post("", response_model=KBOut, status_code=status.HTTP_202_ACCEPTED)
async def create_kb(
    name: str = Form(...),
    destination_id: uuid.UUID = Form(...),
    embedding_provider: str | None = Form(None),
    embedding_model: str | None = Form(None),
    metadata: str = Form("{}"),
    files: list[UploadFile] = File(...),
) -> KBOut:
    return await service.create_kb(
        name=name,
        destination_id=destination_id,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        metadata=_parse_metadata(metadata),
        files=files,
    )


@router.get("", response_model=list[KBOut])
async def list_kbs() -> list[KBOut]:
    return await service.list_kbs()


@router.get("/{kb_id}", response_model=KBOut)
async def get_kb(kb_id: uuid.UUID) -> KBOut:
    return await service.get_kb(kb_id)
