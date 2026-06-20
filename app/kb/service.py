from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from fastapi import HTTPException, UploadFile

from app.config import get_settings
from app.db import queries
from app.kb.ingestion.loaders import SUPPORTED_EXTENSIONS
from app.kb.schemas import KBOut
from app.queue import get_arq

UPLOAD_DIR = Path("data/uploads")


def _to_out(row: dict[str, Any]) -> KBOut:
    return KBOut(
        id=row["id"],
        name=row["name"],
        destination_id=row["destination_id"],
        embedding_model=row["embedding_model"],
        embedding_dim=row["embedding_dim"],
        status=row["status"],
        error=row["error"],
        metadata=row["metadata"] or {},
        created_at=row["created_at"],
    )


async def create_kb(
    *,
    name: str,
    destination_id: uuid.UUID,
    embedding_provider: str | None,
    embedding_model: str | None,
    metadata: dict[str, Any],
    files: list[UploadFile],
) -> KBOut:
    settings = get_settings()

    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required.")
    for f in files:
        if Path(f.filename or "").suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file {f.filename!r}; supported: {', '.join(SUPPORTED_EXTENSIONS)}",
            )

    dest = await queries.get_destination(destination_id)
    if dest is None:
        raise HTTPException(status_code=400, detail="Destination not found.")

    provider = embedding_provider or settings.embedding_provider
    model = embedding_model or settings.embedding_model
    dim = settings.embedding_dim
    stored_meta = {**metadata, "_embedding_provider": provider}

    kb = await queries.create_kb(
        name=name,
        destination_id=destination_id,
        embedding_model=model,
        embedding_dim=dim,
        metadata=stored_meta,
    )
    kb_id = kb["id"]

    kb_dir = UPLOAD_DIR / str(kb_id)
    kb_dir.mkdir(parents=True, exist_ok=True)
    saved: list[tuple[str, str]] = []
    for f in files:
        safe_name = Path(f.filename or "upload").name
        target = kb_dir / safe_name
        target.write_bytes(await f.read())
        saved.append((safe_name, str(target)))

    await get_arq().enqueue_job("ingest_kb", str(kb_id), saved, stored_meta)
    return _to_out(kb)


async def list_kbs() -> list[KBOut]:
    return [_to_out(r) for r in await queries.list_kbs()]


async def get_kb(kb_id: uuid.UUID) -> KBOut:
    row = await queries.get_kb(kb_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return _to_out(row)
