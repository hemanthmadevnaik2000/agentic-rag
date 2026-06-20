from __future__ import annotations

import uuid
from typing import Any

from fastapi import HTTPException

from app import crypto
from app.db import queries
from app.destinations.schemas import DestinationCreate, DestinationOut


def _to_out(row: dict[str, Any]) -> DestinationOut:
    # Explicitly drop secret_enc / key_version: secrets never leave the service.
    return DestinationOut(
        id=row["id"],
        name=row["name"],
        type=row["type"],
        host=row["host"],
        port=row["port"],
        db_name=row["db_name"],
        username=row["username"],
        secret_last4=row["secret_last4"],
        created_at=row["created_at"],
    )


def _validate_milvus(host: str, port: int, secret: str | None) -> None:
    """Best-effort connectivity check so bad credentials fail at registration."""
    try:
        from pymilvus import connections

        alias = f"validate-{uuid.uuid4().hex[:8]}"
        connections.connect(
            alias=alias,
            uri=f"http://{host}:{port}",
            token=secret or "",
        )
        connections.disconnect(alias)
    except Exception as exc:  # noqa: BLE001 - surface as a 400 to the caller
        raise HTTPException(
            status_code=400, detail=f"Could not connect to Milvus: {exc}"
        ) from exc


async def create_destination(payload: DestinationCreate, validate: bool) -> DestinationOut:
    if validate and payload.type == "milvus":
        _validate_milvus(payload.host, payload.port, payload.secret)

    secret_enc = crypto.encrypt(payload.secret) if payload.secret else None
    last4 = crypto.last4(payload.secret) if payload.secret else None

    row = await queries.create_destination(
        name=payload.name,
        type=payload.type,
        host=payload.host,
        port=payload.port,
        db_name=payload.db_name,
        username=payload.username,
        secret_enc=secret_enc,
        key_version=crypto.KEY_VERSION,
        secret_last4=last4,
    )
    return _to_out(row)


async def list_destinations() -> list[DestinationOut]:
    return [_to_out(r) for r in await queries.list_destinations()]


async def get_destination(dest_id: uuid.UUID) -> DestinationOut:
    row = await queries.get_destination(dest_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Destination not found")
    return _to_out(row)


async def delete_destination(dest_id: uuid.UUID) -> None:
    if not await queries.delete_destination(dest_id):
        raise HTTPException(status_code=404, detail="Destination not found")
