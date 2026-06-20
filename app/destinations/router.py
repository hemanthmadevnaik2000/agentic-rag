from __future__ import annotations

import uuid

from fastapi import APIRouter, status

from app.destinations import service
from app.destinations.schemas import DestinationCreate, DestinationOut

router = APIRouter(prefix="/destinations", tags=["destinations"])


@router.post("", response_model=DestinationOut, status_code=status.HTTP_201_CREATED)
async def create_destination(payload: DestinationCreate, validate: bool = True) -> DestinationOut:
    return await service.create_destination(payload, validate=validate)


@router.get("", response_model=list[DestinationOut])
async def list_destinations() -> list[DestinationOut]:
    return await service.list_destinations()


@router.get("/{dest_id}", response_model=DestinationOut)
async def get_destination(dest_id: uuid.UUID) -> DestinationOut:
    return await service.get_destination(dest_id)


@router.delete("/{dest_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_destination(dest_id: uuid.UUID) -> None:
    await service.delete_destination(dest_id)
