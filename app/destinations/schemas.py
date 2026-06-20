from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class DestinationCreate(BaseModel):
    name: str
    type: str = "milvus"
    host: str
    port: int
    db_name: str | None = None
    username: str | None = None
    # Milvus token / password. Write-only: never echoed back.
    secret: str | None = Field(default=None, repr=False)


class DestinationOut(BaseModel):
    id: uuid.UUID
    name: str
    type: str
    host: str
    port: int
    db_name: str | None
    username: str | None
    secret_last4: str | None
    created_at: datetime
