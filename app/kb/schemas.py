from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel


class KBOut(BaseModel):
    id: uuid.UUID
    name: str
    destination_id: uuid.UUID
    embedding_model: str
    embedding_dim: int
    status: str
    error: str | None
    metadata: dict[str, Any]
    created_at: datetime
