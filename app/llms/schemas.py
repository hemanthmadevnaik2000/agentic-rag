from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Provider = Literal["openai", "anthropic", "custom"]


class LLMCreate(BaseModel):
    name: str
    provider: Provider
    model: str
    base_url: str | None = None  # required for custom / self-hosted
    # Write-only: never echoed back.
    api_key: str | None = Field(default=None, repr=False)


class LLMOut(BaseModel):
    id: uuid.UUID
    name: str
    provider: str
    model: str
    base_url: str | None
    secret_last4: str | None
    created_at: datetime
