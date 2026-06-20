from __future__ import annotations

from pydantic import BaseModel, Field


class Answer(BaseModel):
    """Structured, grounded answer enforced on the LLM.

    references holds chunk_id values from the retrieved context. They are not
    hard-validated against the retrieved set yet (self-reported gate), but emitting
    them keeps the upgrade to chunk-id validation a small later change.
    """

    answer: str = Field(description="The answer, grounded only in the provided context.")
    references: list[str] = Field(
        default_factory=list,
        description="chunk_id values from the context that support the answer.",
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Self-assessed confidence from 0 to 1."
    )
