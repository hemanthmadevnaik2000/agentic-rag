from __future__ import annotations

from abc import ABC, abstractmethod


class Embeddings(ABC):
    """Pluggable embedding provider.

    Implementations are synchronous (CPU/IO bound); async callers should wrap
    calls in ``asyncio.to_thread``. ``model_name`` + ``dim`` identify the
    embedding space and are persisted on the KB so query-time uses the same one.
    """

    model_name: str
    dim: int

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        ...
