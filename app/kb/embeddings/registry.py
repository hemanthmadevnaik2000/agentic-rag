from __future__ import annotations

from functools import lru_cache

from app.config import get_settings
from app.kb.embeddings.base import Embeddings


@lru_cache(maxsize=8)
def get_embeddings(provider: str | None = None, model: str | None = None) -> Embeddings:
    """Return a cached embedding provider. Defaults come from settings.

    The instance is cached (model load is expensive) keyed by (provider, model).
    """
    settings = get_settings()
    provider = provider or settings.embedding_provider
    model = model or settings.embedding_model

    if provider == "local":
        from app.kb.embeddings.local import LocalEmbeddings

        return LocalEmbeddings(model)
    if provider == "openai":
        from app.kb.embeddings.openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model, api_key=settings.openai_api_key, dim=settings.embedding_dim)
    raise ValueError(f"Unknown embedding provider: {provider!r}")
