from __future__ import annotations

from functools import lru_cache
from typing import Any

from app.config import get_settings


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)
        self.model_name = model_name

    def rerank(
        self, query: str, candidates: list[dict[str, Any]], top_k: int
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []
        pairs = [(query, c["text"]) for c in candidates]
        scores = self._model.predict(pairs)
        ranked = sorted(
            zip(candidates, scores), key=lambda cs: float(cs[1]), reverse=True
        )
        return [{**c, "rerank_score": float(s)} for c, s in ranked[:top_k]]


@lru_cache(maxsize=4)
def get_reranker(model_name: str | None = None) -> CrossEncoderReranker:
    return CrossEncoderReranker(model_name or get_settings().reranker_model)
