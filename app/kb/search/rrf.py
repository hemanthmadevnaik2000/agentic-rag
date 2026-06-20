"""Reciprocal Rank Fusion: combine ranked lists without score normalization."""
from __future__ import annotations

from typing import Any


def reciprocal_rank_fusion(
    rankings: list[list[dict[str, Any]]],
    k: int = 60,
    id_key: str = "chunk_id",
) -> list[dict[str, Any]]:
    scores: dict[str, float] = {}
    items: dict[str, dict[str, Any]] = {}

    for ranking in rankings:
        for rank, item in enumerate(ranking):
            cid = item[id_key]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            # Keep the richest record we have seen for this id (prefer one with text).
            if cid not in items or (not items[cid].get("text") and item.get("text")):
                items[cid] = item

    fused = [
        {**items[cid], "rrf_score": score}
        for cid, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    ]
    return fused
