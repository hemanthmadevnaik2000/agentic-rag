"""Semantic cache (RedisVL on Redis Stack).

Caches answers keyed by query-embedding similarity, scoped per KB set so a cached
answer is never served against the wrong documents. TTL is native (Redis per-entry).
The cache compares query-to-query, so it stays internally consistent regardless of
the KB embedding space; the scope tag enforces KB correctness. All RedisVL access is
defensive: if the library/server is unavailable the cache simply disables itself.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any

from app.config import get_settings

_cache: Any = None
_enabled: bool = False


def _import_semantic_cache():
    try:
        from redisvl.extensions.cache.llm import SemanticCache  # redisvl >= 0.4
    except ImportError:
        from redisvl.extensions.llmcache import SemanticCache  # redisvl 0.3.x
    return SemanticCache


def _build_vectorizer():
    # Reuse the project default embeddings so we do not load a second model.
    from redisvl.utils.vectorize import CustomTextVectorizer

    from app.kb.embeddings.registry import get_embeddings

    embeddings = get_embeddings()
    return CustomTextVectorizer(embed=lambda text: embeddings.embed_query(text))


def make_scope(kb_ids: list[str]) -> str:
    joined = ",".join(sorted(str(k) for k in kb_ids))
    return hashlib.sha1(joined.encode()).hexdigest()


async def init_cache() -> Any:
    global _cache, _enabled
    settings = get_settings()
    if not settings.semantic_cache_enabled:
        _enabled = False
        return None

    def _build():
        SemanticCache = _import_semantic_cache()
        return SemanticCache(
            name="agentic_rag_cache",
            redis_url=settings.redis_url,
            distance_threshold=settings.semantic_cache_max_distance,
            ttl=settings.semantic_cache_ttl_seconds,
            vectorizer=_build_vectorizer(),
            filterable_fields=[
                {"name": "scope", "type": "tag"},
                {"name": "kb", "type": "tag"},
            ],
        )

    try:
        _cache = await asyncio.to_thread(_build)
        _enabled = True
    except Exception:  # noqa: BLE001 - cache is best-effort; never block startup
        _cache = None
        _enabled = False
    return _cache


async def close_cache() -> None:
    global _cache, _enabled
    _cache = None
    _enabled = False


def cache_enabled() -> bool:
    return _enabled and _cache is not None


async def lookup(scope: str, query: str) -> dict[str, Any] | None:
    if not cache_enabled():
        return None

    def _check():
        from redisvl.query.filter import Tag

        results = _cache.check(
            prompt=query,
            num_results=1,
            filter_expression=(Tag("scope") == scope),
            return_fields=["response"],
        )
        return results

    try:
        results = await asyncio.to_thread(_check)
    except Exception:  # noqa: BLE001
        return None
    if not results:
        return None
    raw = results[0].get("response")
    try:
        return json.loads(raw) if raw else None
    except (TypeError, ValueError):
        return None


async def store(scope: str, kb_ids: list[str], query: str, payload: dict[str, Any]) -> None:
    if not cache_enabled():
        return

    def _store():
        _cache.store(
            prompt=query,
            response=json.dumps(payload),
            filters={"scope": scope, "kb": [str(k) for k in kb_ids]},
        )

    try:
        await asyncio.to_thread(_store)
    except Exception:  # noqa: BLE001 - never fail a turn because caching failed
        pass


async def invalidate_kb(kb_id: str) -> int:
    """Best-effort: drop cache entries tagged with this kb. TTL is the backstop."""
    if not cache_enabled():
        return 0

    def _invalidate():
        from redisvl.query import FilterQuery
        from redisvl.query.filter import Tag

        index = getattr(_cache, "_index", None)
        if index is None:
            return 0
        query = FilterQuery(
            filter_expression=(Tag("kb") == str(kb_id)),
            return_fields=["id"],
            num_results=1000,
        )
        results = index.query(query)
        keys = [r["id"] for r in results] if results else []
        if keys:
            index.drop_keys(keys)
        return len(keys)

    try:
        return await asyncio.to_thread(_invalidate)
    except Exception:  # noqa: BLE001
        return 0
