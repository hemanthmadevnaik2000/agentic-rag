"""Lazy arq (Redis) connection pool for enqueuing ingestion jobs from the API."""
from __future__ import annotations

from arq import create_pool
from arq.connections import ArqRedis, RedisSettings

from app.config import get_settings

_arq: ArqRedis | None = None


async def init_arq() -> ArqRedis:
    global _arq
    if _arq is None:
        _arq = await create_pool(RedisSettings.from_dsn(get_settings().redis_url))
    return _arq


async def close_arq() -> None:
    global _arq
    if _arq is not None:
        await _arq.close()
        _arq = None


def get_arq() -> ArqRedis:
    if _arq is None:
        raise RuntimeError("arq pool not initialized; call init_arq() in app lifespan.")
    return _arq
