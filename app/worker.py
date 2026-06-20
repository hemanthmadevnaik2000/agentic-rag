"""arq worker: runs KB ingestion off the request path.

Start with:  arq app.worker.WorkerSettings
"""
from __future__ import annotations

from typing import Any

from arq.connections import RedisSettings

from app.config import get_settings
from app.db.pool import close_pool, init_pool


async def ingest_kb(
    ctx: dict[str, Any],
    kb_id: str,
    files: list[tuple[str, str]],
    metadata: dict[str, Any],
) -> None:
    from app.kb.ingestion.pipeline import run_ingest

    await run_ingest(kb_id, files, metadata)


async def _startup(ctx: dict[str, Any]) -> None:
    await init_pool()


async def _shutdown(ctx: dict[str, Any]) -> None:
    await close_pool()


class WorkerSettings:
    functions = [ingest_kb]
    on_startup = _startup
    on_shutdown = _shutdown
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
    # Embedding/ingestion can be slow on large docs; allow generous time.
    job_timeout = 3600
