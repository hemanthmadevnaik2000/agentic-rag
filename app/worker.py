"""arq worker: KB ingestion + scheduled session-memory cleanup.

Start with:  arq app.worker.WorkerSettings
"""
from __future__ import annotations

from typing import Any

from arq import cron
from arq.connections import RedisSettings

from app.agent.checkpointer import close_checkpointer, init_checkpointer
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


async def cleanup_sessions(ctx: dict[str, Any]) -> int:
    """Delete checkpoints + conversations past SESSION_TTL_SECONDS (Postgres has no TTL)."""
    from app.agent.checkpointer import delete_expired_sessions

    return await delete_expired_sessions()


async def _startup(ctx: dict[str, Any]) -> None:
    await init_pool()
    await init_checkpointer()


async def _shutdown(ctx: dict[str, Any]) -> None:
    await close_checkpointer()
    await close_pool()


class WorkerSettings:
    functions = [ingest_kb]
    cron_jobs = [cron(cleanup_sessions, minute=7)]  # hourly at :07 (off the herd)
    on_startup = _startup
    on_shutdown = _shutdown
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
    job_timeout = 3600
