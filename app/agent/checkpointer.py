"""Postgres-backed LangGraph checkpointer (per-session memory).

Postgres has no native TTL, so session expiry is handled by delete_expired_sessions(),
run on a schedule from the arq worker. Sessions are keyed by thread_id = conversation id;
last_active_at on the conversations row is the activity clock.
"""
from __future__ import annotations

from typing import Any

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.config import get_settings

_pool: AsyncConnectionPool | None = None
_saver: Any = None


async def init_checkpointer() -> Any:
    global _pool, _saver
    if _saver is None:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        settings = get_settings()
        _pool = AsyncConnectionPool(
            conninfo=settings.database_url,
            max_size=10,
            open=False,
            kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        )
        await _pool.open()
        _saver = AsyncPostgresSaver(_pool)
        await _saver.setup()
    return _saver


async def close_checkpointer() -> None:
    global _pool, _saver
    if _pool is not None:
        await _pool.close()
    _pool = None
    _saver = None


def get_checkpointer() -> Any:
    if _saver is None:
        raise RuntimeError("checkpointer not initialized; call init_checkpointer() first.")
    return _saver


async def _raw_delete_thread(thread_id: str) -> None:
    # Fallback for langgraph versions without adelete_thread.
    assert _pool is not None
    async with _pool.connection() as conn:
        for table in ("checkpoint_writes", "checkpoint_blobs", "checkpoints"):
            await conn.execute(f"DELETE FROM {table} WHERE thread_id = %s", (thread_id,))


async def delete_expired_sessions() -> int:
    """Delete checkpoints + conversation rows for sessions past the TTL."""
    from app.db import queries

    settings = get_settings()
    expired = await queries.expired_conversation_ids(settings.session_ttl_seconds)
    saver = get_checkpointer()
    for conversation_id in expired:
        thread_id = str(conversation_id)
        try:
            await saver.adelete_thread(thread_id)
        except AttributeError:
            await _raw_delete_thread(thread_id)
        await queries.delete_conversation(conversation_id)
    return len(expired)
