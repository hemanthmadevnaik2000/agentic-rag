"""Plain-SQL data access (asyncpg). Small, parameterized functions only."""
from __future__ import annotations

import uuid
from typing import Any

from app.db.pool import get_pool


async def _fetchrow(sql: str, *args: Any) -> dict[str, Any] | None:
    row = await get_pool().fetchrow(sql, *args)
    return dict(row) if row is not None else None


async def _fetch(sql: str, *args: Any) -> list[dict[str, Any]]:
    rows = await get_pool().fetch(sql, *args)
    return [dict(r) for r in rows]


async def _execute_deleted(sql: str, *args: Any) -> bool:
    status = await get_pool().execute(sql, *args)
    return status.rsplit(" ", 1)[-1] != "0"


# --------------------------------------------------------------------------- #
# destinations
# --------------------------------------------------------------------------- #
async def create_destination(
    *, name, type, host, port, db_name, username, secret_enc, key_version, secret_last4
) -> dict[str, Any]:
    return await _fetchrow(  # type: ignore[return-value]
        """
        INSERT INTO destinations
            (name, type, host, port, db_name, username, secret_enc, key_version, secret_last4)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        RETURNING *
        """,
        name, type, host, port, db_name, username, secret_enc, key_version, secret_last4,
    )


async def get_destination(dest_id: uuid.UUID) -> dict[str, Any] | None:
    return await _fetchrow("SELECT * FROM destinations WHERE id = $1", dest_id)


async def list_destinations() -> list[dict[str, Any]]:
    return await _fetch("SELECT * FROM destinations ORDER BY created_at DESC")


async def delete_destination(dest_id: uuid.UUID) -> bool:
    return await _execute_deleted("DELETE FROM destinations WHERE id = $1", dest_id)


# --------------------------------------------------------------------------- #
# llm_registrations
# --------------------------------------------------------------------------- #
async def create_llm(
    *, name, provider, base_url, model, api_key_enc, key_version, secret_last4
) -> dict[str, Any]:
    return await _fetchrow(  # type: ignore[return-value]
        """
        INSERT INTO llm_registrations
            (name, provider, base_url, model, api_key_enc, key_version, secret_last4)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        RETURNING *
        """,
        name, provider, base_url, model, api_key_enc, key_version, secret_last4,
    )


async def get_llm(llm_id: uuid.UUID) -> dict[str, Any] | None:
    return await _fetchrow("SELECT * FROM llm_registrations WHERE id = $1", llm_id)


async def list_llms() -> list[dict[str, Any]]:
    return await _fetch("SELECT * FROM llm_registrations ORDER BY created_at DESC")


async def delete_llm(llm_id: uuid.UUID) -> bool:
    return await _execute_deleted("DELETE FROM llm_registrations WHERE id = $1", llm_id)


# --------------------------------------------------------------------------- #
# knowledge_bases
# --------------------------------------------------------------------------- #
async def create_kb(
    *, name, destination_id, embedding_model, embedding_dim, metadata
) -> dict[str, Any]:
    return await _fetchrow(  # type: ignore[return-value]
        """
        INSERT INTO knowledge_bases
            (name, destination_id, embedding_model, embedding_dim, status, metadata)
        VALUES ($1, $2, $3, $4, 'processing', $5)
        RETURNING *
        """,
        name, destination_id, embedding_model, embedding_dim, metadata,
    )


async def get_kb(kb_id: uuid.UUID) -> dict[str, Any] | None:
    return await _fetchrow("SELECT * FROM knowledge_bases WHERE id = $1", kb_id)


async def list_kbs() -> list[dict[str, Any]]:
    return await _fetch("SELECT * FROM knowledge_bases ORDER BY created_at DESC")


async def get_kbs_by_ids(kb_ids: list[uuid.UUID]) -> list[dict[str, Any]]:
    return await _fetch(
        "SELECT * FROM knowledge_bases WHERE id = ANY($1::uuid[])", kb_ids
    )


async def set_kb_status(kb_id: uuid.UUID, status: str, error: str | None = None) -> None:
    await get_pool().execute(
        "UPDATE knowledge_bases SET status = $2, error = $3 WHERE id = $1",
        kb_id, status, error,
    )


# --------------------------------------------------------------------------- #
# documents
# --------------------------------------------------------------------------- #
async def next_document_version(kb_id: uuid.UUID, filename: str) -> int:
    row = await _fetchrow(
        "SELECT COALESCE(MAX(version), 0) + 1 AS v FROM documents WHERE kb_id = $1 AND filename = $2",
        kb_id, filename,
    )
    return int(row["v"]) if row else 1


async def create_document(
    *, kb_id, filename, version, chunk_count, status
) -> dict[str, Any]:
    return await _fetchrow(  # type: ignore[return-value]
        """
        INSERT INTO documents (kb_id, filename, version, chunk_count, status)
        VALUES ($1, $2, $3, $4, $5)
        RETURNING *
        """,
        kb_id, filename, version, chunk_count, status,
    )


# --------------------------------------------------------------------------- #
# agents
# --------------------------------------------------------------------------- #
async def create_agent(
    *, name, llm_id, kb_ids, confidence_threshold, max_retries, system_prompt
) -> dict[str, Any]:
    return await _fetchrow(  # type: ignore[return-value]
        """
        INSERT INTO agents
            (name, llm_id, kb_ids, confidence_threshold, max_retries, system_prompt)
        VALUES ($1, $2, $3, $4, $5, $6)
        RETURNING *
        """,
        name, llm_id, kb_ids, confidence_threshold, max_retries, system_prompt,
    )


async def get_agent(agent_id: uuid.UUID) -> dict[str, Any] | None:
    return await _fetchrow("SELECT * FROM agents WHERE id = $1", agent_id)


async def list_agents() -> list[dict[str, Any]]:
    return await _fetch("SELECT * FROM agents ORDER BY created_at DESC")


async def delete_agent(agent_id: uuid.UUID) -> bool:
    return await _execute_deleted("DELETE FROM agents WHERE id = $1", agent_id)


# --------------------------------------------------------------------------- #
# conversations / messages
# --------------------------------------------------------------------------- #
async def create_conversation(agent_id: uuid.UUID) -> dict[str, Any]:
    return await _fetchrow(  # type: ignore[return-value]
        "INSERT INTO conversations (agent_id) VALUES ($1) RETURNING *", agent_id
    )


async def add_message(
    *, conversation_id, role, content, metadata
) -> dict[str, Any]:
    return await _fetchrow(  # type: ignore[return-value]
        """
        INSERT INTO messages (conversation_id, role, content, metadata)
        VALUES ($1, $2, $3, $4)
        RETURNING *
        """,
        conversation_id, role, content, metadata,
    )


async def list_messages(conversation_id: uuid.UUID) -> list[dict[str, Any]]:
    return await _fetch(
        "SELECT * FROM messages WHERE conversation_id = $1 ORDER BY created_at ASC",
        conversation_id,
    )
