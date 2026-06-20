"""Websocket chat endpoint with per-session memory.

Protocol:
  - Client connects and sends a first message: {agent_id, kb_ids?, question?, session_id?}.
      * session_id present -> resume that session (memory restored from the checkpointer).
      * session_id absent  -> a new session is created.
  - Server replies with {"type":"session","session_id": "..."} so the client can reuse it.
  - For each question the server streams status events
    (retrieving / reranking / generating / validating) then one final answer.
  - The socket stays open: send more {question} messages to continue the same session.

The session id is the checkpoint thread_id (= conversation id); memory persists across
turns and across reconnects that pass the same session_id.
"""
from __future__ import annotations

import uuid
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from app.agent.runtime import AgentRuntime, RuntimeError_, build_runtime
from app.agent.schemas import Answer
from app.db import queries

_STATUS_BY_NODE = {
    "retrieve": "retrieving",
    "rerank": "reranking",
    "generate": "generating",
    "validate": "validating",
}


def _build_sources(
    references: list[str], retrieved: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Group cited chunk_ids by their source document (filename, version)."""
    by_id = {c.get("chunk_id"): c for c in retrieved}
    grouped: dict[tuple[Any, Any], dict[str, Any]] = {}
    for chunk_id in references:
        chunk = by_id.get(chunk_id)
        if chunk is None:
            continue
        key = (chunk.get("filename"), chunk.get("version"))
        entry = grouped.setdefault(
            key,
            {"filename": chunk.get("filename"), "version": chunk.get("version"), "chunk_ids": []},
        )
        entry["chunk_ids"].append(chunk_id)
    return list(grouped.values())


async def _run_turn(
    ws: WebSocket, runtime: AgentRuntime, session_id: uuid.UUID, question: str
) -> None:
    await queries.add_message(
        conversation_id=session_id, role="user", content=question, metadata={}
    )
    await queries.touch_conversation(session_id)

    config = {"configurable": {"thread_id": str(session_id)}}
    final: dict[str, Any] = {}
    try:
        async for step in runtime.graph.astream(
            {"question": question, "attempts": 0, "feedback": ""},
            config=config,
            stream_mode="updates",
        ):
            for node, update in step.items():
                stage = _STATUS_BY_NODE.get(node)
                if stage:
                    await ws.send_json({"type": "status", "stage": stage})
                if isinstance(update, dict):
                    final.update(update)
    except Exception as exc:  # noqa: BLE001 - surface failure to the client
        await ws.send_json({"type": "error", "message": f"agent error: {exc}"})
        return

    answer: Answer | None = final.get("answer")
    if answer is None:
        await ws.send_json({"type": "error", "message": "no answer produced"})
        return

    rejected = bool(final.get("rejected", False))
    sources = _build_sources(answer.references, final.get("retrieved", []) or [])
    await ws.send_json(
        {
            "type": "answer",
            "answer": answer.answer,
            "references": answer.references,
            "sources": sources,
            "confidence": answer.confidence,
            "rejected": rejected,
        }
    )
    await queries.add_message(
        conversation_id=session_id,
        role="assistant",
        content=answer.answer,
        metadata={
            "references": answer.references,
            "sources": sources,
            "confidence": answer.confidence,
            "rejected": rejected,
        },
    )
    await queries.touch_conversation(session_id)


async def _resolve_session(
    agent_id: uuid.UUID, session_id_raw: Any
) -> tuple[uuid.UUID | None, str | None]:
    """Return (session_id, error_message)."""
    if session_id_raw:
        try:
            session_id = uuid.UUID(str(session_id_raw))
        except ValueError:
            return None, "invalid session_id"
        conv = await queries.get_conversation(session_id)
        if conv is None:
            return None, "session not found"
        if conv["agent_id"] != agent_id:
            return None, "session belongs to a different agent"
        return session_id, None
    conv = await queries.create_conversation(agent_id)
    return conv["id"], None


async def chat_ws(ws: WebSocket) -> None:
    await ws.accept()
    try:
        message = await ws.receive_json()
    except (WebSocketDisconnect, ValueError):
        return

    try:
        agent_id = uuid.UUID(message["agent_id"])
        kb_ids = [uuid.UUID(x) for x in message.get("kb_ids", [])] or None
    except (KeyError, ValueError, TypeError) as exc:
        await ws.send_json({"type": "error", "message": f"invalid request: {exc}"})
        await ws.close()
        return

    try:
        runtime = await build_runtime(agent_id, kb_ids)
    except RuntimeError_ as exc:
        await ws.send_json({"type": "error", "message": str(exc)})
        await ws.close()
        return

    session_id, err = await _resolve_session(agent_id, message.get("session_id"))
    if err is not None:
        await ws.send_json({"type": "error", "message": err})
        await ws.close()
        return
    assert session_id is not None
    await ws.send_json({"type": "session", "session_id": str(session_id)})

    # Process the first message question (if any), then loop for follow-ups.
    while True:
        question = message.get("question")
        if question:
            await _run_turn(ws, runtime, session_id, str(question))
        try:
            message = await ws.receive_json()
        except (WebSocketDisconnect, ValueError):
            break
