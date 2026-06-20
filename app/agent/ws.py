"""Websocket chat endpoint.

Protocol: client sends {agent_id, kb_ids?, question}; server streams status events
(retrieving / reranking / generating / validating) then a single final answer
message after the groundedness gate passes. No token streaming / retraction.
"""
from __future__ import annotations

import uuid
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from app.agent.runtime import RuntimeError_, build_runtime
from app.agent.schemas import Answer
from app.db import queries

_STATUS_BY_NODE = {
    "retrieve": "retrieving",
    "rerank": "reranking",
    "generate": "generating",
    "validate": "validating",
}


async def chat_ws(ws: WebSocket) -> None:
    await ws.accept()
    try:
        payload = await ws.receive_json()
    except (WebSocketDisconnect, ValueError):
        return

    try:
        agent_id = uuid.UUID(payload["agent_id"])
        question = str(payload["question"])
        kb_ids = [uuid.UUID(x) for x in payload.get("kb_ids", [])] or None
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

    conv = await queries.create_conversation(agent_id)
    await queries.add_message(
        conversation_id=conv["id"], role="user", content=question, metadata={}
    )

    final: dict[str, Any] = {}
    try:
        async for step in runtime.graph.astream(
            {"question": question, "attempts": 0, "feedback": ""},
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
        await ws.close()
        return

    answer: Answer | None = final.get("answer")
    if answer is None:
        await ws.send_json({"type": "error", "message": "no answer produced"})
        await ws.close()
        return

    rejected = bool(final.get("rejected", False))
    result = {
        "type": "answer",
        "answer": answer.answer,
        "references": answer.references,
        "confidence": answer.confidence,
        "rejected": rejected,
    }
    await ws.send_json(result)
    await queries.add_message(
        conversation_id=conv["id"],
        role="assistant",
        content=answer.answer,
        metadata={
            "references": answer.references,
            "confidence": answer.confidence,
            "rejected": rejected,
        },
    )
    await ws.close()
