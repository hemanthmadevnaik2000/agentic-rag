"""History-aware query rewriting.

Turns a possibly-elliptical user question into a standalone, canonical query. Used
both for retrieval (so follow-ups resolve) and as the semantic-cache key (so
paraphrases collide). Falls back to the raw question if rewriting fails.
"""
from __future__ import annotations

from typing import Any

from app.agent.llm.base import StructuredLLM
from app.agent.schemas import RewrittenQuery

_REWRITE_SYSTEM = (
    "You rewrite a user question into a single, self-contained search query. "
    "Resolve pronouns and references using the conversation so far so the query "
    "stands on its own. Keep it concise and faithful. Do not answer it."
)


async def rewrite_query(
    llm: StructuredLLM, question: str, history: list[dict[str, Any]]
) -> str:
    hist = ""
    if history:
        lines = [
            f"{h.get('role', 'user')}: {h.get('content', '')}" for h in history[-6:]
        ]
        hist = "Conversation so far:\n" + "\n".join(lines) + "\n\n"
    user = f"{hist}User question: {question}\n\nRewrite it as a standalone query."
    try:
        result = await llm.generate(_REWRITE_SYSTEM, user, RewrittenQuery)
        rewritten = (result.query or "").strip()
        return rewritten or question
    except Exception:  # noqa: BLE001 - never block a turn on rewrite failure
        return question
