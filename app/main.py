from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.agent.cache import close_cache, init_cache
from app.agent.checkpointer import close_checkpointer, init_checkpointer
from app.agent.router import router as agents_router
from app.agent.ws import chat_ws
from app.db.pool import close_pool, init_pool
from app.destinations.router import router as destinations_router
from app.kb.router import router as kb_router
from app.llms.router import router as llms_router
from app.queue import close_arq, init_arq
from app.tracing import configure_tracing


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_tracing()
    await init_pool()
    await init_arq()
    await init_checkpointer()
    await init_cache()
    try:
        yield
    finally:
        await close_cache()
        await close_checkpointer()
        await close_arq()
        await close_pool()


def create_app() -> FastAPI:
    app = FastAPI(title="Agentic RAG", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(destinations_router)
    app.include_router(llms_router)
    app.include_router(kb_router)
    app.include_router(agents_router)
    app.add_api_websocket_route("/ws/chat", chat_ws)
    return app


app = create_app()
