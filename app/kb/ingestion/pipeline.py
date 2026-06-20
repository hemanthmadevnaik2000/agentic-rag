"""KB ingestion orchestration (runs in the arq worker).

For each file: load -> semantic chunk -> embed -> upsert to Milvus (+ optional ES).
Re-ingesting a filename replaces its vectors in the store (latest-only); Postgres
documents keeps a row per version as history. On success, the semantic cache for this
KB is invalidated so stale answers are not served. Blocking calls run in a thread.
"""
from __future__ import annotations

import asyncio
import uuid
from typing import Any

from app import crypto
from app.config import get_settings
from app.db import queries
from app.kb.embeddings.registry import get_embeddings
from app.kb.ingestion.chunker import semantic_chunks
from app.kb.ingestion.loaders import load_document
from app.kb.search.bm25 import BM25Index
from app.kb.vectorstore.milvus import MilvusStore, collection_name


async def run_ingest(
    kb_id_str: str, files: list[tuple[str, str]], metadata: dict[str, Any]
) -> None:
    kb_id = uuid.UUID(kb_id_str)
    settings = get_settings()
    try:
        kb = await queries.get_kb(kb_id)
        if kb is None:
            raise RuntimeError("Knowledge base not found")
        dest = await queries.get_destination(kb["destination_id"])
        if dest is None:
            raise RuntimeError("Destination not found")

        token = crypto.decrypt(dest["secret_enc"]) if dest["secret_enc"] else ""
        host = dest["host"]
        port = dest["port"]
        uri = f"http://{host}:{port}"
        provider = (kb.get("metadata") or {}).get(
            "_embedding_provider"
        ) or settings.embedding_provider

        embeddings = await asyncio.to_thread(
            get_embeddings, provider, kb["embedding_model"]
        )
        store = MilvusStore(uri, token)
        coll = collection_name(kb["embedding_model"], kb["embedding_dim"])
        await asyncio.to_thread(store.ensure_collection, coll, kb["embedding_dim"])

        bm25: BM25Index | None = None
        if settings.enable_bm25:
            bm25 = BM25Index(settings.elasticsearch_url)
            await asyncio.to_thread(bm25.ensure_index)

        user_meta = {
            k: v for k, v in (metadata or {}).items() if not k.startswith("_")
        }

        for filename, path in files:
            text = await asyncio.to_thread(load_document, path)
            chunks = await asyncio.to_thread(semantic_chunks, text, embeddings)
            version = await queries.next_document_version(kb_id, filename)

            if not chunks:
                await queries.create_document(
                    kb_id=kb_id, filename=filename, version=version,
                    chunk_count=0, status="empty",
                )
                continue

            vectors = await asyncio.to_thread(embeddings.embed_documents, chunks)
            rows = []
            for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
                rows.append(
                    {
                        "chunk_id": f"{kb_id}:{filename}:v{version}:{i}",
                        "embedding": vec,
                        "kb_id": str(kb_id),
                        "filename": filename,
                        "version": version,
                        "text": chunk,
                        "metadata": {
                            "filename": filename,
                            "version": version,
                            "kb_id": str(kb_id),
                            **user_meta,
                        },
                    }
                )

            await asyncio.to_thread(store.delete_filename, coll, str(kb_id), filename)
            await asyncio.to_thread(store.upsert_chunks, coll, rows)
            if bm25 is not None:
                await asyncio.to_thread(bm25.delete_filename, str(kb_id), filename)
                await asyncio.to_thread(bm25.index_chunks, rows)

            await queries.create_document(
                kb_id=kb_id, filename=filename, version=version,
                chunk_count=len(chunks), status="ready",
            )

        await queries.set_kb_status(kb_id, "ready")
        # Invalidate stale cached answers for this KB (TTL is the backstop).
        try:
            from app.agent import cache as cache_mod

            await cache_mod.invalidate_kb(str(kb_id))
        except Exception:  # noqa: BLE001
            pass
    except Exception as exc:  # noqa: BLE001 - record failure on the KB row
        await queries.set_kb_status(kb_id, "failed", error=str(exc))
        raise
