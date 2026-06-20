"""Hybrid retrieval: dense (Milvus) + sparse (BM25) -> RRF -> cross-encoder rerank.

Split into fuse + rerank so the agent can surface separate status events. All
attached KBs must share one embedding space (one Milvus collection); the agent
layer resolves that and passes it in via RetrievalTarget.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from app.config import get_settings
from app.kb.embeddings.registry import get_embeddings
from app.kb.search.bm25 import BM25Index
from app.kb.search.rerank import get_reranker
from app.kb.search.rrf import reciprocal_rank_fusion
from app.kb.vectorstore.milvus import MilvusStore, collection_name


@dataclass(frozen=True)
class RetrievalTarget:
    kb_ids: tuple[str, ...]
    embedding_provider: str
    embedding_model: str
    embedding_dim: int
    milvus_uri: str
    milvus_token: str = ""
    enable_bm25: bool = True


async def fuse_candidates(
    query: str, target: RetrievalTarget, candidates: int | None = None
) -> list[dict[str, Any]]:
    settings = get_settings()
    candidates = candidates or settings.retrieval_candidates
    kb_ids = list(target.kb_ids)
    if not kb_ids:
        return []

    embeddings = get_embeddings(target.embedding_provider, target.embedding_model)
    query_vector = await asyncio.to_thread(embeddings.embed_query, query)

    store = MilvusStore(target.milvus_uri, target.milvus_token)
    coll = collection_name(target.embedding_model, target.embedding_dim)
    dense = await asyncio.to_thread(store.search, coll, query_vector, kb_ids, candidates)

    sparse: list[dict[str, Any]] = []
    if target.enable_bm25:
        bm25 = BM25Index(settings.elasticsearch_url)
        sparse = await asyncio.to_thread(bm25.search, query, kb_ids, candidates)

    return reciprocal_rank_fusion([dense, sparse], k=settings.rrf_k)


async def rerank_candidates(
    query: str, fused: list[dict[str, Any]], top_k: int | None = None
) -> list[dict[str, Any]]:
    settings = get_settings()
    top_k = top_k or settings.retrieval_top_k
    reranker = get_reranker()
    return await asyncio.to_thread(reranker.rerank, query, fused, top_k)


async def hybrid_retrieve(
    query: str,
    target: RetrievalTarget,
    top_k: int | None = None,
    candidates: int | None = None,
) -> list[dict[str, Any]]:
    fused = await fuse_candidates(query, target, candidates)
    return await rerank_candidates(query, fused, top_k)
