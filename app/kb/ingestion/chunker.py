"""Semantic chunking via the LangChain SemanticChunker.

Our Embeddings duck-types the LangChain interface (embed_documents/embed_query),
so it can be passed directly as the splitter embedding model.
"""
from __future__ import annotations

from app.kb.embeddings.base import Embeddings


def semantic_chunks(text: str, embeddings: Embeddings) -> list[str]:
    from langchain_experimental.text_splitter import SemanticChunker

    text = text.strip()
    if not text:
        return []
    chunker = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
    docs = chunker.create_documents([text])
    return [d.page_content for d in docs if d.page_content.strip()]
