"""Optional Elasticsearch BM25 index. One shared index, filtered by kb_id."""
from __future__ import annotations

from typing import Any

INDEX = "rag_chunks"


class BM25Index:
    def __init__(self, url: str) -> None:
        from elasticsearch import Elasticsearch

        self._es = Elasticsearch(url)

    def ensure_index(self) -> None:
        if self._es.indices.exists(index=INDEX):
            return
        self._es.indices.create(
            index=INDEX,
            mappings={
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "kb_id": {"type": "keyword"},
                    "filename": {"type": "keyword"},
                    "version": {"type": "integer"},
                    "text": {"type": "text"},
                }
            },
        )

    def delete_filename(self, kb_id: str, filename: str) -> None:
        self._es.delete_by_query(
            index=INDEX,
            query={
                "bool": {
                    "filter": [
                        {"term": {"kb_id": kb_id}},
                        {"term": {"filename": filename}},
                    ]
                }
            },
            refresh=True,
            conflicts="proceed",
        )

    def index_chunks(self, rows: list[dict[str, Any]]) -> None:
        from elasticsearch.helpers import bulk

        actions = [
            {
                "_index": INDEX,
                "_id": r["chunk_id"],
                "_source": {
                    "chunk_id": r["chunk_id"],
                    "kb_id": r["kb_id"],
                    "filename": r["filename"],
                    "version": r["version"],
                    "text": r["text"],
                },
            }
            for r in rows
        ]
        if actions:
            bulk(self._es, actions, refresh=True)

    def search(self, query: str, kb_ids: list[str], top_k: int) -> list[dict[str, Any]]:
        resp = self._es.search(
            index=INDEX,
            size=top_k,
            query={
                "bool": {
                    "must": {"match": {"text": query}},
                    "filter": [{"terms": {"kb_id": kb_ids}}],
                }
            },
        )
        out: list[dict[str, Any]] = []
        for hit in resp["hits"]["hits"]:
            src = hit["_source"]
            out.append(
                {
                    "chunk_id": src["chunk_id"],
                    "kb_id": src["kb_id"],
                    "filename": src["filename"],
                    "version": src.get("version"),
                    "text": src["text"],
                    "bm25_score": float(hit["_score"]),
                }
            )
        return out
