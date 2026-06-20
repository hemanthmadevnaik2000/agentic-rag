"""Milvus store: one collection per embedding space, partitioned by kb_id.

A multi-KB query is a single search with a kb_id-in-list filter. Re-ingesting a
filename replaces that filename vectors (latest-only in the vector store); full
version history is retained in Postgres documents.
"""
from __future__ import annotations

import re
from typing import Any


def collection_name(embedding_model: str, dim: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", embedding_model.lower()).strip("_")
    return f"kb_{slug}_{dim}"


def _q(value: str) -> str:
    return value.replace('"', '\\"')


class MilvusStore:
    def __init__(self, uri: str, token: str = "") -> None:
        from pymilvus import MilvusClient

        self._client = MilvusClient(uri=uri, token=token or "")

    def ensure_collection(self, name: str, dim: int) -> None:
        from pymilvus import DataType

        if self._client.has_collection(name):
            return
        schema = self._client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=512)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field("kb_id", DataType.VARCHAR, max_length=64, is_partition_key=True)
        schema.add_field("filename", DataType.VARCHAR, max_length=1024)
        schema.add_field("version", DataType.INT64)
        schema.add_field("text", DataType.VARCHAR, max_length=65535)
        schema.add_field("metadata", DataType.JSON)

        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200},
        )
        self._client.create_collection(name, schema=schema, index_params=index_params)

    def delete_filename(self, name: str, kb_id: str, filename: str) -> None:
        self._client.delete(
            collection_name=name,
            filter=f'kb_id == "{_q(kb_id)}" and filename == "{_q(filename)}"',
        )

    def upsert_chunks(self, name: str, rows: list[dict[str, Any]]) -> None:
        if rows:
            self._client.insert(collection_name=name, data=rows)

    def search(
        self, name: str, query_vector: list[float], kb_ids: list[str], top_k: int
    ) -> list[dict[str, Any]]:
        quoted = ", ".join(f'"{_q(k)}"' for k in kb_ids)
        results = self._client.search(
            collection_name=name,
            data=[query_vector],
            limit=top_k,
            filter=f"kb_id in [{quoted}]",
            search_params={"metric_type": "COSINE"},
            output_fields=["chunk_id", "kb_id", "filename", "version", "text", "metadata"],
        )
        hits = results[0] if results else []
        out: list[dict[str, Any]] = []
        for hit in hits:
            entity = hit.get("entity", {})
            out.append(
                {
                    "chunk_id": entity.get("chunk_id"),
                    "kb_id": entity.get("kb_id"),
                    "filename": entity.get("filename"),
                    "version": entity.get("version"),
                    "text": entity.get("text"),
                    "metadata": entity.get("metadata"),
                    "dense_score": float(hit.get("distance", 0.0)),
                }
            )
        return out
