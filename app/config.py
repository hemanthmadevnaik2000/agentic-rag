from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Postgres
    database_url: str = "postgresql://rag:rag@localhost:5432/rag"
    # Redis (arq worker / future semantic cache)
    redis_url: str = "redis://localhost:6379/0"
    # Milvus
    milvus_uri: str = "http://localhost:19530"
    milvus_token: str = ""
    # Elasticsearch
    elasticsearch_url: str = "http://localhost:9200"
    enable_bm25: bool = True
    # Secrets encryption: comma-separated Fernet keys, newest first
    app_encryption_keys: str = ""
    # Embeddings / reranker
    embedding_provider: str = "local"  # local | openai
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_dim: int = 768
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    # Retrieval params
    retrieval_top_k: int = 5
    retrieval_candidates: int = 20
    rrf_k: int = 60
    # Sessions / memory (checkpointer cleanup horizon, seconds)
    session_ttl_seconds: int = 86400
    # Tracing
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "agentic-rag"
    # Dev fallback provider keys (registered LLMs live in the DB)
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    @property
    def encryption_keys(self) -> list[str]:
        return [k.strip() for k in self.app_encryption_keys.split(",") if k.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
