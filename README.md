# Agentic RAG

Create knowledge bases (KBs) from documents and chat with a grounded agent that answers strictly from those KBs over a websocket.

## Architecture

Modular monolith (FastAPI) + async ingestion worker (arq):

- **KB module** — ingest PDF/DOCX/txt/md → semantic chunk → embed → Milvus (one collection per embedding space, partitioned by `kb_id`) + optional Elasticsearch BM25.
- **Retrieval** — dense + BM25 → Reciprocal Rank Fusion → cross-encoder rerank.
- **Agent module** — LangGraph: `retrieve → generate (structured) → validate → respond` with a retry edge. Supports OpenAI / Anthropic / self-hosted SLM via a `StructuredLLM` port.
- **Postgres** — backend state (plain SQL via asyncpg). Secrets encrypted at rest (Fernet).
- **Tracing** — LangSmith (optional).

## Quick start

```bash
# 1. Services
docker compose up -d

# 2. Install
python -m venv .venv
. .venv/Scripts/activate
pip install -e ".[dev]"

# 3. Configure
cp .env.example .env
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

# 4. Migrate
python -m app.db.migrate

# 5. Run API + worker
uvicorn app.main:app --reload
arq app.worker.WorkerSettings
```

`GET /health` returns `{"status": "ok"}`.

## Notes

- Set `ENABLE_BM25=false` to skip Elasticsearch (dense-only retrieval).
- Registered LLM keys and Milvus credentials are stored encrypted in Postgres — never returned by the API (only `secret_last4`).
