# Agentic RAG

Build **knowledge bases** from your documents and chat with a **grounded, citation-aware agent** that answers strictly from those knowledge bases over a WebSocket — with **per-session memory**.

![Python](https://img.shields.io/badge/python-3.11+-blue)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)
![LangGraph](https://img.shields.io/badge/agent-LangGraph-1c3c3c)
![Milvus](https://img.shields.io/badge/vectors-Milvus-00a1ea)

---

## Overview

Agentic RAG is a production-shaped (but personal-scale) Retrieval-Augmented Generation system with two halves:

- **Knowledge bases (KBs):** upload documents (PDF / DOCX / TXT / MD); they are semantically chunked, embedded, and pushed to a vector DB (Milvus), with optional keyword indexing in Elasticsearch.
- **Agent:** attach one or more KBs and chat over a WebSocket. The agent retrieves with hybrid search, generates a **structured** answer, and runs it through a **groundedness gate** (confidence + references) before returning it — remembering the conversation within a session.

It is a **modular monolith**: one FastAPI app with `kb` and `agent` modules, plus an async ingestion worker.

---

## Features

- **Document ingestion** — PDF, DOCX, TXT, Markdown, with **semantic chunking**.
- **Hybrid retrieval** — dense vectors (Milvus) + BM25 keyword search (Elasticsearch) fused with **Reciprocal Rank Fusion**, then **cross-encoder reranking**.
- **Grounded answers** — the LLM is forced to return structured output `{answer, references, confidence}`; a validation gate rejects low-confidence / unreferenced answers and retries with feedback before falling back to a safe "I could not find it".
- **Source attribution** — each answer reports the **documents** (filename + version) its cited chunks came from, mapped from the retrieved set.
- **Session memory** — multi-turn conversations persisted with a **Postgres LangGraph checkpointer**; resume a session by `session_id`, with TTL-based cleanup.
- **Multi-provider LLMs** — OpenAI, Anthropic, and self-hosted / custom SLMs behind one `StructuredLLM` interface; providers are registered at runtime via an API.
- **Pluggable embeddings & reranker** — local (sentence-transformers / BGE) by default, swappable for cloud.
- **Secrets encrypted at rest** — vector-DB credentials and LLM API keys are encrypted with Fernet; the API never returns them (only a masked `secret_last4`).
- **Multi-KB queries** — one Milvus collection per embedding space, partitioned by `kb_id`, so querying several KBs is a single filtered search.
- **Streaming UX** — the WebSocket streams pipeline status events (`retrieving -> reranking -> generating -> validating`) and then one final, validated answer.
- **Tracing** — optional LangSmith integration.
- **Async ingestion** — heavy chunk/embed/index work runs in an `arq` worker off the request path.

---

## Architecture

```
                 +---------------------------- FastAPI app ----------------------------+
   HTTP -------->|  /destinations  /llms  /kb  /agents                                 |
   WebSocket --->|  /ws/chat  (sessions, multi-turn)                                   |
                 |      |                 |                          |                 |
                 |      v                 v                          v                 |
                 |  Postgres        arq enqueue              Agent (LangGraph)         |
                 |  (plain SQL,     (ingest job)        retrieve->rerank->generate     |
                 |   secrets enc.,        |                  ->validate->respond        |
                 |   checkpointer)        |                          |                 |
                 +------------------------|--------------------------|-----------------+
                                          v                          | hybrid_retrieve
                                   arq worker                        v
                          load -> chunk -> embed -> write     +---------------+
                          + hourly session TTL cleanup        | Milvus (dense)|
                                          |                   | Elastic (BM25)|
                                          +------------------>+---------------+
```

**Request flow when chatting:** client opens `/ws/chat` with `{agent_id, kb_ids, question, session_id?}` →
the agent runs `retrieve` (dense + BM25 → RRF) → `rerank` (cross-encoder) → `generate`
(structured output from the registered LLM) → `validate` (confidence + references gate,
retry on failure) → `respond`. Each node maps to a status event; prior turns are restored
from the checkpointer keyed by the session id.

---

## Tech stack

| Concern | Choice |
|---|---|
| API / WebSocket | FastAPI + uvicorn |
| Agent orchestration | LangGraph (+ Postgres checkpointer for memory) |
| LLMs | OpenAI / Anthropic / custom SLM (via LangChain chat models) |
| Vector DB | Milvus (one collection per embedding space, `kb_id` partition) |
| Keyword search | Elasticsearch BM25 (optional, `ENABLE_BM25`) |
| Embeddings / rerank | sentence-transformers (BGE) by default; pluggable |
| Relational store | PostgreSQL via asyncpg, **plain SQL** (no ORM) |
| Background jobs | arq + Redis |
| Secrets | cryptography (Fernet / MultiFernet) |
| Tracing | LangSmith (optional) |

---

## Prerequisites

- Python 3.11+
- Docker (for Postgres, Redis, Milvus, Elasticsearch) — or your own instances
- First run downloads the local embedding + reranker models (a few hundred MB)

---

## Quick start

```bash
# 1. Start backing services
docker compose up -d                 # postgres, redis, milvus(+etcd,minio), elasticsearch

# 2. Install
python -m venv .venv
. .venv/Scripts/activate             # Windows; on *nix: . .venv/bin/activate
pip install -e ".[dev]"

# 3. Configure
cp .env.example .env
# generate an encryption key and paste it into APP_ENCRYPTION_KEYS in .env:
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

# 4. Apply the database schema (also creates checkpointer tables on first run)
python -m app.db.migrate

# 5. Run the API and the ingestion/cleanup worker (two shells)
uvicorn app.main:app --reload
arq app.worker.WorkerSettings
```

Health check: `GET http://localhost:8000/health` returns `{"status": "ok"}`.
Interactive API docs: `http://localhost:8000/docs`.

---

## Configuration

All settings come from environment variables (see `.env.example`).

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql://rag:rag@localhost:5432/rag` | Postgres DSN (app + checkpointer) |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis for arq |
| `MILVUS_URI` | `http://localhost:19530` | Milvus endpoint |
| `MILVUS_TOKEN` | _(empty)_ | Milvus auth token (also stored per-destination, encrypted) |
| `ELASTICSEARCH_URL` | `http://localhost:9200` | Elasticsearch endpoint |
| `ENABLE_BM25` | `true` | Toggle BM25; `false` = dense-only retrieval (no Elasticsearch needed) |
| `APP_ENCRYPTION_KEYS` | _(required)_ | Comma-separated Fernet keys, **newest first** (enables rotation) |
| `EMBEDDING_PROVIDER` | `local` | `local` or `openai` |
| `EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5` | Embedding model |
| `EMBEDDING_DIM` | `768` | Must match the embedding model |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Cross-encoder reranker |
| `RETRIEVAL_TOP_K` | `5` | Chunks returned after rerank |
| `RETRIEVAL_CANDIDATES` | `20` | Candidates fetched per retriever before fusion |
| `RRF_K` | `60` | RRF constant |
| `SESSION_TTL_SECONDS` | `86400` | Idle sessions purged after this many seconds (Postgres has no native TTL) |
| `LANGSMITH_TRACING` | `false` | Enable LangSmith tracing |
| `LANGSMITH_API_KEY` | _(empty)_ | LangSmith key |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | _(empty)_ | Dev fallback only; registered LLMs live in the DB |

---

## Usage

### 1. Register a vector-DB destination

```bash
curl -X POST http://localhost:8000/destinations -H "Content-Type: application/json" -d '{
  "name": "local-milvus",
  "type": "milvus",
  "host": "localhost",
  "port": 19530,
  "secret": ""
}'
```

### 2. Register an LLM

```bash
curl -X POST http://localhost:8000/llms -H "Content-Type: application/json" -d '{
  "name": "claude",
  "provider": "anthropic",
  "model": "claude-opus-4-8",
  "api_key": "sk-ant-..."
}'
```

`provider` is `openai`, `anthropic`, or `custom` (custom requires `base_url`). Responses never include the key — only `secret_last4`.

### 3. Create a knowledge base (multipart upload)

```bash
curl -X POST http://localhost:8000/kb \
  -F "name=handbook" \
  -F "destination_id=<destination-uuid>" \
  -F 'metadata={"team":"docs"}' \
  -F "files=@./handbook.pdf" \
  -F "files=@./faq.md"
```

Returns `202 Accepted` with the KB and `status=processing`. Poll `GET /kb/{id}` until `status=ready`.

### 4. Create an agent

```bash
curl -X POST http://localhost:8000/agents -H "Content-Type: application/json" -d '{
  "name": "support-bot",
  "llm_id": "<llm-uuid>",
  "kb_ids": ["<kb-uuid>"],
  "confidence_threshold": 0.5,
  "max_retries": 1
}'
```

### 5. Chat over WebSocket (sessions + memory)

Connect to `ws://localhost:8000/ws/chat` and send a first message:

```json
{ "agent_id": "<agent-uuid>", "kb_ids": ["<kb-uuid>"], "question": "What is the refund policy?", "session_id": "<optional, to resume>" }
```

`kb_ids` and `session_id` are optional. Omit `session_id` to start a new session; pass it to
resume one (memory restored). `kb_ids`, if given, must be a subset of the agent configured
KBs (allowlist). The server first returns the session id:

```json
{ "type": "session", "session_id": "<uuid>" }
```

Then, per question, you receive status events and a final answer:

```json
{ "type": "status", "stage": "retrieving" }
{ "type": "status", "stage": "reranking" }
{ "type": "status", "stage": "generating" }
{ "type": "status", "stage": "validating" }
{ "type": "answer", "answer": "...",
  "references": ["<chunk_id>"],
  "sources": [{ "filename": "handbook.pdf", "version": 1, "chunk_ids": ["<chunk_id>"] }],
  "confidence": 0.82, "rejected": false }
```

**Keep the socket open and send more `{ "question": "..." }` messages to continue the same
session** — the agent remembers prior turns. Reconnecting later with the same `session_id`
restores the memory.

Minimal multi-turn Python client:

```python
import asyncio, json, websockets

async def main():
    async with websockets.connect("ws://localhost:8000/ws/chat") as ws:
        # first turn (no session_id -> new session)
        await ws.send(json.dumps({
            "agent_id": "<agent-uuid>",
            "kb_ids": ["<kb-uuid>"],
            "question": "What is the refund policy?",
        }))
        session_id = None
        async for raw in ws:
            msg = json.loads(raw)
            print(msg)
            if msg["type"] == "session":
                session_id = msg["session_id"]
            if msg["type"] == "answer":
                break
        # follow-up turn on the same session
        await ws.send(json.dumps({"question": "And for digital goods?"}))
        async for raw in ws:
            print(json.loads(raw))

asyncio.run(main())
```

---

## How it works

### Ingestion (worker)
`load -> semantic chunk -> embed -> upsert to Milvus (+ index in Elasticsearch)`.
Each chunk carries metadata `{filename, version, kb_id, ...your key/values}`. Re-ingesting a
filename creates a new version: the latest chunks replace the previous ones in the vector
store, while Postgres `documents` retains a row per version as history.

### Retrieval
Dense search (Milvus, cosine over normalized embeddings) and BM25 (Elasticsearch) each return
candidates; they are fused with **Reciprocal Rank Fusion** and then reordered by a
**cross-encoder reranker**. With `ENABLE_BM25=false` it degrades cleanly to dense-only.

### Agent (LangGraph)
`retrieve -> rerank -> generate -> validate -> respond`. `generate` enforces structured output
`{answer, references, confidence}`. `validate` rejects answers below `confidence_threshold`
or with no references, loops back to `generate` with corrective feedback up to `max_retries`,
then returns a safe fallback. `references` are emitted as `chunk_id`s, and the response also
includes `sources` — the documents (filename + version) those cited chunks came from, mapped
deterministically from the retrieved set.

### Sessions & memory
Each session maps to a checkpoint `thread_id` (the conversation id). The graph is compiled
with a Postgres `AsyncPostgresSaver`, so the accumulating `history` channel is persisted per
thread; resuming a session restores prior turns and follow-up questions get context.
**Postgres has no native TTL**, so `delete_expired_sessions()` runs hourly from the arq worker
and removes checkpoints + conversations idle longer than `SESSION_TTL_SECONDS` (tracked via
`last_active_at`).

### Security
Vector-DB credentials and LLM keys are encrypted with Fernet before they touch Postgres and
decrypted only at point of use; the encryption key lives in `APP_ENCRYPTION_KEYS`, never in the
DB. `query_knowledge_base` binds `kb_ids` server-side, so the model can never widen KB access;
resuming a session also verifies it belongs to the same agent.

---

## Project structure

```
app/
  main.py                 FastAPI app factory + lifespan
  config.py               settings (pydantic-settings)
  crypto.py               Fernet encrypt/decrypt + Secret wrapper
  queue.py                arq pool
  worker.py               arq worker (ingestion job + session TTL cron)
  tracing.py              LangSmith wrapper
  db/                     asyncpg pool, plain-SQL queries, migration runner
  destinations/           Milvus connection CRUD
  llms/                   LLM registration CRUD
  kb/
    ingestion/            loaders, semantic chunker, pipeline
    embeddings/           pluggable embedding providers
    vectorstore/          Milvus store
    search/               bm25, rrf, rerank, retriever
    router.py service.py schemas.py
  agent/
    graph.py              LangGraph definition (+ history channel)
    runtime.py            build a runnable agent from config
    checkpointer.py       Postgres session memory + TTL cleanup
    tools.py              query_knowledge_base (kb_ids bound server-side)
    llm/                  StructuredLLM port + provider adapters
    management.py router.py ws.py schemas.py
migrations/               0001_init.sql, 0002_session_memory.sql
tests/                    crypto + rrf unit tests
docker-compose.yml
```

---

## Testing

```bash
pytest
```

Unit tests cover the encryption round-trip and RRF fusion (no external services needed).
Full end-to-end verification (ingest -> retrieve -> answer) requires the Docker services running.

---

## Design notes

- **Embedding dimension** is taken from `EMBEDDING_DIM` and stored on each KB; it must match the embedding model.
- **Multi-KB chat** requires the attached KBs to share one embedding space *and* one destination — that is what makes a multi-KB query a single filtered search.
- **Document versioning** keeps only the latest version searchable in the vector store; full version history lives in Postgres.
- **Session memory** uses a Postgres checkpointer keyed by session id; since Postgres lacks TTL, an hourly worker job prunes sessions idle beyond `SESSION_TTL_SECONDS`.

---

## Roadmap

- Semantic caching (cache-lookup node at the front of the graph).
- Hard chunk-id reference validation and/or native citations.
- Token streaming over the WebSocket.
- Authentication and multi-tenancy.
- KMS-backed encryption key.

---

## License

No license yet — add one (e.g. MIT) before sharing publicly.
