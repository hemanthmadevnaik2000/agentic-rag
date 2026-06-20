-- Core schema for Agentic RAG. Plain SQL, applied by app/db/migrate.py.

CREATE EXTENSION IF NOT EXISTS pgcrypto;  -- gen_random_uuid()

-- Vector DB connection targets (Milvus). Secret stored encrypted at rest.
CREATE TABLE IF NOT EXISTS destinations (
    id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    name         text NOT NULL,
    type         text NOT NULL DEFAULT 'milvus',
    host         text NOT NULL,
    port         integer NOT NULL,
    db_name      text,
    username     text,
    secret_enc   bytea,
    key_version  integer NOT NULL DEFAULT 1,
    secret_last4 text,
    created_at   timestamptz NOT NULL DEFAULT now()
);

-- Registered LLM providers. api_key stored encrypted at rest.
CREATE TABLE IF NOT EXISTS llm_registrations (
    id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    name         text NOT NULL,
    provider     text NOT NULL,            -- openai | anthropic | custom
    base_url     text,                     -- for self-hosted / custom
    model        text NOT NULL,
    api_key_enc  bytea,
    key_version  integer NOT NULL DEFAULT 1,
    secret_last4 text,
    created_at   timestamptz NOT NULL DEFAULT now()
);

-- A knowledge base: documents ingested into one embedding space + destination.
CREATE TABLE IF NOT EXISTS knowledge_bases (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    name            text NOT NULL,
    destination_id  uuid NOT NULL REFERENCES destinations(id),
    embedding_model text NOT NULL,
    embedding_dim   integer NOT NULL,
    status          text NOT NULL DEFAULT 'processing',  -- processing | ready | failed
    error           text,
    metadata        jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS documents (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    kb_id       uuid NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
    filename    text NOT NULL,
    version     integer NOT NULL DEFAULT 1,
    chunk_count integer NOT NULL DEFAULT 0,
    status      text NOT NULL DEFAULT 'processing',
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS agents (
    id                   uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    name                 text NOT NULL,
    llm_id               uuid NOT NULL REFERENCES llm_registrations(id),
    kb_ids               uuid[] NOT NULL DEFAULT '{}',
    confidence_threshold double precision NOT NULL DEFAULT 0.5,
    max_retries          integer NOT NULL DEFAULT 1,
    system_prompt        text,
    created_at           timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS conversations (
    id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id   uuid NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS messages (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id uuid NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            text NOT NULL,        -- user | assistant | system
    content         text NOT NULL,
    metadata        jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_documents_kb        ON documents(kb_id);
CREATE INDEX IF NOT EXISTS idx_kb_destination      ON knowledge_bases(destination_id);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id);
