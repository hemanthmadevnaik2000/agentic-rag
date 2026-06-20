-- Session memory support: track activity for TTL-based checkpoint cleanup.
ALTER TABLE conversations
    ADD COLUMN IF NOT EXISTS last_active_at timestamptz NOT NULL DEFAULT now();

CREATE INDEX IF NOT EXISTS idx_conversations_last_active
    ON conversations(last_active_at);
