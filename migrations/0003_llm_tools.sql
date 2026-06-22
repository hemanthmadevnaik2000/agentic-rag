-- Tool-calling capability per registered LLM. Drives agent-engine selection:
-- true  -> tool-calling agent (agent_node <-> tool_node loop)
-- false -> prompt pipeline (retrieve -> generate -> validate), for SLMs without tools.
ALTER TABLE llm_registrations
    ADD COLUMN IF NOT EXISTS supports_tools boolean NOT NULL DEFAULT true;
