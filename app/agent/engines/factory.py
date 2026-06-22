"""Engine registry + selection.

Add new engines by registering them here. Selection is driven by the LLM
registration: tool-capable models get the tool-calling loop, others the pipeline.
"""
from __future__ import annotations

from typing import Any

from app.agent.engines.base import AgentEngine
from app.agent.engines.pipeline import PipelineEngine
from app.agent.engines.tool_calling import ToolCallingEngine

ENGINES: dict[str, type[AgentEngine]] = {
    "tool_calling": ToolCallingEngine,
    "pipeline": PipelineEngine,
}


def engine_name_for_llm(llm_row: dict[str, Any]) -> str:
    return "tool_calling" if llm_row.get("supports_tools", True) else "pipeline"


def get_engine(name: str) -> AgentEngine:
    cls = ENGINES.get(name)
    if cls is None:
        raise ValueError(f"Unknown agent engine: {name!r}")
    return cls()


def select_engine(llm_row: dict[str, Any]) -> AgentEngine:
    return get_engine(engine_name_for_llm(llm_row))
