"""LangSmith tracing wrapper. No-op unless configured, so the rest of the code
never has to care whether tracing is on."""
from __future__ import annotations

import os

from app.config import get_settings


def configure_tracing() -> None:
    settings = get_settings()
    if settings.langsmith_tracing and settings.langsmith_api_key:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
