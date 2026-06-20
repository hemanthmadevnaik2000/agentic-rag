"""Tiny plain-SQL migration runner.

Applies migrations/*.sql in filename order, tracking applied files in a
schema_migrations table. Run with:  python -m app.db.migrate
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import asyncpg

from app.config import get_settings

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"


async def run_migrations() -> None:
    conn = await asyncpg.connect(dsn=get_settings().database_url)
    try:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                filename   text PRIMARY KEY,
                applied_at timestamptz NOT NULL DEFAULT now()
            )
            """
        )
        applied = {
            r["filename"]
            for r in await conn.fetch("SELECT filename FROM schema_migrations")
        }
        for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
            if path.name in applied:
                continue
            sql = path.read_text(encoding="utf-8")
            async with conn.transaction():
                await conn.execute(sql)
                await conn.execute(
                    "INSERT INTO schema_migrations(filename) VALUES($1)", path.name
                )
            print(f"applied {path.name}")
        print("migrations up to date")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(run_migrations())
