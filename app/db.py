import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite

from . import keychain

logger = logging.getLogger(__name__)

DB_PATH   = Path.home() / ".transcribe" / "archive.db"
AUDIO_DIR = Path.home() / ".transcribe" / "audio"

# Bump this and append a migration to _MIGRATIONS when the schema changes.
SCHEMA_VERSION = 1


def _connect() -> aiosqlite.Connection:
    return aiosqlite.connect(DB_PATH)


async def _configure(db: aiosqlite.Connection) -> None:
    # WAL lets readers and a writer coexist; busy_timeout makes a second
    # writer wait instead of failing immediately with "database is locked".
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA busy_timeout=5000")


async def _column_exists(db: aiosqlite.Connection, table: str, column: str) -> bool:
    async with db.execute(f"PRAGMA table_info({table})") as cur:
        return any(row[1] == column for row in await cur.fetchall())


async def _get_schema_version(db: aiosqlite.Connection) -> int:
    await db.execute(
        "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)"
    )
    async with db.execute("SELECT version FROM schema_version") as cur:
        row = await cur.fetchone()
    return row[0] if row else 0


async def _set_schema_version(db: aiosqlite.Connection, version: int) -> None:
    await db.execute("DELETE FROM schema_version")
    await db.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))


async def _migrate_to_v1(db: aiosqlite.Connection) -> None:
    """Baseline migration — brings any pre-versioning database up to date.

    Legacy databases predate the schema_version table, so we can't know which
    of the incremental ALTERs already ran; column checks make each step
    idempotent instead of guessing with a swallowed exception.
    """
    for table, column, ddl in (
        ("archive",  "speaker_names", "ALTER TABLE archive ADD COLUMN speaker_names TEXT NOT NULL DEFAULT '{}'"),
        ("archive",  "summary",       "ALTER TABLE archive ADD COLUMN summary TEXT"),
        ("archive",  "audio_ext",     "ALTER TABLE archive ADD COLUMN audio_ext TEXT"),
        ("settings", "prompt_style",  "ALTER TABLE settings ADD COLUMN prompt_style TEXT NOT NULL DEFAULT 'meeting'"),
        ("settings", "custom_prompt", "ALTER TABLE settings ADD COLUMN custom_prompt TEXT NOT NULL DEFAULT ''"),
    ):
        if not await _column_exists(db, table, column):
            await db.execute(ddl)

    # Move any legacy plaintext api_key out of SQLite into the OS keychain.
    async with db.execute("SELECT api_key FROM settings WHERE id = 1") as cur:
        row = await cur.fetchone()
    if row and row[0]:
        keychain.set_api_key(row[0])
        await db.execute("UPDATE settings SET api_key = '' WHERE id = 1")


_MIGRATIONS = [
    (1, _migrate_to_v1),
]


async def init_db() -> None:
    DB_PATH.parent.mkdir(exist_ok=True)
    AUDIO_DIR.mkdir(exist_ok=True)
    async with _connect() as db:
        await _configure(db)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS archive (
                id            TEXT PRIMARY KEY,
                filename      TEXT NOT NULL,
                created_at    TEXT NOT NULL,
                duration_s    REAL,
                speaker_count INTEGER,
                result_json   TEXT NOT NULL,
                speaker_names TEXT NOT NULL DEFAULT '{}',
                summary       TEXT,
                audio_ext     TEXT
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                id            INTEGER PRIMARY KEY CHECK (id = 1),
                provider      TEXT NOT NULL DEFAULT 'lmstudio',
                base_url      TEXT NOT NULL DEFAULT 'http://localhost:1234',
                model         TEXT NOT NULL DEFAULT '',
                api_key       TEXT NOT NULL DEFAULT '',
                prompt_style  TEXT NOT NULL DEFAULT 'meeting',
                custom_prompt TEXT NOT NULL DEFAULT ''
            )
        """)
        # FTS5 index over filename + concatenated transcript text.
        await db.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS archive_fts "
            "USING fts5(id UNINDEXED, filename, transcript, tokenize='unicode61')"
        )

        version = await _get_schema_version(db)
        for target, migrate in _MIGRATIONS:
            if version < target:
                logger.info("Migrating database schema to version %d", target)
                await migrate(db)
                version = target
        await _set_schema_version(db, version)

        # Self-healing FTS backfill: index any archive rows missing from the
        # search index (rows that pre-date it, or drift). Kept in sync by the
        # save/update/delete helpers below; this query is cheap at this
        # app's scale.
        async with db.execute(
            "SELECT id, filename, result_json FROM archive "
            "WHERE id NOT IN (SELECT id FROM archive_fts)"
        ) as cur:
            missing = await cur.fetchall()
        for row_id, fname, result_json in missing:
            try:
                segments = json.loads(result_json)
            except json.JSONDecodeError:
                logger.warning("Archive entry %s has unparseable result_json; indexing without transcript", row_id)
                segments = []
            await db.execute(
                "INSERT INTO archive_fts (id, filename, transcript) VALUES (?, ?, ?)",
                (row_id, fname, _segments_to_text(segments)),
            )
        await db.commit()


def _segments_to_text(segments: list) -> str:
    return "\n".join(s.get("text", "").strip() for s in segments if s.get("text"))


async def _fts_upsert(db, entry_id: str, filename: str, transcript: str) -> None:
    await db.execute("DELETE FROM archive_fts WHERE id = ?", (entry_id,))
    await db.execute(
        "INSERT INTO archive_fts (id, filename, transcript) VALUES (?, ?, ?)",
        (entry_id, filename, transcript),
    )


async def save_transcription(job_id: str, filename: str, segments: list, audio_ext: str | None = None) -> None:
    duration = segments[-1]["end"] if segments else 0
    speakers = len({s["speaker"] for s in segments})
    created_at = datetime.now(timezone.utc).isoformat()
    async with _connect() as db:
        await _configure(db)
        await db.execute(
            """INSERT INTO archive (id, filename, created_at, duration_s, speaker_count, result_json, speaker_names, audio_ext)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (job_id, filename, created_at, duration, speakers, json.dumps(segments), "{}", audio_ext),
        )
        await _fts_upsert(db, job_id, filename, _segments_to_text(segments))
        await db.commit()


async def list_archive() -> list[dict]:
    async with _connect() as db:
        await _configure(db)
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """SELECT id, filename, created_at, duration_s, speaker_count,
                      CASE WHEN summary IS NOT NULL AND summary != '' THEN 1 ELSE 0 END AS has_summary
               FROM archive ORDER BY created_at DESC"""
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]


async def get_archive_entry(entry_id: str) -> dict | None:
    async with _connect() as db:
        await _configure(db)
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM archive WHERE id = ?", (entry_id,)) as cur:
            row = await cur.fetchone()
    if row is None:
        return None
    d = dict(row)
    d["segments"]      = json.loads(d.pop("result_json"))
    d["speaker_names"] = json.loads(d.get("speaker_names") or "{}")
    return d


async def update_speaker_names(entry_id: str, names: dict) -> bool:
    async with _connect() as db:
        await _configure(db)
        cur = await db.execute(
            "UPDATE archive SET speaker_names = ? WHERE id = ?",
            (json.dumps(names), entry_id),
        )
        await db.commit()
        return cur.rowcount > 0


async def update_filename(entry_id: str, filename: str) -> bool:
    async with _connect() as db:
        await _configure(db)
        cur = await db.execute(
            "UPDATE archive SET filename = ? WHERE id = ?",
            (filename, entry_id),
        )
        if cur.rowcount > 0:
            await db.execute(
                "UPDATE archive_fts SET filename = ? WHERE id = ?",
                (filename, entry_id),
            )
        await db.commit()
        return cur.rowcount > 0


async def get_settings() -> dict:
    async with _connect() as db:
        await _configure(db)
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT provider, base_url, model, prompt_style, custom_prompt FROM settings WHERE id = 1"
        ) as cur:
            row = await cur.fetchone()
    if row is None:
        return {
            "provider": "lmstudio",
            "base_url": "http://localhost:1234",
            "model": "",
            "api_key": keychain.get_api_key(),
            "prompt_style": "meeting",
            "custom_prompt": "",
        }
    cfg = dict(row)
    cfg["api_key"] = keychain.get_api_key()
    return cfg


async def save_settings(
    provider: str,
    base_url: str,
    model: str,
    api_key: str,
    prompt_style: str = "meeting",
    custom_prompt: str = "",
) -> None:
    keychain.set_api_key(api_key)
    async with _connect() as db:
        await _configure(db)
        await db.execute(
            """INSERT OR REPLACE INTO settings
               (id, provider, base_url, model, api_key, prompt_style, custom_prompt)
               VALUES (1, ?, ?, ?, '', ?, ?)""",
            (provider, base_url, model, prompt_style, custom_prompt),
        )
        await db.commit()


async def save_summary(entry_id: str, summary: str) -> bool:
    async with _connect() as db:
        await _configure(db)
        cur = await db.execute(
            "UPDATE archive SET summary = ? WHERE id = ?", (summary, entry_id)
        )
        await db.commit()
        return cur.rowcount > 0


async def delete_archive_entry(entry_id: str) -> bool:
    async with _connect() as db:
        await _configure(db)
        async with db.execute("SELECT audio_ext FROM archive WHERE id = ?", (entry_id,)) as cur:
            row = await cur.fetchone()
        cur = await db.execute("DELETE FROM archive WHERE id = ?", (entry_id,))
        await db.execute("DELETE FROM archive_fts WHERE id = ?", (entry_id,))
        await db.commit()
        if cur.rowcount > 0 and row and row[0]:
            audio_path = AUDIO_DIR / f"{entry_id}{row[0]}"
            try:
                audio_path.unlink()
            except OSError:
                pass
        return cur.rowcount > 0


_FTS_RESERVED = {"and", "or", "not", "near"}


def _sanitize_fts_query(q: str) -> str:
    """Turn free-form user input into a safe FTS5 MATCH expression.

    FTS5 raises on stray operators, quotes, and many Unicode punctuation
    marks. We strip everything that isn't a word char or whitespace,
    lowercase tokens, drop any that collide with FTS5's reserved operators
    (so a user typing 'foo AND bar' isn't required to literally have 'and'
    in the document), then AND the rest together with prefix matching.
    """
    cleaned = re.sub(r"[^\w\s]", " ", q, flags=re.UNICODE)
    tokens = [t.lower() for t in cleaned.split() if t and t.lower() not in _FTS_RESERVED]
    return " ".join(f"{t}*" for t in tokens)


async def search_archive(q: str, limit: int = 50) -> list[dict]:
    match = _sanitize_fts_query(q)
    if not match:
        return []
    async with _connect() as db:
        await _configure(db)
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """
            SELECT a.id, a.filename, a.created_at, a.duration_s, a.speaker_count,
                   CASE WHEN a.summary IS NOT NULL AND a.summary != '' THEN 1 ELSE 0 END AS has_summary,
                   snippet(archive_fts, 2, '<mark>', '</mark>', '…', 10) AS snippet
            FROM archive_fts
            JOIN archive a ON a.id = archive_fts.id
            WHERE archive_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (match, limit),
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]
