"""SQLite-backed persistence for sources, documents, chat history and settings.

This is all of the application's non-vector state: everything the API layer needs
that is not a chunk embedding or its text (that lives in LanceDB, built in a later
task). Every table stores its Pydantic model as an opaque JSON blob via
``model_dump_json`` / ``model_validate_json``, so this schema does not have to
track ``api.schemas`` field-for-field while that file stays frozen. The handful of
columns pulled out alongside the JSON (``source_id``, ``path``, ``session_id``,
``created_at``) exist only so SQL can filter or order rows; they are always kept in
sync with the matching value inside the JSON blob, never a second source of truth.

``MetaStore`` is a plain persistence object, not a ``StorePort`` implementation —
that seam is built on top of it in a later task.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ragcore.api.schemas import (
    AppSettings,
    ChatMessage,
    ChatSession,
    Document,
    Source,
    SourceCreate,
)
from ragcore.brain.cards import DocCard

_SETTINGS_KEY = "app"
_UPSERT_DOCUMENT = (
    "INSERT INTO documents (id, source_id, path, json) VALUES (?, ?, ?, ?) "
    "ON CONFLICT(id) DO UPDATE SET "
    "source_id = excluded.source_id, path = excluded.path, json = excluded.json"
)
_UPSERT_SHA = (
    "INSERT INTO shas (path, sha) VALUES (?, ?) ON CONFLICT(path) DO UPDATE SET sha = excluded.sha"
)
_INSERT_CARD = "INSERT OR REPLACE INTO doc_cards (doc_id, vector, json) VALUES (?, ?, ?)"


def _placeholders(values: list) -> str:
    return ",".join("?" * len(values))


def _card_row(card: DocCard) -> tuple[str, bytes, str]:
    payload = {"summary_chunk_ids": card.summary_chunk_ids, "terms": card.terms}
    return card.doc_id, card.vector.astype(np.float32).tobytes(), json.dumps(payload)


def _card(doc_id: str, blob: bytes, raw: str) -> DocCard:
    data = json.loads(raw)
    return DocCard(
        doc_id, np.frombuffer(blob, dtype=np.float32), data["summary_chunk_ids"], data["terms"]
    )


def _now() -> datetime:
    return datetime.now(tz=UTC)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


class MetaStore:
    """All application state that isn't vectors or chunk text.

    Safe to share across FastAPI's threadpool: the connection is opened with
    ``check_same_thread=False`` and every method holds an internal lock while it
    touches the connection, since the ``sqlite3`` module does not serialize access
    to a single connection across threads on its own.
    """

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        # Under WAL, NORMAL skips the fsync per commit; a power cut can lose the
        # last commits, which only means those files are indexed again.
        self._conn.execute("PRAGMA synchronous=NORMAL")
        with self._conn:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS sources (id TEXT PRIMARY KEY, json TEXT NOT NULL)"
            )
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS documents ("
                "id TEXT PRIMARY KEY, source_id TEXT NOT NULL, path TEXT NOT NULL, "
                "json TEXT NOT NULL)"
            )
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS shas (path TEXT PRIMARY KEY, sha TEXT NOT NULL)"
            )
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, json TEXT NOT NULL)"
            )
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS messages ("
                "id TEXT PRIMARY KEY, session_id TEXT NOT NULL, created_at TEXT NOT NULL, "
                "json TEXT NOT NULL)"
            )
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, json TEXT NOT NULL)"
            )
            self._conn.execute("CREATE TABLE IF NOT EXISTS removed (path TEXT PRIMARY KEY)")
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS doc_cards ("
                "doc_id TEXT PRIMARY KEY, vector BLOB NOT NULL, json TEXT NOT NULL)"
            )

    # ------------------------------------------------------------------ sources

    def add_source(self, payload: SourceCreate) -> Source:
        source = Source(id=_new_id("src"), added_at=_now(), **payload.model_dump())
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO sources (id, json) VALUES (?, ?)",
                (source.id, source.model_dump_json()),
            )
        return source

    def replace_sources(self, sources: Iterable[Source]) -> None:
        rows = [(source.id, source.model_dump_json()) for source in sources]
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM sources")
            self._conn.executemany("INSERT INTO sources (id, json) VALUES (?, ?)", rows)

    def list_sources(self) -> dict[str, Source]:
        with self._lock:
            rows = self._conn.execute("SELECT id, json FROM sources").fetchall()
        return {row[0]: Source.model_validate_json(row[1]) for row in rows}

    def remove_source(self, source_id: str) -> list[str]:
        """Delete a source, its documents and their sha entries; return the doc ids."""
        with self._lock, self._conn:
            rows = self._conn.execute(
                "SELECT id, path FROM documents WHERE source_id = ?", (source_id,)
            ).fetchall()
            doc_ids = [row[0] for row in rows]
            paths = [row[1] for row in rows]
            self._conn.execute("DELETE FROM documents WHERE source_id = ?", (source_id,))
            self._delete_cards(doc_ids)
            if paths:
                placeholders = ",".join("?" * len(paths))
                self._conn.execute(f"DELETE FROM shas WHERE path IN ({placeholders})", paths)
            self._conn.execute("DELETE FROM sources WHERE id = ?", (source_id,))
        return doc_ids

    # ---------------------------------------------------------------- documents

    def save_documents(
        self, docs: list[Document], shas: list[tuple[str, str]], cards: Iterable[DocCard] = ()
    ) -> None:
        """Upsert documents, ``(path, sha)`` digests and doc cards in one transaction.

        Every saved document drops its old card first, so one that no longer has chunks
        keeps none.
        """
        ids = [d.id for d in docs]
        with self._lock, self._conn:
            self._conn.executemany(
                _UPSERT_DOCUMENT, [(d.id, d.source_id, d.path, d.model_dump_json()) for d in docs]
            )
            self._conn.executemany(_UPSERT_SHA, shas)
            self._delete_cards(ids)
            self._conn.executemany(_INSERT_CARD, [_card_row(c) for c in cards])

    def _delete_cards(self, doc_ids: list[str]) -> None:
        """Caller holds the lock and the transaction."""
        if doc_ids:
            self._conn.execute(
                f"DELETE FROM doc_cards WHERE doc_id IN ({_placeholders(doc_ids)})", doc_ids
            )

    def save_cards(self, cards: Iterable[DocCard]) -> None:
        with self._lock, self._conn:
            self._conn.executemany(_INSERT_CARD, [_card_row(c) for c in cards])

    def load_cards(self, doc_ids: list[str] | None = None) -> dict[str, DocCard]:
        query, params = "SELECT doc_id, vector, json FROM doc_cards", []
        if doc_ids is not None:
            if not doc_ids:
                return {}
            query, params = f"{query} WHERE doc_id IN ({_placeholders(doc_ids)})", doc_ids
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return {row[0]: _card(*row) for row in rows}

    def card_ids(self) -> set[str]:
        with self._lock:
            return {row[0] for row in self._conn.execute("SELECT doc_id FROM doc_cards")}

    def list_documents(self) -> dict[str, Document]:
        with self._lock:
            rows = self._conn.execute("SELECT id, json FROM documents").fetchall()
        return {row[0]: Document.model_validate_json(row[1]) for row in rows}

    def get_document(self, doc_id: str) -> Document | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT json FROM documents WHERE id = ?", (doc_id,)
            ).fetchone()
        return Document.model_validate_json(row[0]) if row else None

    def delete_documents(self, doc_ids: list[str]) -> None:
        if not doc_ids:
            return
        placeholders = ",".join("?" * len(doc_ids))
        with self._lock, self._conn:
            self._conn.execute(f"DELETE FROM documents WHERE id IN ({placeholders})", doc_ids)
            self._delete_cards(doc_ids)

    # --------------------------------------------------------------------- shas

    def sha_index(self) -> dict[str, str]:
        with self._lock:
            rows = self._conn.execute("SELECT path, sha FROM shas").fetchall()
        return dict(rows)

    def clear_shas(self) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM shas")

    # ------------------------------------------------------------ removed paths

    def removed_paths(self) -> set[str]:
        """Files the user took out of the library; rescans must not bring them back."""
        with self._lock:
            return {row[0] for row in self._conn.execute("SELECT path FROM removed")}

    def replace_removed_paths(self, paths: Iterable[str]) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM removed")
            self._conn.executemany("INSERT INTO removed (path) VALUES (?)", [(p,) for p in paths])

    # -------------------------------------------------------------------- wipe

    def wipe(self, *, keep_connections: bool) -> None:
        """Clear sources, documents, shas, removed paths, sessions and messages.

        Connections are not part of this store's schema (see ``ports.StorePort``);
        ``keep_connections`` is accepted only so callers routed through this store
        can pass the same flag the settings endpoint takes, and has no effect here.
        """
        del keep_connections
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM sources")
            self._conn.execute("DELETE FROM documents")
            self._conn.execute("DELETE FROM shas")
            self._conn.execute("DELETE FROM sessions")
            self._conn.execute("DELETE FROM messages")
            self._conn.execute("DELETE FROM removed")
            self._conn.execute("DELETE FROM doc_cards")

    # -------------------------------------------------------------------- chats

    def create_session(self, title: str | None, scope_doc_id: str | None) -> ChatSession:
        now = _now()
        session = ChatSession(
            id=_new_id("chat"),
            title=title or "New chat",
            message_count=0,
            scope_doc_id=scope_doc_id,
            created_at=now,
            updated_at=now,
        )
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO sessions (id, json) VALUES (?, ?)",
                (session.id, session.model_dump_json()),
            )
        return session

    def list_sessions(self) -> dict[str, ChatSession]:
        with self._lock:
            rows = self._conn.execute("SELECT id, json FROM sessions").fetchall()
        return {row[0]: ChatSession.model_validate_json(row[1]) for row in rows}

    def append_message(self, message: ChatMessage) -> None:
        """Insert the message and refresh its session's denormalized fields.

        Mirrors the stub's in-memory ``Store.append_message``: ``message_count`` and
        ``updated_at`` are kept current on the session row, and a still-untitled
        session takes its title from the first user message.
        """
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO messages (id, session_id, created_at, json) VALUES (?, ?, ?, ?)",
                (
                    message.id,
                    message.session_id,
                    message.created_at.isoformat(),
                    message.model_dump_json(),
                ),
            )
            row = self._conn.execute(
                "SELECT json FROM sessions WHERE id = ?", (message.session_id,)
            ).fetchone()
            if row is None:
                return
            session = ChatSession.model_validate_json(row[0])
            (count,) = self._conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = ?", (message.session_id,)
            ).fetchone()
            session.message_count = count
            session.updated_at = message.created_at
            if session.title == "New chat" and message.role == "user":
                session.title = message.text[:48].strip() or session.title
            self._conn.execute(
                "UPDATE sessions SET json = ? WHERE id = ?",
                (session.model_dump_json(), session.id),
            )

    def list_messages(self) -> dict[str, list[ChatMessage]]:
        with self._lock:
            session_ids = [
                row[0] for row in self._conn.execute("SELECT id FROM sessions").fetchall()
            ]
            rows = self._conn.execute(
                "SELECT session_id, json FROM messages ORDER BY created_at, id"
            ).fetchall()
        grouped: dict[str, list[ChatMessage]] = {sid: [] for sid in session_ids}
        for session_id, json_blob in rows:
            grouped.setdefault(session_id, []).append(ChatMessage.model_validate_json(json_blob))
        return grouped

    # ----------------------------------------------------------------- settings

    def load_settings(self, default: AppSettings) -> AppSettings:
        """Return the persisted settings, initializing them from ``default`` once."""
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT json FROM settings WHERE key = ?", (_SETTINGS_KEY,)
            ).fetchone()
            if row is not None:
                return AppSettings.model_validate_json(row[0])
            self._conn.execute(
                "INSERT INTO settings (key, json) VALUES (?, ?)",
                (_SETTINGS_KEY, default.model_dump_json()),
            )
        return default.model_copy(deep=True)

    def save_settings(self, settings: AppSettings) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO settings (key, json) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET json = excluded.json",
                (_SETTINGS_KEY, settings.model_dump_json()),
            )

    # ------------------------------------------------------------------- close

    def close(self) -> None:
        with self._lock:
            self._conn.close()
