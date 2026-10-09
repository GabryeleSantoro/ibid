"""In-memory fixture store.

Stands in for SQLite plus LanceDB. Every method here maps to something the real
store will do; nothing above this layer knows the data is fake.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ragcore.api.schemas import (
    AppSettings,
    ChatMessage,
    ChatProject,
    ChatSession,
    Connection,
    Document,
    DocumentChunkRef,
    DocumentContent,
    DocumentPage,
    EvalSet,
    Folder,
    IndexStats,
    InstalledModel,
    Source,
    SourceCreate,
)
from ragcore.config import Config
from ragcore.local_connection import LOCAL_ID, local_connection
from ragcore.stub.corpus import LoadedDoc, find_fixture_dir, load_corpus
from ragcore.stub.retrieval import Retriever

logger = logging.getLogger("ragcore.library")

SCHEMA_VERSION = 1
EMBED_MODEL = "embeddinggemma-300M-qat-Q4_0"
EMBED_DIM = 768
RERANK_MODEL = "Qwen3-Reranker-0.6B-Q8_0"


def _now() -> datetime:
    return datetime.now(tz=UTC)


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


class Store:
    # Set by a store whose vectors came from another embedder; health reports it.
    index_blocked: str | None = None

    def __init__(self, config: Config) -> None:
        self.config = config
        self.started_at = _now()

        self.sources: dict[str, Source] = {}
        self.documents: dict[str, Document] = {}
        self.loaded: dict[str, LoadedDoc] = {}
        self.connections: dict[str, Connection] = {}
        self.folders: dict[str, Folder] = {}
        # Files the user removed from the library; rescans must not bring them back.
        self.removed_paths: set[str] = set()
        # Connection id -> API key. In memory only, for this process' lifetime:
        # the durable copy lives in the OS keychain, held by the Rust shell.
        self.secrets: dict[str, str] = {}
        self.models: dict[str, InstalledModel] = {}
        self.sessions: dict[str, ChatSession] = {}
        self.projects: dict[str, ChatProject] = {}
        self.messages: dict[str, list[ChatMessage]] = {}
        self.retriever = Retriever([])

        self.settings = AppSettings(
            onboarded=True,
            storage_path=str(config.data_dir),
            active_connection_id=None,
        )
        self.eval_sets = [
            EvalSet(name="base", n_questions=50, description="Public corpus, committed"),
            EvalSet(name="private", n_questions=32, description="Personal documents, local only"),
        ]

        self.load_chats()
        self._seed()
        self.load_connections()
        self.load_folders()

    # ------------------------------------------------------------------ seeding

    def _seed(self) -> None:
        self._seed_models()
        if self.load_library():
            return
        # First run only: afterwards the sample docs are the user's to keep or remove.
        fixture_dir = find_fixture_dir()
        if fixture_dir:
            source = self.add_source(
                SourceCreate(path=str(fixture_dir), include_globs=["**/*.md", "**/*.txt"])
            )
            self.ingest_source(source.id)
        # Slide conversions written before the library was persisted.
        converted = self.config.data_dir / "global-files"
        if converted.is_dir():
            source = self.add_source(
                SourceCreate(path=str(converted), include_globs=["**/*.md"], watch=True)
            )
            self.ingest_source(source.id)

    def _seed_models(self) -> None:
        shipped = [
            InstalledModel(
                id="embed-gemma-300m",
                role="embedding",
                name="EmbeddingGemma 300M",
                repo_id="ggml-org/embeddinggemma-300M-qat-q4_0-GGUF",
                filename="embeddinggemma-300M-qat-Q4_0.gguf",
                quant="Q4_0",
                size_bytes=277_852_192,
                sha256="50d28e22432a148f6f8a86eab3700f92add5d1f54baf7790675a2a4dadbccf26",
                active=True,
                shipped=True,
                downloaded_at=_now() - timedelta(days=2),
                context_length=2048,
            ),
            InstalledModel(
                id="rerank-qwen3-0.6b",
                role="reranking",
                name="Qwen3 Reranker 0.6B",
                repo_id="ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF",
                filename="qwen3-reranker-0.6b-q8_0.gguf",
                quant="Q8_0",
                size_bytes=639_000_000,
                sha256="b" * 64,
                active=True,
                shipped=True,
                downloaded_at=_now() - timedelta(days=2),
                context_length=32768,
            ),
        ]
        for model in shipped:
            self.models[model.id] = model

    # ------------------------------------------------------------------ sources

    def add_source(self, payload: SourceCreate) -> Source:
        source = Source(id=_id("src"), added_at=_now(), **payload.model_dump())
        self.sources[source.id] = source
        self.save_library()
        return source

    def remove_source(self, source_id: str) -> int:
        self.sources.pop(source_id, None)
        removed = [d for d in self.documents.values() if d.source_id == source_id]
        for doc in removed:
            self.documents.pop(doc.id, None)
            self.loaded.pop(doc.id, None)
            # Re-adding the folder later brings its files back.
            self.removed_paths.discard(doc.path)
        self.rebuild_index()
        self.save_library()
        return len(removed)

    def remove_document(self, doc_id: str) -> None:
        """Out of the index for good: rescans and restarts skip the file too."""
        document = self.documents.pop(doc_id)
        self.loaded.pop(doc_id, None)
        self.removed_paths.add(document.path)
        logger.info("removed from the library: %s", document.path)
        self.rebuild_index()
        self.save_library()

    # ------------------------------------------------------------- persistence

    # Sources and removals are stored, not documents: every start re-reads the
    # files, so edits made on disk while the app was closed are picked up.
    # ponytail: re-parses everything at startup (~0.13 s per PDF); cache parsed
    # pages by path+mtime once libraries outgrow the shell's 120 s ready timeout.
    # The packaged sample docs live in PyInstaller's per-launch temp dir, so their
    # path is saved as this placeholder and resolved again on every start.
    _FIXTURES = "<bundled-fixtures>"

    def _library_path(self) -> Path:
        return self.config.data_dir / "library.json"

    def load_library(self) -> bool:
        """Restore the saved sources and re-index them. False on a first run."""
        path = self._library_path()
        if not path.is_file():
            return False
        data = json.loads(path.read_text())
        self.removed_paths = set(data.get("removed_paths", []))
        fixture_dir = find_fixture_dir()
        for raw in data["sources"]:
            stale_sample = raw["path"].endswith("fixtures/docs") and not Path(raw["path"]).is_dir()
            if raw["path"] == self._FIXTURES or stale_sample:
                if fixture_dir is None:
                    continue
                raw = {**raw, "path": str(fixture_dir)}
            source = Source.model_validate(raw)
            self.sources[source.id] = source
            self.ingest_source(source.id)
        return True

    def save_library(self) -> None:
        path = self._library_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        fixture_dir = str(find_fixture_dir())
        sources = [s.model_dump(mode="json") for s in self.sources.values()]
        for source in sources:
            if source["path"] == fixture_dir:
                source["path"] = self._FIXTURES
        payload = {
            "sources": sources,
            "removed_paths": sorted(self.removed_paths),
        }
        path.write_text(json.dumps(payload, indent=2))

    @staticmethod
    def scan(source: Source) -> list[LoadedDoc] | None:
        """Parse a source's files. Touches no store state, so it may run in a thread."""
        directory = Path(source.path)
        if not directory.is_dir():
            logger.warning("source folder missing, nothing indexed: %s", directory)
            return None
        return load_corpus(
            directory,
            include_globs=source.include_globs,
            exclude_globs=source.exclude_globs,
            max_file_mb=source.max_file_mb,
        )

    def ingest_source(
        self, source_id: str, scanned: list[LoadedDoc] | None = None
    ) -> list[Document]:
        """Load every file under a source and mark it indexed.

        Pass ``scanned`` (from ``scan``, run off the event loop) to skip the parse.
        """
        source = self.sources.get(source_id)
        if source is None:
            return []
        directory = Path(source.path)
        if scanned is None:
            scanned = self.scan(source)
            if scanned is None:
                return []
        started = time.perf_counter()

        docs: list[Document] = []
        for loaded in scanned:
            if str(loaded.path) in self.removed_paths:
                continue
            document = Document(
                id=loaded.doc_id,
                source_id=source_id,
                path=str(loaded.path),
                title=loaded.title,
                ext=loaded.ext,
                mime=loaded.mime,
                size_bytes=loaded.size_bytes,
                n_pages=len(loaded.pages),
                n_chunks=len(loaded.chunks),
                lang="en",
                status="indexed",
                keywords=loaded.keywords,
                mtime=loaded.mtime,
                indexed_at=_now(),
            )
            self.documents[document.id] = document
            self.loaded[document.id] = loaded
            docs.append(document)

        owned = [d for d in self.documents.values() if d.source_id == source_id]
        source.document_count = len(owned)
        source.indexed_count = len([d for d in owned if d.status == "indexed"])
        source.last_scan_at = _now()
        self.rebuild_index()
        elapsed = time.perf_counter() - started
        logger.info("indexed %d documents from %s in %.1fs", len(docs), directory, elapsed)
        return docs

    async def ingest_source_async(self, source_id: str) -> list[Document]:
        """``ingest_source`` with the parse moved off the event loop."""
        source = self.sources.get(source_id)
        scanned = await asyncio.to_thread(self.scan, source) if source else None
        return self.ingest_source(source_id, scanned)

    def rebuild_index(self) -> None:
        chunks = [c for doc in self.loaded.values() for c in doc.chunks]
        self.retriever = Retriever(chunks)

    # ---------------------------------------------------------------- documents

    def doc_meta(self) -> dict[str, dict]:
        return {
            d.id: {"source_id": d.source_id, "ext": d.ext, "lang": d.lang, "mtime": d.mtime}
            for d in self.documents.values()
        }

    def content(self, doc_id: str) -> DocumentContent | None:
        loaded = self.loaded.get(doc_id)
        document = self.documents.get(doc_id)
        if loaded is None or document is None:
            return None
        page_text = {p.page: p.text for p in loaded.pages}
        return DocumentContent(
            doc_id=doc_id,
            title=document.title,
            path=document.path,
            n_pages=len(loaded.pages),
            pages=[
                DocumentPage(page=p.page, section_path=p.section_path, text=p.text)
                for p in loaded.pages
            ],
            chunks=[
                DocumentChunkRef(
                    chunk_id=c.chunk_id,
                    page=c.page,
                    char_start=c.char_start,
                    char_end=c.char_end,
                    # Sliced from the page rather than copied off the chunk: the
                    # reader highlights by offset, so the two have to agree
                    # character for character.
                    text=page_text.get(c.page, "")[c.char_start : c.char_end],
                )
                for c in loaded.chunks
            ],
        )

    # ------------------------------------------------------------------- stats

    def index_stats(self) -> IndexStats:
        chunks = sum(len(d.chunks) for d in self.loaded.values())
        return IndexStats(
            documents=len(self.documents),
            chunks=chunks,
            parents=sum(len(d.pages) for d in self.loaded.values()),
            topics=0,
            size_bytes=sum(d.size_bytes for d in self.documents.values()),
            embed_model=EMBED_MODEL,
            embed_dim=EMBED_DIM,
            reranker_model=RERANK_MODEL,
            schema_version=SCHEMA_VERSION,
            last_indexed_at=max(
                (d.indexed_at for d in self.documents.values() if d.indexed_at), default=None
            ),
        )

    # ------------------------------------------------------------------- chats

    def create_project(self, name: str) -> ChatProject:
        now = _now()
        project = ChatProject(
            id=_id("project"),
            name=name.strip() or "Untitled project",
            created_at=now,
            updated_at=now,
        )
        self.projects[project.id] = project
        self.save_chats()
        return project

    def create_session(
        self,
        title: str | None,
        scope_doc_id: str | None = None,
        project_id: str | None = None,
    ) -> ChatSession:
        now = _now()
        session = ChatSession(
            id=_id("chat"),
            title=title or "New chat",
            message_count=0,
            scope_doc_id=scope_doc_id,
            project_id=project_id,
            created_at=now,
            updated_at=now,
        )
        self.sessions[session.id] = session
        self.messages[session.id] = []
        self.save_chats()
        return session

    def append_message(self, message: ChatMessage) -> None:
        self.messages.setdefault(message.session_id, []).append(message)
        session = self.sessions.get(message.session_id)
        if session:
            session.message_count = len(self.messages[message.session_id])
            session.updated_at = message.created_at
            if session.title == "New chat" and message.role == "user":
                session.title = message.text[:48].strip() or session.title
        self.save_chats()

    def new_id(self, prefix: str) -> str:
        return _id(prefix)

    # ponytail: rewrites every chat on each change; move to MetaStore's SQLite
    # tables if long histories make saving noticeable.
    def load_chats(self) -> None:
        path = self.config.data_dir / "chats.json"
        if not path.is_file():
            return
        data = json.loads(path.read_text())
        self.projects = {
            p.id: p for p in (ChatProject.model_validate(raw) for raw in data["projects"])
        }
        self.sessions = {
            s.id: s for s in (ChatSession.model_validate(raw) for raw in data["sessions"])
        }
        self.messages = {
            session_id: [ChatMessage.model_validate(raw) for raw in items]
            for session_id, items in data["messages"].items()
        }

    def save_chats(self) -> None:
        path = self.config.data_dir / "chats.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "projects": [p.model_dump(mode="json") for p in self.projects.values()],
            "sessions": [s.model_dump(mode="json") for s in self.sessions.values()],
            "messages": {
                session_id: [m.model_dump(mode="json") for m in items]
                for session_id, items in self.messages.items()
            },
        }
        path.write_text(json.dumps(payload))

    # ------------------------------------------------------------- connections

    # Connections (never their keys) are the one thing the stub keeps on disk:
    # without them every restart sends the user back to the connection dialog.
    def _connections_path(self) -> Path:
        return self.config.data_dir / "connections.json"

    def load_connections(self) -> None:
        path = self._connections_path()
        if not path.is_file():
            return
        data = json.loads(path.read_text())
        self.connections = {
            c.id: c for c in (Connection.model_validate(raw) for raw in data["connections"])
        }
        self.settings.active_connection_id = data.get("active_connection_id")

    def save_connections(self) -> None:
        path = self._connections_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "connections": [c.model_dump(mode="json") for c in self.connections.values()],
            "active_connection_id": self.settings.active_connection_id,
        }
        path.write_text(json.dumps(payload, indent=2))

    def load_folders(self) -> None:
        path = self.config.data_dir / "folders.json"
        if path.is_file():
            raw = json.loads(path.read_text())
            self.folders = {f.id: f for f in (Folder.model_validate(item) for item in raw)}

    def save_folders(self) -> None:
        path = self.config.data_dir / "folders.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps([f.model_dump(mode="json") for f in self.folders.values()], indent=2)
        )

    # ------------------------------------------------------------------ wipe

    def wipe(self, *, keep_connections: bool) -> None:
        """Back to a first run: library, index and chats go; connections only if asked."""
        self.sources.clear()
        self.documents.clear()
        self.loaded.clear()
        self.sessions.clear()
        self.messages.clear()
        self.removed_paths.clear()
        self.rebuild_index()
        self.save_library()
        self.save_chats()
        if not keep_connections:
            self.connections.clear()
            self.settings.active_connection_id = None
            self.secrets.clear()
            self.save_connections()
        self.settings.onboarded = False
        self.save_settings()

    def save_settings(self) -> None:
        """The stub keeps settings in memory only."""

    def active_connection(self) -> Connection | None:
        """The one connection that answers, chosen by the user in the app."""
        active_id = self.settings.active_connection_id or ""
        if active_id == LOCAL_ID:
            return local_connection(active=True)
        return self.connections.get(active_id)
