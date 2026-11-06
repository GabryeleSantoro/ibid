"""The seam between the API layer and whatever is answering it.

The API layer was written against the stub's shape, so these Protocols describe
that shape rather than an idealised one. A real backend has to fit the existing
routes; the routes are contract and do not bend.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ragcore.api.schemas import (
    AppSettings,
    ChatMessage,
    ChatProject,
    ChatSession,
    Connection,
    Document,
    DocumentContent,
    Folder,
    IndexStats,
    InstalledModel,
    QueryFilters,
    RetrievalSettings,
    RetrievedChunk,
    Source,
    SourceCreate,
    StageLatency,
    Topic,
    TopicTree,
)

if TYPE_CHECKING:
    from ragcore.brain.answer import GlobalContext
    from ragcore.brain.route import Routed


@runtime_checkable
class RetrieverPort(Protocol):
    async def asearch(
        self,
        query: str,
        *,
        settings: RetrievalSettings,
        filters: QueryFilters,
        doc_meta: dict[str, dict],
    ) -> tuple[list[RetrievedChunk], StageLatency, int]:
        """Returns the kept chunks, per-stage latency, and the candidate count."""
        ...


@runtime_checkable
class BrainPort(Protocol):
    """The topic map: ``brain.topics.Brain`` (real backend) or ``brain.null.NullBrain`` (stub)."""

    building: bool

    def tree(self) -> TopicTree: ...
    def topic(self, topic_id: str) -> Topic | None: ...
    async def start(self) -> None: ...
    async def recluster(self) -> None: ...
    async def route(self, question: str) -> Routed: ...
    async def for_query(
        self,
        vector: list[float] | None,
        *,
        search: str,
        question: str,
        filters: QueryFilters,
        doc_meta: dict[str, dict],
        topic_id: str | None,
        budget: int,
    ) -> GlobalContext | None: ...


@runtime_checkable
class StorePort(Protocol):
    settings: AppSettings
    sources: dict[str, Source]
    documents: dict[str, Document]
    connections: dict[str, Connection]
    folders: dict[str, Folder]
    removed_paths: set[str]
    secrets: dict[str, str]
    models: dict[str, InstalledModel]
    sessions: dict[str, ChatSession]
    projects: dict[str, ChatProject]
    messages: dict[str, list[ChatMessage]]
    retriever: RetrieverPort
    brain: BrainPort
    index_blocked: str | None

    def add_source(self, payload: SourceCreate) -> Source: ...
    def remove_source(self, source_id: str) -> int: ...
    def remove_document(self, doc_id: str) -> None: ...
    def rename_document(self, doc_id: str, title: str) -> Document: ...
    def save_library(self) -> None: ...
    def save_chats(self) -> None: ...
    async def ingest_source_async(self, source_id: str) -> list[Document]: ...
    def rebuild_index(self) -> None: ...
    def save_connections(self) -> None: ...
    def save_folders(self) -> None: ...
    def doc_meta(self) -> dict[str, dict]: ...
    def content(self, doc_id: str) -> DocumentContent | None: ...
    def index_stats(self) -> IndexStats: ...
    def create_project(self, name: str) -> ChatProject: ...
    def create_session(
        self,
        title: str | None,
        scope_doc_id: str | None = None,
        project_id: str | None = None,
    ) -> ChatSession: ...
    def append_message(self, message: ChatMessage) -> None: ...
    def new_id(self, prefix: str) -> str: ...
    def active_connection(self) -> Connection | None: ...
    def wipe(self, *, keep_connections: bool) -> None: ...
    def save_settings(self) -> None: ...


@runtime_checkable
class AnswerEngine(Protocol):
    def stream(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        directives: set[str],
        *,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        history: Sequence[tuple[str, str]] = (),
    ) -> AsyncIterator[str]:
        """Yields answer text pieces. Directives are stub-only and ignored by real engines;
        history is earlier (role, text) turns, oldest first."""
        ...
