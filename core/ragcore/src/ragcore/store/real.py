"""The real library: files parsed, chunked and embedded into LanceDB, state in SQLite.

Only sources, documents, the index and settings are real here. Chats,
connections, folders and the model inventory are inherited from the stub store
unchanged — the same files in the data dir — so switching backends keeps them.

The working set (sources, documents, removed paths) lives in memory and is
written through on every change: routes read these dicts on every request and
mutate objects in place before calling ``save_library``, so reading through to
SQLite would be slower and would silently drop those writes.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import shutil
import threading
import time
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from ragcore.api.schemas import (
    AppSettings,
    Document,
    DocumentChunkRef,
    DocumentContent,
    DocumentPage,
    IndexStats,
    Source,
    SourceCreate,
)
from ragcore.config import Config
from ragcore.ingest.chunk import Chunk, chunk_document
from ragcore.ingest.parse import SUPPORTED_SUFFIXES, ParsedDoc, parse
from ragcore.ingest.walk import FoundFile, walk_source
from ragcore.models.embed import EMBED_DIM, EmbedClient
from ragcore.retrieve.hybrid import HybridRetriever
from ragcore.store.lance import ChunkRow, VectorStore
from ragcore.store.meta import MetaStore
from ragcore.stub.store import EMBED_MODEL, RERANK_MODEL, SCHEMA_VERSION, Store

logger = logging.getLogger("ragcore.library")

# Rows per LanceDB write. Every write is a fragment, and many small ones slow search.
FLUSH_ROWS = 1024
# Embedding requests in flight. llama-server is compute-bound; a second request
# only hides the round trip between batches (~7% measured on EmbeddingGemma).
MAX_IN_FLIGHT = 2


def _now() -> datetime:
    return datetime.now(tz=UTC)


def doc_id_for(path: str) -> str:
    """Stable per path and opaque: safe inside SQL predicates and as a file name."""
    return "doc_" + hashlib.sha1(path.encode()).hexdigest()[:16]


# Statuses whose sha is trusted on rescan: an unchanged file is not read again.
_SETTLED = frozenset({"indexed", "skipped"})
NO_TEXT_LAYER = "no text found, not even by OCR"


@dataclass(slots=True)
class _Parsed:
    doc_id: str
    found: FoundFile
    parsed: ParsedDoc | None = None
    chunks: list[Chunk] = field(default_factory=list)
    error: str | None = None
    vectors: list[list[float]] = field(default_factory=list)


def _parse(doc_id: str, found: FoundFile) -> _Parsed:
    """Runs in a worker thread. A file that fails to parse is that file's error only."""
    try:
        parsed = parse(found.path)
    except Exception as exc:  # noqa: BLE001 - a corrupt user file must not abort its source
        logger.warning("could not parse %s: %s", found.path, exc)
        return _Parsed(doc_id, found, error=str(exc) or type(exc).__name__)
    return _Parsed(doc_id, found, parsed, chunk_document(doc_id, parsed))


class RealStore(Store):
    def __init__(
        self,
        config: Config,
        *,
        embedder: EmbedClient,
        embed_model_id: str = EMBED_MODEL,
        reranker=None,
    ) -> None:
        self.meta = MetaStore(config.data_dir / "app.db")
        self.vectors = VectorStore(config.data_dir / "index")
        self.embedder = embedder
        self.embed_model_id = embed_model_id
        self.index_blocked: str | None = None
        self._pages_dir = config.data_dir / "pages"
        self._ingest_lock = asyncio.Lock()
        self._documents_lock = threading.Lock()
        super().__init__(config)
        self.retriever = HybridRetriever(self.vectors, embedder, reranker=reranker)

    def _seed(self) -> None:
        """Called by ``Store.__init__``: load the saved library instead of scanning fixtures."""
        self._seed_models()
        self.settings = self.meta.load_settings(AppSettings(storage_path=str(self.config.data_dir)))
        self.sources = self.meta.list_sources()
        self.documents = self.meta.list_documents()
        self.removed_paths = self.meta.removed_paths()
        self.index_blocked = self._check_index()

    # -------------------------------------------------------------- index guard

    def _index_meta(self) -> dict[str, str]:
        return {
            "schema_version": str(SCHEMA_VERSION),
            "embed_model": self.embed_model_id,
            "embed_dim": str(EMBED_DIM),
            "reranker_model": RERANK_MODEL,
        }

    def _check_index(self) -> str | None:
        """Vectors from another embedder live in another space: searching them is noise."""
        stored = self.vectors.read_meta()
        if not stored:
            self.vectors.write_meta(self._index_meta())
            return None
        if (stored.get("embed_model"), stored.get("embed_dim")) != (
            self.embed_model_id,
            str(EMBED_DIM),
        ):
            return (
                f"Index was built with {stored.get('embed_model')!r}; this build ships "
                f"{self.embed_model_id!r}: re-index to continue."
            )
        return None

    def rebuild_index(self) -> None:
        """Drop every vector; the next scan embeds each file again with this build's model."""
        self.vectors.clear()
        self.meta.clear_shas()
        self.vectors.write_meta(self._index_meta())
        self.index_blocked = None
        queued = [
            d.model_copy(update={"status": "queued", "n_chunks": 0})
            for d in self.documents.values()
        ]
        self.meta.save_documents(queued, [])
        self._update_documents(put=queued)

    # ------------------------------------------------------------------ sources

    def add_source(self, payload: SourceCreate) -> Source:
        source = self.meta.add_source(payload)
        self.sources[source.id] = source
        return source

    def save_library(self) -> None:
        self.meta.replace_sources(self.sources.values())
        self.meta.replace_removed_paths(self.removed_paths)

    def remove_source(self, source_id: str) -> int:
        source = self.sources.pop(source_id, None)
        doc_ids = self.meta.remove_source(source_id)
        self._forget(doc_ids)
        if source is not None:
            # Re-adding the folder later brings its removed files back.
            root = Path(source.path)
            self.removed_paths = {p for p in self.removed_paths if not Path(p).is_relative_to(root)}
        self.save_library()
        return len(doc_ids)

    def remove_document(self, doc_id: str) -> None:
        """Out of the index for good: rescans and restarts skip the file too."""
        document = self.documents[doc_id]
        self.removed_paths.add(document.path)
        self.meta.delete_documents([doc_id])
        self._forget([doc_id])
        if source := self.sources.get(document.source_id):
            self._recount(source)
        self.save_library()
        logger.info("removed from the library: %s", document.path)

    def _update_documents(self, *, put: Iterable[Document] = (), drop: Iterable[str] = ()) -> None:
        """Swap in an updated copy instead of mutating in place.

        Ingestion writes from a worker thread while queries iterate
        ``documents``; a reader keeps the dict it started with. The lock stops
        two writers (ingest and a threadpool route) from losing each other's update.
        """
        with self._documents_lock:
            documents = dict(self.documents)
            for doc_id in drop:
                documents.pop(doc_id, None)
            for document in put:
                documents[document.id] = document
            self.documents = documents

    def _forget(self, doc_ids: list[str]) -> None:
        self.vectors.delete_by_doc(doc_ids)
        self._update_documents(drop=doc_ids)
        for doc_id in doc_ids:
            (self._pages_dir / f"{doc_id}.json").unlink(missing_ok=True)

    def _drop(self, doc_ids: list[str]) -> None:
        self.meta.delete_documents(doc_ids)
        self._forget(doc_ids)

    def _recount(self, source: Source) -> None:
        owned = [d for d in self.documents.values() if d.source_id == source.id]
        source.document_count = len(owned)
        source.indexed_count = sum(d.status == "indexed" for d in owned)
        source.error_count = sum(d.status == "error" for d in owned)

    # ---------------------------------------------------------------- ingestion

    async def ingest_source_async(self, source_id: str) -> list[Document]:
        """Bring the index in line with a source folder; returns its documents.

        Unchanged files come back as ``skipped`` copies without being read,
        parsed or embedded; files gone from the folder leave the index.
        """
        async with self._ingest_lock:
            if self.index_blocked:
                raise RuntimeError(self.index_blocked)
            source = self.sources.get(source_id)
            if source is None:
                return []
            root = Path(source.path)
            if not root.is_dir():
                logger.warning("source folder missing, nothing indexed: %s", root)
                return []
            started = time.perf_counter()

            shas = self.meta.sha_index()
            known = {
                d.path: (d.size_bytes, d.mtime, shas[d.path])
                for d in self.documents.values()
                if d.source_id == source_id and d.status in _SETTLED and d.path in shas
            }
            found = await asyncio.to_thread(
                walk_source,
                root,
                include_globs=source.include_globs,
                exclude_globs=source.exclude_globs,
                max_file_mb=source.max_file_mb,
                suffixes=SUPPORTED_SUFFIXES,
                known=known,
            )
            found = [f for f in found if str(f.path) not in self.removed_paths]
            ids = [doc_id_for(str(f.path)) for f in found]

            present = set(ids)
            gone = [
                d.id
                for d in self.documents.values()
                if d.source_id == source_id and d.id not in present
            ]
            results: dict[str, Document] = {}
            todo: list[tuple[str, FoundFile]] = []
            for doc_id, f in zip(ids, found, strict=True):
                current = self.documents.get(doc_id)
                if (
                    current is not None
                    and current.status in _SETTLED
                    and shas.get(str(f.path)) == f.sha256
                ):
                    results[doc_id] = current.model_copy(update={"status": "skipped"})
                else:
                    todo.append((doc_id, f))

            if gone:
                await asyncio.to_thread(self._drop, gone)
            indexed = await self._index(source_id, todo)
            if source_id not in self.sources:
                # Removed (or wiped) while indexing: writes already in flight
                # landed after its cleanup, so they go now.
                await asyncio.to_thread(self._drop, [d.id for d in indexed])
                return []
            for document in indexed:
                results[document.id] = document
            if todo or gone:
                await asyncio.to_thread(self.vectors.optimize)

            self._recount(source)
            source.last_scan_at = _now()
            self.save_library()
            logger.info(
                "%s: %d indexed, %d unchanged, %d gone in %.1fs",
                root,
                len(todo),
                len(found) - len(todo),
                len(gone),
                time.perf_counter() - started,
            )
            return [results[doc_id] for doc_id in ids]

    async def _index(self, source_id: str, todo: list[tuple[str, FoundFile]]) -> list[Document]:
        """Parse, embed and write ``todo`` while keeping the embedder busy.

        Parsing runs one file ahead in a worker thread, serially because pdfium
        is not thread-safe, so it overlaps with embedding instead of adding to it.
        Chunks of consecutive files share requests of at least ``embed_batch``
        texts, at most ``MAX_IN_FLIGHT`` requests run at once, and writes land in
        groups of ``FLUSH_ROWS`` rows.
        """
        batch = self.settings.performance.embed_batch
        written: list[Document] = []
        group: list[_Parsed] = []
        in_flight: deque[tuple[list[_Parsed], asyncio.Task[list[list[float]]]]] = deque()
        ready: list[_Parsed] = []

        def parse_ahead(i: int) -> asyncio.Future[_Parsed] | None:
            if i >= len(todo):
                return None
            return asyncio.ensure_future(asyncio.to_thread(_parse, *todo[i]))

        def send() -> None:
            texts = [c.embed_text for p in group for c in p.chunks]
            task = asyncio.create_task(self.embedder.embed(texts, batch_size=batch))
            in_flight.append((group.copy(), task))
            group.clear()

        async def write(*, force: bool = False) -> None:
            if ready and (force or sum(len(p.chunks) for p in ready) >= FLUSH_ROWS):
                written.extend(await asyncio.to_thread(self._write, source_id, ready.copy()))
                ready.clear()

        async def settle_oldest() -> None:
            files, task = in_flight.popleft()
            vectors = iter(await task)
            for p in files:
                p.vectors = [next(vectors) for _ in p.chunks]
            ready.extend(files)
            await write()

        upcoming = parse_ahead(0)
        try:
            for i in range(len(todo)):
                if source_id not in self.sources:
                    break
                current = await upcoming
                upcoming = parse_ahead(i + 1)
                if not current.chunks:  # a parse error or an empty file: nothing to embed
                    ready.append(current)
                    continue
                group.append(current)
                if sum(len(p.chunks) for p in group) >= batch:
                    send()
                    if len(in_flight) >= MAX_IN_FLIGHT:
                        await settle_oldest()
            if group:
                send()
            while in_flight:
                await settle_oldest()
            await write(force=True)
        finally:
            for _, task in in_flight:
                task.cancel()
            if upcoming is not None:
                upcoming.cancel()
        return written

    def _write(self, source_id: str, files: list[_Parsed]) -> list[Document]:
        """One LanceDB delete + add and one SQLite transaction for a group of files.

        A file's old chunks go first, whether it changed or an interrupted run
        left rows behind; its sha is recorded last, so an interrupted write is
        simply redone by the next scan.
        """
        self.vectors.delete_by_doc([p.doc_id for p in files])
        self.vectors.add_chunks(
            [
                ChunkRow(
                    chunk_id=c.chunk_id,
                    doc_id=c.doc_id,
                    doc_title=c.doc_title,
                    page_start=c.page_start,
                    page_end=c.page_end,
                    section_path=c.section_path,
                    text=c.text,
                    embed_text=c.embed_text,
                    n_tokens=c.n_tokens,
                    vector=vector,
                )
                for p in files
                for c, vector in zip(p.chunks, p.vectors, strict=True)
            ]
        )
        now = _now()
        documents: list[Document] = []
        for p in files:
            self._save_pages(p)
            documents.append(self._document(source_id, p, now))
        self.meta.save_documents(
            documents, [(str(p.found.path), p.found.sha256) for p in files if p.error is None]
        )
        self._update_documents(put=documents)
        return documents

    def _save_pages(self, p: _Parsed) -> None:
        """What the reader view needs, since chunk rows carry no character offsets."""
        path = self._pages_dir / f"{p.doc_id}.json"
        if p.parsed is None:
            path.unlink(missing_ok=True)
            return
        self._pages_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "pages": [[pg.page, pg.section_path, pg.text] for pg in p.parsed.pages],
                    "chunks": [
                        [c.chunk_id, c.page_start, c.char_start, c.char_end] for c in p.chunks
                    ],
                }
            )
        )

    @staticmethod
    def _document(source_id: str, p: _Parsed, now: datetime) -> Document:
        found = p.found
        scan = (
            p.parsed is not None
            and p.parsed.needs_ocr
            and not any(pg.text for pg in p.parsed.pages)
        )
        return Document(
            id=p.doc_id,
            source_id=source_id,
            path=str(found.path),
            title=p.parsed.title if p.parsed else found.path.stem,
            ext=found.ext,
            mime=found.mime,
            size_bytes=found.size_bytes,
            n_pages=len(p.parsed.pages) if p.parsed else None,
            n_chunks=len(p.chunks),
            status="error" if p.error else "skipped" if scan else "indexed",
            error=p.error or (NO_TEXT_LAYER if scan else None),
            mtime=found.mtime,
            indexed_at=None if p.error else now,
        )

    # ---------------------------------------------------------------- documents

    def content(self, doc_id: str) -> DocumentContent | None:
        document = self.documents.get(doc_id)
        if document is None:
            return None
        try:
            data = json.loads((self._pages_dir / f"{doc_id}.json").read_text())
        except FileNotFoundError:
            return None
        pages = [DocumentPage(page=n, section_path=s, text=t) for n, s, t in data["pages"]]
        text = {page.page: page.text for page in pages}
        return DocumentContent(
            doc_id=doc_id,
            title=document.title,
            path=document.path,
            n_pages=len(pages),
            pages=pages,
            chunks=[
                DocumentChunkRef(
                    chunk_id=chunk_id,
                    page=page,
                    char_start=start,
                    char_end=end,
                    text=text.get(page, "")[start:end],
                )
                for chunk_id, page, start, end in data["chunks"]
            ],
        )

    def index_stats(self) -> IndexStats:
        documents = self.documents.values()
        return IndexStats(
            documents=len(self.documents),
            chunks=self.vectors.chunk_count(),
            parents=sum(d.n_pages or 0 for d in documents),
            topics=0,
            size_bytes=sum(d.size_bytes for d in documents),
            embed_model=self.embed_model_id,
            embed_dim=EMBED_DIM,
            reranker_model=RERANK_MODEL,
            schema_version=SCHEMA_VERSION,
            last_indexed_at=max((d.indexed_at for d in documents if d.indexed_at), default=None),
        )

    # ----------------------------------------------------------------- lifecycle

    def save_settings(self) -> None:
        self.meta.save_settings(self.settings)

    def wipe(self, *, keep_connections: bool) -> None:
        super().wipe(keep_connections=keep_connections)
        self.meta.wipe(keep_connections=keep_connections)
        shutil.rmtree(self._pages_dir, ignore_errors=True)

    def close(self) -> None:
        self.meta.close()
