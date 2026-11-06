from __future__ import annotations

from fastapi import APIRouter, Query

from ragcore.api.deps import JobsDep, StoreDep
from ragcore.api.errors import api_error
from ragcore.api.schemas import (
    Document,
    DocumentContent,
    DocumentList,
    DocumentPatch,
    DocumentStatus,
    Job,
    Ok,
    ReindexRequest,
)

router = APIRouter(tags=["documents"])


@router.get("/documents", response_model=DocumentList)
def list_documents(
    store: StoreDep,
    source_id: str | None = None,
    folder_id: str | None = None,
    status: DocumentStatus | None = None,
    q: str | None = None,
    ext: str | None = None,
    offset: int = 0,
    limit: int = Query(default=100, le=500),
) -> DocumentList:
    items = list(store.documents.values())
    if source_id:
        items = [d for d in items if d.source_id == source_id]
    if folder_id:
        folder = store.folders.get(folder_id)
        members = set(folder.doc_ids) if folder else set()
        items = [d for d in items if d.id in members]
    if status:
        items = [d for d in items if d.status == status]
    if ext:
        items = [d for d in items if d.ext == ext]
    if q:
        needle = q.lower()
        items = [d for d in items if needle in d.title.lower() or needle in d.path.lower()]

    items.sort(key=lambda d: d.title.lower())
    return DocumentList(
        items=items[offset : offset + limit], total=len(items), offset=offset, limit=limit
    )


@router.get("/documents/{doc_id}", response_model=Document)
def get_document(doc_id: str, store: StoreDep) -> Document:
    document = store.documents.get(doc_id)
    if document is None:
        raise api_error(404, "document_not_found", "document not found")
    return document


@router.patch("/documents/{doc_id}", response_model=Document)
def rename_document(doc_id: str, payload: DocumentPatch, store: StoreDep) -> Document:
    if doc_id not in store.documents:
        raise api_error(404, "document_not_found", "document not found")
    title = payload.title.strip()
    if not title:
        raise api_error(422, "document_title_empty", "document title cannot be empty")
    return store.rename_document(doc_id, title)


@router.get("/documents/{doc_id}/content", response_model=DocumentContent)
def get_content(doc_id: str, store: StoreDep) -> DocumentContent:
    content = store.content(doc_id)
    if content is None:
        raise api_error(404, "document_not_found", "document not found")
    return content


@router.post("/index/rebuild", response_model=Job)
async def rebuild(payload: ReindexRequest, store: StoreDep, jobs: JobsDep) -> Job:
    targets = payload.doc_ids or [
        d.id for d in store.documents.values()
        if payload.source_id is None or d.source_id == payload.source_id
    ]
    label = "Full re-index" if payload.full else f"Re-indexing {len(targets)} documents"
    job = jobs.create("reindex", label, total=max(len(targets), 1), source_id=payload.source_id)
    return jobs.start(
        job, jobs.staged_runner([("Parsing", 1.2), ("Chunking", 0.8), ("Embedding", 2.0)])
    )


@router.delete("/documents/{doc_id}", response_model=Ok)
def remove_document(doc_id: str, store: StoreDep) -> Ok:
    if doc_id not in store.documents:
        raise api_error(404, "document_not_found", "document not found")
    store.remove_document(doc_id)
    return Ok()
