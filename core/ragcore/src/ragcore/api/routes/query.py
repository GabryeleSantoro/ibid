"""The query route: retrieve, stream an answer, validate its citations.

Frame order is fixed and the UI depends on it:
    start -> mode -> sources -> token* -> citations -> done
with `error` replacing everything from the point it occurs.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from fastapi import APIRouter, Request

from ragcore.api.deps import AnswererDep, StoreDep
from ragcore.api.errors import api_error
from ragcore.api.schemas import (
    ChatMessage,
    CitationsEvent,
    DoneEvent,
    ErrorEvent,
    ModeEvent,
    Ok,
    QueryFilters,
    QueryMode,
    QueryRequest,
    RetrievalSettings,
    RetrievedChunk,
    SourcesEvent,
    StageLatency,
    StartEvent,
    TokenEvent,
)
from ragcore.api.sse import frame, sse_response
from ragcore.citations import extract_citations, parse_directives
from ragcore.llm import LlmError, global_prompt, system_prompt_for

router = APIRouter(tags=["query"])

_cancelled: set[str] = set()


def project_filters(store, session_id: str, filters: QueryFilters) -> QueryFilters:
    """Restrict chats to global knowledge plus the current project's folders."""
    session = store.sessions.get(session_id)
    global_ids = [source.id for source in store.sources.values() if source.project_id is None]
    allowed = set(global_ids)

    if session is not None and session.project_id is not None:
        project = store.projects.get(session.project_id)
        if project is not None:
            allowed = {
                source.id
                for source in store.sources.values()
                if source.project_id == project.id
            }
            if project.use_global_sources:
                allowed.update(global_ids)

    scoped = filters.model_copy(deep=True)
    requested = scoped.source_ids
    scoped.source_ids = (
        sorted(allowed)
        if requested is None
        else [source_id for source_id in requested if source_id in allowed]
    )
    allowed_documents = {
        document.id
        for document in store.documents.values()
        if document.source_id in allowed
    }
    requested_documents = scoped.doc_ids
    scoped.doc_ids = (
        sorted(allowed_documents)
        if requested_documents is None
        else [doc_id for doc_id in requested_documents if doc_id in allowed_documents]
    )
    return scoped


MIN_PIECES_FOR_SPEED = 30


def tokens_per_second(pieces: int, first_at: float | None, last_at: float | None) -> float | None:
    """Pieces per second from the first to the last one; load and prompt time excluded."""
    if pieces < MIN_PIECES_FOR_SPEED or first_at is None or last_at is None:
        return None
    span = last_at - first_at
    return (pieces - 1) / span if span > 0 else None


def chat_history(messages: list[ChatMessage], budget_tokens: int) -> list[tuple[str, str]]:
    """Earlier answered turns as (role, text), oldest first, newest kept within the budget.

    A question with no answer after it (failed or cancelled) is left out: providers
    reject two user turns in a row, and an empty assistant turn.
    """
    pairs = [
        (asked.text, answered.text)
        for asked, answered in zip(messages, messages[1:], strict=False)
        if asked.role == "user" and answered.role == "assistant" and answered.text.strip()
    ]
    kept: list[tuple[str, str]] = []
    spent = 0
    for question, answer in reversed(pairs):
        # ponytail: ~4 chars per token; count with the model's tokenizer if budgets run tight.
        spent += (len(question) + len(answer)) // 4
        if spent > budget_tokens:
            break
        kept.append((question, answer))
    return [turn for q, a in reversed(kept) for turn in (("user", q), ("assistant", a))]


def retrieval_query(question: str, history: list[tuple[str, str]]) -> str:
    """A follow-up ("and on a laptop?") searched alone finds nothing; prefix the last question.

    ponytail: concatenation, not an LLM rewrite. A follow-up that changes topic drags the
    previous question into the search; add a condenser call if eval shows it hurts.
    """
    previous = next((text for role, text in reversed(history) if role == "user"), None)
    return f"{previous}\n{question}" if previous else question


def retitled(store, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Index rows keep the title the file had when embedded; a rename since then wins."""
    for chunk in chunks:
        if (document := store.documents.get(chunk.doc_id)) is not None:
            chunk.doc_title = document.title
    return chunks


@dataclass(slots=True)
class Retrieval:
    mode: Literal["local", "global"]
    reason: str
    scope: str | None
    chunks: list[RetrievedChunk]
    latency: StageLatency
    candidates: int
    system_prompt: str


async def retrieve(
    store,
    question: str,
    *,
    mode: QueryMode,
    topic_id: str | None,
    filters: QueryFilters,
    history: list[tuple[str, str]],
    settings: RetrievalSettings,
    lang: str | None,
) -> Retrieval:
    """Route, then gather passages: a few per topic when global, by search when local."""
    routed = await store.brain.route(question)
    chosen, reason = routed.mode, routed.reason
    if topic_id is not None:
        chosen, reason = "global", "Asked about one topic"
    elif mode != "auto":
        chosen, reason = mode, "Set manually"
    extra = store.settings.chat_extra_instructions
    search = retrieval_query(question, history)
    if chosen == "global":
        started = time.perf_counter()
        context = await store.brain.for_query(
            routed.vector,
            search=search,
            question=question,
            filters=filters,
            doc_meta=store.doc_meta(),
            topic_id=topic_id,
            budget=settings.global_token_budget,
        )
        if context is not None:
            return Retrieval(
                "global",
                reason,
                context.scope_label,
                retitled(store, context.chunks),
                StageLatency(dense_ms=(time.perf_counter() - started) * 1000),
                context.candidates,
                system_prompt_for(lang, extra, base=global_prompt(context.outline)),
            )
        chosen, reason = "local", "No topic map yet"
    chunks, latency, candidates = await store.retriever.asearch(
        search, settings=settings, filters=filters, doc_meta=store.doc_meta()
    )
    return Retrieval(
        chosen, reason, None, retitled(store, chunks), latency, candidates,
        system_prompt_for(lang, extra),
    )


@router.post("/query")
async def query(payload: QueryRequest, request: Request, store: StoreDep, answerer: AnswererDep):
    if payload.topic_id is not None and store.brain.topic(payload.topic_id) is None:
        raise api_error(404, "topic_not_found", "topic not found")
    query_id = payload.query_id or f"q_{uuid.uuid4().hex[:10]}"
    question, directives = parse_directives(payload.q)

    session_id = payload.session_id
    if session_id is None or session_id not in store.sessions:
        session_id = store.create_session(question[:48]).id

    async def events() -> AsyncIterator[str]:
        started = time.perf_counter()
        yield frame("start", StartEvent(query_id=query_id, session_id=session_id))

        filters = project_filters(store, session_id, payload.filters)
        history = chat_history(
            store.messages.get(session_id, []), store.settings.retrieval.history_token_budget
        )
        found = await retrieve(
            store,
            question,
            mode=payload.mode,
            topic_id=payload.topic_id,
            filters=filters,
            history=history,
            settings=store.settings.retrieval,
            lang=payload.lang,
        )
        chunks, latency = found.chunks, found.latency
        yield frame("mode", ModeEvent(mode=found.mode, reason=found.reason, scope=found.scope))
        yield frame(
            "sources",
            SourcesEvent(chunks=chunks, candidates=found.candidates, kept=len(chunks)),
        )

        store.append_message(
            ChatMessage(
                id=store.new_id("msg"),
                session_id=session_id,
                role="user",
                text=question,
                created_at=datetime.now(tz=UTC),
            )
        )

        connection = store.active_connection()
        stream = answerer.stream(
            question, chunks, directives, system_prompt=found.system_prompt, history=history
        )

        answer: list[str] = []
        first_token_at: float | None = None
        last_piece_at: float | None = None
        pieces = 0
        try:
            async for piece in stream:
                if query_id in _cancelled:
                    break
                if await request.is_disconnected():
                    break
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                pieces += 1
                last_piece_at = time.perf_counter()
                answer.append(piece)
                yield frame("token", TokenEvent(text=piece))
        except Exception as exc:  # noqa: BLE001 - reported to the UI as an error frame
            logging.getLogger("ragcore.query").error("answer failed: %s", exc)
            yield frame(
                "error",
                ErrorEvent(
                    message=str(exc),
                    retryable=True,
                    code=exc.code if isinstance(exc, LlmError) else None,
                    params=exc.params if isinstance(exc, LlmError) else {},
                ),
            )
            return
        finally:
            close = getattr(stream, "aclose", None)
            if close is not None:
                await close()
            _cancelled.discard(query_id)

        text = "".join(answer)
        citations, dropped, grounding = extract_citations(text, chunks)
        yield frame(
            "citations",
            CitationsEvent(citations=citations, dropped=dropped, grounding=grounding),
        )

        latency.llm_first_token_ms = (first_token_at - started) * 1000 if first_token_at else 0.0
        latency.total_ms = (time.perf_counter() - started) * 1000

        message = ChatMessage(
            id=store.new_id("msg"),
            session_id=session_id,
            role="assistant",
            text=text,
            mode=found.mode,
            citations=citations,
            chunks=chunks,
            grounding=grounding,
            created_at=datetime.now(tz=UTC),
        )
        store.append_message(message)

        yield frame(
            "done",
            DoneEvent(
                message_id=message.id,
                latency=latency,
                connection_id=connection.id if connection else None,
                remote=bool(connection and connection.is_remote),
                tokens_per_s=(
                    tokens_per_second(pieces, first_token_at, last_piece_at)
                    if connection and connection.kind == "local"
                    else None
                ),
            ),
        )

    return sse_response(events())


@router.post("/query/{query_id}/cancel", response_model=Ok)
async def cancel_query(query_id: str) -> Ok:
    _cancelled.add(query_id)

    # The generator clears it on its way out; this is only a safety net for a
    # cancel that arrives after the stream already finished.
    async def expire() -> None:
        await asyncio.sleep(30)
        _cancelled.discard(query_id)

    with contextlib.suppress(RuntimeError):
        asyncio.create_task(expire())
    return Ok()
