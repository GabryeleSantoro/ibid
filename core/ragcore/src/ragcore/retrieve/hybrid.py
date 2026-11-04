"""Retrieval: dense and BM25 legs, fused by reciprocal rank, then an optional reranker."""

from __future__ import annotations

import logging
import time

import httpx

from ragcore.api.schemas import (
    QueryFilters,
    RetrievalSettings,
    RetrievedChunk,
    StageLatency,
)
from ragcore.retrieve.pack import pack
from ragcore.store.lance import ChunkRow, VectorStore
from ragcore.stub.retrieval import allowed

logger = logging.getLogger(__name__)


def rrf(ranked: list[list[str]], k: int) -> dict[str, float]:
    """Reciprocal rank fusion. Rank is 1-based; k damps the head of each list."""
    fused: dict[str, float] = {}
    for leg in ranked:
        for rank, chunk_id in enumerate(leg, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return fused


def to_chunk(
    row: ChunkRow, *, dense_rank: int | None, bm25_rank: int | None, score: float
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=row["chunk_id"],
        doc_id=row["doc_id"],
        doc_title=row["doc_title"],
        page_start=int(row["page_start"]),
        page_end=int(row["page_end"]),
        section_path=row["section_path"] or None,
        text=row["text"],
        dense_rank=dense_rank,
        bm25_rank=bm25_rank,
        rrf_score=round(score, 6),
        rerank_score=score,
    )


class HybridRetriever:
    def __init__(self, vectors: VectorStore, embedder, *, reranker=None) -> None:
        self.vectors = vectors
        self.embedder = embedder
        self.reranker = reranker

    async def asearch(
        self,
        query: str,
        *,
        settings: RetrievalSettings,
        filters: QueryFilters,
        doc_meta: dict[str, dict],
    ) -> tuple[list[RetrievedChunk], StageLatency, int]:
        latency = StageLatency()
        docs = {doc_id for doc_id in doc_meta if allowed(doc_id, filters, doc_meta)}
        if not query.strip() or not docs:
            return [], latency, 0

        t0 = time.perf_counter()
        [vector] = await self.embedder.embed([query], query=True)
        latency.embed_ms = (time.perf_counter() - t0) * 1000

        # An unfiltered query skips the IN list; chunks of unlisted documents
        # (a write the library never saw) are dropped below instead.
        # ponytail: LanceDB calls block the event loop; to_thread them if a big
        # index makes queries stall ingest progress.
        t0 = time.perf_counter()
        scoped = sorted(docs) if len(docs) < len(doc_meta) else None
        dense = [cid for cid, _ in self.vectors.dense(vector, settings.dense_top_k, doc_ids=scoped)]
        latency.dense_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        bm25 = [cid for cid, _ in self.vectors.fts(query, settings.bm25_top_k, doc_ids=scoped)]
        latency.bm25_ms = (time.perf_counter() - t0) * 1000

        fused = rrf([dense, bm25], settings.rrf_k)
        order = sorted(fused, key=lambda cid: -fused[cid])[: settings.rerank_candidates]
        dense_rank = {cid: rank for rank, cid in enumerate(dense, start=1)}
        bm25_rank = {cid: rank for rank, cid in enumerate(bm25, start=1)}
        scored = [
            to_chunk(
                row,
                dense_rank=dense_rank.get(row["chunk_id"]),
                bm25_rank=bm25_rank.get(row["chunk_id"]),
                score=fused[row["chunk_id"]],
            )
            for row in self.vectors.get(order)
            if row["doc_id"] in docs
        ]
        candidates = len(scored)

        if self.reranker is not None and scored:
            t0 = time.perf_counter()
            try:
                ranked = await self.reranker.rerank(
                    query, [c.text for c in scored], top_n=settings.top_k
                )
            except httpx.HTTPError as exc:
                # A reranker that is down or still loading costs precision, not the answer.
                logger.warning("rerank skipped, keeping fused order: %s", exc)
            else:
                # Only reranker probabilities are on the scale `min_score` describes;
                # cosine and RRF scores sit near 0.0-0.05 and would all be dropped.
                scored = [
                    scored[i].model_copy(update={"rerank_score": score})
                    for i, score in ranked
                    if score >= settings.min_score
                ]
            latency.rerank_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        kept = pack(scored[: settings.top_k], settings.context_token_budget)
        latency.pack_ms = (time.perf_counter() - t0) * 1000
        latency.total_ms = (
            latency.embed_ms
            + latency.dense_ms
            + latency.bm25_ms
            + latency.rerank_ms
            + latency.pack_ms
        )
        return kept, latency, candidates
