"""Context for a question about the whole library: a few passages from every topic.

Topics are ranked by how close their centroid sits to the question; each gives its first
allowed representative passage plus its best dense hits. Passages interleave across topics
before packing, so a tight budget trims every topic a little instead of dropping the last
ones whole.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from itertools import zip_longest

import httpx
import numpy as np

from ragcore.api.schemas import QueryFilters, RetrievedChunk
from ragcore.retrieve.hybrid import to_chunk
from ragcore.retrieve.pack import pack
from ragcore.stub.retrieval import allowed

MAX_TOPICS = 8
PER_TOPIC = 3  # 1 representative + 2 dense hits


@dataclass(frozen=True, slots=True)
class GlobalContext:
    chunks: list[RetrievedChunk]
    outline: str
    scope_label: str | None
    n_topics: int
    candidates: int


@dataclass(frozen=True, slots=True)
class _Candidate:
    label: str
    centroid: np.ndarray
    doc_ids: frozenset[str]
    rep_chunk_ids: list[str]
    sublabels: list[str]


def _candidates(brain, docs: set[str], topic_id: str | None):
    """The scope's children; a leaf topic or no tree at all: one pseudo-topic per card."""
    nodes = brain.nodes
    scope = nodes.get(topic_id) if topic_id else None
    children = [n for n in nodes.values() if n.parent_id == (scope.id if scope else None)]
    if children:
        return scope, [
            _Candidate(
                n.label,
                n.centroid,
                frozenset(n.doc_ids) & docs,
                n.rep_chunk_ids,
                [c.label for c in nodes.values() if c.parent_id == n.id],
            )
            for n in children
        ]
    pool = docs & set(scope.doc_ids) if scope else docs
    documents = brain.store.documents
    return scope, [
        _Candidate(
            documents[d].title if d in documents else d,
            card.vector,
            frozenset([d]),
            card.summary_chunk_ids[:1],
            [],
        )
        for d, card in brain.store.meta.load_cards(sorted(pool)).items()
    ]


async def for_query(
    brain,
    vector: list[float] | None,
    *,
    search: str,
    question: str,
    filters: QueryFilters,
    doc_meta: dict[str, dict],
    topic_id: str | None,
    budget: int,
) -> GlobalContext | None:
    """None when there is nothing to stand on: no cards, no allowed documents, a blocked index."""
    store = brain.store
    if store.index_blocked:
        return None
    docs = {d for d in doc_meta if allowed(d, filters, doc_meta)}
    if not docs:
        return None
    scope, candidates = _candidates(brain, docs, topic_id)
    candidates = [c for c in candidates if c.doc_ids]
    if not candidates:
        return None
    if vector is not None:
        q = np.asarray(vector, dtype=np.float32)
        candidates.sort(key=lambda c: -float(c.centroid @ q))
    else:  # the embedder is down: the biggest topics first
        candidates.sort(key=lambda c: -len(c.doc_ids))
    kept = candidates[:MAX_TOPICS]

    # A follow-up searches with the previous question too, as in local mode.
    dense_vector = vector
    if vector is not None and search != question:
        with contextlib.suppress(httpx.HTTPError):  # the question's own vector still works
            [dense_vector] = await store.embedder.embed([search], query=True)

    rep_doc = {
        row["chunk_id"]: row["doc_id"]
        for row in store.vectors.get([cid for c in kept for cid in c.rep_chunk_ids])
    }
    per_topic: list[list[str]] = []
    for c in kept:
        rep = [cid for cid in c.rep_chunk_ids if rep_doc.get(cid) in c.doc_ids][:1]
        hits = []
        if dense_vector is not None:
            found = store.vectors.dense(dense_vector, PER_TOPIC, doc_ids=sorted(c.doc_ids))
            hits = [cid for cid, _ in found]
        per_topic.append(list(dict.fromkeys(rep + hits))[:PER_TOPIC])

    order = [cid for round_ in zip_longest(*per_topic) for cid in round_ if cid is not None]
    rows = {row["chunk_id"]: row for row in store.vectors.get(order)}
    ranked = [cid for cid in order if cid in rows]
    # Scores fall with interleave position, so pack()'s score order is the round-robin order.
    chunks = [
        to_chunk(rows[cid], dense_rank=None, bm25_rank=None, score=round(1 - i / len(ranked), 4))
        for i, cid in enumerate(ranked)
    ]
    outline = "\n".join(
        f"- {c.label} ({len(c.doc_ids)} documents)"
        + (f"; subtopics: {', '.join(c.sublabels)}" if c.sublabels else "")
        for c in kept
    )
    return GlobalContext(
        chunks=pack(chunks, budget),
        outline=outline,
        scope_label=scope.label if scope else None,
        n_topics=len(kept),
        candidates=len(chunks),
    )
