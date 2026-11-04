"""The topic map over doc cards: rebuilt by reclustering, kept current by assignment between.

One per RealStore. Between reclusters a new or changed document joins the leaf whose
centroid is nearest and a removed one leaves; each counts toward ``changed_since`` until
15% of the cards have moved and the brain schedules its own recluster.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, datetime

import httpx
import numpy as np

from ragcore.api.schemas import Topic, TopicTree
from ragcore.brain.cards import DocCard
from ragcore.brain.cluster import Node, build_tree
from ragcore.brain.route import Examples, Routed, embed_examples, keyword_route, score_route
from ragcore.store.meta import EMPTY_BRAIN_STATE

logger = logging.getLogger("ragcore.brain")

MIN_CARDS = 6
RECLUSTER_SHARE = 0.15


def to_topic(node: Node) -> Topic:
    return Topic(
        id=node.id,
        parent_id=node.parent_id,
        level=node.level,
        label=node.label,
        keywords=node.keywords,
        doc_ids=node.doc_ids,
        rep_chunk_ids=node.rep_chunk_ids,
    )


def _lineage(nodes: dict[str, Node], node_id: str | None) -> Iterator[str]:
    while node_id is not None:
        yield node_id
        node_id = nodes[node_id].parent_id


class Brain:
    def __init__(self, store) -> None:
        self.store = store
        self.nodes: dict[str, Node] = store.meta.load_topics()
        self.state: dict = store.meta.brain_state()
        self.building = False
        # nodes and state are swapped, never mutated: readers keep the dict they started with.
        # Ingest (event loop), the sync remove routes (threadpool) and a recluster (thread)
        # all write, so writes take this lock.
        self._lock = threading.Lock()
        self._recluster = asyncio.Lock()
        self._generation = 0  # bumped by clear(); a recluster started before it is discarded
        self._task: asyncio.Task[None] | None = None
        self._examples: Examples | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    # ------------------------------------------------------------------- reads

    def tree(self) -> TopicTree:
        nodes, state = self.nodes, self.state
        status = "building" if self.building else "ready" if nodes else "none"
        return TopicTree(
            status=status,
            built_at=state["built_at"],
            changed_since=state["changed_since"],
            topics=[to_topic(n) for n in nodes.values()],
        )

    def topic(self, topic_id: str) -> Topic | None:
        node = self.nodes.get(topic_id)
        return to_topic(node) if node else None

    async def route(self, question: str) -> Routed:
        """Embed the question (and, once, the examples); keywords alone if the embedder is down."""
        embedder = self.store.embedder
        try:
            if self._examples is None:
                self._examples = await embed_examples(embedder)
            [vector] = await embedder.embed([question], query=True)
        except httpx.HTTPError as exc:
            logger.warning("router could not embed, routing by keywords: %s", exc)
            return keyword_route(question)
        return score_route(question, vector, self._examples)

    # ----------------------------------------------------------------- changes

    async def start(self) -> None:
        """At startup: cards for documents indexed before cards existed, then a recluster if due."""
        self._loop = asyncio.get_running_loop()
        built = {c.doc_id: c for c in await asyncio.to_thread(self.store.backfill_cards)}
        if built:
            self._apply(built, set(built), len(built))
        self._kick()

    async def changed(self, added: list[str], removed: list[str]) -> None:
        """End of an ingest: re-place what was (re)indexed, drop what left the folder."""
        self._loop = asyncio.get_running_loop()
        cards = self.store.meta.load_cards(added) if added else {}
        self._apply(cards, {*added, *removed}, len(added) + len(removed))
        self._kick()

    def removed(self, doc_ids: list[str]) -> None:
        """From the sync remove routes, which run in a worker thread."""
        self._apply({}, set(doc_ids), len(doc_ids))
        if self._loop is not None:
            with contextlib.suppress(RuntimeError):  # the loop is closed: nothing to schedule on
                self._loop.call_soon_threadsafe(self._kick)

    def clear(self) -> None:
        """The index was rebuilt or the library wiped: cards and topics go with the vectors."""
        with self._lock:
            self._generation += 1
            self.store.meta.clear_brain()
            self.nodes, self.state = {}, dict(EMPTY_BRAIN_STATE)

    def _apply(self, cards: dict[str, DocCard], drop: set[str], count: int) -> None:
        """Take ``drop`` out of every topic, put each card in its nearest leaf, persist."""
        with self._lock:
            nodes = dict(self.nodes)
            dirty: set[str] = set()
            for node in self.nodes.values():
                if drop.intersection(node.doc_ids):
                    kept = [d for d in node.doc_ids if d not in drop]
                    nodes[node.id] = replace(node, doc_ids=kept)
                    dirty.add(node.id)
            parents = {n.parent_id for n in nodes.values()}
            leaves = [n.id for n in nodes.values() if n.id not in parents]
            if leaves and cards:
                centroids = np.stack([nodes[i].centroid for i in leaves])
                for card in cards.values():
                    leaf = leaves[int(np.argmax(centroids @ card.vector))]
                    for node_id in _lineage(nodes, leaf):
                        node = nodes[node_id]
                        nodes[node_id] = replace(node, doc_ids=[*node.doc_ids, card.doc_id])
                        dirty.add(node_id)
            state = {**self.state, "changed_since": self.state["changed_since"] + count}
            self.store.meta.save_topics([nodes[i] for i in dirty], state)
            self.nodes, self.state = nodes, state

    # --------------------------------------------------------------- recluster

    def _due(self) -> bool:
        if self.store.index_blocked:
            return False
        cards = len(self.store.meta.card_ids())
        if not self.nodes:
            return cards >= MIN_CARDS
        return self.state["changed_since"] >= RECLUSTER_SHARE * cards

    def _kick(self) -> None:
        """Schedule a background recluster if one is due and none is pending. Loop thread only."""
        if (self._task is None or self._task.done()) and self._due():
            self._task = asyncio.get_running_loop().create_task(self._auto())

    async def _auto(self) -> None:
        try:
            await self.recluster()
        except Exception:  # noqa: BLE001 - the previous tree keeps serving
            logger.exception("recluster failed; keeping the previous topic map")

    async def settle(self) -> None:
        """Wait for a scheduled recluster: the CLI exits right after its ingest."""
        if self._task is not None:
            await self._task

    async def recluster(self) -> None:
        """Rebuild the tree from every card. One at a time: a second caller waits, then rebuilds."""
        async with self._recluster:
            self.building = True
            try:
                await asyncio.to_thread(self._rebuild)
            finally:
                self.building = False

    def _rebuild(self) -> None:
        meta = self.store.meta
        generation = self._generation
        cards = meta.load_cards()
        ids = sorted(cards)
        nodes: list[Node] = []
        if len(ids) >= MIN_CARDS:
            nodes = build_tree(
                ids,
                np.stack([cards[i].vector for i in ids]),
                [cards[i].terms for i in ids],
                [cards[i].summary_chunk_ids[0] for i in ids],
            )
        state = {"built_at": datetime.now(tz=UTC).isoformat(), "changed_since": 0}
        with self._lock:
            if generation != self._generation:
                return  # cleared while building: these cards are gone
            meta.replace_topics(nodes, state)
            self.nodes, self.state = {n.id: n for n in nodes}, state
        logger.info("topic map rebuilt: %d topics over %d documents", len(nodes), len(ids))
        # Documents indexed or removed while the tree was being built.
        current = meta.card_ids()
        late, gone = current - set(ids), set(ids) - current
        if late or gone:
            self._apply(meta.load_cards(sorted(late)), late | gone, len(late) + len(gone))
