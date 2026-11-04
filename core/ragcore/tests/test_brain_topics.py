"""The brain over a real store with fake models: cards, topics, assignment, global context."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ragcore.api.schemas import QueryFilters, SourceCreate
from ragcore.brain import topics as topics_module
from ragcore.brain.cards import build_card
from ragcore.config import Config
from ragcore.models.embed import EMBED_DIM
from ragcore.models.fakes import FakeEmbedClient
from ragcore.store.real import RealStore, doc_id_for

WORDS = [
    "tomato soil compost",
    "telescope orbit planet",
    "budget savings interest",
    "flour yeast dough",
    "chain saddle gears",
    "embedding chunk rerank",
    "violin chord melody",
    "glacier valley summit",
]


def library(root: Path, n: int) -> Path:
    folder = root / "library"
    folder.mkdir(exist_ok=True)
    for i in range(n):
        words = WORDS[i % len(WORDS)]
        text = f"# Note {i}\n\n## Body\n\n{words} {words} detail{i}.\n"
        (folder / f"note{i}.md").write_text(text)
    return folder


def build(tmp_path: Path, embedder: FakeEmbedClient | None = None) -> RealStore:
    return RealStore(
        Config(data_dir=tmp_path, backend="real"), embedder=embedder or FakeEmbedClient()
    )


async def ingested(tmp_path: Path, n: int):
    store = build(tmp_path)
    source = store.add_source(SourceCreate(path=str(library(tmp_path, n))))
    await store.ingest_source_async(source.id)
    return store, source


# -------------------------------------------------------------------- cards


async def test_ingest_writes_a_card_per_document(tmp_path: Path) -> None:
    store, _ = await ingested(tmp_path, 3)

    cards = store.meta.load_cards()

    assert set(cards) == set(store.documents)
    card = cards[sorted(cards)[0]]
    assert card.vector.shape == (EMBED_DIM,)
    assert abs(float(np.linalg.norm(card.vector)) - 1) < 1e-5
    assert 1 <= len(card.summary_chunk_ids) <= 3
    assert card.terms and all(len(t) >= 3 for t in card.terms)
    assert "the" not in card.terms


async def test_a_file_that_loses_its_text_loses_its_card(tmp_path: Path) -> None:
    store, source = await ingested(tmp_path, 3)
    path = tmp_path / "library" / "note0.md"
    path.write_text("")

    await store.ingest_source_async(source.id)

    assert doc_id_for(str(path)) not in store.meta.card_ids()


async def test_removing_a_document_deletes_its_card(tmp_path: Path) -> None:
    store, _ = await ingested(tmp_path, 3)
    doc_id = sorted(store.documents)[0]

    store.remove_document(doc_id)

    assert doc_id not in store.meta.card_ids()


async def test_removing_a_source_deletes_its_cards(tmp_path: Path) -> None:
    store, source = await ingested(tmp_path, 3)

    store.remove_source(source.id)

    assert store.meta.card_ids() == set()


async def test_backfill_builds_cards_for_documents_indexed_before_cards(tmp_path: Path) -> None:
    store, _ = await ingested(tmp_path, 3)
    before = store.meta.load_cards()
    with store.meta._conn:  # a library indexed before doc cards existed
        store.meta._conn.execute("DELETE FROM doc_cards")

    built = store.backfill_cards()

    after = store.meta.load_cards()
    assert {c.doc_id for c in built} == set(store.documents) == set(after)
    for doc_id, card in after.items():
        assert np.allclose(card.vector, before[doc_id].vector, atol=1e-5)
        assert set(card.summary_chunk_ids) == set(before[doc_id].summary_chunk_ids)


async def test_backfill_skips_a_blocked_index(tmp_path: Path) -> None:
    store, _ = await ingested(tmp_path, 3)
    with store.meta._conn:
        store.meta._conn.execute("DELETE FROM doc_cards")
    store.index_blocked = "another embedder"

    assert store.backfill_cards() == []


async def test_wipe_deletes_the_cards(tmp_path: Path) -> None:
    store, _ = await ingested(tmp_path, 3)

    store.wipe(keep_connections=True)

    assert store.meta.card_ids() == set()


# ------------------------------------------------------------------- topics


def leaves(store: RealStore) -> list:
    parents = {n.parent_id for n in store.brain.nodes.values()}
    return [n for n in store.brain.nodes.values() if n.id not in parents]


async def test_six_cards_build_a_tree_over_every_document(tmp_path: Path) -> None:
    store, _ = await ingested(tmp_path, 8)

    await store.brain.settle()  # the ingest scheduled it: no tree yet, 8 cards

    tree = store.brain.tree()
    assert tree.status == "ready"
    assert tree.built_at is not None and tree.changed_since == 0
    top = [t for t in tree.topics if t.level == 0]
    assert sorted(d for t in top for d in t.doc_ids) == sorted(store.documents)
    assert all(t.label and t.rep_chunk_ids for t in top)
    assert store.index_stats().topics == len(tree.topics)


async def test_under_six_cards_there_is_no_tree(tmp_path: Path) -> None:
    store, _ = await ingested(tmp_path, 5)

    await store.brain.recluster()

    assert store.brain.tree().status == "none"
    assert store.brain.tree().topics == []


async def test_the_tree_survives_a_restart(tmp_path: Path) -> None:
    store, _ = await ingested(tmp_path, 8)
    await store.brain.settle()
    before = store.brain.tree()
    store.close()

    after = build(tmp_path).brain.tree()

    assert after.topics == before.topics
    assert after.built_at == before.built_at


async def test_a_new_document_joins_its_nearest_leaf(tmp_path: Path) -> None:
    store, source = await ingested(tmp_path, 8)
    await store.brain.settle()
    path = tmp_path / "library" / "fresh.md"
    path.write_text("# Fresh\n\nviolin chord melody tomato.\n")

    await store.ingest_source_async(source.id)

    doc_id = doc_id_for(str(path))
    card = store.meta.load_cards([doc_id])[doc_id]
    nearest = max(leaves(store), key=lambda n: float(n.centroid @ card.vector))
    holders = {n.id for n in store.brain.nodes.values() if doc_id in n.doc_ids}
    lineage, node_id = set(), nearest.id
    while node_id is not None:
        lineage.add(node_id)
        node_id = store.brain.nodes[node_id].parent_id
    assert holders == lineage
    assert store.brain.tree().changed_since == 1


async def test_a_removed_document_leaves_every_topic(tmp_path: Path) -> None:
    store, _ = await ingested(tmp_path, 8)
    await store.brain.settle()
    doc_id = sorted(store.documents)[0]

    store.remove_document(doc_id)

    assert all(doc_id not in t.doc_ids for t in store.brain.tree().topics)
    assert store.brain.tree().changed_since == 1


async def test_churn_past_fifteen_percent_schedules_a_recluster(tmp_path: Path) -> None:
    store, source = await ingested(tmp_path, 10)
    await store.brain.settle()
    built = store.brain.tree().built_at
    for i in (0, 1):  # 2 of 10 documents: 20%
        (tmp_path / "library" / f"note{i}.md").write_text(f"# Note {i}\n\nrewritten {i} text.\n")

    await store.ingest_source_async(source.id)
    await store.brain.settle()

    tree = store.brain.tree()
    assert tree.built_at > built
    assert tree.changed_since == 0


async def test_rebuilding_the_index_clears_the_brain(tmp_path: Path) -> None:
    store, _ = await ingested(tmp_path, 8)
    await store.brain.settle()

    store.rebuild_index()

    assert store.meta.card_ids() == set()
    assert store.brain.tree().topics == []
    store.close()
    assert build(tmp_path).brain.tree().topics == []


async def test_wipe_clears_the_brain(tmp_path: Path) -> None:
    store, _ = await ingested(tmp_path, 8)
    await store.brain.settle()

    store.wipe(keep_connections=True)

    assert store.brain.tree().status == "none"
    store.close()
    reopened = build(tmp_path)
    assert reopened.brain.tree().topics == []
    assert reopened.brain.tree().changed_since == 0


async def test_a_doc_added_during_a_recluster_joins_the_new_tree(
    tmp_path: Path, monkeypatch
) -> None:
    store, _ = await ingested(tmp_path, 8)
    await store.brain.settle()
    gone = sorted(store.documents)[0]
    [vector] = await store.embedder.embed(["late arrival words"])
    late = build_card("doc_late", ["doc_late:1:0"], ["late arrival words"], [vector])
    real_build = topics_module.build_tree

    def racing(*args, **kwargs):
        nodes = real_build(*args, **kwargs)
        store.meta.save_cards([late])  # an ingest that landed while the tree was building
        store.meta.delete_documents([gone])  # and a removal
        return nodes

    monkeypatch.setattr(topics_module, "build_tree", racing)
    await store.brain.recluster()

    held = {d for t in store.brain.tree().topics for d in t.doc_ids}
    assert "doc_late" in held
    assert gone not in held


async def test_a_recluster_that_fails_keeps_the_previous_tree(tmp_path: Path, monkeypatch) -> None:
    store, _ = await ingested(tmp_path, 8)
    await store.brain.settle()
    before = store.brain.tree().topics

    def broken(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(topics_module, "build_tree", broken)
    with pytest.raises(RuntimeError, match="boom"):
        await store.brain.recluster()

    assert store.brain.tree().topics == before
    assert store.brain.tree().status == "ready"


async def test_start_backfills_and_builds_a_tree(tmp_path: Path) -> None:
    store, _ = await ingested(tmp_path, 8)
    await store.brain.settle()
    store.brain.clear()  # an older library: documents indexed, no cards, no tree
    store.close()
    reopened = build(tmp_path)

    await reopened.brain.start()
    await reopened.brain.settle()

    assert reopened.meta.card_ids() == set(reopened.documents)
    assert reopened.brain.tree().status == "ready"


# ----------------------------------------------------------- global context


async def global_context(store: RealStore, vector, **overrides):
    args = {
        "search": "what is here",
        "question": "what is here",
        "filters": QueryFilters(),
        "doc_meta": store.doc_meta(),
        "topic_id": None,
        "budget": 6144,
    }
    return await store.brain.for_query(vector, **(args | overrides))


async def question_vector(store: RealStore) -> list[float]:
    [vector] = await store.embedder.embed(["what is here"], query=True)
    return vector


async def test_a_small_library_answers_global_from_cards(tmp_path: Path) -> None:
    store, _ = await ingested(tmp_path, 3)

    context = await global_context(store, await question_vector(store))

    assert store.brain.tree().status == "none"
    assert context is not None and context.n_topics == 3
    assert {c.doc_id for c in context.chunks} == set(store.documents)
    assert len(context.outline.splitlines()) == 3


async def test_global_context_respects_the_document_filter(tmp_path: Path) -> None:
    store, _ = await ingested(tmp_path, 8)
    await store.brain.settle()
    keep = sorted(store.documents)[:2]

    context = await global_context(
        store, await question_vector(store), filters=QueryFilters(doc_ids=keep)
    )

    assert context is not None and context.chunks
    assert {c.doc_id for c in context.chunks} <= set(keep)


async def test_global_context_without_a_vector_uses_representatives(tmp_path: Path) -> None:
    store, _ = await ingested(tmp_path, 8)
    await store.brain.settle()
    top = [n for n in store.brain.nodes.values() if n.parent_id is None]

    context = await global_context(store, None)

    assert context is not None and context.n_topics == len(top)
    assert len(context.chunks) == len(top)  # one representative each, no dense hits
    assert {c.chunk_id for c in context.chunks} <= {cid for n in top for cid in n.rep_chunk_ids}


async def test_global_passages_interleave_across_topics(tmp_path: Path) -> None:
    store, _ = await ingested(tmp_path, 8)
    await store.brain.settle()
    top = [n for n in store.brain.nodes.values() if n.parent_id is None]
    owner = {d: n.id for n in top for d in n.doc_ids}

    context = await global_context(store, await question_vector(store))

    first_round = [owner[c.doc_id] for c in context.chunks[: len(top)]]
    assert len(set(first_round)) == len(top)
    assert all(len(c.text.split()) for c in context.chunks)


async def test_a_blocked_index_has_no_global_context(tmp_path: Path) -> None:
    store, _ = await ingested(tmp_path, 3)
    store.index_blocked = "another embedder"

    assert await global_context(store, await question_vector(store)) is None
