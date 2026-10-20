"""The brain over a real store with fake models: cards, topics, assignment, global context."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ragcore.api.schemas import SourceCreate
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
