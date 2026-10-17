"""The real store end to end, with a fake embedder: ingest, re-ingest, remove, guard."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest
from ragcore.api.schemas import Source, SourceCreate
from ragcore.config import Config
from ragcore.models.fakes import FakeEmbedClient
from ragcore.store.real import RealStore


class CountingEmbedClient(FakeEmbedClient):
    """Records how many texts each embedding request carried."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    async def embed(
        self, texts: list[str], *, batch_size: int = 32, query: bool = False
    ) -> list[list[float]]:
        self.calls.append(len(texts))
        return await super().embed(texts, batch_size=batch_size, query=query)


class GatedEmbedClient(FakeEmbedClient):
    """Holds every request until released, so a test can act mid-ingest."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def embed(
        self, texts: list[str], *, batch_size: int = 32, query: bool = False
    ) -> list[list[float]]:
        self.started.set()
        await self.release.wait()
        return await super().embed(texts, batch_size=batch_size, query=query)


def corpus(root: Path) -> Path:
    docs = root / "corpus"
    docs.mkdir()
    (docs / "reranking.md").write_text(
        "# Reranking\n\n## Cross encoders\n\nCross encoders score query passage pairs.\n"
    )
    (docs / "chunking.md").write_text(
        "# Chunking\n\n## Sizes\n\nSmaller chunks retrieve precisely.\n"
    )
    return docs


def build(tmp_path: Path, embedder: FakeEmbedClient | None = None, **kwargs: str) -> RealStore:
    config = Config(data_dir=tmp_path, backend="real")
    return RealStore(config, embedder=embedder or FakeEmbedClient(), **kwargs)


def add(store: RealStore, folder: Path) -> Source:
    return store.add_source(SourceCreate(path=str(folder)))


async def test_ingesting_a_source_indexes_its_documents(tmp_path: Path) -> None:
    store = build(tmp_path)
    source = add(store, corpus(tmp_path))

    documents = await store.ingest_source_async(source.id)

    assert len(documents) == 2
    assert {d.status for d in documents} == {"indexed"}
    assert store.index_stats().chunks > 0
    assert store.sources[source.id].indexed_count == 2


async def test_reingesting_unchanged_files_is_a_no_op(tmp_path: Path) -> None:
    store = build(tmp_path)
    source = add(store, corpus(tmp_path))
    await store.ingest_source_async(source.id)
    chunks_before = store.index_stats().chunks

    again = await store.ingest_source_async(source.id)

    assert {d.status for d in again} == {"skipped"}
    assert {d.status for d in store.documents.values()} == {"indexed"}
    assert store.index_stats().chunks == chunks_before


async def test_changed_file_is_reindexed(tmp_path: Path) -> None:
    docs = corpus(tmp_path)
    store = build(tmp_path)
    source = add(store, docs)
    await store.ingest_source_async(source.id)

    (docs / "reranking.md").write_text(
        "# Reranking\n\n## Cross encoders\n\nRewritten body with different words entirely.\n"
    )
    again = await store.ingest_source_async(source.id)

    statuses = {d.title: d.status for d in again}
    assert statuses["Reranking"] == "indexed"
    assert statuses["Chunking"] == "skipped"
    content = store.content(next(d.id for d in again if d.title == "Reranking"))
    assert content is not None
    assert "Rewritten" in " ".join(c.text for c in content.chunks)
    assert store.index_stats().chunks == sum(d.n_chunks for d in store.documents.values())


async def test_removing_a_source_drops_its_chunks(tmp_path: Path) -> None:
    store = build(tmp_path)
    source = add(store, corpus(tmp_path))
    await store.ingest_source_async(source.id)

    removed = store.remove_source(source.id)

    assert removed == 2
    assert store.index_stats().chunks == 0
    assert store.documents == {}
    assert store.sources == {}


async def test_document_content_is_readable_for_the_reader_view(tmp_path: Path) -> None:
    store = build(tmp_path)
    source = add(store, corpus(tmp_path))
    documents = await store.ingest_source_async(source.id)

    content = store.content(documents[0].id)

    assert content is not None
    assert content.n_pages == len(content.pages)
    assert all(c.page <= content.n_pages for c in content.chunks)
    page_text = {p.page: p.text for p in content.pages}
    assert all(
        c.text and c.text == page_text[c.page][c.char_start : c.char_end] for c in content.chunks
    )


async def test_a_different_embed_model_blocks_the_index(tmp_path: Path) -> None:
    store = build(tmp_path)
    source = add(store, corpus(tmp_path))
    await store.ingest_source_async(source.id)
    store.close()

    reopened = build(tmp_path, embed_model_id="some-other-model")

    assert reopened.index_blocked is not None
    assert "re-index" in reopened.index_blocked


async def test_a_blocked_index_refuses_to_ingest_until_rebuilt(tmp_path: Path) -> None:
    store = build(tmp_path)
    source = add(store, corpus(tmp_path))
    await store.ingest_source_async(source.id)
    store.close()
    reopened = build(tmp_path, embed_model_id="some-other-model")

    with pytest.raises(RuntimeError, match="re-index"):
        await reopened.ingest_source_async(source.id)

    reopened.rebuild_index()
    assert reopened.index_blocked is None
    assert reopened.index_stats().chunks == 0
    documents = await reopened.ingest_source_async(source.id)
    assert {d.status for d in documents} == {"indexed"}
    reopened.close()
    assert build(tmp_path, embed_model_id="some-other-model").index_blocked is None


async def test_a_restart_restores_the_library_without_re_embedding(tmp_path: Path) -> None:
    store = build(tmp_path)
    source = add(store, corpus(tmp_path))
    await store.ingest_source_async(source.id)
    store.close()

    embedder = CountingEmbedClient()
    reopened = build(tmp_path, embedder)
    again = await reopened.ingest_source_async(source.id)

    assert {d.title for d in reopened.documents.values()} == {"Reranking", "Chunking"}
    assert {d.status for d in again} == {"skipped"}
    assert embedder.calls == []


async def test_small_documents_share_embedding_requests(tmp_path: Path) -> None:
    docs = tmp_path / "notes"
    docs.mkdir()
    for i in range(40):
        (docs / f"note{i}.md").write_text(f"# Note {i}\n\nBody number {i}.\n")
    embedder = CountingEmbedClient()
    store = build(tmp_path, embedder)
    source = add(store, docs)

    await store.ingest_source_async(source.id)

    assert sum(embedder.calls) == store.index_stats().chunks
    assert len(embedder.calls) <= 2


async def test_a_deleted_file_leaves_the_index_on_rescan(tmp_path: Path) -> None:
    docs = corpus(tmp_path)
    store = build(tmp_path)
    source = add(store, docs)
    await store.ingest_source_async(source.id)

    (docs / "chunking.md").unlink()
    again = await store.ingest_source_async(source.id)

    assert [d.title for d in again] == ["Reranking"]
    assert {d.title for d in store.documents.values()} == {"Reranking"}
    assert store.sources[source.id].document_count == 1
    assert store.index_stats().chunks == store.documents[again[0].id].n_chunks


async def test_a_removed_document_stays_out_after_a_restart(tmp_path: Path) -> None:
    store = build(tmp_path)
    source = add(store, corpus(tmp_path))
    documents = await store.ingest_source_async(source.id)
    chunking = next(d for d in documents if d.title == "Chunking")
    store.remove_document(chunking.id)
    store.close()

    reopened = build(tmp_path)
    again = await reopened.ingest_source_async(source.id)

    assert [d.title for d in again] == ["Reranking"]
    assert chunking.id not in reopened.documents


async def test_a_quote_in_a_file_name_is_harmless(tmp_path: Path) -> None:
    docs = tmp_path / "quoted"
    docs.mkdir()
    (docs / "it's here.md").write_text("# Quoted\n\nA body worth indexing.\n")
    store = build(tmp_path)
    source = add(store, docs)

    [document] = await store.ingest_source_async(source.id)
    store.remove_document(document.id)

    assert re.fullmatch(r"[A-Za-z0-9_]+", document.id)
    assert store.index_stats().chunks == 0


async def test_an_unreadable_file_is_an_error_not_a_failed_scan(tmp_path: Path) -> None:
    docs = corpus(tmp_path)
    (docs / "broken.pdf").write_bytes(b"not a pdf at all")
    (docs / "photo.png").write_bytes(b"\x89PNG\r\n")
    store = build(tmp_path)
    source = add(store, docs)

    documents = await store.ingest_source_async(source.id)

    statuses = {Path(d.path).name: d.status for d in documents}
    assert statuses == {"reranking.md": "indexed", "chunking.md": "indexed", "broken.pdf": "error"}
    assert store.sources[source.id].error_count == 1


async def test_wipe_empties_the_library_and_the_index(tmp_path: Path) -> None:
    store = build(tmp_path)
    source = add(store, corpus(tmp_path))
    await store.ingest_source_async(source.id)

    store.wipe(keep_connections=True)

    assert store.documents == {}
    assert store.sources == {}
    assert store.index_stats().chunks == 0
    store.close()
    reopened = build(tmp_path)
    assert reopened.documents == {}
    assert reopened.sources == {}


async def test_settings_survive_a_restart(tmp_path: Path) -> None:
    store = build(tmp_path)
    store.settings.onboarded = True
    store.save_settings()
    store.close()

    assert build(tmp_path).settings.onboarded is True


async def test_a_reader_mid_iteration_survives_an_ingest(tmp_path: Path) -> None:
    docs = corpus(tmp_path)
    store = build(tmp_path)
    source = add(store, docs)
    await store.ingest_source_async(source.id)
    (docs / "fresh.md").write_text("# Fresh\n\nA new note.\n")

    reading = iter(store.documents.values())
    next(reading)
    await store.ingest_source_async(source.id)

    assert len(list(reading)) == 1
    assert len(store.documents) == 3


async def test_removing_a_source_mid_ingest_leaves_nothing_behind(tmp_path: Path) -> None:
    embedder = GatedEmbedClient()
    store = build(tmp_path, embedder)
    source = add(store, corpus(tmp_path))

    ingest = asyncio.create_task(store.ingest_source_async(source.id))
    await embedder.started.wait()
    store.remove_source(source.id)
    embedder.release.set()
    await ingest

    assert store.documents == {}
    assert store.index_stats().chunks == 0


async def test_a_pdf_with_no_text_layer_is_skipped_with_a_reason_once(
    tmp_path: Path, text_pdf, monkeypatch: pytest.MonkeyPatch
) -> None:
    scans = tmp_path / "scans"
    scans.mkdir()
    (scans / "scan.pdf").write_bytes(text_pdf("", ""))
    store = build(tmp_path)
    source = add(store, scans)

    [first] = await store.ingest_source_async(source.id)

    def never(path: Path):
        raise AssertionError(f"re-parsed an unchanged scan: {path}")

    monkeypatch.setattr("ragcore.store.real.parse", never)
    [again] = await store.ingest_source_async(source.id)

    for document in (first, again):
        assert document.status == "skipped"
        assert "OCR" in (document.error or "")


async def test_pdf_citations_keep_their_page_past_a_blank_one(tmp_path: Path, text_pdf) -> None:
    pdfs = tmp_path / "pdfs"
    pdfs.mkdir()
    (pdfs / "paper.pdf").write_bytes(
        text_pdf("Cross encoders score pairs.", "", "Hybrid search merges legs.")
    )
    store = build(tmp_path)
    [document] = await store.ingest_source_async(add(store, pdfs).id)

    content = store.content(document.id)

    assert content is not None
    assert content.n_pages == 3
    assert [c.page for c in content.chunks if "Hybrid" in c.text] == [3]


async def test_a_scan_ocr_reads_is_indexed(tmp_path: Path, text_pdf, monkeypatch) -> None:
    from ragcore.ingest import ocr

    monkeypatch.setattr(ocr, "read_page", lambda page: "Rerankers rescore the fused candidates.")
    docs = tmp_path / "scans"
    docs.mkdir()
    (docs / "scan.pdf").write_bytes(text_pdf(""))
    store = build(tmp_path)
    source = add(store, docs)

    [document] = await store.ingest_source_async(source.id)

    assert document.status == "indexed"
    assert document.n_chunks >= 1


async def test_a_scan_ocr_cannot_read_is_skipped_with_a_reason(
    tmp_path: Path, text_pdf, monkeypatch
) -> None:
    from ragcore.ingest import ocr

    monkeypatch.setattr(ocr, "read_page", lambda page: "")
    docs = tmp_path / "scans"
    docs.mkdir()
    (docs / "blank.pdf").write_bytes(text_pdf(""))
    store = build(tmp_path)
    source = add(store, docs)

    [document] = await store.ingest_source_async(source.id)

    assert document.status == "skipped"
    assert document.error == "no text found, not even by OCR"


async def test_indexed_documents_carry_their_language(tmp_path: Path) -> None:
    docs = tmp_path / "it"
    docs.mkdir()
    (docs / "nota.md").write_text(
        "# Nota\n\nIl recupero dei passaggi combina la ricerca semantica con quella per "
        "parole chiave. I risultati vengono poi riordinati da un modello che valuta quanto "
        "ogni passaggio risponde alla domanda dell'utente, e solo i migliori arrivano al "
        "modello linguistico che scrive la risposta.\n"
    )
    store = build(tmp_path)
    source = add(store, docs)

    [document] = await store.ingest_source_async(source.id)

    assert document.lang == "it"
