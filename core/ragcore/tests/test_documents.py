"""The library listing: filters, paging, and what the reader is handed."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from ragcore.api.app import create_app


def a_source(client: TestClient, folder: Path, **extra) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    return client.post(
        "/sources", json={"path": str(folder), "include_globs": ["**/*.md"], **extra}
    ).json()


def test_the_fixture_corpus_is_listed(client: TestClient) -> None:
    listing = client.get("/documents").json()

    assert listing["total"] > 0
    assert listing["items"][0]["status"] == "indexed"


def test_documents_come_back_sorted_by_title(client: TestClient) -> None:
    titles = [d["title"] for d in client.get("/documents").json()["items"]]

    assert titles == sorted(titles, key=str.lower)


def test_paging_reports_the_full_total_not_the_page_size(client: TestClient) -> None:
    first = client.get("/documents", params={"limit": 2}).json()

    assert len(first["items"]) <= 2
    assert first["total"] > len(first["items"])
    assert (first["offset"], first["limit"]) == (0, 2)


def test_an_offset_skips_the_earlier_documents(client: TestClient) -> None:
    everything = client.get("/documents").json()["items"]

    page = client.get("/documents", params={"offset": 1, "limit": 1}).json()

    assert page["items"][0]["id"] == everything[1]["id"]
    assert page["offset"] == 1


def test_an_offset_past_the_end_returns_an_empty_page(client: TestClient) -> None:
    listing = client.get("/documents", params={"offset": 9999}).json()

    assert listing["items"] == []
    assert listing["total"] > 0


def test_the_page_size_is_capped(client: TestClient) -> None:
    assert client.get("/documents", params={"limit": 5000}).status_code == 422


def test_filtering_by_source_keeps_only_that_sources_documents(
    client: TestClient, tmp_path: Path
) -> None:
    folder = tmp_path / "notes"
    folder.mkdir()
    (folder / "one.md").write_text("# One\n\nReranking reorders candidates.\n")
    source = a_source(client, folder)

    listing = client.get("/documents", params={"source_id": source["id"]}).json()

    assert listing["total"] == 1
    assert listing["items"][0]["source_id"] == source["id"]


def test_filtering_by_an_unknown_source_is_empty_not_an_error(client: TestClient) -> None:
    assert client.get("/documents", params={"source_id": "src_nope"}).json()["total"] == 0


def test_filtering_by_extension(client: TestClient) -> None:
    listing = client.get("/documents", params={"ext": ".md"}).json()

    assert listing["total"] > 0
    assert {d["ext"] for d in listing["items"]} == {".md"}
    assert client.get("/documents", params={"ext": ".docx"}).json()["total"] == 0


def test_filtering_by_status(client: TestClient) -> None:
    assert client.get("/documents", params={"status": "indexed"}).json()["total"] > 0
    assert client.get("/documents", params={"status": "error"}).json()["total"] == 0


def test_an_unknown_status_is_rejected(client: TestClient) -> None:
    assert client.get("/documents", params={"status": "haunted"}).status_code == 422


def test_the_search_box_matches_title_or_path_case_insensitively(
    client: TestClient, tmp_path: Path
) -> None:
    folder = tmp_path / "notes"
    folder.mkdir()
    (folder / "Cobalt-Launch.md").write_text("# Cobalt Launch\n\nQuarterly numbers.\n")
    a_source(client, folder)

    assert client.get("/documents", params={"q": "COBALT"}).json()["total"] == 1
    assert client.get("/documents", params={"q": "cobalt-launch"}).json()["total"] == 1
    assert client.get("/documents", params={"q": "zzz"}).json()["total"] == 0


def test_filters_combine(client: TestClient, tmp_path: Path) -> None:
    folder = tmp_path / "notes"
    folder.mkdir()
    (folder / "alpha.md").write_text("# Alpha\n\nFirst.\n")
    (folder / "beta.md").write_text("# Beta\n\nSecond.\n")
    source = a_source(client, folder)

    listing = client.get(
        "/documents", params={"source_id": source["id"], "q": "alpha", "ext": ".md"}
    ).json()

    assert [d["title"] for d in listing["items"]] == ["Alpha"]


# ----------------------------------------------------------------- one document


def test_a_document_can_be_fetched_by_id(client: TestClient) -> None:
    listed = client.get("/documents", params={"limit": 1}).json()["items"][0]

    assert client.get(f"/documents/{listed['id']}").json() == listed


def test_an_unknown_document_is_a_404(client: TestClient) -> None:
    assert client.get("/documents/doc_nope").status_code == 404
    assert client.get("/documents/doc_nope/content").status_code == 404
    assert client.delete("/documents/doc_nope").status_code == 404


def test_content_pages_and_chunks_agree_character_for_character(client: TestClient) -> None:
    doc_id = client.get("/documents", params={"limit": 1}).json()["items"][0]["id"]

    content = client.get(f"/documents/{doc_id}/content").json()

    assert content["n_pages"] == len(content["pages"])
    pages = {page["page"]: page["text"] for page in content["pages"]}
    for chunk in content["chunks"]:
        assert chunk["text"] == pages[chunk["page"]][chunk["char_start"] : chunk["char_end"]]


def test_deleting_a_document_removes_it_from_the_index(client: TestClient) -> None:
    before = client.get("/documents").json()
    doc_id = before["items"][0]["id"]

    assert client.delete(f"/documents/{doc_id}").json() == {"ok": True}

    after = client.get("/documents").json()
    assert after["total"] == before["total"] - 1
    assert client.get(f"/documents/{doc_id}/content").status_code == 404


def test_a_deleted_documents_chunks_stop_being_retrievable(
    client: TestClient, read_events
) -> None:
    """Deleting must rebuild the retriever, not just the listing."""
    with client.stream("POST", "/query", json={"q": "reciprocal rank fusion"}) as response:
        chunks = dict(read_events(response))["sources"]["chunks"]
    assert chunks, "nothing retrieved, so this test would prove nothing"
    doc_id = chunks[0]["doc_id"]

    client.delete(f"/documents/{doc_id}")

    with client.stream("POST", "/query", json={"q": "reciprocal rank fusion"}) as response:
        after = dict(read_events(response))["sources"]["chunks"]
    assert doc_id not in {chunk["doc_id"] for chunk in after}


# --------------------------------------------------------------------- rebuild


def test_a_full_rebuild_is_labelled_as_such(client: TestClient) -> None:
    job = client.post("/index/rebuild", json={"full": True}).json()

    assert job["kind"] == "reindex"
    assert job["label"] == "Full re-index"


def test_rebuilding_one_source_counts_only_its_documents(
    client: TestClient, tmp_path: Path
) -> None:
    folder = tmp_path / "notes"
    folder.mkdir()
    (folder / "one.md").write_text("# One\n\nBody.\n")
    source = a_source(client, folder)

    job = client.post("/index/rebuild", json={"source_id": source["id"]}).json()

    assert job["total"] == 1
    assert job["source_id"] == source["id"]


def test_rebuilding_a_named_set_of_documents_ignores_the_source_filter(
    client: TestClient,
) -> None:
    doc_ids = [d["id"] for d in client.get("/documents", params={"limit": 2}).json()["items"]]

    job = client.post("/index/rebuild", json={"doc_ids": doc_ids}).json()

    assert job["total"] == len(doc_ids)


def test_rebuilding_an_empty_index_still_produces_a_job(client: TestClient) -> None:
    client.post("/settings/wipe", json={"confirm": "DELETE"})

    job = client.post("/index/rebuild", json={"full": True}).json()

    # A zero total would render as a divide-by-zero progress bar.
    assert job["total"] == 1


both_backends = pytest.mark.parametrize("client", ["stub", "real"], indirect=True)


@both_backends
def test_a_renamed_document_keeps_its_name_in_answers_and_after_a_restart(
    client: TestClient, read_events
) -> None:
    with client.stream(
        "POST", "/query", json={"q": "How does reranking work?", "mode": "local"}
    ) as response:
        doc_id = dict(read_events(response))["sources"]["chunks"][0]["doc_id"]
    path = client.get(f"/documents/{doc_id}").json()["path"]

    renamed = client.patch(f"/documents/{doc_id}", json={"title": "  My notes  "})

    assert renamed.status_code == 200
    assert renamed.json()["title"] == "My notes"
    assert renamed.json()["path"] == path  # the file on disk is untouched
    with client.stream(
        "POST", "/query", json={"q": "How does reranking work?", "mode": "local"}
    ) as response:
        chunks = dict(read_events(response))["sources"]["chunks"]
    assert {c["doc_title"] for c in chunks if c["doc_id"] == doc_id} == {"My notes"}
    with TestClient(create_app(client.app.state.config)) as restarted:
        restarted.headers["Authorization"] = client.headers["Authorization"]
        assert restarted.get(f"/documents/{doc_id}").json()["title"] == "My notes"


def test_a_rename_needs_a_name_and_a_document(client: TestClient) -> None:
    doc_id = client.get("/documents").json()["items"][0]["id"]

    assert client.patch(f"/documents/{doc_id}", json={"title": "   "}).status_code == 422
    assert client.patch("/documents/doc_missing", json={"title": "x"}).status_code == 404
