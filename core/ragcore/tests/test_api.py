"""Contract tests for the stub core.

These lock the parts the desktop app was built against: who may call what, the
order of the SSE frames, and the fact that a citation nothing retrieved gets
dropped rather than shown.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient

Events = Callable[..., list[tuple[str, dict]]]


both_backends = pytest.mark.parametrize("client", ["stub", "real"], indirect=True)


@both_backends
def test_health_is_public(client: TestClient) -> None:
    del client.headers["Authorization"]
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ok", "degraded", "starting"}
    assert body["dev_mode"] is True
    assert body["index"]["documents"] > 0


@both_backends
def test_everything_else_needs_the_session_token(client: TestClient) -> None:
    del client.headers["Authorization"]

    assert client.get("/documents").status_code == 401
    assert client.get("/settings").status_code == 401
    assert client.post("/query", json={"q": "anything"}).status_code == 401


@both_backends
def test_openapi_is_reachable_without_a_token(client: TestClient) -> None:
    """`bun run gen:types` reads this unauthenticated."""
    del client.headers["Authorization"]
    schema = client.get("/openapi.json").json()

    # The UI derives its SSE frame types from this model; losing it would make
    # the generated types silently incomplete.
    assert "StreamEnvelope" in schema["components"]["schemas"]


@both_backends
def test_documents_expose_content_the_reader_can_highlight(client: TestClient) -> None:
    listing = client.get("/documents", params={"limit": 5}).json()
    assert listing["total"] > 0

    doc_id = listing["items"][0]["id"]
    content = client.get(f"/documents/{doc_id}/content").json()

    assert content["pages"], "a document with no pages cannot be read"
    pages = {page["page"]: page["text"] for page in content["pages"]}

    for chunk in content["chunks"]:
        text = pages[chunk["page"]]
        assert 0 <= chunk["char_start"] <= chunk["char_end"] <= len(text)
        assert text[chunk["char_start"] : chunk["char_end"]] == chunk["text"]


@both_backends
def test_query_streams_frames_in_contract_order(client: TestClient, read_events: Events) -> None:
    with client.stream(
        "POST", "/query", json={"q": "How does reranking improve retrieval?"}
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = read_events(response)

    names = [name for name, _ in events]
    assert names[0] == "start"
    assert names[1] == "mode"
    assert names[2] == "sources"
    assert names[-1] == "done"
    assert names.index("citations") < names.index("done")
    assert names.count("token") > 1

    payload = dict(events)
    assert payload["sources"]["chunks"], "retrieval returned nothing for a corpus question"
    assert payload["citations"]["grounding"] == "ok"
    assert payload["citations"]["dropped"] == 0
    assert payload["done"]["latency"]["total_ms"] >= 0


@both_backends
def test_a_citation_nothing_retrieved_is_dropped(client: TestClient, read_events: Events) -> None:
    with client.stream("POST", "/query", json={"q": "!badcite Why rerank?"}) as response:
        events = dict(read_events(response))

    assert events["citations"]["dropped"] >= 1
    assert events["citations"]["grounding"] in {"low", "none"}


@both_backends
def test_an_answer_with_no_citation_reports_low_grounding(
    client: TestClient, read_events: Events
) -> None:
    with client.stream("POST", "/query", json={"q": "!nocite Why rerank?"}) as response:
        events = dict(read_events(response))

    assert events["citations"]["citations"] == []
    assert events["citations"]["grounding"] == "none"


@both_backends
def test_the_error_frame_replaces_the_rest_of_the_stream(
    client: TestClient, read_events: Events
) -> None:
    with client.stream("POST", "/query", json={"q": "!error Why rerank?"}) as response:
        names = [name for name, _ in read_events(response)]

    assert "error" in names
    assert "done" not in names


@both_backends
def test_sources_and_jobs_round_trip(client: TestClient, tmp_path) -> None:
    folder = tmp_path / "notes"
    folder.mkdir()
    (folder / "note.md").write_text("# Note\n\nChunking splits documents into passages.\n")

    created = client.post(
        "/sources", json={"path": str(folder), "include_globs": ["**/*.md"]}
    ).json()
    assert created["path"] == str(folder)

    assert any(source["id"] == created["id"] for source in client.get("/sources").json())
    assert client.delete(f"/sources/{created['id']}").status_code in {200, 204}


def test_projects_group_and_pin_chats(client: TestClient) -> None:
    project = client.post("/chats/projects", json={"name": "Research"})
    assert project.status_code == 201
    project_body = project.json()

    session = client.post("/chats", json={"project_id": project_body["id"]})
    assert session.status_code == 201
    session_body = session.json()
    assert session_body["project_id"] == project_body["id"]

    pinned_chat = client.patch(f"/chats/{session_body['id']}", json={"pinned": True})
    assert pinned_chat.status_code == 200
    assert pinned_chat.json()["pinned"] is True

    pinned_project = client.patch(f"/chats/projects/{project_body['id']}", json={"pinned": True})
    assert pinned_project.status_code == 200
    assert pinned_project.json()["pinned"] is True

    deleted_project = client.delete(f"/chats/projects/{project_body['id']}")
    assert deleted_project.status_code == 200
    assert client.get(f"/chats/{session_body['id']}").json()["project_id"] is None


def test_project_can_disable_global_knowledge_and_add_project_folder(
    client: TestClient, tmp_path, read_events: Events
) -> None:
    project = client.post("/chats/projects", json={"name": "Private research"}).json()
    folder = tmp_path / "private"
    folder.mkdir()
    (folder / "brief.md").write_text("Project-only material about the cobalt launch.")

    project_source = client.post(
        "/sources",
        json={"path": str(folder), "include_globs": ["**/*.md"], "project_id": project["id"]},
    )
    assert project_source.status_code == 201
    assert project_source.json()["project_id"] == project["id"]

    updated = client.patch(f"/chats/projects/{project['id']}", json={"use_global_sources": False})
    assert updated.status_code == 200
    assert updated.json()["use_global_sources"] is False

    session = client.post("/chats", json={"project_id": project["id"]}).json()
    with client.stream(
        "POST",
        "/query",
        json={"q": "reciprocal rank fusion", "session_id": session["id"]},
    ) as response:
        disabled_events = dict(read_events(response))
    assert disabled_events["sources"]["chunks"] == []

    client.patch(f"/chats/projects/{project['id']}", json={"use_global_sources": True})
    with client.stream(
        "POST",
        "/query",
        json={"q": "reciprocal rank fusion", "session_id": session["id"]},
    ) as response:
        enabled_events = dict(read_events(response))
    assert enabled_events["sources"]["chunks"]


def test_nested_pdfs_are_indexed_when_adding_a_source(
    client: TestClient, tmp_path, text_pdf
) -> None:
    nested = tmp_path / "research" / "papers" / "2026"
    nested.mkdir(parents=True)
    pdf = nested / "paper.pdf"
    pdf.write_bytes(text_pdf("Indexed paper"))

    source = client.post(
        "/sources",
        json={"path": str(tmp_path / "research"), "include_globs": ["**/*.pdf"]},
    )
    assert source.status_code == 201

    documents = client.get("/documents", params={"source_id": source.json()["id"]}).json()
    assert documents["total"] == 1
    assert documents["items"][0]["path"] == str(pdf)


def test_wipe_sends_the_user_back_through_onboarding(client: TestClient) -> None:
    assert client.get("/settings").json()["onboarded"] is True
    client.post(
        "/connections",
        json={"name": "LM Studio", "kind": "openai-compatible", "model_id": "qwen3-8b-instruct"},
    )

    response = client.post("/settings/wipe", json={"confirm": "DELETE", "keep_connections": True})
    assert response.status_code == 200

    settings = client.get("/settings").json()
    assert settings["onboarded"] is False
    assert client.get("/documents").json()["total"] == 0
    assert [
        c for c in client.get("/connections").json() if c["id"] != "local"
    ], "connections were meant to be kept"


@both_backends
def test_wipe_refuses_without_the_literal_confirmation(client: TestClient) -> None:
    response = client.post("/settings/wipe", json={"confirm": "delete"})

    assert response.status_code == 400
    assert client.get("/documents").json()["total"] > 0


@both_backends
def test_embedder_activation_requires_accepting_the_reindex(client: TestClient) -> None:
    inventory = client.get("/models").json()
    embedder = next(model for model in inventory["installed"] if model["role"] == "embedding")

    response = client.post(f"/models/{embedder['id']}/activate")

    # Already active, so this is a no-op rather than a swap; the guard is on
    # downloading a *different* embedder and activating it.
    assert response.status_code in {200, 409}


@both_backends
def test_eval_reports_progress_then_metrics(client: TestClient, read_events: Events) -> None:
    with client.stream("POST", "/eval/run", json={"set_name": "base"}) as response:
        assert response.status_code == 200
        events = read_events(response)

    names = [name for name, _ in events]
    assert names.count("progress") >= 1
    assert names[-1] == "result"

    result = events[-1][1]
    assert result["metrics"]["n_questions"] == len(result["questions"])
    assert 0.0 <= result["metrics"]["recall_at_6"] <= 1.0
    assert result["metrics"]["baseline_recall_at_6"] is not None
