"""The query route: routing, scoping, session handling and cancellation."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from ragcore.api.routes.query import project_filters, route_mode, tokens_per_second
from ragcore.api.schemas import QueryFilters

# ----------------------------------------------------------------- mode router


@pytest.mark.parametrize(
    "question",
    [
        "What are the main themes across the corpus?",
        "Summarize all documents",
        "Compare the two approaches",
        "What topics does this library cover?",
    ],
)
def test_a_corpus_wide_question_routes_to_global(question: str) -> None:
    mode, reason = route_mode(question)

    assert mode == "global"
    assert "asks about the corpus" in reason


def test_a_passage_question_routes_to_local() -> None:
    mode, reason = route_mode("How does reciprocal rank fusion combine two rankings?")

    assert mode == "local"
    assert reason


def test_the_router_ignores_case() -> None:
    assert route_mode("SUMMARIZE ALL the notes")[0] == "global"


def test_the_route_reports_a_manual_mode_as_manual(client: TestClient, read_events) -> None:
    with client.stream(
        "POST", "/query", json={"q": "How does reranking work?", "mode": "global"}
    ) as response:
        mode = dict(read_events(response))["mode"]

    assert mode["mode"] == "global"
    assert mode["reason"] == "Set manually"


def test_auto_is_the_default_mode(client: TestClient, read_events) -> None:
    with client.stream("POST", "/query", json={"q": "main themes across the notes"}) as response:
        mode = dict(read_events(response))["mode"]

    assert mode["mode"] == "global"
    assert mode["reason"] != "Set manually"


# --------------------------------------------------------------- project scope


def test_a_chat_outside_a_project_sees_only_global_folders(
    client: TestClient, tmp_path: Path
) -> None:
    project = client.post("/chats/projects", json={"name": "Research"}).json()
    private = tmp_path / "private"
    private.mkdir()
    (private / "brief.md").write_text("Project-only material.")
    project_source = client.post(
        "/sources",
        json={"path": str(private), "include_globs": ["**/*.md"], "project_id": project["id"]},
    ).json()
    session = client.post("/chats", json={}).json()
    store = client.app.state.store

    scoped = project_filters(store, session["id"], QueryFilters())

    assert project_source["id"] not in scoped.source_ids
    assert scoped.source_ids, "global folders are still in scope"


def test_a_project_chat_sees_its_own_folders_plus_global_ones(
    client: TestClient, tmp_path: Path
) -> None:
    project = client.post("/chats/projects", json={"name": "Research"}).json()
    private = tmp_path / "private"
    private.mkdir()
    (private / "brief.md").write_text("Project-only material.")
    project_source = client.post(
        "/sources",
        json={"path": str(private), "include_globs": ["**/*.md"], "project_id": project["id"]},
    ).json()
    session = client.post("/chats", json={"project_id": project["id"]}).json()
    store = client.app.state.store

    scoped = project_filters(store, session["id"], QueryFilters())

    assert project_source["id"] in scoped.source_ids
    assert len(scoped.source_ids) == 2


def test_a_requested_source_outside_the_scope_is_dropped(
    client: TestClient, tmp_path: Path
) -> None:
    """A filter from the UI narrows the scope; it can never widen it."""
    project = client.post("/chats/projects", json={"name": "Research"}).json()
    private = tmp_path / "private"
    private.mkdir()
    (private / "brief.md").write_text("Project-only material.")
    project_source = client.post(
        "/sources",
        json={"path": str(private), "include_globs": ["**/*.md"], "project_id": project["id"]},
    ).json()
    session = client.post("/chats", json={}).json()
    store = client.app.state.store

    scoped = project_filters(
        store, session["id"], QueryFilters(source_ids=[project_source["id"]])
    )

    assert scoped.source_ids == []
    chunks, _, _ = store.retriever.search(
        "project-only material",
        settings=store.settings.retrieval,
        filters=scoped,
        doc_meta=store.doc_meta(),
    )
    assert chunks == []


def test_an_unknown_session_falls_back_to_global_scope(client: TestClient) -> None:
    store = client.app.state.store

    scoped = project_filters(store, "chat_nope", QueryFilters())

    assert scoped.source_ids == sorted(store.sources)


def test_other_filters_survive_the_scoping(client: TestClient) -> None:
    session = client.post("/chats", json={}).json()
    store = client.app.state.store

    scoped = project_filters(store, session["id"], QueryFilters(exts=[".md"], langs=["en"]))

    assert scoped.exts == [".md"]
    assert scoped.langs == ["en"]


# -------------------------------------------------------------------- sessions


def test_a_query_without_a_session_opens_one(client: TestClient, read_events) -> None:
    with client.stream("POST", "/query", json={"q": "Why rerank?"}) as response:
        start = dict(read_events(response))["start"]

    assert client.get(f"/chats/{start['session_id']}").status_code == 200


def test_a_query_with_a_session_id_appends_to_it(client: TestClient, read_events) -> None:
    session = client.post("/chats", json={"title": "Existing"}).json()

    with client.stream(
        "POST", "/query", json={"q": "Why rerank?", "session_id": session["id"]}
    ) as response:
        start = dict(read_events(response))["start"]

    assert start["session_id"] == session["id"]
    assert client.get("/chats").json() == [client.get(f"/chats/{session['id']}").json()]


def test_a_query_naming_a_session_that_is_gone_opens_a_fresh_one(
    client: TestClient, read_events
) -> None:
    with client.stream(
        "POST", "/query", json={"q": "Why rerank?", "session_id": "chat_deleted"}
    ) as response:
        start = dict(read_events(response))["start"]

    assert start["session_id"] != "chat_deleted"
    assert client.get(f"/chats/{start['session_id']}").status_code == 200


def test_the_client_can_choose_the_query_id(client: TestClient, read_events) -> None:
    """The UI mints it up front so cancel works before the first frame."""
    with client.stream(
        "POST", "/query", json={"q": "Why rerank?", "query_id": "q_chosen"}
    ) as response:
        start = dict(read_events(response))["start"]

    assert start["query_id"] == "q_chosen"


def test_the_done_frame_reports_no_connection_while_the_stub_answers(
    client: TestClient, read_events
) -> None:
    with client.stream("POST", "/query", json={"q": "Why rerank?"}) as response:
        done = dict(read_events(response))["done"]

    assert done["connection_id"] is None
    assert done["remote"] is False
    assert done["latency"]["total_ms"] >= done["latency"]["llm_first_token_ms"] > 0


# ---------------------------------------------------------------- cancellation


def test_cancelling_before_the_stream_starts_cuts_it_short(
    client: TestClient, read_events
) -> None:
    assert client.post("/query/q_early/cancel").json() == {"ok": True}

    with client.stream(
        "POST", "/query", json={"q": "!slow Why rerank?", "query_id": "q_early"}
    ) as response:
        names = [name for name, _ in read_events(response)]

    # Cancellation stops the tokens, but the stream still closes cleanly so the
    # partial answer is saved rather than lost.
    assert names.count("token") == 0
    assert names[-1] == "done"


def test_cancelling_an_id_that_never_ran_is_still_ok(client: TestClient) -> None:
    assert client.post("/query/q_ghost/cancel").status_code == 200


def test_a_cancelled_id_does_not_poison_the_next_query_that_reuses_it(
    client: TestClient, read_events
) -> None:
    client.post("/query/q_reused/cancel")
    with client.stream(
        "POST", "/query", json={"q": "Why rerank?", "query_id": "q_reused"}
    ) as response:
        list(read_events(response))

    with client.stream(
        "POST", "/query", json={"q": "Why rerank?", "query_id": "q_reused"}
    ) as response:
        names = [name for name, _ in read_events(response)]

    assert names.count("token") > 1


# ----------------------------------------------------------------- persistence


def test_the_assistant_turn_keeps_its_chunks_and_grounding(
    client: TestClient, read_events
) -> None:
    with client.stream("POST", "/query", json={"q": "How does reranking work?"}) as response:
        events = dict(read_events(response))

    session_id = events["start"]["session_id"]
    [_, assistant] = client.get(f"/chats/{session_id}/messages").json()

    assert assistant["id"] == events["done"]["message_id"]
    assert assistant["grounding"] == events["citations"]["grounding"]
    assert len(assistant["chunks"]) == len(events["sources"]["chunks"])


def test_a_directive_is_stripped_from_the_stored_question(
    client: TestClient, read_events
) -> None:
    with client.stream("POST", "/query", json={"q": "!nocite Why rerank?"}) as response:
        session_id = dict(read_events(response))["start"]["session_id"]

    [user, _] = client.get(f"/chats/{session_id}/messages").json()

    assert user["text"] == "Why rerank?"


def test_a_failed_generation_leaves_no_assistant_turn(
    client: TestClient, read_events
) -> None:
    with client.stream("POST", "/query", json={"q": "!error Why rerank?"}) as response:
        events = dict(read_events(response))

    messages = client.get(f"/chats/{events['start']['session_id']}/messages").json()

    assert [m["role"] for m in messages] == ["user"]
    assert events["error"]["retryable"] is True


def test_a_failed_generation_carries_a_code_the_ui_can_translate(
    client: TestClient, read_events
) -> None:
    with client.stream("POST", "/query", json={"q": "!error Why rerank?"}) as response:
        events = dict(read_events(response))

    assert events["error"]["code"] == "generation_failed"
    assert "Generation failed" in events["error"]["message"]


def test_the_query_language_reaches_the_answer_engine(client: TestClient, read_events) -> None:
    from ragcore.api import deps

    captured: dict = {}

    class Recorder:
        async def stream(
            self, question, chunks, directives, *, system_prompt=None, max_tokens=None
        ):
            captured["system_prompt"] = system_prompt
            yield "ok"

    client.app.dependency_overrides[deps.get_answerer] = Recorder
    try:
        with client.stream("POST", "/query", json={"q": "Why rerank?", "lang": "fr"}) as response:
            read_events(response)
    finally:
        client.app.dependency_overrides.pop(deps.get_answerer, None)

    assert captured["system_prompt"].endswith(
        "Write the answer in French. Keep the [document_id:page] markers unchanged."
    )


def test_a_query_without_a_language_keeps_the_default_prompt(
    client: TestClient, read_events
) -> None:
    from ragcore.api import deps
    from ragcore.llm import SYSTEM_PROMPT

    captured: dict = {}

    class Recorder:
        async def stream(
            self, question, chunks, directives, *, system_prompt=None, max_tokens=None
        ):
            captured["system_prompt"] = system_prompt
            yield "ok"

    client.app.dependency_overrides[deps.get_answerer] = Recorder
    try:
        with client.stream("POST", "/query", json={"q": "Why rerank?"}) as response:
            read_events(response)
    finally:
        client.app.dependency_overrides.pop(deps.get_answerer, None)

    assert captured["system_prompt"] == SYSTEM_PROMPT


def test_speed_counts_from_the_first_piece() -> None:
    assert tokens_per_second(101, 5.0, 15.0) == 10.0


def test_short_answers_have_no_speed() -> None:
    assert tokens_per_second(29, 0.0, 10.0) is None


def test_an_answer_that_arrived_at_once_has_no_speed() -> None:
    assert tokens_per_second(100, 3.0, 3.0) is None
    assert tokens_per_second(0, None, None) is None


def test_stub_answers_report_no_speed(client: TestClient, read_events) -> None:
    with client.stream("POST", "/query", json={"q": "How does reranking work?"}) as response:
        done = dict(read_events(response))["done"]

    assert done["tokens_per_s"] is None


def test_extra_instructions_reach_the_answer_engine(client: TestClient, read_events) -> None:
    from ragcore.api import deps

    captured: dict = {}

    class Recorder:
        async def stream(
            self, question, chunks, directives, *, system_prompt=None, max_tokens=None
        ):
            captured["system_prompt"] = system_prompt
            yield "ok"

    client.patch("/settings", json={"chat_extra_instructions": "Answer in bullets"})
    client.app.dependency_overrides[deps.get_answerer] = Recorder
    try:
        with client.stream("POST", "/query", json={"q": "Why rerank?"}) as response:
            read_events(response)
    finally:
        client.app.dependency_overrides.pop(deps.get_answerer, None)

    assert captured["system_prompt"].endswith("Answer in bullets")
