"""Settings and the wipe that sends the user back through onboarding."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_defaults_are_what_the_ui_renders_on_first_run(client: TestClient) -> None:
    settings = client.get("/settings").json()

    assert settings["telemetry"] is False
    assert settings["active_connection_id"] is None
    assert settings["retrieval"]["top_k"] == 6
    assert settings["performance"]["profile"] == "balanced"


def test_a_patch_changes_only_the_fields_it_names(client: TestClient) -> None:
    before = client.get("/settings").json()

    patched = client.patch("/settings", json={"telemetry": True}).json()

    assert patched["telemetry"] is True
    assert patched["storage_path"] == before["storage_path"]
    assert patched["retrieval"] == before["retrieval"]


def test_retrieval_settings_can_be_replaced_wholesale(client: TestClient) -> None:
    retrieval = client.get("/settings").json()["retrieval"] | {"top_k": 12, "min_score": 0.1}

    patched = client.patch("/settings", json={"retrieval": retrieval}).json()

    assert patched["retrieval"]["top_k"] == 12
    assert patched["retrieval"]["min_score"] == 0.1


def test_changed_retrieval_settings_reach_the_next_query(
    client: TestClient, read_events
) -> None:
    retrieval = client.get("/settings").json()["retrieval"] | {"top_k": 1, "min_score": 0.0}
    client.patch("/settings", json={"retrieval": retrieval})

    with client.stream("POST", "/query", json={"q": "How does reranking work?"}) as response:
        sources = dict(read_events(response))["sources"]

    assert len(sources["chunks"]) <= 1


def test_a_patch_persists_across_requests(client: TestClient) -> None:
    client.patch("/settings", json={"telemetry": True})

    assert client.get("/settings").json()["telemetry"] is True


def test_wipe_clears_the_library_and_the_chat_history(client: TestClient) -> None:
    with client.stream("POST", "/query", json={"q": "Why rerank?"}) as response:
        list(response.iter_lines())
    assert client.get("/chats").json()

    client.post("/settings/wipe", json={"confirm": "DELETE"})

    assert client.get("/documents").json()["total"] == 0
    assert client.get("/sources").json() == []
    assert client.get("/chats").json() == []


def test_wipe_can_drop_the_connections_too(client: TestClient) -> None:
    client.post(
        "/connections",
        json={"name": "LM Studio", "kind": "openai-compatible", "model_id": "qwen3-8b-instruct"},
    )

    client.post("/settings/wipe", json={"confirm": "DELETE", "keep_connections": False})

    assert [c for c in client.get("/connections").json() if c["id"] != "local"] == []
    assert client.get("/settings").json()["active_connection_id"] is None


def test_wipe_keeps_connections_by_default(client: TestClient) -> None:
    client.post(
        "/connections",
        json={"name": "LM Studio", "kind": "openai-compatible", "model_id": "qwen3-8b-instruct"},
    )

    client.post("/settings/wipe", json={"confirm": "DELETE"})

    assert [c for c in client.get("/connections").json() if c["id"] != "local"]


def test_a_wiped_index_answers_that_it_knows_nothing(client: TestClient, read_events) -> None:
    client.post("/settings/wipe", json={"confirm": "DELETE"})

    with client.stream("POST", "/query", json={"q": "Why rerank?"}) as response:
        events = dict(read_events(response))

    assert events["sources"]["chunks"] == []
    assert events["citations"]["grounding"] == "none"


def test_an_empty_confirmation_is_refused(client: TestClient) -> None:
    assert client.post("/settings/wipe", json={"confirm": ""}).status_code == 400


def test_extra_instructions_default_empty_and_round_trip(client: TestClient) -> None:
    assert client.get("/settings").json()["chat_extra_instructions"] == ""
    patched = client.patch("/settings", json={"chat_extra_instructions": "Be brief"}).json()
    assert patched["chat_extra_instructions"] == "Be brief"
    assert client.get("/settings").json()["chat_extra_instructions"] == "Be brief"


def test_over_long_extra_instructions_are_rejected(client: TestClient) -> None:
    response = client.patch("/settings", json={"chat_extra_instructions": "x" * 1001})
    assert response.status_code == 422
    assert client.get("/settings").json()["chat_extra_instructions"] == ""


def test_settings_saved_before_the_field_existed_still_load() -> None:
    from ragcore.api.schemas import AppSettings

    old = AppSettings(storage_path="/tmp/x").model_dump_json(exclude={"chat_extra_instructions"})
    assert AppSettings.model_validate_json(old).chat_extra_instructions == ""
