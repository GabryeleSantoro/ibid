"""Contract tests for the reachability probe run before generation."""

from __future__ import annotations

import httpx

_LM_STUDIO = {
    "name": "LM Studio",
    "kind": "openai-compatible",
    "base_url": "http://localhost:1234/v1",
    "model_id": "qwen3-8b-instruct",
}


def test_probe_reports_ok_when_the_model_is_offered(client, monkeypatch) -> None:
    async def fake_get(self, url, headers=None):
        assert url == "http://localhost:1234/v1/models"
        return httpx.Response(200, json={"data": [{"id": "qwen3-8b-instruct"}]})

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    connection_id = client.post("/connections", json=_LM_STUDIO).json()["id"]

    response = client.post("/connections/test", json={"connection_id": connection_id})

    assert response.status_code == 200
    result = response.json()
    assert result == {
        "ok": True,
        "reachable": True,
        "model_found": True,
        "streaming": True,
        "tokens_per_second": None,
        "latency_ms": result["latency_ms"],
        "error": None,
    }


def test_probe_reports_unreachable_when_the_endpoint_refuses_the_connection(
    client, monkeypatch
) -> None:
    async def fake_get(self, url, headers=None):
        raise httpx.ConnectError("connection refused", request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    connection_id = client.post("/connections", json=_LM_STUDIO).json()["id"]

    response = client.post("/connections/test", json={"connection_id": connection_id})

    assert response.status_code == 200
    result = response.json()
    assert result["ok"] is False
    assert result["reachable"] is False
    assert "ConnectError" in result["error"]


def test_probe_rejects_an_unknown_connection_id(client) -> None:
    response = client.post("/connections/test", json={"connection_id": "missing"})

    assert response.status_code == 404


# ------------------------------------------------------------------- lifecycle

_ANTHROPIC = {"name": "Claude", "kind": "anthropic", "model_id": "claude-sonnet-4"}


def test_the_first_connection_becomes_the_active_one(client) -> None:
    created = client.post("/connections", json=_LM_STUDIO)

    assert created.status_code == 201
    assert created.json()["active"] is True
    assert client.get("/settings").json()["active_connection_id"] == created.json()["id"]


def test_a_second_connection_does_not_steal_the_active_slot(client) -> None:
    first = client.post("/connections", json=_LM_STUDIO).json()

    second = client.post("/connections", json=_ANTHROPIC).json()

    assert second["active"] is False
    assert client.get("/settings").json()["active_connection_id"] == first["id"]


def test_a_localhost_endpoint_is_not_treated_as_remote(client) -> None:
    assert client.post("/connections", json=_LM_STUDIO).json()["is_remote"] is False


def test_an_internet_endpoint_is_remote(client) -> None:
    connection = client.post(
        "/connections",
        json={
            "name": "OpenRouter",
            "kind": "openai-compatible",
            "base_url": "https://openrouter.ai/api/v1",
            "model_id": "google/gemma-3-27b-it",
        },
    ).json()

    assert connection["is_remote"] is True


def test_anthropic_is_always_remote(client) -> None:
    """There is no local Anthropic endpoint, whatever base URL is typed in."""
    connection = client.post(
        "/connections", json={**_ANTHROPIC, "base_url": "http://localhost:1234/v1"}
    ).json()

    assert connection["is_remote"] is True


def test_the_api_key_is_never_echoed_back(client) -> None:
    created = client.post("/connections", json={**_LM_STUDIO, "api_key": "sk-secret"}).json()

    assert "api_key" not in created
    assert created["has_api_key"] is True
    assert "sk-secret" not in client.get("/connections").text


def test_a_connection_without_a_key_says_so(client) -> None:
    assert client.post("/connections", json=_LM_STUDIO).json()["has_api_key"] is False


def test_editing_a_connection_updates_its_fields(client) -> None:
    connection = client.post("/connections", json=_LM_STUDIO).json()

    updated = client.patch(
        f"/connections/{connection['id']}",
        json={
            "name": "LM Studio (laptop)",
            "kind": "openai-compatible",
            "base_url": "https://example.test/v1",
            "model_id": "qwen3-14b",
            "max_output_tokens": 8192,
        },
    ).json()

    assert updated["name"] == "LM Studio (laptop)"
    assert updated["model_id"] == "qwen3-14b"
    assert updated["max_output_tokens"] == 8192
    # The URL moved off localhost, so the remote flag has to follow.
    assert updated["is_remote"] is True


def test_an_edit_that_omits_the_key_keeps_the_stored_one(client) -> None:
    connection = client.post("/connections", json={**_LM_STUDIO, "api_key": "sk-keep"}).json()

    updated = client.patch(f"/connections/{connection['id']}", json=_LM_STUDIO).json()

    assert updated["has_api_key"] is True


def test_an_empty_key_clears_the_stored_one(client) -> None:
    connection = client.post("/connections", json={**_LM_STUDIO, "api_key": "sk-drop"}).json()

    updated = client.patch(
        f"/connections/{connection['id']}", json={**_LM_STUDIO, "api_key": ""}
    ).json()

    assert updated["has_api_key"] is False


def test_editing_an_unknown_connection_is_a_404(client) -> None:
    assert client.patch("/connections/conn_nope", json=_LM_STUDIO).status_code == 404
    assert client.post("/connections/conn_nope/activate").status_code == 404
    assert client.delete("/connections/conn_nope").status_code == 404


def test_activating_a_connection_deactivates_the_others(client) -> None:
    first = client.post("/connections", json=_LM_STUDIO).json()
    second = client.post("/connections", json=_ANTHROPIC).json()

    activated = client.post(f"/connections/{second['id']}/activate").json()

    assert activated["active"] is True
    assert client.get("/settings").json()["active_connection_id"] == second["id"]
    states = {c["id"]: c["active"] for c in client.get("/connections").json() if c["id"] != "local"}
    assert states == {first["id"]: False, second["id"]: True}


def test_deleting_the_active_connection_promotes_another(client) -> None:
    first = client.post("/connections", json=_LM_STUDIO).json()
    second = client.post("/connections", json=_ANTHROPIC).json()

    client.delete(f"/connections/{first['id']}")

    assert client.get("/settings").json()["active_connection_id"] == second["id"]


def test_deleting_the_last_connection_leaves_nothing_active(client) -> None:
    connection = client.post("/connections", json=_LM_STUDIO).json()

    assert client.delete(f"/connections/{connection['id']}").json() == {"ok": True}

    assert [c for c in client.get("/connections").json() if c["id"] != "local"] == []
    assert client.get("/settings").json()["active_connection_id"] is None


# ------------------------------------------------------------------ more probes


def test_the_probe_reports_the_model_missing_when_the_endpoint_offers_others(
    client, monkeypatch
) -> None:
    async def fake_get(self, url, headers=None):
        return httpx.Response(200, json={"data": [{"id": "some-other-model"}]})

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    connection_id = client.post("/connections", json=_LM_STUDIO).json()["id"]

    result = client.post("/connections/test", json={"connection_id": connection_id}).json()

    assert result["reachable"] is True
    assert (result["ok"], result["model_found"]) == (False, False)
    assert "qwen3-8b-instruct" in result["error"]


def test_the_probe_reports_an_http_error_as_reachable_but_not_ok(client, monkeypatch) -> None:
    async def fake_get(self, url, headers=None):
        return httpx.Response(401, json={"error": "bad key"})

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    connection_id = client.post("/connections", json=_LM_STUDIO).json()["id"]

    result = client.post("/connections/test", json={"connection_id": connection_id}).json()

    assert (result["ok"], result["reachable"]) == (False, True)
    assert result["error"] == "HTTP 401"


def test_an_openai_connection_without_a_base_url_fails_before_any_request(client) -> None:
    result = client.post(
        "/connections/test", json={"kind": "openai-compatible", "model_id": "qwen3-8b-instruct"}
    ).json()

    assert result["ok"] is False
    assert result["error"] == "No base URL set"
    assert result["latency_ms"] is None


def test_probing_anthropic_uses_its_own_endpoint_and_header(client, monkeypatch) -> None:
    seen: dict = {}

    async def fake_get(self, url, headers=None):
        seen.update(url=url, headers=headers or {})
        return httpx.Response(200, json={"data": [{"id": "claude-sonnet-4"}]})

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    connection_id = client.post(
        "/connections", json={**_ANTHROPIC, "api_key": "sk-ant-test"}
    ).json()["id"]

    result = client.post("/connections/test", json={"connection_id": connection_id}).json()

    assert result["ok"] is True
    assert seen["url"] == "https://api.anthropic.com/v1/models"
    assert seen["headers"]["anthropic-version"] == "2023-06-01"
    # Anthropic authenticates with x-api-key, not a bearer token.
    assert seen["headers"]["x-api-key"] == "sk-ant-test"
    assert "Authorization" not in seen["headers"]


def test_probing_a_saved_connection_reuses_its_stored_key(client, monkeypatch) -> None:
    seen: dict = {}

    async def fake_get(self, url, headers=None):
        seen.update(headers or {})
        return httpx.Response(200, json={"data": [{"id": "qwen3-8b-instruct"}]})

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    connection_id = client.post(
        "/connections", json={**_LM_STUDIO, "api_key": "sk-stored"}
    ).json()["id"]

    client.post("/connections/test", json={"connection_id": connection_id})

    assert seen["Authorization"] == "Bearer sk-stored"


def test_a_connection_can_be_probed_before_it_is_saved(client, monkeypatch) -> None:
    """The dialog tests the typed-in values, which have no id yet."""
    async def fake_get(self, url, headers=None):
        assert url == "https://example.test/v1/models"
        return httpx.Response(200, json={"data": [{"id": "qwen3-14b"}]})

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)

    result = client.post(
        "/connections/test",
        json={
            "kind": "openai-compatible",
            "base_url": "https://example.test/v1/",
            "model_id": "qwen3-14b",
        },
    ).json()

    assert result["ok"] is True
    assert [
        c for c in client.get("/connections").json() if c["id"] != "local"
    ] == [], "probing must not save anything"


def test_the_probe_matches_a_model_id_by_substring(client, monkeypatch) -> None:
    """Servers prefix ids with the repo, so an exact match alone is too strict."""
    async def fake_get(self, url, headers=None):
        return httpx.Response(200, json={"data": [{"id": "lmstudio/qwen3-8b-instruct-gguf"}]})

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)

    result = client.post(
        "/connections/test",
        json={
            "kind": "openai-compatible",
            "base_url": "http://localhost:1234/v1",
            "model_id": "qwen3-8b-instruct",
        },
    ).json()

    assert result["model_found"] is True


def test_a_body_that_is_not_a_model_list_does_not_crash_the_probe(client, monkeypatch) -> None:
    async def fake_get(self, url, headers=None):
        return httpx.Response(200, text="<html>login</html>")

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)

    result = client.post(
        "/connections/test",
        json={
            "kind": "openai-compatible",
            "base_url": "http://localhost:1234/v1",
            "model_id": "qwen3-8b-instruct",
        },
    ).json()

    assert (result["ok"], result["reachable"]) == (False, True)


def test_connections_survive_a_restart_but_their_keys_do_not(client, tmp_path) -> None:
    from fastapi.testclient import TestClient
    from ragcore.api.app import create_app

    created = client.post("/connections", json={**_LM_STUDIO, "api_key": "sk-1"}).json()

    with TestClient(create_app(client.app.state.config)) as restarted:
        restarted.headers["Authorization"] = client.headers["Authorization"]
        [reloaded] = [c for c in restarted.get("/connections").json() if c["id"] != "local"]
        store = restarted.app.state.store

        assert reloaded == created
        assert store.active_connection().id == created["id"]
        assert created["id"] not in store.secrets
        assert "sk-1" not in (tmp_path / "connections.json").read_text()

        response = restarted.put(f"/connections/{created['id']}/secret", json={"api_key": "sk-1"})

        assert response.status_code == 200
        assert store.secrets[created["id"]] == "sk-1"
