"""The built-in model shows up as a connection the user can activate once it is installed."""

from __future__ import annotations

_REMOTE = {
    "name": "OpenRouter",
    "kind": "openai-compatible",
    "base_url": "https://openrouter.ai/api/v1",
    "model_id": "x/y",
}


class _FakeLocal:
    def __init__(self, state: str) -> None:
        self._state = state

    def status(self) -> str:
        return self._state

    def stop(self) -> None:
        pass


def test_local_is_listed_first_and_inactive(client) -> None:
    first = client.get("/connections").json()[0]

    assert first["id"] == "local"
    assert first["kind"] == "local"
    assert first["is_remote"] is False
    assert first["has_api_key"] is False
    assert first["active"] is False


def test_activation_is_refused_until_the_model_is_installed(client) -> None:
    client.app.state.local_llm = _FakeLocal("missing")

    response = client.post("/connections/local/activate")

    assert response.status_code == 409
    assert response.json()["code"] == "local_model_missing"
    assert client.get("/connections").json()[0]["active"] is False


def test_an_installed_model_can_be_activated_and_answers(client) -> None:
    client.app.state.local_llm = _FakeLocal("ready")

    response = client.post("/connections/local/activate")

    assert response.status_code == 200
    assert client.get("/connections").json()[0]["active"] is True
    assert client.app.state.store.active_connection().kind == "local"


def test_it_cannot_be_edited_or_deleted(client) -> None:
    assert client.delete("/connections/local").status_code == 404
    body = {**_REMOTE, "name": "x"}
    assert client.patch("/connections/local", json=body).status_code == 404


def test_the_first_remote_connection_does_not_steal_active_from_local(client) -> None:
    client.app.state.local_llm = _FakeLocal("ready")
    client.post("/connections/local/activate")

    created = client.post("/connections", json=_REMOTE).json()

    assert created["active"] is False
    assert client.get("/connections").json()[0]["active"] is True


def test_activating_a_remote_connection_deactivates_local(client) -> None:
    client.app.state.local_llm = _FakeLocal("ready")
    client.post("/connections/local/activate")
    remote = client.post("/connections", json=_REMOTE).json()

    client.post(f"/connections/{remote['id']}/activate")

    listed = client.get("/connections").json()
    assert listed[0]["active"] is False
    assert [c["active"] for c in listed[1:]] == [True]


def test_slide_conversion_says_the_built_in_model_cannot_do_it(client) -> None:
    client.app.state.local_llm = _FakeLocal("ready")
    client.post("/connections/local/activate")

    response = client.post("/conversions/slides", json={"slide_ids": ["x"]})

    assert response.status_code == 409
    assert response.json()["code"] == "local_model_unsupported_for_slides"
