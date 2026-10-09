from __future__ import annotations


def test_suggestions_return_the_topics_so_the_ui_can_phrase_them_in_its_language(client) -> None:
    doc_id = client.get("/documents", params={"limit": 1}).json()["items"][0]["id"]

    body = client.post("/suggestions", json={"doc_ids": [doc_id]}).json()

    assert body["topics"]
    assert len(body["topics"]) <= 3
    pairs = zip(body["topics"], body["questions"], strict=True)
    assert all(topic in question for topic, question in pairs)


def test_no_documents_means_no_suggestions(client) -> None:
    body = client.post("/suggestions", json={"doc_ids": []}).json()

    assert body["questions"] == [] and body["topics"] == []


def test_local_model_questions_replace_the_templates_when_installed(client) -> None:
    doc_id = client.get("/documents", params={"limit": 1}).json()["items"][0]["id"]

    class Fake:
        def status(self) -> str:
            return "ready"

        def stop(self) -> None:
            pass

        async def generate(self, system: str, user: str, max_tokens: int = 200) -> str:
            return "1. skipped line\nWhat is A?\n- What is B?\nWhat is C?"

    client.app.state.local_llm = Fake()
    body = client.post("/suggestions", json={"doc_ids": [doc_id]}).json()

    assert body["source"] == "model"
    assert body["questions"] == ["What is A?", "What is B?", "What is C?"]


def test_install_routes_start_downloads_inside_the_event_loop(tmp_path, monkeypatch):
    """Sync routes run in a thread with no loop; create_task there 500s."""
    from fastapi.testclient import TestClient
    from ragcore.api.app import create_app
    from ragcore.config import Config
    from ragcore.local_llm import LocalLLM

    async def noop(self):
        pass

    monkeypatch.setattr(LocalLLM, "_install", noop)
    config = Config(token="t", data_dir=tmp_path)
    with TestClient(create_app(config)) as client:
        response = client.post("/suggestions/model", headers={"Authorization": "Bearer t"})
    assert response.status_code == 200
