"""The topic map over HTTP."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

real = pytest.mark.parametrize("client", ["real"], indirect=True)


def test_the_stub_has_no_topic_map(client: TestClient) -> None:
    assert client.get("/brain/topics").json() == {
        "status": "none",
        "built_at": None,
        "changed_since": 0,
        "topics": [],
    }


@real
def test_topics_come_flat_with_parents_first(client: TestClient, wait_for_topics) -> None:
    wait_for_topics(client)

    tree = client.get("/brain/topics").json()

    seen: set[str] = set()
    for topic in tree["topics"]:
        assert topic["parent_id"] is None or topic["parent_id"] in seen
        seen.add(topic["id"])
    docs = {d["id"] for d in client.get("/documents").json()["items"]}
    top = [t for t in tree["topics"] if t["level"] == 0]
    assert sorted(d for t in top for d in t["doc_ids"]) == sorted(docs)
    assert client.get("/health").json()["index"]["topics"] == len(tree["topics"])


@real
def test_recluster_returns_a_job_that_finishes(client: TestClient, wait_for_topics) -> None:
    wait_for_topics(client)

    job = client.post("/brain/recluster").json()

    assert job["kind"] == "recluster"
    deadline = time.monotonic() + 15
    while (state := client.get(f"/jobs/{job['id']}").json()["state"]) in ("queued", "running"):
        assert time.monotonic() < deadline
        time.sleep(0.02)
    assert state == "done"


def test_recluster_is_refused_while_the_index_is_blocked(client: TestClient) -> None:
    client.app.state.store.index_blocked = "built with another embedder"

    response = client.post("/brain/recluster")

    assert response.status_code == 409
    assert response.json()["code"] == "index_blocked"
