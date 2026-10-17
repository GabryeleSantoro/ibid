"""Golden-set evaluation: dev-only, and honest about what it measured."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_the_committed_golden_sets_are_listed(client: TestClient) -> None:
    names = {s["name"]: s for s in client.get("/eval/sets").json()}

    assert names["base"]["n_questions"] == 20


def test_a_private_set_in_the_data_dir_is_listed_too(client: TestClient, tmp_path) -> None:
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "mine.jsonl").write_text('{"q": "x", "expected": ["a.md"]}\n')

    names = {s["name"] for s in client.get("/eval/sets").json()}

    assert {"base", "mine"} <= names


def test_an_unknown_set_is_a_404(client: TestClient) -> None:
    response = client.post("/eval/run", json={"set_name": "nonexistent"})

    assert response.status_code == 404
    assert "nonexistent" in response.json()["detail"]


def test_eval_is_refused_in_a_production_build(client: TestClient) -> None:
    client.app.state.config.dev_mode = False

    response = client.post("/eval/run", json={"set_name": "base"})

    assert response.status_code == 403


def test_progress_counts_up_to_the_question_total(client: TestClient, read_events) -> None:
    with client.stream("POST", "/eval/run", json={"set_name": "base"}) as response:
        events = read_events(response)

    progress = [payload for name, payload in events if name == "progress"]
    total = progress[0]["total"]
    assert [p["completed"] for p in progress] == list(range(1, total + 1))
    assert all(p["question"] for p in progress)


def test_the_metrics_are_internally_consistent(client: TestClient, read_events) -> None:
    with client.stream("POST", "/eval/run", json={"set_name": "base"}) as response:
        result = read_events(response)[-1][1]

    metrics = result["metrics"]
    assert metrics["set_name"] == "base"
    assert metrics["n_questions"] == len(result["questions"])
    for key in ("recall_at_1", "recall_at_6", "mrr", "ndcg_at_6"):
        assert 0.0 <= metrics[key] <= 1.0
    # Recall@1 counts a subset of what recall@6 counts.
    assert metrics["recall_at_1"] <= metrics["recall_at_6"]
    assert metrics["latency_p50"]["total_ms"] <= metrics["latency_p95"]["total_ms"]


def test_a_question_that_hit_carries_the_rank_it_hit_at(client: TestClient, read_events) -> None:
    with client.stream("POST", "/eval/run", json={"set_name": "base"}) as response:
        questions = read_events(response)[-1][1]["questions"]

    for question in questions:
        assert (question["rank"] is not None) == question["hit"]
        if question["hit"]:
            assert question["rank"] >= 1


def test_the_baseline_comparison_is_opt_out(client: TestClient, read_events) -> None:
    with client.stream(
        "POST", "/eval/run", json={"set_name": "base", "compare_baseline": False}
    ) as response:
        metrics = read_events(response)[-1][1]["metrics"]

    assert metrics["baseline_recall_at_6"] is None


def test_an_empty_index_scores_zero(client: TestClient, read_events) -> None:
    client.post("/settings/wipe", json={"confirm": "DELETE"})

    with client.stream("POST", "/eval/run", json={"set_name": "base"}) as response:
        events = read_events(response)

    assert events[-1][0] == "result"
    metrics = events[-1][1]["metrics"]
    assert metrics["n_questions"] == 20
    assert metrics["recall_at_6"] == 0.0
