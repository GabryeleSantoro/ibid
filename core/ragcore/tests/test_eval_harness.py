"""The eval harness: golden-set parsing, hit matching and metric math."""

from __future__ import annotations

import math
from fnmatch import fnmatch
from pathlib import Path

import pytest
from ragcore.api.schemas import EvalQuestionResult, RetrievedChunk
from ragcore.eval import (
    GOLDEN_DIR,
    coverage,
    first_hit,
    gate_failures,
    load_golden,
    matches,
    summarize,
)
from ragcore.ingest.parse import parse

FIXTURE_DOCS = Path(__file__).resolve().parents[3] / "fixtures" / "docs"
BRAIN_DOCS = Path(__file__).resolve().parents[3] / "fixtures" / "brain"


def chunk(doc_id: str, section_path: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{doc_id}:1", doc_id=doc_id, doc_title="T", page_start=1, page_end=1,
        section_path=section_path, text="x",
    )


def test_an_expected_section_matches_by_file_name_and_heading_prefix() -> None:
    c = chunk("d1", "Reranking with Cross-Encoders > The minimum-score gate")

    assert matches(c, "/any/where/reranking.md", "reranking.md#The minimum-score")
    assert matches(c, "/any/where/reranking.md", "reranking.md")
    assert not matches(c, "/any/where/reranking.md", "reranking.md#Candidate count")
    assert not matches(c, "/any/where/hybrid-search.md", "reranking.md")
    assert not matches(c, None, "reranking.md")


def test_first_hit_is_the_rank_of_the_first_matching_chunk() -> None:
    chunks = [chunk("a", "A > One"), chunk("b", "B > Two"), chunk("b", "B > Three")]
    paths = {"a": "/x/a.md", "b": "/x/b.md"}

    assert first_hit(chunks, paths, ("b.md#Three",)) == 3
    assert first_hit(chunks, paths, ("b.md",)) == 2
    assert first_hit(chunks, paths, ("missing.md",)) is None


def test_a_malformed_golden_line_names_its_line_number(tmp_path: Path) -> None:
    golden = tmp_path / "bad.jsonl"
    golden.write_text('{"q": "fine", "expected": ["a.md"]}\n{"q": "no expected"}\n')

    with pytest.raises(ValueError, match="bad.jsonl:2"):
        load_golden(golden)


def test_the_metrics_follow_their_definitions() -> None:
    results = [
        EvalQuestionResult(q="a", type="local", hit=True, rank=1, ndcg=1.0),
        EvalQuestionResult(q="b", type="local", hit=True, rank=3, ndcg=1 / math.log2(4)),
        EvalQuestionResult(q="c", type="local", hit=False, rank=None, ndcg=0.0),
    ]

    metrics = summarize("t", results, baseline_recall=0.9)

    assert metrics.recall_at_1 == round(1 / 3, 4)
    assert metrics.recall_at_6 == round(2 / 3, 4)
    assert metrics.mrr == round((1 + 1 / 3) / 3, 4)
    assert metrics.ndcg_at_6 == round(1.5 / 3, 4)
    assert metrics.baseline_recall_at_6 == 0.9


def test_every_base_question_points_at_a_real_fixture_section() -> None:
    """Guards against golden rot: renaming a fixture heading must fail here, not in CI eval."""
    headings = {
        path.name: [page.section_path.split(" > ")[-1] for page in parse(path).pages]
        for path in FIXTURE_DOCS.glob("*.md")
    }
    questions = load_golden(GOLDEN_DIR / "base.jsonl")

    assert len(questions) == 20
    for question in questions:
        for expected in question.expected:
            name, _, section = expected.partition("#")
            assert name in headings, expected
            assert any(h.startswith(section) for h in headings[name]), expected


def test_coverage_is_the_share_of_expected_patterns_present() -> None:
    chunks = [chunk("doc_a", ""), chunk("doc_b", "")]
    paths = {"doc_a": "/x/garden-tomatoes.md", "doc_b": "/x/astro-moon.md"}

    assert coverage(chunks, paths, ["garden-*.md", "astro-*.md", "cook-*.md", "rag-*.md"]) == 0.5
    assert coverage([], paths, ["garden-*.md"]) == 0.0


def test_global_rows_report_coverage_and_routing_not_recall() -> None:
    results = [
        EvalQuestionResult(q="a", type="local", hit=True, rank=1, ndcg=1.0, routed="local"),
        EvalQuestionResult(q="b", type="global", hit=True, coverage=0.5, routed="global"),
        EvalQuestionResult(q="c", type="global", hit=True, coverage=1.0, routed="local"),
    ]

    metrics = summarize("t", results)

    assert metrics.recall_at_6 == 1.0  # the local row only
    assert metrics.global_coverage == 0.75
    assert metrics.router_accuracy == round(2 / 3, 4)


def test_the_gate_holds_each_metric_to_its_baseline() -> None:
    metrics = summarize(
        "t", [EvalQuestionResult(q="b", type="global", hit=True, coverage=0.7, routed="global")]
    )
    ok = {"recall_at_6": 0.0, "global_coverage": 0.72, "router_accuracy": 1.0}
    low = {"recall_at_6": 0.0, "global_coverage": 0.8, "router_accuracy": 1.0}

    assert gate_failures(metrics, ok) == []
    failed = gate_failures(metrics, low)
    assert len(failed) == 1 and failed[0].startswith("global coverage")
    assert gate_failures(metrics, {"recall_at_6": 0.0}) == [
        "global coverage: no baseline value",
        "router accuracy: no baseline value",
    ]


def test_the_brain_set_points_at_files_that_exist() -> None:
    names = [path.name for path in BRAIN_DOCS.glob("*.md")]
    questions = load_golden(GOLDEN_DIR / "brain.jsonl")

    assert len(names) == 40
    assert sum(q.type == "global" for q in questions) == 15
    assert sum(q.type == "local" for q in questions) == 15
    for question in questions:
        for expected in question.expected:
            assert any(fnmatch(n, expected.partition("#")[0]) for n in names), expected


def test_one_router_flip_in_twenty_rows_stays_within_the_gate() -> None:
    def metrics(wrong: int):
        rows = [
            EvalQuestionResult(q=str(i), type="local", hit=True, rank=1, routed="local")
            for i in range(20 - wrong)
        ]
        rows += [
            EvalQuestionResult(q=f"w{i}", type="local", hit=True, rank=1, routed="global")
            for i in range(wrong)
        ]
        return summarize("t", rows)

    assert gate_failures(metrics(1), {"recall_at_6": 1.0, "router_accuracy": 1.0}) == []
    assert gate_failures(metrics(3), {"recall_at_6": 1.0, "router_accuracy": 0.9}) == []
    assert len(gate_failures(metrics(2), {"recall_at_6": 1.0, "router_accuracy": 1.0})) == 1
