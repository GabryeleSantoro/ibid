"""The eval harness: golden-set parsing, hit matching and metric math."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from ragcore.api.schemas import EvalQuestionResult, RetrievedChunk
from ragcore.eval import GOLDEN_DIR, first_hit, load_golden, matches, summarize
from ragcore.ingest.parse import parse

FIXTURE_DOCS = Path(__file__).resolve().parents[3] / "fixtures" / "docs"


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
