"""Golden-set retrieval evaluation, shared by ``ragcore eval`` and the dev-only route.

A golden line is ``{"q": ..., "expected": ["file.md#Section prefix", ...], "type": "local"}``.
An expected entry matches a retrieved chunk from a file with that name whose section
heading starts with the text after ``#`` (no ``#``: any chunk of that file). File names,
not doc ids: ids hash absolute paths and differ on every machine.
"""

from __future__ import annotations

import json
import math
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from pathlib import Path

from ragcore.api.schemas import (
    EvalMetrics,
    EvalQuestionResult,
    QueryFilters,
    RetrievalSettings,
    RetrievedChunk,
    StageLatency,
)

K = 6
# The CI gate's tolerance: one question of the 20-question base set. (The roadmap's
# "2 points" assumed a 50-question set, where one question is 2 points.)
MAX_RECALL_DROP = 0.05
# The committed sets; exists in a dev checkout only, never in the packaged sidecar.
GOLDEN_DIR = next(
    (
        parent / "eval" / "golden"
        for parent in Path(__file__).resolve().parents
        if (parent / "eval" / "golden").is_dir()
    ),
    None,
)


@dataclass(frozen=True, slots=True)
class GoldenQuestion:
    q: str
    expected: tuple[str, ...]
    type: str = "local"


def load_golden(path: Path) -> list[GoldenQuestion]:
    questions: list[GoldenQuestion] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            questions.append(
                GoldenQuestion(
                    q=row["q"], expected=tuple(row["expected"]), type=row.get("type", "local")
                )
            )
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"{path.name}:{number}: {exc!r}") from exc
    return questions


def golden_sets(data_dir: Path) -> list[Path]:
    """The committed sets (dev checkout) plus private ones the user put in ``<data dir>/eval``."""
    dirs = [d for d in (GOLDEN_DIR, data_dir / "eval") if d is not None and d.is_dir()]
    return [path for d in dirs for path in sorted(d.glob("*.jsonl"))]


def matches(chunk: RetrievedChunk, path: str | None, expected: str) -> bool:
    name, _, section = expected.partition("#")
    if path is None or Path(path).name != name:
        return False
    return (chunk.section_path or "").split(" > ")[-1].startswith(section)


def first_hit(
    chunks: list[RetrievedChunk], paths: dict[str, str], expected: Iterable[str]
) -> int | None:
    expected = tuple(expected)
    return next(
        (
            rank
            for rank, chunk in enumerate(chunks[:K], start=1)
            if any(matches(chunk, paths.get(chunk.doc_id), e) for e in expected)
        ),
        None,
    )


async def score_questions(
    retriever,
    questions: list[GoldenQuestion],
    *,
    settings: RetrievalSettings,
    doc_meta: dict[str, dict],
    paths: dict[str, str],
) -> AsyncIterator[EvalQuestionResult]:
    for question in questions:
        chunks, latency, _ = await retriever.asearch(
            question.q, settings=settings, filters=QueryFilters(), doc_meta=doc_meta
        )
        rank = first_hit(chunks, paths, question.expected)
        yield EvalQuestionResult(
            q=question.q,
            type=question.type,
            hit=rank is not None,
            rank=rank,
            # One relevant target per question, so the ideal DCG is 1.
            ndcg=1 / math.log2(rank + 1) if rank else 0.0,
            latency_ms=round(latency.total_ms, 2),
        )


def summarize(
    set_name: str, results: list[EvalQuestionResult], baseline_recall: float | None = None
) -> EvalMetrics:
    n = max(len(results), 1)

    def percentile(p: float) -> StageLatency:
        values = sorted(r.latency_ms for r in results) or [0.0]
        return StageLatency(total_ms=round(values[min(int(len(values) * p), len(values) - 1)], 2))

    return EvalMetrics(
        set_name=set_name,
        n_questions=len(results),
        recall_at_1=round(sum(r.rank == 1 for r in results) / n, 4),
        recall_at_6=round(sum(r.hit for r in results) / n, 4),
        mrr=round(sum(1 / r.rank for r in results if r.rank) / n, 4),
        ndcg_at_6=round(sum(r.ndcg for r in results) / n, 4),
        latency_p50=percentile(0.5),
        latency_p95=percentile(0.95),
        baseline_recall_at_6=baseline_recall,
    )


def baseline_recall(path: Path) -> float | None:
    """recall@6 from a report ``ragcore eval --report`` wrote; None if there is none."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))["metrics"]["recall_at_6"]
    except (FileNotFoundError, KeyError, TypeError, json.JSONDecodeError):
        return None
