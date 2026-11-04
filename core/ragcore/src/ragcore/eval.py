"""Golden-set retrieval evaluation, shared by ``ragcore eval`` and the dev-only route.

A golden line is ``{"q": ..., "expected": ["file.md#Section prefix", ...], "type": "local"}``.
An expected entry matches a retrieved chunk from a file with that name whose section
heading starts with the text after ``#`` (no ``#``: any chunk of that file). File names,
not doc ids: ids hash absolute paths and differ on every machine.
Expected file names are ``fnmatch`` patterns (``garden-*.md``).
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from fnmatch import fnmatch
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
MAX_BRAIN_DROP = 0.04  # one router question of 30; 2 points is under one question
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
    if path is None or not fnmatch(Path(path).name, name):
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


def coverage(chunks: list[RetrievedChunk], paths: dict[str, str], expected: Iterable[str]) -> float:
    """Share of expected entries that at least one packed passage matches."""
    expected = tuple(expected)
    if not expected:
        return 0.0
    found = sum(any(matches(c, paths.get(c.doc_id), e) for c in chunks) for e in expected)
    return found / len(expected)


async def score_questions(
    retriever,
    questions: list[GoldenQuestion],
    *,
    settings: RetrievalSettings,
    doc_meta: dict[str, dict],
    paths: dict[str, str],
    brain=None,
) -> AsyncIterator[EvalQuestionResult]:
    """With a brain, every row is routed too and global rows score the packed global context."""
    for question in questions:
        routed = await brain.route(question.q) if brain is not None else None
        route = {"routed": routed.mode, "route_reason": routed.reason} if routed else {}
        if routed is not None and question.type == "global":
            started = time.perf_counter()
            context = await brain.for_query(
                routed.vector,
                search=question.q,
                question=question.q,
                filters=QueryFilters(),
                doc_meta=doc_meta,
                topic_id=None,
                budget=settings.global_token_budget,
            )
            share = coverage(context.chunks if context else [], paths, question.expected)
            yield EvalQuestionResult(
                q=question.q,
                type="global",
                hit=share > 0,
                coverage=round(share, 4),
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
                **route,
            )
            continue
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
            **route,
        )


def summarize(
    set_name: str, results: list[EvalQuestionResult], baseline_recall: float | None = None
) -> EvalMetrics:
    local = [r for r in results if r.coverage is None]
    n = max(len(local), 1)
    covered = [r.coverage for r in results if r.coverage is not None]
    routed = [r for r in results if r.routed is not None]

    def percentile(p: float) -> StageLatency:
        values = sorted(r.latency_ms for r in results) or [0.0]
        return StageLatency(total_ms=round(values[min(int(len(values) * p), len(values) - 1)], 2))

    return EvalMetrics(
        set_name=set_name,
        n_questions=len(results),
        recall_at_1=round(sum(r.rank == 1 for r in local) / n, 4),
        recall_at_6=round(sum(r.hit for r in local) / n, 4),
        mrr=round(sum(1 / r.rank for r in local if r.rank) / n, 4),
        ndcg_at_6=round(sum(r.ndcg for r in local) / n, 4),
        latency_p50=percentile(0.5),
        latency_p95=percentile(0.95),
        baseline_recall_at_6=baseline_recall,
        global_coverage=round(sum(covered) / len(covered), 4) if covered else None,
        router_accuracy=(
            round(sum(r.routed == r.type for r in routed) / len(routed), 4) if routed else None
        ),
    )


def baseline_metrics(path: Path) -> dict | None:
    """The metrics block of a report ``ragcore eval --report`` wrote; None if there is none."""
    try:
        metrics = json.loads(path.read_text(encoding="utf-8"))["metrics"]
    except (FileNotFoundError, KeyError, TypeError, json.JSONDecodeError):
        return None
    return metrics if isinstance(metrics, dict) else None


def baseline_recall(path: Path) -> float | None:
    metrics = baseline_metrics(path)
    return metrics.get("recall_at_6") if metrics else None


def gate_failures(metrics: EvalMetrics, baseline: dict) -> list[str]:
    """Every measured metric may fall at most its tolerance below the committed baseline."""
    checks = (
        ("recall@6", metrics.recall_at_6, "recall_at_6", MAX_RECALL_DROP),
        ("global coverage", metrics.global_coverage, "global_coverage", MAX_BRAIN_DROP),
        ("router accuracy", metrics.router_accuracy, "router_accuracy", MAX_BRAIN_DROP),
    )
    failures: list[str] = []
    for name, value, key, tolerance in checks:
        if value is None:
            continue
        base = baseline.get(key)
        if base is None:
            failures.append(f"{name}: no baseline value")
        elif value < base - tolerance:
            failures.append(
                f"{name} {value:.2f} is more than {tolerance} below the baseline {base:.2f}"
            )
    return failures
