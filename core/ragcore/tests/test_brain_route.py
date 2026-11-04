"""Local or global: example similarity, keyword hints in the dead zone, keywords alone offline."""

from __future__ import annotations

from pathlib import Path

import httpx
import numpy as np
import pytest
from ragcore.brain.null import NullBrain
from ragcore.brain.route import EXAMPLES, keyword_route, score_route
from ragcore.config import Config
from ragcore.models.fakes import FakeEmbedClient
from ragcore.store.real import RealStore

# One global example along e0, one local along e1: the score is q[0] - q[1].
EXAMPLES_2D = (np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]]))


def test_a_clear_global_score_routes_global_and_reports_it() -> None:
    routed = score_route("what is in here", [0.6, 0.4], EXAMPLES_2D)

    assert routed.mode == "global"
    assert routed.reason == "Global (router 0.20)"
    assert routed.vector == [0.6, 0.4]


def test_a_small_lead_for_global_is_not_enough() -> None:
    # Local questions score up to 0.12 on the eval sets; global ones mostly above 0.15.
    routed = score_route("what is in here", [0.55, 0.45], EXAMPLES_2D)

    assert routed.mode == "local"
    assert routed.reason == "Answerable from individual passages (router 0.10)"


def test_a_clear_local_score_routes_local() -> None:
    routed = score_route("what is in here", [0.47, 0.53], EXAMPLES_2D)

    assert routed.mode == "local"
    assert routed.reason == "Local (router -0.06)"


@pytest.mark.parametrize(
    ("question", "mode"),
    [("what are the main themes across my notes", "global"), ("what is in here", "local")],
)
def test_the_dead_zone_falls_back_to_keyword_hints(question: str, mode: str) -> None:
    routed = score_route(question, [0.51, 0.49], EXAMPLES_2D)

    assert routed.mode == mode
    assert routed.reason.endswith("(router 0.02)")


@pytest.mark.parametrize(
    "question",
    [
        "What are the main themes across the corpus?",
        "Summarize all documents",
        "Compare the two approaches",
        "What topics does this library cover?",
    ],
)
def test_a_corpus_wide_phrase_routes_global(question: str) -> None:
    routed = keyword_route(question)

    assert routed.mode == "global"
    assert "asks about the corpus" in routed.reason


def test_a_passage_question_routes_local() -> None:
    assert keyword_route("How does reciprocal rank fusion combine two rankings?").mode == "local"


def test_keyword_hints_ignore_case() -> None:
    assert keyword_route("SUMMARIZE ALL the notes").mode == "global"


def test_the_examples_cover_both_modes_and_five_languages() -> None:
    kinds = [kind for kind, _ in EXAMPLES]

    assert len(EXAMPLES) >= 28
    assert kinds.count("global") >= 14 and kinds.count("local") >= 14


class DownEmbedClient(FakeEmbedClient):
    async def embed(self, texts, *, batch_size=32, query=False):
        raise httpx.ConnectError("embedder down")


class CountingEmbedClient(FakeEmbedClient):
    def __init__(self) -> None:
        self.calls: list[int] = []

    async def embed(self, texts, *, batch_size=32, query=False):
        self.calls.append(len(texts))
        return await super().embed(texts, batch_size=batch_size, query=query)


def store_with(tmp_path: Path, embedder) -> RealStore:
    return RealStore(Config(data_dir=tmp_path, backend="real"), embedder=embedder)


async def test_an_embedder_outage_routes_by_keywords(tmp_path: Path) -> None:
    store = store_with(tmp_path, DownEmbedClient())

    routed = await store.brain.route("main themes across my notes")

    assert routed.mode == "global"
    assert routed.vector is None


async def test_the_examples_are_embedded_once(tmp_path: Path) -> None:
    embedder = CountingEmbedClient()
    store = store_with(tmp_path, embedder)

    await store.brain.route("first question")
    routed = await store.brain.route("second question")

    assert embedder.calls == [len(EXAMPLES), 1, 1]
    assert routed.vector is not None


async def test_the_stub_brain_routes_by_keywords() -> None:
    assert (await NullBrain().route("Summarize all documents")).mode == "global"
