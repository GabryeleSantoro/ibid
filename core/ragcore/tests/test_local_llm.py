"""LocalLLM keeps the model in RAM only while an answer needs it."""

from __future__ import annotations

import asyncio

from ragcore import local_llm
from ragcore.local_llm import LocalLLM


def _llm(tmp_path, monkeypatch):
    monkeypatch.setattr(local_llm, "IDLE_S", 0.05)
    llm = LocalLLM(tmp_path)
    started: list[int] = []
    stops: list[int] = []

    async def fake_ensure() -> str:
        started.append(1)
        return "http://127.0.0.1:1"

    llm._ensure_server = fake_ensure
    llm.stop = lambda: stops.append(1)
    return llm, started, stops


def test_nothing_starts_until_a_lease_is_taken(tmp_path, monkeypatch) -> None:
    _, started, _ = _llm(tmp_path, monkeypatch)
    assert started == []


def test_the_server_is_not_stopped_while_a_lease_is_held(tmp_path, monkeypatch) -> None:
    llm, started, stops = _llm(tmp_path, monkeypatch)

    async def scenario() -> None:
        async with llm.lease() as url:
            assert url == "http://127.0.0.1:1"
            await asyncio.sleep(0.15)
            assert stops == []
        await asyncio.sleep(0.15)
        assert stops == [1]

    asyncio.run(scenario())
    assert started == [1]


def test_overlapping_leases_keep_it_alive_until_the_last_one_ends(tmp_path, monkeypatch) -> None:
    llm, _, stops = _llm(tmp_path, monkeypatch)

    async def scenario() -> None:
        async with llm.lease():
            async with llm.lease():
                pass
            await asyncio.sleep(0.15)
            assert stops == []
        await asyncio.sleep(0.15)
        assert stops == [1]

    asyncio.run(scenario())


def test_a_lease_is_released_when_its_body_raises(tmp_path, monkeypatch) -> None:
    llm, _, stops = _llm(tmp_path, monkeypatch)

    async def scenario() -> None:
        try:
            async with llm.lease():
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        await asyncio.sleep(0.15)
        assert stops == [1]

    asyncio.run(scenario())
