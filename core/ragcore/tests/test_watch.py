"""Watched sources rescan themselves shortly after their files change."""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from ragcore.watch import DEBOUNCE_S, SourceWatcher


class FakeStore:
    def __init__(self, *sources: SimpleNamespace, fail: bool = False) -> None:
        self.sources = {s.id: s for s in sources}
        self.ingested: list[str] = []
        self.fail = fail

    async def ingest_source_async(self, source_id: str) -> list:
        self.ingested.append(source_id)
        if self.fail:
            raise RuntimeError("embedder down")
        return []


def source(path: Path, *, watch: bool = True, id: str = "src_1") -> SimpleNamespace:
    return SimpleNamespace(id=id, path=str(path), watch=watch)


def wait_for(predicate, timeout: float = 5.0) -> bool:
    """The OS reports file events with some latency (FSEvents batches them)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


@pytest.fixture
def watching(tmp_path: Path):
    started: list[SourceWatcher] = []

    def make(store: FakeStore) -> SourceWatcher:
        watcher = SourceWatcher(store)
        watcher.start()
        watcher.sync()
        started.append(watcher)
        return watcher

    yield make
    for watcher in started:
        watcher.stop()


async def test_new_documents_trigger_one_scan_after_the_debounce(tmp_path, watching) -> None:
    store = FakeStore(source(tmp_path))
    watcher = watching(store)

    (tmp_path / "a.md").write_text("# A")
    (tmp_path / "b.md").write_text("# B")
    assert wait_for(lambda: "src_1" in watcher.pending)

    await watcher.tick(now=time.monotonic())
    assert store.ingested == []
    await watcher.tick(now=time.monotonic() + DEBOUNCE_S + 0.1)
    assert store.ingested == ["src_1"]


async def test_hidden_and_unsupported_changes_are_ignored(tmp_path, watching) -> None:
    store = FakeStore(source(tmp_path))
    watcher = watching(store)

    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "index").write_text("x")
    (tmp_path / "photo.jpg").write_bytes(b"x")
    (tmp_path / "~$report.docx").write_bytes(b"x")
    time.sleep(1.5)

    assert "src_1" not in watcher.pending


async def test_deleting_a_document_triggers_a_scan(tmp_path, watching) -> None:
    doomed = tmp_path / "old.md"
    doomed.write_text("# Old")
    store = FakeStore(source(tmp_path))
    watcher = watching(store)

    doomed.unlink()

    assert wait_for(lambda: "src_1" in watcher.pending)


async def test_only_watched_existing_sources_are_observed(tmp_path, watching) -> None:
    folder = tmp_path / "docs"
    folder.mkdir()
    store = FakeStore(source(folder), source(tmp_path / "missing", id="src_2"))
    watcher = watching(store)
    assert set(watcher.watched) == {"src_1"}

    store.sources["src_1"].watch = False
    watcher.sync()
    assert watcher.watched == {}

    store.sources["src_1"].watch = True
    watcher.sync()
    shutil.rmtree(folder)
    watcher.sync()
    assert watcher.watched == {}


async def test_a_source_removed_before_its_scan_is_not_scanned(tmp_path, watching) -> None:
    store = FakeStore(source(tmp_path))
    watcher = watching(store)
    watcher.touch("src_1")
    del store.sources["src_1"]

    await watcher.tick(now=time.monotonic() + DEBOUNCE_S + 0.1)

    assert store.ingested == []


async def test_a_failed_scan_is_logged_not_raised(tmp_path, watching, caplog) -> None:
    store = FakeStore(source(tmp_path), fail=True)
    watcher = watching(store)
    watcher.touch("src_1")

    await watcher.tick(now=time.monotonic() + DEBOUNCE_S + 0.1)

    assert store.ingested == ["src_1"]
    assert "embedder down" in caplog.text
