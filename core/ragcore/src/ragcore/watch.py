"""Rescan a watched source shortly after its files change.

A watchdog observer (FSEvents, inotify, ReadDirectoryChangesW) marks a source
dirty; ``tick`` rescans it once nothing has changed for ``DEBOUNCE_S``. The
rescan is ``ingest_source_async``: unchanged files are skipped by sha, so a
spurious event costs a directory walk, not an embedding pass.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer
from watchdog.observers.api import ObservedWatch

from ragcore.ingest.parse import SUPPORTED_SUFFIXES
from ragcore.ingest.walk import is_hidden

logger = logging.getLogger("ragcore.watch")

DEBOUNCE_S = 2.0  # copying 500 files in is one rescan, not 500
TICK_S = 1.0


class _Handler(FileSystemEventHandler):
    def __init__(self, watcher: SourceWatcher, source_id: str, root: Path) -> None:
        self.watcher = watcher
        self.source_id = source_id
        self.root = root

    def on_any_event(self, event: FileSystemEvent) -> None:
        if event.event_type in ("opened", "closed_no_write"):
            return
        # FSEvents adds "directory modified" for the parent of every change; the
        # file's own event carries what matters, so this one would only add noise.
        if event.is_directory and event.event_type == "modified":
            return
        paths = (event.src_path, getattr(event, "dest_path", "") or "")
        if any(self._relevant(str(p), event.is_directory) for p in paths if p):
            self.watcher.touch(self.source_id)

    def _relevant(self, path: str, is_directory: bool) -> bool:
        try:
            relative = Path(path).resolve().relative_to(self.root)
        except ValueError:
            return False
        # The folder itself: FSEvents also reports its creation if that happened just
        # before the watch began. Its removal is sync()'s job, not a rescan's.
        if not relative.parts or is_hidden(relative):
            return False
        return is_directory or Path(path).suffix.lower() in SUPPORTED_SUFFIXES


class SourceWatcher:
    def __init__(self, store) -> None:
        self.store = store
        self.pending: dict[str, float] = {}
        self.watched: dict[str, tuple[str, ObservedWatch]] = {}
        self._observer = Observer()
        self._lock = threading.Lock()

    def touch(self, source_id: str) -> None:
        """Called from the observer thread."""
        with self._lock:
            self.pending[source_id] = time.monotonic()

    def start(self) -> None:
        self._observer.start()

    def stop(self) -> None:
        self._observer.stop()
        self._observer.join(timeout=5)

    def sync(self) -> None:
        """Observe exactly the sources that want watching and still exist."""
        wanted = {
            s.id: s.path
            for s in self.store.sources.values()
            if s.watch and Path(s.path).is_dir()
        }
        for source_id, (path, watch) in list(self.watched.items()):
            if wanted.get(source_id) != path:
                with contextlib.suppress(KeyError, OSError):  # its folder may be gone already
                    self._observer.unschedule(watch)
                del self.watched[source_id]
        for source_id, path in wanted.items():
            if source_id in self.watched:
                continue
            handler = _Handler(self, source_id, Path(path).resolve())
            try:
                watch = self._observer.schedule(handler, path, recursive=True)
            except OSError as exc:
                logger.warning("cannot watch %s: %s", path, exc)
                continue
            self.watched[source_id] = (path, watch)

    def _due(self, now: float) -> list[str]:
        with self._lock:
            ready = [sid for sid, at in self.pending.items() if now - at >= DEBOUNCE_S]
            for source_id in ready:
                del self.pending[source_id]
        return ready

    async def tick(self, now: float | None = None) -> None:
        self.sync()
        for source_id in self._due(time.monotonic() if now is None else now):
            if source_id not in self.store.sources:
                continue
            try:
                await self.store.ingest_source_async(source_id)
            except Exception as exc:  # noqa: BLE001 - one bad scan must not stop watching
                logger.warning("rescan of %s after a change failed: %s", source_id, exc)

    async def run(self) -> None:
        self.start()
        try:
            while True:
                await self.tick()
                await asyncio.sleep(TICK_S)
        finally:
            self.stop()
