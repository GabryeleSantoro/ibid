"""Finding files worth indexing, and fingerprinting them.

The digest is what makes re-indexing a no-op: a file whose sha256 is unchanged
is skipped without being parsed or embedded (Task 9 checks it against
``MetaStore.sha_index`` before doing either).

Semantics later tasks (8, 9, 14) rely on, decided here deliberately:

- **Globs are matched relative to ``root`` and include subfolders.**
  ``include_globs``/``exclude_globs`` use recursive matching, so both
  ``"*.md"`` and ``"**/*.md"`` find Markdown files below ``root``. Patterns
  never see an absolute path. Task 9 forwards user-supplied globs unchanged,
  so this is the contract a caller's glob strings must be written against.
- **Exclude always wins.** A file is kept only if it matches at least one
  ``include_globs`` pattern *and* matches none of the ``exclude_globs``
  patterns. A file matching both is dropped — excludes are a hard veto, not a
  second vote.
- **``max_file_mb`` is mebibytes** (``max_file_mb * 1024 * 1024`` bytes), and
  the cap is inclusive: a file of exactly that many bytes is kept; only a file
  *strictly larger* is skipped.
- **Hidden files and directories are always skipped**, regardless of
  ``include_globs``. ``pathlib.Path.glob`` matches dotfiles by default (unlike
  a shell), so a naive ``"**/*.md"`` would otherwise descend into things like
  ``.git``, ``.obsidian`` or ``.venv``. Any path whose name, or an ancestor
  directory's name (relative to ``root``), starts with ``"."`` is excluded
  unconditionally — a caller cannot opt back in via ``include_globs``.
  Office lock files (``~$report.docx``, written while a document is open) are
  skipped the same way.
- **Symlinked files are indexed like any other file; symlinked directories are
  never traversed.** On Python >=3.13 (this repo's floor), ``Path.glob``'s
  recursive ``**`` defaults to ``recurse_symlinks=False``, so a symlink cycle
  under ``root`` cannot make the walk recurse forever. We rely on that default
  rather than re-implementing cycle detection.
- **``mime`` comes from a small static extension table, not the stdlib
  ``mimetypes`` module.** ``mimetypes.guess_type`` reads the host's installed
  mime database, which varies by platform and can produce surprising results
  for uncommon extensions (e.g. ``.xyz`` resolves to ``chemical/x-xyz`` on this
  machine). A fixed table keeps ``mime`` deterministic across every machine
  this runs on; an extension outside the table gets
  ``"application/octet-stream"``.
- **A file that vanishes or becomes unreadable mid-walk is skipped, not
  fatal.** Task 9 runs this against a real user folder, where a sync client
  can delete a file, or permissions can change, between glob discovery and
  the ``stat``/hash below. If that raises, only that one file is dropped
  (with a logged warning naming the path and the error); every other file
  already found in this source is still returned. The walk never aborts and
  discards a whole source over one bad file.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

MIME_BY_EXT = {
    ".md": "text/markdown",
    ".txt": "text/plain",
    ".pdf": "application/pdf",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".html": "text/html",
    ".htm": "text/html",
    ".csv": "text/csv",
}

READ_CHUNK = 1024 * 1024  # stream digests in 1 MiB blocks; never read a whole file into memory


@dataclass(slots=True)
class FoundFile:
    path: Path
    sha256: str
    size_bytes: int
    mtime: datetime
    ext: str
    mime: str


def sha256_of(path: Path) -> str:
    """Digest a file's bytes, streaming so a large file is never read whole.

    The digest depends only on content, not on how the file happens to be
    chunked while reading it, so it is stable across file sizes and re-runs.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(READ_CHUNK):
            digest.update(block)
    return digest.hexdigest()


def is_hidden(relative: Path) -> bool:
    """True if any path segment starts with a dot, or is an Office lock file (``~$name``)."""
    return any(part.startswith((".", "~$")) for part in relative.parts)


def _glob_recursive(root: Path, pattern: str) -> set[Path]:
    """Match a pattern below ``root``, including files in nested directories.

    ``Path.glob("*.md")`` only checks the root itself. Source patterns are
    intended to describe document types for the whole source, so patterns
    without an explicit ``**`` are expanded with ``rglob`` as well.
    """
    if "**" in pattern:
        return set(root.glob(pattern))
    return set(root.rglob(pattern))


def walk_source(
    root: Path,
    *,
    include_globs: list[str],
    exclude_globs: list[str],
    max_file_mb: int,
    suffixes: Collection[str] | None = None,
    known: Mapping[str, tuple[int, datetime, str]] | None = None,
) -> list[FoundFile]:
    """Find files under ``root`` worth indexing.

    ``suffixes`` (lowercased, with the dot) drops other files before they are
    stat'd or hashed. ``known`` maps a path to the ``(size_bytes, mtime,
    sha256)`` seen last time; a file whose size and mtime both still match
    reuses that digest instead of being read again.

    See the module docstring for the exact glob, hidden-file, symlink, size
    and failure-handling rules this applies. Results are sorted by path for a
    deterministic order. A file that disappears or becomes unreadable between
    being matched and being stat'd/hashed is skipped (with a logged warning)
    rather than raising out of this function.
    """
    cap_bytes = max_file_mb * 1024 * 1024

    included: set[Path] = set()
    for pattern in include_globs:
        included.update(p for p in _glob_recursive(root, pattern) if p.is_file())
    for pattern in exclude_globs:
        included.difference_update(_glob_recursive(root, pattern))

    out: list[FoundFile] = []
    for path in sorted(included):
        ext = path.suffix.lower()
        if is_hidden(path.relative_to(root)) or (suffixes is not None and ext not in suffixes):
            continue
        try:
            stat = path.stat()
            if stat.st_size > cap_bytes:
                continue
            mtime = datetime.fromtimestamp(stat.st_mtime, tz=UTC)
            # ponytail: size+mtime is the change signal (as make/rsync use it); an
            # edit that preserves both goes unseen until the file is touched.
            last = known.get(str(path)) if known else None
            if last is not None and last[0] == stat.st_size and last[1] == mtime:
                digest = last[2]
            else:
                digest = sha256_of(path)
        except OSError as exc:
            # Realistically FileNotFoundError (deleted between glob and stat) or
            # PermissionError (unreadable) — both are OSError subclasses. Skip
            # this one file rather than losing every file already found.
            logger.warning("skipping %s: %s", path, exc)
            continue
        out.append(
            FoundFile(
                path=path,
                sha256=digest,
                size_bytes=stat.st_size,
                mtime=mtime,
                ext=ext,
                mime=MIME_BY_EXT.get(ext, "application/octet-stream"),
            )
        )
    return out
