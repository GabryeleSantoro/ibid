"""Walking a source folder: globs, size cap, stable digests."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
from ragcore.ingest.walk import sha256_of, walk_source


def build_corpus(root: Path) -> None:
    (root / "notes").mkdir(parents=True)
    (root / "notes" / "reranking.md").write_text("# Reranking\n\nCross-encoders reorder.")
    (root / "notes" / "chunking.txt").write_text("Chunking splits documents.")
    (root / "notes" / "ignore.log").write_text("noise")
    (root / "notes" / "huge.md").write_text("x" * (2 * 1024 * 1024))


def test_include_globs_select_files(tmp_path: Path) -> None:
    build_corpus(tmp_path)

    found = walk_source(tmp_path, include_globs=["**/*.md"], exclude_globs=[], max_file_mb=100)

    assert sorted(f.path.name for f in found) == ["huge.md", "reranking.md"]


def test_include_glob_without_double_star_still_scans_subfolders(tmp_path: Path) -> None:
    build_corpus(tmp_path)

    found = walk_source(tmp_path, include_globs=["*.md"], exclude_globs=[], max_file_mb=100)

    assert sorted(f.path.name for f in found) == ["huge.md", "reranking.md"]


def test_exclude_globs_win(tmp_path: Path) -> None:
    build_corpus(tmp_path)

    found = walk_source(
        tmp_path, include_globs=["**/*"], exclude_globs=["**/*.log", "**/huge.md"], max_file_mb=100
    )

    assert sorted(f.path.name for f in found) == ["chunking.txt", "reranking.md"]


def test_size_cap_skips_large_files(tmp_path: Path) -> None:
    build_corpus(tmp_path)

    found = walk_source(tmp_path, include_globs=["**/*.md"], exclude_globs=[], max_file_mb=1)

    assert [f.path.name for f in found] == ["reranking.md"]


def test_digest_is_stable_and_content_sensitive(tmp_path: Path) -> None:
    build_corpus(tmp_path)
    target = tmp_path / "notes" / "reranking.md"
    before = sha256_of(target)
    assert before == sha256_of(target)

    target.write_text("# Reranking\n\nChanged.")

    assert sha256_of(target) != before


def test_mime_is_derived_from_the_extension(tmp_path: Path) -> None:
    build_corpus(tmp_path)

    found = walk_source(tmp_path, include_globs=["**/*.txt"], exclude_globs=[], max_file_mb=100)

    assert found[0].mime == "text/plain"


def test_hidden_files_and_directories_are_skipped(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config.md").write_text("hidden dir file")
    (tmp_path / ".dotfile.md").write_text("hidden dotfile")
    (tmp_path / "visible.md").write_text("# Visible\n\nText.")

    found = walk_source(tmp_path, include_globs=["**/*.md"], exclude_globs=[], max_file_mb=100)

    assert [f.path.name for f in found] == ["visible.md"]


def test_a_file_that_disappears_mid_walk_does_not_abort_the_walk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file gone by the time it's stat'd is skipped; the rest of the source still comes back.

    ``walk_source`` itself already calls ``.is_file()`` (which stats internally)
    once per candidate during discovery, before its own per-file ``stat()``/hash
    step. So the file must survive that first stat — real discovery would have
    seen it too — and only vanish on the *second* stat, which is what the walk's
    own try/except around ``stat``/hash is meant to catch.
    """
    (tmp_path / "keep.md").write_text("# Keep\n\nStays.")
    victim = tmp_path / "vanish.md"
    victim.write_text("# Vanish\n\nGone before the second stat.")
    (tmp_path / "zzz.md").write_text("# Zzz\n\nAfter the gap.")

    real_stat = Path.stat
    victim_stats = 0

    def flaky_stat(self: Path, *args: object, **kwargs: object) -> object:
        nonlocal victim_stats
        if self == victim:
            victim_stats += 1
            if victim_stats > 1:
                victim.unlink()
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", flaky_stat)

    found = walk_source(tmp_path, include_globs=["**/*.md"], exclude_globs=[], max_file_mb=100)

    assert [f.path.name for f in found] == ["keep.md", "zzz.md"]


def test_a_suffix_filter_drops_other_files_before_they_are_hashed(tmp_path: Path) -> None:
    build_corpus(tmp_path)

    found = walk_source(
        tmp_path, include_globs=["**/*"], exclude_globs=[], max_file_mb=100, suffixes={".md"}
    )

    assert sorted(f.path.name for f in found) == ["huge.md", "reranking.md"]


def test_an_unchanged_file_reuses_its_known_digest(tmp_path: Path) -> None:
    path = tmp_path / "note.md"
    path.write_text("# Note\n\nBody.")
    [first] = walk_source(tmp_path, include_globs=["*.md"], exclude_globs=[], max_file_mb=1)
    known = {str(path): (first.size_bytes, first.mtime, "digest-from-last-scan")}

    [again] = walk_source(
        tmp_path, include_globs=["*.md"], exclude_globs=[], max_file_mb=1, known=known
    )

    assert again.sha256 == "digest-from-last-scan"


def test_a_file_whose_mtime_moved_is_hashed_again(tmp_path: Path) -> None:
    path = tmp_path / "note.md"
    path.write_text("# Note\n\nBody.")
    [first] = walk_source(tmp_path, include_globs=["*.md"], exclude_globs=[], max_file_mb=1)
    stale = first.mtime - timedelta(seconds=5)
    known = {str(path): (first.size_bytes, stale, "digest-from-last-scan")}

    [again] = walk_source(
        tmp_path, include_globs=["*.md"], exclude_globs=[], max_file_mb=1, known=known
    )

    assert again.sha256 == sha256_of(path)


def test_office_lock_files_are_skipped(tmp_path: Path) -> None:
    (tmp_path / "report.docx").write_bytes(b"x")
    (tmp_path / "~$report.docx").write_bytes(b"x")

    found = walk_source(tmp_path, include_globs=["**/*"], exclude_globs=[], max_file_mb=100)

    assert [f.path.name for f in found] == ["report.docx"]
