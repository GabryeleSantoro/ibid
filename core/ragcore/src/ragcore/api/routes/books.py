"""Markdown documents, in the user's order, compiled into one PDF book via Tectonic."""

from __future__ import annotations

import asyncio
import hashlib
import io
import platform
import shutil
import tarfile
from collections import Counter
from pathlib import Path

import httpx
from fastapi import APIRouter

from ragcore.api.deps import ConfigDep, StoreDep
from ragcore.api.errors import api_error
from ragcore.api.routes.conversions import _file_stem
from ragcore.api.schemas import BookRequest, BookResult
from ragcore.book import to_latex

router = APIRouter(prefix="/books", tags=["books"])

TECTONIC_VERSION = "0.17.0"
# Pinned: we execute this binary, so a swapped release asset must not run.
_TECTONIC = {
    "arm64": (
        "aarch64-apple-darwin",
        "a3f1cac7c5678f01661a92212f58480ae3b0634115d880dbc59e2953ded45667",
    ),
    "x86_64": (
        "x86_64-apple-darwin",
        "7c90ef5b6ddb1eb1937e4337add5237b79338e4b9676459fa91187d24d6cdf80",
    ),
}
# The proxy gives a plain call 300 s. The first compile also fetches LaTeX packages.
COMPILE_TIMEOUT_S = 240
_install_lock = asyncio.Lock()


async def _tectonic(data_dir: Path) -> Path:
    """Tectonic on PATH, else the copy in the data dir, downloaded on first use."""
    if found := shutil.which("tectonic"):
        return Path(found)
    binary = data_dir / "tectonic" / "tectonic"
    async with _install_lock:
        if binary.exists():
            return binary
        asset = _TECTONIC.get(platform.machine()) if platform.system() == "Darwin" else None
        if asset is None:
            raise api_error(
                501, "book_engine_unsupported", "no Tectonic build for this platform; "
                "install tectonic and put it on PATH"
            )
        triple, sha256 = asset
        url = (
            "https://github.com/tectonic-typesetting/tectonic/releases/download/"
            f"tectonic%40{TECTONIC_VERSION}/tectonic-{TECTONIC_VERSION}-{triple}.tar.gz"
        )
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=120) as client:
                response = await client.get(url)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise api_error(
                502, "book_engine_download", f"could not download Tectonic: {exc}"
            ) from exc
        if hashlib.sha256(response.content).hexdigest() != sha256:
            raise api_error(502, "book_engine_download", "Tectonic download failed its checksum")
        binary.parent.mkdir(parents=True, exist_ok=True)
        part = binary.with_suffix(".part")
        with tarfile.open(fileobj=io.BytesIO(response.content)) as tar:
            try:
                member = tar.extractfile("tectonic")
            except KeyError:
                member = None
            if member is None:
                raise api_error(502, "book_engine_download", "tectonic missing from the archive")
            part.write_bytes(member.read())
        part.chmod(0o755)
        part.replace(binary)
        return binary


async def _compile(tectonic: Path, tex: Path) -> bool:
    """Write the PDF next to `tex`. True when LaTeX reported errors it compiled through."""
    pdf = tex.with_suffix(".pdf")
    pdf.unlink(missing_ok=True)
    proc = await asyncio.create_subprocess_exec(
        str(tectonic), "--untrusted", "--chatter", "minimal", "--keep-logs",
        "-Z", "continue-on-errors", "--outdir", str(tex.parent), str(tex),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        output, _ = await asyncio.wait_for(proc.communicate(), COMPILE_TIMEOUT_S)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        raise api_error(504, "book_compile_timeout", "LaTeX compilation timed out") from None
    if not pdf.exists():
        tail = " / ".join(output.decode(errors="replace").strip().splitlines()[-6:])
        raise api_error(
            422, "book_compile_failed", f"LaTeX compilation failed: {tail}", detail=tail
        )
    return proc.returncode != 0


@router.post("", response_model=BookResult)
async def create_book(payload: BookRequest, store: StoreDep, config: ConfigDep) -> BookResult:
    chapters: list[tuple[str, str]] = []
    languages: Counter[str] = Counter()
    for doc_id in payload.document_ids:
        document = store.documents.get(doc_id)
        if document is None:
            raise api_error(404, "document_not_found", "document not found")
        if document.ext.lower() not in (".md", ".markdown"):
            raise api_error(
                422, "book_not_markdown", f"{document.title} is not Markdown", title=document.title
            )
        try:
            text = Path(document.path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            raise api_error(
                404, "book_source_missing", f"{document.path} is missing", title=document.title
            ) from None
        chapters.append((document.title, text))
        if document.lang:
            languages[document.lang] += 1

    title = payload.title.strip() or chapters[0][0]
    folder = config.data_dir / "books" / _file_stem(title)
    folder.mkdir(parents=True, exist_ok=True)
    tex = folder / f"{folder.name}.tex"
    language = languages.most_common(1)[0][0] if languages else None
    tex.write_text(to_latex(title, chapters, language), encoding="utf-8")

    had_errors = await _compile(await _tectonic(config.data_dir), tex)
    pdf = tex.with_suffix(".pdf")
    return BookResult(tex_path=str(tex), pdf_path=str(pdf), had_errors=had_errors)
