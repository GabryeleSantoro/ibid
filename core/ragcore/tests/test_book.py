from __future__ import annotations

import asyncio
import hashlib
import io
import tarfile
from pathlib import Path

import httpx
import pytest
from ragcore.api.errors import ApiError
from ragcore.api.routes import books
from ragcore.book import to_latex


def _chapter(markdown: str) -> str:
    latex = to_latex("Libro", [("Capitolo", markdown)], "it")
    return latex.split(r"\chapter{Capitolo}", 1)[1].split(r"\end{document}", 1)[0]


def test_text_is_escaped_and_math_passes_through() -> None:
    body = _chapter(r"Costo 50% & co_x #1 {a} ~ \ e $x_t \in \mathbb{R}$ e \(h\).")
    assert r"50\% \& co\_x \#1 \{a\} \textasciitilde{} \textbackslash{}" in body
    assert r"\(x_t \in \mathbb{R}\)" in body
    assert r"\(h\)" in body


def test_markdown_escapes_inside_math_are_undone() -> None:
    assert r"\(v^*(s) + x_t\)" in _chapter(r"$v^\*(s) + x\_t$")


def test_symbols_the_text_font_lacks_go_through_the_math_font() -> None:
    assert r"x$₁₂$ e $α$" in _chapter("x₁₂ e α")
    assert r"\section{Con \texorpdfstring{\(\gamma\)}{\textbackslash{}gamma}}" in _chapter(
        "## Con $\\gamma$"
    )


def test_bracket_display_math_right_under_text_is_still_math() -> None:
    body = _chapter("REINFORCE:\n\\[\n\\nabla J = \\mathbb{E}\n\\]\ndove \\(G_t\\)")
    assert "\\[\n\\nabla J = \\mathbb{E}\n\\]" in body
    assert r"\textbackslash{}nabla" not in body


def test_html_sub_and_sup_become_latex_and_other_tags_stay_text() -> None:
    body = _chapter("*s*<sub>opt</sub> e x<sup>2 e List<T> <sub>aperto")
    assert r"\emph{s}\textsubscript{opt} e x\textsuperscript{2 e List<T> " in body
    assert body.rstrip().endswith(r"\textsubscript{aperto}}")


def test_currency_is_not_math() -> None:
    assert r"costa 5\$ e 10\$" in _chapter("costa 5$ e 10$")


def test_display_math_and_environments() -> None:
    body = _chapter("$$\ny = 1\n\n+ 2\n$$\n\n$$\n\\begin{align}a&=b\\end{align}\n$$")
    assert "\\[\ny = 1\n+ 2\n\\]" in body
    assert "\\[\n\\begin{align}" not in body
    assert "\\begin{align}a&=b\\end{align}" in body


def test_math_that_touches_files_is_rendered_as_text() -> None:
    body = _chapter(r"$\input{/etc/passwd}$ e $^^5cinput{x}$ ma $\left( x \right)$")
    assert r"\input{" not in body
    assert r"\texttt{\textbackslash{}input\{/etc/passwd\}}" in body
    assert "^^5c" not in body
    assert r"\(\left( x \right)\)" in body


@pytest.mark.parametrize(
    "math",
    [
        r"\InputIfFileExists{../../.ssh/id_rsa}{}{}",
        r"\scantokens\expandafter{\string\in put}",
        r"\ExplSyntaxOn \use:c{in put}",
        r"\def\x{1}",
        r"\begin{filecontents}{x}",
        r"\openin1=x \read1 to\y",
    ],
)
def test_math_that_could_build_a_file_read_is_rendered_as_text(math: str) -> None:
    assert r"\(" not in _chapter(f"${math}$")


def test_ordinary_math_commands_pass() -> None:
    math = r"\bullet \operatorname{arg\,max}_a \mathbb{E}[x] \le \text{se } \tfrac{1}{2}"
    assert rf"\({math}\)" in _chapter(f"${math}$")


def test_leading_h1_becomes_the_chapter_and_levels_shift() -> None:
    body = _chapter("# Titolo\n\nIntro\n\n## Sez\n\n### Sub")
    assert "Titolo" not in body
    assert r"\section{Sez}" in body
    assert r"\subsection{Sub}" in body


def test_extra_h1s_become_sections() -> None:
    body = _chapter("# Uno\n\n# Due\n\n## Tre")
    assert r"\section{Due}" in body
    assert r"\subsection{Tre}" in body


def test_lists_tables_quotes_code_and_links() -> None:
    body = _chapter(
        "1. [uno]\n   - *a*\n\ntesto\n\n3. **due**\n\n| A | B |\n|---|---|\n| [x] | |\n\n"
        "> citazione\n\n```\nif [a]:\n  b\n```\n\n[sito](https://e.it/a%20b#c) [x](ftp://h/f)"
    )
    assert r"\item{} [uno]" in body
    assert r"\emph{a}" in body
    assert r"\begin{enumerate}[start=3]" in body
    assert r"\textbf{due}" in body
    assert r"\begin{longtable}" in body and body.count("p{\\dimexpr\\linewidth/2") == 2
    assert r"{}[x] & {} \\" in body
    assert r"\begin{quote}" in body
    assert r"\mbox{}if\ [a]:\newline" in body and r"\mbox{}\ \ b" in body
    assert r"\href{https://e.it/a\%20b\#c}{sito}" in body
    assert "ftp" not in body and "{x}" in body


def test_chapters_keep_their_order_and_language() -> None:
    latex = to_latex("Dispense", [("Secondo", "b"), ("Primo", "a")], "it")
    assert latex.index(r"\chapter{Secondo}") < latex.index(r"\chapter{Primo}")
    assert "italian]{babel}" in latex
    assert r"\title{Dispense}" in latex
    assert "english]{babel}" in to_latex("T", [("C", "x")], None)


@pytest.fixture
def fake_engine(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    compiled: list[Path] = []

    async def tectonic(_data_dir: Path) -> Path:
        return Path("tectonic")

    async def compile_pdf(_tectonic: Path, tex: Path) -> bool:
        compiled.append(tex)
        tex.with_suffix(".pdf").write_bytes(b"%PDF-1.7")
        return False

    monkeypatch.setattr(books, "_tectonic", tectonic)
    monkeypatch.setattr(books, "_compile", compile_pdf)
    return compiled


def _markdown_ids(client) -> list[str]:
    items = client.get("/documents", params={"ext": ".md"}).json()["items"]
    assert len(items) >= 2
    return [items[1]["id"], items[0]["id"]]


def test_book_writes_tex_and_pdf_in_the_chosen_order(client, fake_engine) -> None:
    ids = _markdown_ids(client)
    titles = [client.get(f"/documents/{i}").json()["title"] for i in ids]

    response = client.post("/books", json={"title": "Le mie dispense", "document_ids": ids})

    assert response.status_code == 200, response.text
    result = response.json()
    tex = Path(result["tex_path"])
    assert tex.name == "Le mie dispense.tex"
    assert Path(result["pdf_path"]).read_bytes() == b"%PDF-1.7"
    assert result["had_errors"] is False
    latex = tex.read_text(encoding="utf-8")
    first, second = (latex.index(rf"\chapter{{{t}}}") for t in titles)
    assert first < second
    assert fake_engine == [tex]


def test_book_rejects_documents_that_are_not_markdown(client, fake_engine, tmp_path) -> None:
    folder = tmp_path / "notes"
    folder.mkdir()
    (folder / "plain.txt").write_text("just text")
    client.post("/sources", json={"path": str(folder), "include_globs": ["**/*.txt"]})
    other = client.get("/documents", params={"ext": ".txt"}).json()["items"][0]
    response = client.post("/books", json={"title": "x", "document_ids": [other["id"]]})
    assert response.status_code == 422
    assert response.json()["code"] == "book_not_markdown"
    assert fake_engine == []


def test_book_needs_known_documents(client, fake_engine) -> None:
    response = client.post("/books", json={"title": "x", "document_ids": ["nope"]})
    assert response.status_code == 404
    assert client.post("/books", json={"title": "x", "document_ids": []}).status_code == 422


def _tarball(payload: bytes) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        info = tarfile.TarInfo("tectonic")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


def test_tectonic_is_installed_only_when_the_checksum_matches(tmp_path, monkeypatch) -> None:
    archive = _tarball(b"#!/bin/sh\n")
    client = httpx.AsyncClient
    transport = httpx.MockTransport(lambda _request: httpx.Response(200, content=archive))
    monkeypatch.setattr(books.httpx, "AsyncClient", lambda **kw: client(transport=transport, **kw))
    monkeypatch.setattr(books.shutil, "which", lambda _name: None)
    monkeypatch.setattr(books.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(books.platform, "machine", lambda: "arm64")

    monkeypatch.setattr(books, "_TECTONIC", {"arm64": ("triple", "0" * 64)})
    with pytest.raises(ApiError) as error:
        asyncio.run(books._tectonic(tmp_path))
    assert error.value.code == "book_engine_download"
    assert not (tmp_path / "tectonic" / "tectonic").exists()

    digest = hashlib.sha256(archive).hexdigest()
    monkeypatch.setattr(books, "_TECTONIC", {"arm64": ("triple", digest)})
    binary = asyncio.run(books._tectonic(tmp_path))
    assert binary.read_bytes() == b"#!/bin/sh\n"
    assert binary.stat().st_mode & 0o100
