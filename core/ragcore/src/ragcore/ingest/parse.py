"""File to pages.

A "page" is whatever unit a citation can point at. For PDFs (Task 14) that is
a real page; for markdown/text it is a level-two section, which is the
closest honest analogue and is what the reader view highlights against. Task
8 chunks within a page's text; it never crosses a page boundary.

``parse()`` dispatches on suffix through the ``_PARSERS`` table below. PDFs
are opened with ``pypdfium2`` rather than being decoded as UTF-8: a PDF is a
binary container, and reading it as text returns font tables and drawing
operators instead of the words visible on the slide. A PDF page with no
extractable text layer (for example, a slide exported as an image) is run
through OCR; ``ParsedDoc.needs_ocr`` is set when even OCR finds too little.

Semantics decided here, that Task 8/9 depend on:

- **``section_path`` separator is ``" > "``**, most-general first ((document
  title) > (heading)). It is one segment deep for now because we only split
  on level-two (``##``) headings; deeper headings (``###`` and beyond) stay
  inside their enclosing section's text rather than creating more pages.
- **Text before the first level-two heading** (including all of a level-one
  title line) becomes page 1, with ``section_path`` equal to the document
  title alone (no ``" > "`` suffix) — it is the document's own top-level
  section, not a subsection of itself. If there is no such preamble text
  (the document starts directly at a ``##`` heading), no page is created for
  it.
- **The title** is the first level-one (``#``) heading's text if the document
  has one, otherwise the filename stem. This applies to both markdown and
  plain text (plain text never has a level-one heading, so it always falls
  back to the stem).
- **Encoding**: files are read as UTF-8 with ``errors="replace"``. Real
  corpora contain files that are not valid UTF-8 (legacy exports, mixed
  encodings); refusing to ingest them, or crashing the whole walk on one bad
  file, is worse than replacing the undecodable bytes with U+FFFD and
  indexing what does decode. A file that is mostly non-UTF-8 will simply
  chunk and embed poorly, which is a quality problem, not a crash.
- **A page with empty text (after stripping) is dropped, not returned.** An
  empty file, or a markdown file that is nothing but headings with no body
  under any of them, would otherwise produce one or more pages whose ``text``
  is ``""``. Task 8's chunker already skips those, but a document that
  reports ``n_pages > 0`` while producing zero chunks looks indexed when it
  is not. Dropping empty pages here means a document with no usable text
  comes back with ``pages == []``, which Task 9 can record honestly instead
  of silently pretending something was indexed. The kept pages are
  renumbered ``1..n`` contiguously — a dropped page never leaves a gap in the
  sequence.
"""

from __future__ import annotations

import posixpath
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from zipfile import ZipFile

from ragcore.ingest import ocr

_HEADING = re.compile(r"^(#{1,3})\s+(.*)$", re.MULTILINE)
_NOTES_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide"
_HYPERLINK_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink"


OCR_THRESHOLD = 20


class UnsupportedFormat(Exception):
    """Raised by ``parse()`` when no handler is registered for a file's suffix."""


@dataclass(slots=True)
class ParsedPage:
    page: int
    section_path: str
    text: str


@dataclass(slots=True)
class ParsedDoc:
    title: str
    pages: list[ParsedPage]
    needs_ocr: bool = False


def _title(text: str, fallback: str) -> str:
    match = next((m for m in _HEADING.finditer(text) if len(m.group(1)) == 1), None)
    return match.group(2).strip() if match else fallback


def _paginate(sections: list[tuple[str, str]], *, keep_empty: bool = False) -> list[ParsedPage]:
    """Number sections 1..n. Empty ones are dropped unless they are physical pages
    (PDF pages, slides), where dropping one would shift every later citation."""
    return [
        ParsedPage(page=i, section_path=section_path, text=body)
        for i, (section_path, body) in enumerate(
            ((section_path, body) for section_path, body in sections if body or keep_empty),
            start=1,
        )
    ]


def _markdown(path: Path) -> ParsedDoc:
    text = path.read_text(encoding="utf-8", errors="replace")
    title = _title(text, path.stem)
    marks = [m for m in _HEADING.finditer(text) if len(m.group(1)) == 2]

    if not marks:
        return ParsedDoc(title=title, pages=_paginate([(title, text.strip())]))

    sections: list[tuple[str, str]] = [(title, text[: marks[0].start()].strip())]
    for i, mark in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        heading = mark.group(2).strip()
        sections.append((f"{title} > {heading}", text[mark.end() : end].strip()))

    return ParsedDoc(title=title, pages=_paginate(sections))


def _plain(path: Path) -> ParsedDoc:
    text = path.read_text(encoding="utf-8", errors="replace")
    return ParsedDoc(title=path.stem, pages=_paginate([(path.stem, text.strip())]))


def _clean_pdf_text(text: str) -> str:
    """Normalize text returned by PDFium without changing its character offsets."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()


def _pdf(path: Path) -> ParsedDoc:
    """Extract one page per PDF page using PDFium's text layer.

    A page whose text layer is under OCR_THRESHOLD characters is run through OCR;
    if that still finds too little, the page is kept (later page numbers stay
    true) and the document is flagged.
    """
    try:
        import pypdfium2 as pdfium
    except ModuleNotFoundError as exc:  # pragma: no cover - packaging guard
        raise RuntimeError(
            "PDF support requires pypdfium2; reinstall the ragcore dependencies"
        ) from exc

    document = pdfium.PdfDocument(str(path))
    sections: list[tuple[str, str]] = []
    needs_ocr = False
    try:
        for page_number in range(len(document)):
            page = document[page_number]
            text_page = None
            try:
                text_page = page.get_textpage()
                text = _clean_pdf_text(text_page.get_text_range())
                if len(text) < OCR_THRESHOLD:
                    # ponytail: serial, ~0.5-1 s per scanned page; a 300-page scan takes minutes.
                    text = _clean_pdf_text(ocr.read_page(page)) or text
            finally:
                if text_page is not None:
                    text_page.close()
                page.close()

            sections.append((f"{path.stem} > Page {page_number + 1}", text))
            needs_ocr = needs_ocr or len(text) < OCR_THRESHOLD
    finally:
        document.close()

    return ParsedDoc(
        title=path.stem, pages=_paginate(sections, keep_empty=True), needs_ocr=needs_ocr
    )


def _slide_relationships(archive: ZipFile, slide_name: str) -> dict[str, tuple[str, str]]:
    """``rId`` -> ``(relationship type, target)`` for one slide's ``.rels`` part."""
    rels_name = slide_name.replace("ppt/slides/", "ppt/slides/_rels/") + ".rels"
    if rels_name not in archive.namelist():
        return {}
    root = ET.fromstring(archive.read(rels_name))
    return {rel.get("Id"): (rel.get("Type", ""), rel.get("Target", "")) for rel in root}


def _notes_text(archive: ZipFile, rels: dict[str, tuple[str, str]]) -> str:
    target = next((target for kind, target in rels.values() if kind == _NOTES_REL), None)
    if target is None:
        return ""
    notes_path = posixpath.normpath(posixpath.join("ppt/slides", target))
    if notes_path not in archive.namelist():
        return ""
    root = ET.fromstring(archive.read(notes_path))
    return " ".join(
        node.text.strip()
        for node in root.iter()
        if node.tag.endswith("}t") and node.text and node.text.strip()
    )


def _slide_links(root: ET.Element, rels: dict[str, tuple[str, str]]) -> list[str]:
    r_ns = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
    links: list[str] = []
    seen: set[str] = set()
    for node in root.iter():
        if not node.tag.endswith("}hlinkClick"):
            continue
        rid = node.get(f"{r_ns}id")
        if not rid or rid not in rels:
            continue
        kind, target = rels[rid]
        if kind == _HYPERLINK_REL and target and target not in seen:
            seen.add(target)
            links.append(target)
    return links


def _pptx(path: Path) -> ParsedDoc:
    """Extract visible text, speaker notes and external links, one page per slide."""
    with ZipFile(path) as archive:
        slide_names = sorted(
            name for name in archive.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)
        )
        slide_names.sort(key=lambda name: int(re.search(r"slide(\d+)\.xml$", name).group(1)))
        sections: list[tuple[str, str]] = []
        for slide_number, name in enumerate(slide_names, start=1):
            root = ET.fromstring(archive.read(name))
            text = " ".join(
                node.text.strip()
                for node in root.iter()
                if node.tag.endswith("}t") and node.text and node.text.strip()
            )
            rels = _slide_relationships(archive, name)
            notes = _notes_text(archive, rels)
            links = _slide_links(root, rels)

            parts = [text] if text else []
            if notes:
                parts.append(f"Note del relatore: {notes}")
            if links:
                parts.append("Link nella slide: " + ", ".join(links))
            sections.append((f"{path.stem} > Slide {slide_number}", "\n\n".join(parts)))
    return ParsedDoc(title=path.stem, pages=_paginate(sections, keep_empty=True))


_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_DOCX_HEADINGS = frozenset({"heading 1", "heading 2"})


def _docx_style_names(archive: ZipFile) -> dict[str, str]:
    """styleId -> lowercased style name. Localized Word translates ids, not names."""
    if "word/styles.xml" not in archive.namelist():
        return {}
    root = ET.fromstring(archive.read("word/styles.xml"))
    return {
        style.get(f"{_W}styleId", ""): name.get(f"{_W}val", "").lower()
        for style in root.iter(f"{_W}style")
        if (name := style.find(f"{_W}name")) is not None
    }


def _docx(path: Path) -> ParsedDoc:
    """Paragraph text (table cells included), one page per Heading 1/2 section."""
    with ZipFile(path) as archive:
        styles = _docx_style_names(archive)
        body = ET.fromstring(archive.read("word/document.xml"))

    title: str | None = None
    sections: list[tuple[str | None, list[str]]] = [(None, [])]
    for paragraph in body.iter(f"{_W}p"):
        text = "".join(node.text or "" for node in paragraph.iter(f"{_W}t")).strip()
        if not text:
            continue
        style_ref = paragraph.find(f"{_W}pPr/{_W}pStyle")
        style = styles.get(style_ref.get(f"{_W}val", "")) if style_ref is not None else None
        if style == "title" and title is None:
            title = text
        elif style in _DOCX_HEADINGS:
            sections.append((text, []))
        else:
            sections[-1][1].append(text)

    title = title or path.stem
    return ParsedDoc(
        title=title,
        pages=_paginate(
            [
                (title if heading is None else f"{title} > {heading}", "\n".join(lines))
                for heading, lines in sections
            ]
        ),
    )


class _HtmlText(HTMLParser):
    """Visible text, split at <h1>/<h2> the way markdown splits at ``##``."""

    _SKIP = frozenset({"script", "style", "noscript", "template", "svg"})
    _BREAK = frozenset({"p", "div", "li", "br", "tr", "td", "th", "pre", "blockquote",
                        "section", "article", "h3", "h4", "h5", "h6"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.sections: list[tuple[str | None, list[str]]] = [(None, [])]
        self._skip = 0
        self._capture: list[str] | None = None  # the <title> or heading being read
        self._in_title = False

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in self._SKIP:
            self._skip += 1
        elif self._skip:  # e.g. an icon's <svg><title>: not the page title
            return
        elif tag == "title":
            self._in_title, self._capture = True, []
        elif tag in ("h1", "h2"):
            self._capture = []
        elif tag in self._BREAK:
            self.sections[-1][1].append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP:
            self._skip = max(0, self._skip - 1)
        elif tag == "title" and self._capture is not None:
            self.title = " ".join("".join(self._capture).split())
            self._in_title, self._capture = False, None
        elif tag in ("h1", "h2") and self._capture is not None and not self._in_title:
            heading = " ".join("".join(self._capture).split())
            self._capture = None
            if heading:
                self.sections.append((heading, []))

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        # Source newlines are layout, not breaks; only block tags (above) start a new line.
        # ponytail: <pre> loses its line breaks too.
        text = re.sub(r"\s+", " ", data)
        (self._capture if self._capture is not None else self.sections[-1][1]).append(text)


def _tidy(parts: list[str]) -> str:
    """Collapse runs of whitespace inside lines; drop blank lines."""
    lines = (" ".join(line.split()) for line in "".join(parts).splitlines())
    return "\n".join(line for line in lines if line)


def _html(path: Path) -> ParsedDoc:
    # ponytail: read as UTF-8 like markdown; a legacy <meta charset> page decodes with U+FFFD.
    reader = _HtmlText()
    reader.feed(path.read_text(encoding="utf-8", errors="replace"))
    reader.close()
    first_heading = next((h for h, _ in reader.sections if h), None)
    title = reader.title or first_heading or path.stem
    return ParsedDoc(
        title=title,
        pages=_paginate(
            [
                (title if heading is None else f"{title} > {heading}", _tidy(parts))
                for heading, parts in reader.sections
            ]
        ),
    )


_PARSERS: dict[str, Callable[[Path], ParsedDoc]] = {
    ".md": _markdown,
    ".txt": _plain,
    ".pdf": _pdf,
    ".pptx": _pptx,
    ".docx": _docx,
    ".html": _html,
    ".htm": _html,
}
SUPPORTED_SUFFIXES = frozenset(_PARSERS)


def parse(path: Path) -> ParsedDoc:
    """Parse ``path`` into a ``ParsedDoc``, dispatching on its lowercased suffix."""
    handler = _PARSERS.get(path.suffix.lower())
    if handler is None:
        raise UnsupportedFormat(f"no parser for {path.suffix!r}")
    return handler(path)
