"""Parsing to pages. Page numbers are a correctness requirement: citations carry them."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest
from ragcore.ingest.parse import UnsupportedFormat, parse


def test_markdown_splits_into_pages_on_level_two_headings(tmp_path: Path) -> None:
    source = tmp_path / "reranking.md"
    source.write_text(
        "# Reranking\n\nIntro text.\n\n"
        "## Cross encoders\n\nThey score pairs.\n\n"
        "## Latency\n\nThey are slower.\n"
    )

    parsed = parse(source)

    assert parsed.title == "Reranking"
    assert [p.page for p in parsed.pages] == [1, 2, 3]
    assert parsed.pages[1].section_path == "Reranking > Cross encoders"
    assert "score pairs" in parsed.pages[1].text


def test_markdown_without_headings_is_one_page(tmp_path: Path) -> None:
    source = tmp_path / "flat.md"
    source.write_text("Just a paragraph with no headings at all.")

    parsed = parse(source)

    assert [p.page for p in parsed.pages] == [1]
    assert parsed.title == "flat"


def test_plain_text_is_one_page(tmp_path: Path) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("Chunking splits documents into retrievable units.")

    parsed = parse(source)

    assert len(parsed.pages) == 1
    assert parsed.needs_ocr is False


def test_unknown_extension_raises(tmp_path: Path) -> None:
    source = tmp_path / "thing.xyz"
    source.write_text("?")

    with pytest.raises(UnsupportedFormat):
        parse(source)


def test_pdf_extracts_slide_text_page_by_page(tmp_path: Path, monkeypatch) -> None:
    from ragcore.ingest import ocr

    monkeypatch.setattr(ocr, "read_page", lambda page: "")  # the fake page cannot render

    class FakeTextPage:
        def __init__(self, text: str) -> None:
            self.text = text

        def get_text_range(self) -> str:
            return self.text

        def close(self) -> None:
            pass

    class FakePage:
        def __init__(self, text: str) -> None:
            self.text = text

        def get_textpage(self) -> FakeTextPage:
            return FakeTextPage(self.text)

        def close(self) -> None:
            pass

    class FakeDocument:
        def __init__(self, _path: str) -> None:
            self.pages = [FakePage("Slide title\r\nElaborated slide text."), FakePage("")]

        def __len__(self) -> int:
            return len(self.pages)

        def __getitem__(self, index: int) -> FakePage:
            return self.pages[index]

        def close(self) -> None:
            pass

    monkeypatch.setitem(sys.modules, "pypdfium2", SimpleNamespace(PdfDocument=FakeDocument))
    source = tmp_path / "presentation.pdf"
    source.write_bytes(b"not decoded as UTF-8")

    parsed = parse(source)

    assert parsed.title == "presentation"
    assert [page.page for page in parsed.pages] == [1, 2]
    assert parsed.pages[0].text == "Slide title\nElaborated slide text."
    assert parsed.pages[1].text == ""
    assert parsed.needs_ocr is True


def test_pdf_page_numbers_survive_a_page_without_text(tmp_path: Path, text_pdf) -> None:
    source = tmp_path / "paper.pdf"
    source.write_bytes(text_pdf("Cross encoders score pairs.", "", "Hybrid search merges legs."))

    parsed = parse(source)

    assert [page.page for page in parsed.pages] == [1, 2, 3]
    assert parsed.pages[1].text == ""
    assert "Hybrid search" in parsed.pages[2].text
    assert parsed.needs_ocr is True


def test_a_pdf_page_with_a_near_empty_text_layer_is_flagged_for_ocr(
    tmp_path: Path, text_pdf
) -> None:
    clean, scanned = tmp_path / "clean.pdf", tmp_path / "scanned.pdf"
    clean.write_bytes(text_pdf("Cross encoders score query passage pairs."))
    scanned.write_bytes(text_pdf("Cross encoders score query passage pairs.", "12"))

    assert parse(clean).needs_ocr is False
    assert parse(scanned).needs_ocr is True
    assert parse(scanned).pages[1].text == "12"


def test_pptx_extracts_visible_text_in_slide_order(tmp_path: Path) -> None:
    source = tmp_path / "presentation.pptx"
    slide = (
        '<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
        'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
        "<a:t>Second slide content</a:t></p:sld>"
    )
    with ZipFile(source, "w") as archive:
        archive.writestr("ppt/slides/slide2.xml", slide)
        archive.writestr("ppt/slides/slide1.xml", slide.replace("Second", "First"))

    parsed = parse(source)

    assert [page.page for page in parsed.pages] == [1, 2]
    assert parsed.pages[0].text == "First slide content"
    assert parsed.pages[1].text == "Second slide content"


def test_a_slide_without_text_keeps_the_later_slide_numbers(tmp_path: Path) -> None:
    source = tmp_path / "deck.pptx"
    slide = (
        '<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
        'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">{}</p:sld>'
    )
    with ZipFile(source, "w") as archive:
        archive.writestr("ppt/slides/slide1.xml", slide.format("<a:t>First</a:t>"))
        archive.writestr("ppt/slides/slide2.xml", slide.format(""))
        archive.writestr("ppt/slides/slide3.xml", slide.format("<a:t>Third</a:t>"))

    parsed = parse(source)

    assert [(page.page, page.text) for page in parsed.pages] == [
        (1, "First"),
        (2, ""),
        (3, "Third"),
    ]


def test_pptx_appends_speaker_notes_when_present(tmp_path: Path) -> None:
    source = tmp_path / "presentation.pptx"
    slide = (
        '<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
        'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
        "<a:t>Slide content</a:t></p:sld>"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide" '
        'Target="../notesSlides/notesSlide1.xml"/>'
        "</Relationships>"
    )
    notes = (
        '<p:notes xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
        'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
        "<a:t>Remember to mention the Q3 numbers.</a:t></p:notes>"
    )
    with ZipFile(source, "w") as archive:
        archive.writestr("ppt/slides/slide1.xml", slide)
        archive.writestr("ppt/slides/_rels/slide1.xml.rels", rels)
        archive.writestr("ppt/notesSlides/notesSlide1.xml", notes)

    parsed = parse(source)

    assert parsed.pages[0].text == (
        "Slide content\n\nNote del relatore: Remember to mention the Q3 numbers."
    )


def test_pptx_appends_external_hyperlinks_when_present(tmp_path: Path) -> None:
    source = tmp_path / "presentation.pptx"
    slide = (
        '<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
        'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<a:r><a:rPr><a:hlinkClick r:id="rId2"/></a:rPr><a:t>Learn more</a:t></a:r>'
        "</p:sld>"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId2" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink" '
        'Target="https://example.com/reference" TargetMode="External"/>'
        "</Relationships>"
    )
    with ZipFile(source, "w") as archive:
        archive.writestr("ppt/slides/slide1.xml", slide)
        archive.writestr("ppt/slides/_rels/slide1.xml.rels", rels)

    parsed = parse(source)

    assert parsed.pages[0].text == ("Learn more\n\nLink nella slide: https://example.com/reference")


def test_pptx_slide_without_rels_file_is_unaffected(tmp_path: Path) -> None:
    source = tmp_path / "presentation.pptx"
    slide = (
        '<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
        'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
        "<a:t>Plain slide</a:t></p:sld>"
    )
    with ZipFile(source, "w") as archive:
        archive.writestr("ppt/slides/slide1.xml", slide)

    parsed = parse(source)

    assert parsed.pages[0].text == "Plain slide"


def test_a_page_without_a_text_layer_gets_its_text_from_ocr(
    tmp_path: Path, text_pdf, monkeypatch
) -> None:
    from ragcore.ingest import ocr

    monkeypatch.setattr(ocr, "read_page", lambda page: "Words recovered from the scanned image")
    source = tmp_path / "scan.pdf"
    source.write_bytes(text_pdf("Cross encoders score pairs.", ""))

    parsed = parse(source)

    assert parsed.pages[1].text == "Words recovered from the scanned image"
    assert parsed.needs_ocr is False


def test_a_page_ocr_cannot_read_stays_flagged(tmp_path: Path, text_pdf, monkeypatch) -> None:
    from ragcore.ingest import ocr

    monkeypatch.setattr(ocr, "read_page", lambda page: "")
    source = tmp_path / "blank.pdf"
    source.write_bytes(text_pdf(""))

    parsed = parse(source)

    assert parsed.pages[0].text == ""
    assert parsed.needs_ocr is True


def test_the_real_ocr_engine_reads_rendered_text(tmp_path: Path, text_pdf) -> None:
    import pypdfium2 as pdfium
    from ragcore.ingest import ocr

    source = tmp_path / "printed.pdf"
    source.write_bytes(text_pdf("Hello OCR world"))
    document = pdfium.PdfDocument(str(source))
    try:
        text = ocr.read_page(document[0])
    finally:
        document.close()

    assert "hello" in text.lower()


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _paragraph(style: str | None, text: str) -> str:
    props = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    return f"<w:p>{props}<w:r><w:t>{text}</w:t></w:r></w:p>"


def _write_docx(path: Path, body: str, styles: dict[str, str]) -> None:
    """A minimal .docx: the document body XML plus styleId -> style name."""
    style_xml = "".join(
        f'<w:style w:type="paragraph" w:styleId="{sid}"><w:name w:val="{name}"/></w:style>'
        for sid, name in styles.items()
    )
    with ZipFile(path, "w") as archive:
        archive.writestr(
            "word/document.xml",
            f'<w:document xmlns:w="{W_NS}"><w:body>{body}</w:body></w:document>',
        )
        archive.writestr("word/styles.xml", f'<w:styles xmlns:w="{W_NS}">{style_xml}</w:styles>')


def test_docx_splits_into_pages_on_headings(tmp_path: Path) -> None:
    source = tmp_path / "report.docx"
    body = "".join(
        _paragraph(style, text)
        for style, text in [
            ("Title", "Quarterly Report"),
            (None, "Intro paragraph."),
            ("Heading1", "Results"),
            (None, "Revenue grew."),
            ("Heading2", "Risks"),
            (None, "Supply is tight."),
        ]
    )
    _write_docx(
        source, body, {"Title": "Title", "Heading1": "heading 1", "Heading2": "heading 2"}
    )

    parsed = parse(source)

    assert parsed.title == "Quarterly Report"
    assert [p.section_path for p in parsed.pages] == [
        "Quarterly Report",
        "Quarterly Report > Results",
        "Quarterly Report > Risks",
    ]
    assert parsed.pages[1].text == "Revenue grew."


def test_docx_headings_are_found_in_a_localized_word(tmp_path: Path) -> None:
    # Italian Word names the style id "Titolo1"; the style name stays "heading 1".
    source = tmp_path / "relazione.docx"
    body = _paragraph(None, "Premessa.") + _paragraph("Titolo1", "Risultati") + _paragraph(
        None, "I ricavi sono cresciuti."
    )
    _write_docx(source, body, {"Titolo1": "heading 1"})

    parsed = parse(source)

    assert parsed.title == "relazione"
    assert parsed.pages[1].section_path == "relazione > Risultati"
    assert parsed.pages[1].text == "I ricavi sono cresciuti."


def test_docx_table_cells_are_kept(tmp_path: Path) -> None:
    source = tmp_path / "table.docx"
    cell = "<w:tc>" + _paragraph(None, "{}") + "</w:tc>"
    row = "<w:tr>" + cell.format("Model") + cell.format("EmbeddingGemma") + "</w:tr>"
    _write_docx(source, _paragraph(None, "Specs.") + f"<w:tbl>{row}</w:tbl>", {})

    parsed = parse(source)

    assert parsed.pages[0].text == "Specs.\nModel\nEmbeddingGemma"
