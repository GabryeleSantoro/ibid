# Roadmap Gaps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close seven gaps between `plan/ibid-build-plan.md` and the code: multi-turn chat memory, OCR for scanned PDFs, a real eval harness with a CI gate, DOCX/HTML/CSV parsers, a folder watcher, per-document language detection, and a license scan in CI.

**Architecture:** Every change slots into an existing seam. Chat history is threaded through `AnswerEngine.stream` → `llm_stream` → `_request`. OCR, the new parsers and language detection live in `ingest/parse.py` (OCR engine in a new `ingest/ocr.py`), so both backends get them through `parse()`. The watcher is a new `watch.py` that calls the existing `store.ingest_source_async`. The eval harness is a new `eval.py` shared by the dev-only route and a new `ragcore eval` CLI command. License checks are one stdlib script plus `cargo-deny`.

**Tech Stack:** Python 3.13, FastAPI, pytest; new runtime deps `rapidocr` + `onnxruntime` (OCR), `py3langid` (language), `watchdog` (watcher). DOCX/HTML/CSV use the stdlib only (`zipfile` + `xml.etree`, `html.parser`, `csv`), matching how PPTX is already parsed. GitHub Actions, `cargo-deny`.

**Spec:** `plan/ibid-build-plan.md` (Phase 1 ingestion, Phase 2 eval harness, Phase 3 chat history, §6 risks: license scan). The audit that produced this list is in the 2026-10-17 conversation; item numbers there: 1, 3, 4, 5, 6, 7, 10.

## Global Constraints

- Python `>=3.13`; ruff line-length 100, rules `E, F, I, UP, B, SIM` (`uv run ruff check .` must pass).
- No GPL/AGPL runtime dependencies (`pymupdf`/`pymupdf4llm` are banned by the spec's license guard).
- Citation invariant (`citations.py`): a citation not pointing at a passage retrieved *for this turn* is discarded. History must not change that.
- The UI talks to the sidecar only through `lib/ipc.ts` → Tauri → Rust proxy.
- `requires_models` tests stay excluded by default; new unit tests must run without llama-server.
- CLAUDE.md: run `scripts/install-local.sh` after each task; never cut a release.
- Every new runtime dependency that PyInstaller can't see must be added to `scripts/build-sidecar.sh` in the same task.

## Review Focus

1. **A follow-up after a failed or cancelled answer.** The session then holds a user turn with no answer after it. Providers reject two user turns in a row, so history must skip unanswered turns (Task 1: `test_an_unanswered_question_is_left_out_of_the_history`).
2. **An Excel CSV exported on an Italian machine** (`;` separator, UTF-8 BOM). It must still split into columns, with no `﻿` in the first header (Task 5: `test_csv_reads_an_excel_semicolon_export_with_a_bom`).
3. **A DOCX saved by a localized Word.** Style ids are translated (`Titolo1`), but the style *name* stays `heading 1` (Task 3: `test_docx_headings_are_found_in_a_localized_word`).
4. **Churn inside a watched folder that isn't a document change**: `.git`, `.obsidian`, images, and the parent-directory "modified" event FSEvents adds to every file change. None of it may trigger a rescan (Task 7: `test_hidden_and_unsupported_changes_are_ignored`).
5. **Word's lock file `~$report.docx`**, present while a document is open. It must never be indexed as a broken document (Task 3: `test_office_lock_files_are_skipped`).

---

### Task 1: Multi-turn chat memory

The LLM currently sees only the current question (`llm.py:93,104`), and retrieval searches the raw follow-up. This task sends earlier answered turns as real chat messages, within `RetrievalSettings.history_token_budget` (already exists, default 2048), and searches follow-ups together with the previous question.

**Files:**
- Modify: `core/ragcore/src/ragcore/llm.py` (`_request`, `_stream`, `llm_stream`)
- Modify: `core/ragcore/src/ragcore/ports.py` (`AnswerEngine.stream`)
- Modify: `core/ragcore/src/ragcore/backend.py` (`ConnectionAnswerEngine.stream`)
- Modify: `core/ragcore/src/ragcore/api/routes/query.py` (new `chat_history`, `retrieval_query`; wire into `events()`)
- Test: `core/ragcore/tests/test_llm.py`, `core/ragcore/tests/test_query.py`

**Interfaces:**
- Produces: `llm.History = Sequence[tuple[str, str]]`, i.e. `(role, text)` pairs with role `"user" | "assistant"`, oldest first.
- Produces: `llm_stream(..., history: History = ())` and `AnswerEngine.stream(..., history: Sequence[tuple[str, str]] = ())`.
- Produces: `query.chat_history(messages: list[ChatMessage], budget_tokens: int) -> list[tuple[str, str]]` and `query.retrieval_query(question: str, history: list[tuple[str, str]]) -> str`.

- [ ] **Step 1: Write the failing LLM tests** (append to `tests/test_llm.py`)

```python
def test_earlier_turns_go_between_the_system_prompt_and_the_question() -> None:
    history = [("user", "What is RRF?"), ("assistant", "A fusion method.")]

    _, captured = run(connection(), history=history)

    messages = captured["payload"]["messages"]
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[1]["content"] == "What is RRF?"
    assert messages[2]["content"] == "A fusion method."
    assert "What is reranking?" in messages[3]["content"]


def test_anthropic_gets_the_history_as_alternating_messages() -> None:
    _, captured = run(
        connection(kind="anthropic"),
        lines=ANTHROPIC_LINES,
        history=[("user", "q"), ("assistant", "a")],
    )

    assert [m["role"] for m in captured["payload"]["messages"]] == ["user", "assistant", "user"]


def test_no_history_sends_only_the_question() -> None:
    _, captured = run(connection())

    assert [m["role"] for m in captured["payload"]["messages"]] == ["system", "user"]
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest core/ragcore/tests/test_llm.py -k "history or earlier_turns" -v`
Expected: FAIL with `TypeError: llm_stream() got an unexpected keyword argument 'history'`

- [ ] **Step 3: Thread `history` through `llm.py`**

Add after the imports (next to `from collections.abc import AsyncIterator`, change it to `from collections.abc import AsyncIterator, Sequence`):

```python
# Earlier turns of the chat as (role, text), oldest first; role is "user" or "assistant".
History = Sequence[tuple[str, str]]


def _messages(history: History, user: str) -> list[dict]:
    return [
        *({"role": role, "content": text} for role, text in history),
        {"role": "user", "content": user},
    ]
```

In `_request`, add a keyword parameter `history: History = ()` after `max_tokens`. Replace the Anthropic body line `"messages": [{"role": "user", "content": user}],` with `"messages": _messages(history, user),`. Replace the OpenAI-compatible `"messages": [...]` block with:

```python
        "messages": [{"role": "system", "content": system_prompt}, *_messages(history, user)],
```

In `_stream`, add the keyword parameter `history: History = (),` after `extra_body` and pass `history=history` into the `_request(...)` call.

In `llm_stream`, add the keyword parameter `history: History = (),` after `local=None` and pass `history=history` into **both** `_stream(...)` calls.

- [ ] **Step 4: Widen the port and the engine**

`ports.py`, `AnswerEngine.stream`: add `history: Sequence[tuple[str, str]] = (),` after `max_tokens` (import `Sequence` from `collections.abc` next to `AsyncIterator`). Update the docstring to: `"""Yields answer text pieces. Directives are stub-only and ignored by real engines; history is earlier (role, text) turns, oldest first."""`

`backend.py`, `ConnectionAnswerEngine.stream`: add the same parameter and pass `history=history` to `llm_stream(...)`. The scripted branch ignores it. (Import `Sequence` from `collections.abc`.)

- [ ] **Step 5: Run the LLM tests to verify they pass**

Run: `uv run pytest core/ragcore/tests/test_llm.py -v`
Expected: all PASS

- [ ] **Step 6: Write the failing route tests** (append to `tests/test_query.py`; add `from datetime import UTC, datetime`, and add `ChatMessage` to the `ragcore.api.schemas` import; import `chat_history, retrieval_query` from `ragcore.api.routes.query`)

```python
# -------------------------------------------------------------------- history


def _msg(role: str, text: str) -> ChatMessage:
    return ChatMessage(
        id=f"m_{role}_{text[:8]}", session_id="s", role=role, text=text,
        created_at=datetime.now(tz=UTC),
    )


def test_history_pairs_each_question_with_its_answer() -> None:
    messages = [_msg("user", "first"), _msg("assistant", "one")]

    assert chat_history(messages, 2048) == [("user", "first"), ("assistant", "one")]


def test_an_unanswered_question_is_left_out_of_the_history() -> None:
    messages = [
        _msg("user", "first"), _msg("assistant", "one"),
        _msg("user", "failed"),
        _msg("user", "third"), _msg("assistant", ""),
        _msg("user", "fourth"), _msg("assistant", "four"),
    ]

    assert chat_history(messages, 2048) == [
        ("user", "first"), ("assistant", "one"), ("user", "fourth"), ("assistant", "four"),
    ]


def test_the_oldest_turns_go_first_when_the_budget_is_spent() -> None:
    messages = [
        _msg("user", "a" * 400), _msg("assistant", "b" * 400),
        _msg("user", "new"), _msg("assistant", "reply"),
    ]

    assert chat_history(messages, 100) == [("user", "new"), ("assistant", "reply")]


def test_a_follow_up_is_searched_with_the_previous_question() -> None:
    history = [("user", "How does reranking work?"), ("assistant", "It rescores.")]

    assert retrieval_query("And on a laptop?", history) == (
        "How does reranking work?\nAnd on a laptop?"
    )
    assert retrieval_query("Fresh question", []) == "Fresh question"


class _HistorySpy:
    """Answers with a fixed sentence and records the history it was handed."""

    def __init__(self) -> None:
        self.histories: list[list[tuple[str, str]]] = []

    async def stream(
        self, question, chunks, directives, *, system_prompt=None, max_tokens=None, history=()
    ):
        self.histories.append(list(history))
        yield "Answer."


def test_a_follow_up_in_the_same_chat_carries_the_earlier_turn(
    client: TestClient, read_events, monkeypatch
) -> None:
    spy = _HistorySpy()
    client.app.state.answerer = spy
    store = client.app.state.store
    searched: list[str] = []
    original = store.retriever.asearch

    async def recording_search(query, **kwargs):
        searched.append(query)
        return await original(query, **kwargs)

    monkeypatch.setattr(store.retriever, "asearch", recording_search)

    with client.stream("POST", "/query", json={"q": "How does reranking work?"}) as response:
        session_id = dict(read_events(response))["start"]["session_id"]
    with client.stream(
        "POST", "/query", json={"q": "And on a laptop?", "session_id": session_id}
    ) as response:
        read_events(response)

    assert spy.histories == [
        [],
        [("user", "How does reranking work?"), ("assistant", "Answer.")],
    ]
    assert searched == ["How does reranking work?", "How does reranking work?\nAnd on a laptop?"]
```

- [ ] **Step 7: Run them to verify they fail**

Run: `uv run pytest core/ragcore/tests/test_query.py -k "history or follow_up or unanswered or budget" -v`
Expected: FAIL with `ImportError: cannot import name 'chat_history'`

- [ ] **Step 8: Implement in `api/routes/query.py`**

Add below `tokens_per_second`:

```python
def chat_history(messages: list[ChatMessage], budget_tokens: int) -> list[tuple[str, str]]:
    """Earlier answered turns as (role, text), oldest first, newest kept within the budget.

    A question with no answer after it (failed or cancelled) is left out: providers
    reject two user turns in a row, and an empty assistant turn.
    """
    pairs = [
        (asked.text, answered.text)
        for asked, answered in zip(messages, messages[1:], strict=False)
        if asked.role == "user" and answered.role == "assistant" and answered.text.strip()
    ]
    kept: list[tuple[str, str]] = []
    spent = 0
    for question, answer in reversed(pairs):
        # ponytail: ~4 chars per token; count with the model's tokenizer if budgets run tight.
        spent += (len(question) + len(answer)) // 4
        if spent > budget_tokens:
            break
        kept.append((question, answer))
    return [turn for q, a in reversed(kept) for turn in (("user", q), ("assistant", a))]


def retrieval_query(question: str, history: list[tuple[str, str]]) -> str:
    """A follow-up ("and on a laptop?") searched alone finds nothing; prefix the last question.

    ponytail: concatenation, not an LLM rewrite. A follow-up that changes topic drags the
    previous question into the search; add a condenser call if eval shows it hurts.
    """
    previous = next((text for role, text in reversed(history) if role == "user"), None)
    return f"{previous}\n{question}" if previous else question
```

In `events()`, right after `filters = project_filters(store, session_id, payload.filters)`, add:

```python
        history = chat_history(
            store.messages.get(session_id, []), store.settings.retrieval.history_token_budget
        )
```

Change the retriever call's first argument from `question` to `retrieval_query(question, history)`. Add `history=history,` to the `answerer.stream(...)` call. The history must be read *before* `store.append_message(...)` stores the current question; this placement already guarantees it.

- [ ] **Step 9: Run the whole Python suite**

Run: `uv run pytest -q && uv run ruff check .`
Expected: all PASS. The fakes in `test_conversions.py` don't accept `history`, which is fine: the conversion route never passes it.

- [ ] **Step 10: Commit**

```bash
git add core/ragcore/src/ragcore/{llm.py,ports.py,backend.py,api/routes/query.py} core/ragcore/tests/{test_llm.py,test_query.py}
git commit -m "feat(chat): send earlier turns to the LLM and search follow-ups in context"
```

---

### Task 2: OCR for scanned PDF pages

Pages whose text layer is under `OCR_THRESHOLD` (20 chars) currently stay empty, and a fully scanned PDF is marked `skipped`. This task runs RapidOCR on those pages. `rapidocr` 3.x ships its PP-OCRv6 det/rec ONNX models inside the wheel, so nothing is downloaded at runtime. It needs `onnxruntime` installed separately.

**Files:**
- Create: `core/ragcore/src/ragcore/ingest/ocr.py`
- Modify: `core/ragcore/src/ragcore/ingest/parse.py` (`_pdf`)
- Modify: `core/ragcore/src/ragcore/store/real.py` (`NO_TEXT_LAYER` text)
- Modify: `core/ragcore/pyproject.toml` (via `uv add`), `scripts/build-sidecar.sh`
- Test: `core/ragcore/tests/test_ingest_parse.py`, `core/ragcore/tests/test_real_store.py`

**Interfaces:**
- Produces: `ocr.read_page(page: pypdfium2.PdfPage) -> str`, the recognised lines joined by `\n` (`""` if none).

- [ ] **Step 1: Add the dependencies**

Run: `uv add --package ragcore "rapidocr>=3.9.2" "onnxruntime>=1.22"`
Expected: `core/ragcore/pyproject.toml` lists both; `uv.lock` updated.

- [ ] **Step 2: Write the failing tests** (append to `tests/test_ingest_parse.py`)

```python
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
```

- [ ] **Step 3: Run them to verify they fail**

Run: `uv run pytest core/ragcore/tests/test_ingest_parse.py -k ocr -v`
Expected: FAIL with `ImportError: cannot import name 'ocr' from 'ragcore.ingest'`

- [ ] **Step 4: Create `ingest/ocr.py`**

```python
"""Text from page images, for PDF pages that have no text layer.

RapidOCR ships its detection and recognition models inside the wheel, so this
never downloads anything. The engine (onnxruntime + OpenCV) is heavy to import
and load, so it is built on the first scanned page, once per process.
"""

from __future__ import annotations

from functools import cache

# 216 dpi: small print stays legible to the recogniser; RapidOCR caps the long side at 2000 px.
RENDER_SCALE = 3


@cache
def _engine():
    from rapidocr import RapidOCR

    return RapidOCR(params={"Global.log_level": "warning"})


def read_page(page) -> str:
    """OCR one pypdfium2 page; lines joined in the order the recogniser reports them.

    ponytail: the bundled recogniser is the default PP-OCRv6 one; if accented Latin
    text (à, è, ß) comes back mangled, switch ``Rec.lang_type`` to the latin model.
    """
    bitmap = page.render(scale=RENDER_SCALE)
    try:
        result = _engine()(bitmap.to_numpy())
    finally:
        bitmap.close()
    return "\n".join(result.txts or ())
```

- [ ] **Step 5: Call it from `_pdf` in `parse.py`**

Add `from ragcore.ingest import ocr` to the imports. Import the module, not the function, so tests can monkeypatch `ocr.read_page`. In `_pdf`, replace the block that extracts the text so OCR runs while the page is still open:

```python
            try:
                text_page = page.get_textpage()
                text = _clean_pdf_text(text_page.get_text_range())
                if len(text) < OCR_THRESHOLD:
                    # ponytail: serial, ~0.5-1 s per scanned page; a 300-page scan takes minutes.
                    text = _clean_pdf_text(ocr.read_page(page)) or text
            finally:
```

Keep `needs_ocr = needs_ocr or len(text) < OCR_THRESHOLD` where it is (after the block): a page OCR couldn't read stays flagged. Update the `_pdf` docstring's second paragraph to: `A page whose text layer is under OCR_THRESHOLD characters is run through OCR; if that still finds too little, the page is kept (later page numbers stay true) and the document is flagged.` Update the module docstring sentence about `needs_ocr` the same way.

- [ ] **Step 6: Run the parse tests**

Run: `uv run pytest core/ragcore/tests/test_ingest_parse.py -v`
Expected: all PASS, including the two existing OCR-flag tests (a blank page OCRs to `""`, and `"12"` OCRs to at most `"12"`). The first real-engine test takes a few seconds.

- [ ] **Step 7: Write the failing store test** (append to `tests/test_real_store.py`)

```python
async def test_a_scan_ocr_reads_is_indexed(tmp_path: Path, text_pdf, monkeypatch) -> None:
    from ragcore.ingest import ocr

    monkeypatch.setattr(ocr, "read_page", lambda page: "Rerankers rescore the fused candidates.")
    docs = tmp_path / "scans"
    docs.mkdir()
    (docs / "scan.pdf").write_bytes(text_pdf(""))
    store = build(tmp_path)
    source = add(store, docs)

    [document] = await store.ingest_source_async(source.id)

    assert document.status == "indexed"
    assert document.n_chunks >= 1


async def test_a_scan_ocr_cannot_read_is_skipped_with_a_reason(
    tmp_path: Path, text_pdf, monkeypatch
) -> None:
    from ragcore.ingest import ocr

    monkeypatch.setattr(ocr, "read_page", lambda page: "")
    docs = tmp_path / "scans"
    docs.mkdir()
    (docs / "blank.pdf").write_bytes(text_pdf(""))
    store = build(tmp_path)
    source = add(store, docs)

    [document] = await store.ingest_source_async(source.id)

    assert document.status == "skipped"
    assert document.error == "no text found, not even by OCR"
```

- [ ] **Step 8: Run to verify the second fails**, then change `real.py`: `NO_TEXT_LAYER = "no text found, not even by OCR"`.

Run: `uv run pytest core/ragcore/tests/test_real_store.py -k scan -v`
Expected after the change: PASS. Then grep the UI for the old text: `git grep -n "OCR not available" apps/desktop/src`. If any locale or component matches, update it to the new meaning.

- [ ] **Step 9: Bundle the OCR engine**

In `scripts/build-sidecar.sh`, add two lines after `--collect-all pypdfium2 \`:

```bash
  --collect-all rapidocr \
  --collect-all onnxruntime \
```

- [ ] **Step 10: Full suite, license check, commit**

Run: `uv run pytest -q && uv run ruff check .`
Expected: PASS

```bash
git add core/ragcore/pyproject.toml uv.lock scripts/build-sidecar.sh core/ragcore/src/ragcore/ingest/{ocr.py,parse.py} core/ragcore/src/ragcore/store/real.py core/ragcore/tests/{test_ingest_parse.py,test_real_store.py}
git commit -m "feat(ingest): OCR scanned PDF pages with RapidOCR"
```

---

### Task 3: DOCX parser

PPTX is already parsed with `zipfile` + `ElementTree`. DOCX gets the same treatment: no `python-docx`. Pages split at Heading 1/2, the way Markdown splits at `##`. Heading detection uses the style *name* from `styles.xml`, because localized Word translates the style *id* (`Titolo1`) but not the name (`heading 1`).

**Files:**
- Modify: `core/ragcore/src/ragcore/ingest/parse.py` (new `_docx`, register `.docx`)
- Modify: `core/ragcore/src/ragcore/ingest/walk.py` (`MIME_BY_EXT`, skip `~$` lock files)
- Test: `core/ragcore/tests/test_ingest_parse.py`, `core/ragcore/tests/test_ingest_walk.py`

**Interfaces:**
- Produces: `.docx` in `SUPPORTED_SUFFIXES`; `walk._is_hidden` also skips names starting with `~$`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_ingest_parse.py`)

```python
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
```

And append to `tests/test_ingest_walk.py` (it already imports `walk_source`; check the import line):

```python
def test_office_lock_files_are_skipped(tmp_path: Path) -> None:
    (tmp_path / "report.docx").write_bytes(b"x")
    (tmp_path / "~$report.docx").write_bytes(b"x")

    found = walk_source(tmp_path, include_globs=["**/*"], exclude_globs=[], max_file_mb=100)

    assert [f.path.name for f in found] == ["report.docx"]
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest core/ragcore/tests/test_ingest_parse.py core/ragcore/tests/test_ingest_walk.py -k "docx or lock" -v`
Expected: FAIL with `UnsupportedFormat: no parser for '.docx'`, and the lock file listed.

- [ ] **Step 3: Implement `_docx` in `parse.py`** (add below `_pptx`)

```python
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
```

Register it: add `".docx": _docx,` to `_PARSERS`.

- [ ] **Step 4: Update `walk.py`**

Add to `MIME_BY_EXT`:

```python
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
```

Change `_is_hidden`:

```python
def _is_hidden(relative: Path) -> bool:
    """True if any path segment starts with a dot, or is an Office lock file (``~$name``)."""
    return any(part.startswith((".", "~$")) for part in relative.parts)
```

Add one bullet to the module docstring under "Hidden files": `Office lock files (~$report.docx, written while a document is open) are skipped the same way.`

- [ ] **Step 5: Run the tests**

Run: `uv run pytest core/ragcore/tests/test_ingest_parse.py core/ragcore/tests/test_ingest_walk.py -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add core/ragcore/src/ragcore/ingest/{parse.py,walk.py} core/ragcore/tests/{test_ingest_parse.py,test_ingest_walk.py}
git commit -m "feat(ingest): parse DOCX with the stdlib; skip Office lock files"
```

---

### Task 4: HTML parser

Uses stdlib `html.parser`, not `selectolax`. Visible text only: `script`, `style`, `noscript`, `template` and `svg` are dropped. Pages split at `<h1>`/`<h2>`. The title comes from `<title>`, else the first heading, else the file stem.

**Files:**
- Modify: `core/ragcore/src/ragcore/ingest/parse.py` (new `_HtmlText`, `_html`; register `.html`, `.htm`)
- Modify: `core/ragcore/src/ragcore/ingest/walk.py` (`MIME_BY_EXT`)
- Test: `core/ragcore/tests/test_ingest_parse.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_html_keeps_visible_text_and_splits_on_headings(tmp_path: Path) -> None:
    source = tmp_path / "page.html"
    source.write_text(
        "<html><head><title>Hybrid Search</title><style>p {color: red}</style>"
        "<script>var x = 1;</script></head><body>"
        "<p>Intro &amp; overview.</p>"
        "<h2>Fusion</h2><p>RRF merges   two\nrankings.</p><ul><li>k = 60</li></ul>"
        "<h2>Filters</h2><p>Filter before fusion.</p>"
        "</body></html>"
    )

    parsed = parse(source)

    assert parsed.title == "Hybrid Search"
    assert [p.section_path for p in parsed.pages] == [
        "Hybrid Search",
        "Hybrid Search > Fusion",
        "Hybrid Search > Filters",
    ]
    assert parsed.pages[0].text == "Intro & overview."
    assert parsed.pages[1].text == "RRF merges two rankings.\nk = 60"
    assert "var x" not in " ".join(p.text for p in parsed.pages)


def test_html_without_a_title_falls_back_to_the_first_heading(tmp_path: Path) -> None:
    source = tmp_path / "notes.htm"
    source.write_text("<h1>Release notes</h1><p>Fixed the converter.</p>")

    parsed = parse(source)

    assert parsed.title == "Release notes"
    assert parsed.pages[-1].text == "Fixed the converter."
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest core/ragcore/tests/test_ingest_parse.py -k html -v`
Expected: FAIL with `UnsupportedFormat`

- [ ] **Step 3: Implement** (add `from html.parser import HTMLParser` to the imports; code below `_docx`)

```python
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
```

Register `".html": _html,` and `".htm": _html,` in `_PARSERS`. Add to `MIME_BY_EXT`: `".html": "text/html",` and `".htm": "text/html",`.

Note on the second test: `<h1>Release notes</h1>` opens a section named "Release notes", and the title also falls back to it, so its section_path is `"Release notes > Release notes"`. The test checks only `title` and the last page's text, which is deliberate: section naming for a lone `<h1>` is cosmetic.

- [ ] **Step 4: Run tests, commit**

Run: `uv run pytest core/ragcore/tests/test_ingest_parse.py -v && uv run ruff check .`
Expected: PASS

```bash
git add core/ragcore/src/ragcore/ingest/{parse.py,walk.py} core/ragcore/tests/test_ingest_parse.py
git commit -m "feat(ingest): parse HTML with the stdlib parser"
```

---

### Task 5: CSV parser

"CSV as row groups" (spec, Phase 1). Each page holds `CSV_ROWS_PER_PAGE` rows, and every row is rendered as `column: value; column: value`, so each chunk keeps its column names. The delimiter is sniffed, because Excel on an Italian/German/French machine writes `;`. The UTF-8 BOM is stripped.

**Files:**
- Modify: `core/ragcore/src/ragcore/ingest/parse.py` (new `_csv`; register `.csv`)
- Modify: `core/ragcore/src/ragcore/ingest/walk.py` (`MIME_BY_EXT`)
- Modify: `apps/desktop/src/features/library/add-source-dialog.tsx:26` (`DEFAULT_INCLUDE`)
- Test: `core/ragcore/tests/test_ingest_parse.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_csv_rows_keep_their_column_names(tmp_path: Path) -> None:
    source = tmp_path / "models.csv"
    source.write_text("name,dim\nEmbeddingGemma,768\nQwen3-Embedding,1024\n")

    parsed = parse(source)

    assert parsed.title == "models"
    assert parsed.pages[0].section_path == "models > Rows 1-2"
    assert parsed.pages[0].text == "name: EmbeddingGemma; dim: 768\nname: Qwen3-Embedding; dim: 1024"


def test_csv_reads_an_excel_semicolon_export_with_a_bom(tmp_path: Path) -> None:
    source = tmp_path / "fatture.csv"
    source.write_bytes("﻿cliente;importo\nRossi;1.200,50\n".encode())

    parsed = parse(source)

    assert parsed.pages[0].text == "cliente: Rossi; importo: 1.200,50"


def test_csv_pages_hold_a_fixed_number_of_rows(tmp_path: Path) -> None:
    from ragcore.ingest.parse import CSV_ROWS_PER_PAGE

    source = tmp_path / "big.csv"
    rows = "\n".join(f"{i},x" for i in range(CSV_ROWS_PER_PAGE + 1))
    source.write_text(f"n,v\n{rows}\n")

    parsed = parse(source)

    assert [p.section_path for p in parsed.pages] == [
        f"big > Rows 1-{CSV_ROWS_PER_PAGE}",
        f"big > Rows {CSV_ROWS_PER_PAGE + 1}-{CSV_ROWS_PER_PAGE + 1}",
    ]


def test_a_header_only_csv_has_no_pages(tmp_path: Path) -> None:
    source = tmp_path / "empty.csv"
    source.write_text("a,b\n")

    assert parse(source).pages == []
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest core/ragcore/tests/test_ingest_parse.py -k csv -v`
Expected: FAIL with `UnsupportedFormat`

- [ ] **Step 3: Implement** (add `import csv` and `import io` to the imports)

```python
CSV_ROWS_PER_PAGE = 50


def _csv(path: Path) -> ParsedDoc:
    """Rows as ``column: value`` lines, CSV_ROWS_PER_PAGE rows per page.

    Every row repeats its column names, so a chunk cut from the middle of the
    file still says what each value is.
    """
    # ponytail: reads the whole file; max_file_mb (default 100) bounds it.
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
    except csv.Error:  # one column, or too irregular to tell
        dialect = csv.excel
    rows = [row for row in csv.reader(io.StringIO(text), dialect) if any(c.strip() for c in row)]
    if len(rows) < 2:
        return ParsedDoc(title=path.stem, pages=[])
    header, body = [h.strip() for h in rows[0]], rows[1:]
    sections = []
    for start in range(0, len(body), CSV_ROWS_PER_PAGE):
        group = body[start : start + CSV_ROWS_PER_PAGE]
        lines = [
            "; ".join(f"{h}: {v.strip()}" for h, v in zip(header, row, strict=False) if v.strip())
            for row in group
        ]
        sections.append((f"{path.stem} > Rows {start + 1}-{start + len(group)}", "\n".join(lines)))
    return ParsedDoc(title=path.stem, pages=_paginate(sections))
```

Register `".csv": _csv,`. Add to `MIME_BY_EXT`: `".csv": "text/csv",`.

- [ ] **Step 4: Update the UI default globs** in `add-source-dialog.tsx:26`:

```ts
const DEFAULT_INCLUDE =
  "**/*.md, **/*.txt, **/*.pdf, **/*.pptx, **/*.docx, **/*.html, **/*.htm, **/*.csv";
```

- [ ] **Step 5: Run tests, commit**

Run: `uv run pytest -q && uv run ruff check . && (cd apps/desktop && bun run typecheck)`
Expected: PASS

```bash
git add core/ragcore/src/ragcore/ingest/{parse.py,walk.py} core/ragcore/tests/test_ingest_parse.py apps/desktop/src/features/library/add-source-dialog.tsx
git commit -m "feat(ingest): parse CSV as row groups with sniffed delimiters"
```

---

### Task 6: Per-document language detection

`Document.lang`, `QueryFilters.langs` and the filter in `stub/retrieval.allowed` already exist. Only detection is missing. `py3langid` (BSD-3, numpy only, ~4.5 MB model) runs once per parsed document.

**Files:**
- Modify: `core/ragcore/src/ragcore/ingest/parse.py` (`ParsedDoc.lang`, `detect_language`, `parse()`)
- Modify: `core/ragcore/src/ragcore/store/real.py` (`_document` sets `lang`)
- Modify: `core/ragcore/pyproject.toml` (via `uv add`), `scripts/build-sidecar.sh`
- Test: `core/ragcore/tests/test_ingest_parse.py`, `core/ragcore/tests/test_real_store.py`

**Interfaces:**
- Produces: `ParsedDoc.lang: str | None` (ISO 639-1), `parse.detect_language(text: str) -> str | None`.

- [ ] **Step 1: Add the dependency**

Run: `uv add --package ragcore "py3langid>=0.4.0"`

- [ ] **Step 2: Write the failing tests** (append to `tests/test_ingest_parse.py`)

```python
ITALIAN = (
    "Il recupero dei passaggi combina la ricerca semantica con quella per parole chiave. "
    "I risultati vengono poi riordinati da un modello che valuta quanto ogni passaggio "
    "risponde alla domanda dell'utente, e solo i migliori arrivano al modello linguistico."
)
ENGLISH = (
    "Passage retrieval combines semantic search with keyword search. The results are then "
    "reordered by a model that judges how well each passage answers the user's question, "
    "and only the best of them reach the language model that writes the answer."
)


def test_language_is_detected_per_document() -> None:
    from ragcore.ingest.parse import detect_language

    assert detect_language(ITALIAN) == "it"
    assert detect_language(ENGLISH) == "en"


def test_too_little_text_has_no_language() -> None:
    from ragcore.ingest.parse import detect_language

    assert detect_language("ok") is None


def test_parse_fills_the_language(tmp_path: Path) -> None:
    source = tmp_path / "nota.md"
    source.write_text(f"# Nota\n\n{ITALIAN}\n")

    assert parse(source).lang == "it"
```

Append to `tests/test_real_store.py`:

```python
async def test_indexed_documents_carry_their_language(tmp_path: Path) -> None:
    docs = tmp_path / "it"
    docs.mkdir()
    (docs / "nota.md").write_text(
        "# Nota\n\nIl recupero dei passaggi combina la ricerca semantica con quella per "
        "parole chiave. I risultati vengono poi riordinati da un modello che valuta quanto "
        "ogni passaggio risponde alla domanda dell'utente, e solo i migliori arrivano al "
        "modello linguistico che scrive la risposta.\n"
    )
    store = build(tmp_path)
    source = add(store, docs)

    [document] = await store.ingest_source_async(source.id)

    assert document.lang == "it"
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest core/ragcore/tests -k "language" -v`
Expected: FAIL with `ImportError: cannot import name 'detect_language'`

- [ ] **Step 4: Implement in `parse.py`**

Add the field to `ParsedDoc`: `lang: str | None = None`. Add:

```python
LANG_MIN_CHARS = 200  # below this py3langid guesses; no answer beats a wrong one
LANG_SAMPLE_CHARS = 5000


def detect_language(text: str) -> str | None:
    """ISO 639-1 code for ``text``, or None when there is too little to tell."""
    sample = text[:LANG_SAMPLE_CHARS]
    if len(sample.strip()) < LANG_MIN_CHARS:
        return None
    import py3langid  # loads its model on first use

    return py3langid.classify(sample)[0]
```

Change `parse()` to:

```python
def parse(path: Path) -> ParsedDoc:
    """Parse ``path`` into a ``ParsedDoc``, dispatching on its lowercased suffix."""
    handler = _PARSERS.get(path.suffix.lower())
    if handler is None:
        raise UnsupportedFormat(f"no parser for {path.suffix!r}")
    doc = handler(path)
    doc.lang = detect_language(" ".join(page.text for page in doc.pages))
    return doc
```

In `real.py` `_document`, add `lang=p.parsed.lang if p.parsed else None,` to the `Document(...)` call. Documents indexed before this change keep `lang=None` until they change or the index is rebuilt. That's acceptable; say so in the commit body.

- [ ] **Step 5: Bundle the model**: in `scripts/build-sidecar.sh`, add `--collect-data py3langid \` after the onnxruntime line.

- [ ] **Step 6: Run tests, commit**

Run: `uv run pytest -q && uv run ruff check .`
Expected: PASS

```bash
git add core/ragcore/pyproject.toml uv.lock scripts/build-sidecar.sh core/ragcore/src/ragcore/ingest/parse.py core/ragcore/src/ragcore/store/real.py core/ragcore/tests/{test_ingest_parse.py,test_real_store.py}
git commit -m "feat(ingest): detect each document's language

Documents indexed earlier keep lang=None until they change or the index is rebuilt."
```

---

### Task 7: Folder watcher

`SourceCreate.watch` (default `True`) and the UI's "Watch" switch already exist, but nothing reads them. A `watchdog` observer marks a source dirty on relevant events. A 1 s asyncio loop rescans each dirty source once nothing has changed for `DEBOUNCE_S`, so copying in 500 files costs one rescan. The rescan is the existing `ingest_source_async`: unchanged files are skipped by sha, and removed files leave the index.

**Files:**
- Create: `core/ragcore/src/ragcore/watch.py`
- Modify: `core/ragcore/src/ragcore/ingest/walk.py` (rename `_is_hidden` → `is_hidden`, public)
- Modify: `core/ragcore/src/ragcore/backend.py` (`Backend.watcher`, built under the same gate as `servers`)
- Modify: `core/ragcore/src/ragcore/api/app.py` (start/stop in `lifespan`)
- Modify: `apps/desktop/src/features/library/library-view.tsx:60` (poll documents so watcher scans show up)
- Modify: `core/ragcore/pyproject.toml` (via `uv add`), `scripts/build-sidecar.sh`
- Test: `core/ragcore/tests/test_watch.py`

**Interfaces:**
- Consumes: `walk.is_hidden(relative: Path) -> bool` (Task 3 added the `~$` rule), `parse.SUPPORTED_SUFFIXES`.
- Produces: `SourceWatcher(store)` with `start()`, `stop()`, `sync()`, `touch(source_id)`, `async tick(now: float | None = None)`, `async run()`, and attributes `pending: dict[str, float]` and `watched: dict[str, tuple[str, ObservedWatch]]`; `DEBOUNCE_S`.

- [ ] **Step 1: Add the dependency**

Run: `uv add --package ragcore "watchdog>=6.0.0"`

- [ ] **Step 2: Write the failing tests** (`core/ragcore/tests/test_watch.py`)

```python
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
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest core/ragcore/tests/test_watch.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ragcore.watch'`

- [ ] **Step 4: Make `is_hidden` public**: in `walk.py`, rename `_is_hidden` to `is_hidden` (definition and its one call site in `walk_source`). Run `git grep -n "_is_hidden"` to confirm nothing else uses the old name.

- [ ] **Step 5: Create `watch.py`**

```python
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
        if is_hidden(relative):
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
```

- [ ] **Step 6: Run the watcher tests**

Run: `uv run pytest core/ragcore/tests/test_watch.py -v`
Expected: PASS. If `test_hidden_and_unsupported_changes_are_ignored` fails on macOS, print the events in `on_any_event` to see which one slipped through. Don't widen the sleep.

- [ ] **Step 7: Wire it into the app**

`backend.py`: add the field `watcher: object | None = None` to `Backend`. In `build_backend`, after `servers` is set:

```python
    # Same gate as the model servers: the real backend with real models. Tests drive
    # SourceWatcher directly instead of racing a background rescan.
    watcher = None
    if servers is not None:
        from ragcore.watch import SourceWatcher

        watcher = SourceWatcher(store)
```

Pass `watcher=watcher` to `Backend(...)`.

`api/app.py` `lifespan`: after `await backend.servers.start()`, add

```python
        watch_task = asyncio.create_task(backend.watcher.run()) if backend.watcher else None
```

and in the `finally:` block, first thing:

```python
            if watch_task is not None:
                watch_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await watch_task
```

Add `import asyncio` and `import contextlib` to `app.py` if they are missing.

- [ ] **Step 8: Let the library show watcher scans**: in `library-view.tsx`, add `refetchInterval: 5000,` to the `documents` `useQuery({...})` options (line ~60), next to `queryFn`.

- [ ] **Step 9: Bundle watchdog's platform observers**: in `scripts/build-sidecar.sh`, add `--collect-submodules watchdog \`. The observer class is picked at import time per OS, and PyInstaller can't see that.

- [ ] **Step 10: Full suite, commit**

Run: `uv run pytest -q && uv run ruff check . && (cd apps/desktop && bun run typecheck && bun run test)`
Expected: PASS

```bash
git add core/ragcore/pyproject.toml uv.lock scripts/build-sidecar.sh core/ragcore/src/ragcore/{watch.py,backend.py,api/app.py,ingest/walk.py} core/ragcore/tests/test_watch.py apps/desktop/src/features/library/library-view.tsx
git commit -m "feat(library): rescan watched folders shortly after their files change"
```

---

### Task 8: Real eval harness (`ragcore eval` + route)

Today `/eval/run` builds questions from the first sentence of each chunk, computes "nDCG" as `1/rank`, and picks the baseline with `random.uniform`. This task replaces that with a golden set and real metrics, shared by the route and a new CLI command.

Golden format, one JSON object per line: `{"q": "...", "expected": ["reranking.md#The minimum-score gate"], "type": "local"}`. An expected entry matches a retrieved chunk from a file with that *name* whose section heading starts with the text after `#`; without `#`, any chunk of the file matches. File names are used instead of doc ids because ids hash absolute paths, which differ on every machine. The base set is the 20 fixture-corpus queries already validated in `scripts/eval/model_eval.py`. Its README/plan queries are left out because those files change with every edit.

**Files:**
- Create: `core/ragcore/src/ragcore/eval.py`
- Create: `eval/golden/base.jsonl`
- Modify: `core/ragcore/src/ragcore/api/routes/evals.py` (rewrite)
- Modify: `core/ragcore/src/ragcore/ports.py`, `core/ragcore/src/ragcore/stub/store.py` (delete `eval_sets`)
- Modify: `core/ragcore/src/ragcore/cli.py` (new `eval` command)
- Test: `core/ragcore/tests/test_eval_harness.py` (new), `core/ragcore/tests/test_evals.py`, `core/ragcore/tests/test_cli.py`

**Interfaces:**
- Produces (in `ragcore.eval`): `K = 6`; `MAX_RECALL_DROP = 0.05`; `GOLDEN_DIR: Path | None`; `GoldenQuestion(q: str, expected: tuple[str, ...], type: str = "local")`; `load_golden(path: Path) -> list[GoldenQuestion]`; `golden_sets(data_dir: Path) -> list[Path]`; `matches(chunk, path: str | None, expected: str) -> bool`; `first_hit(chunks, paths: dict[str, str], expected) -> int | None`; `async score_questions(retriever, questions, *, settings, doc_meta, paths) -> AsyncIterator[EvalQuestionResult]`; `summarize(set_name: str, results: list[EvalQuestionResult], baseline_recall: float | None = None) -> EvalMetrics`; `baseline_recall(path: Path) -> float | None`.
- Baseline file convention: `<set>.baseline.json` next to `<set>.jsonl`, holding an `EvalResult` JSON (what `ragcore eval --report` writes).

- [ ] **Step 1: Create `eval/golden/base.jsonl`**

```jsonl
{"q": "why can't I just change the embedding model in an app update", "expected": ["embedding-models.md#Why the embedder"]}
{"q": "how much storage does a million 1024-dimension vectors take", "expected": ["embedding-models.md#Choosing a dimension"]}
{"q": "what does the k constant do in reciprocal rank fusion", "expected": ["hybrid-search.md#Reciprocal rank fusion"]}
{"q": "why keyword search alone misses paraphrased questions", "expected": ["hybrid-search.md#Two kinds of miss"]}
{"q": "make the system say it found nothing instead of returning irrelevant passages", "expected": ["reranking.md#The minimum-score gate"]}
{"q": "can I replace the reranker without rebuilding the index", "expected": ["reranking.md#Swapping rerankers"]}
{"q": "how many candidates to send to the cross-encoder on a laptop without a GPU", "expected": ["reranking.md#Candidate count"]}
{"q": "problem with cutting documents every N tokens", "expected": ["chunking-strategies.md#Why fixed-size"]}
{"q": "send the bigger surrounding passage to the LLM but search on small ones", "expected": ["chunking-strategies.md#Parent and child"]}
{"q": "how many golden questions do I need to start evaluating", "expected": ["evaluation-runbook.md#Build the golden set"]}
{"q": "stop retrieval quality silently regressing in CI", "expected": ["evaluation-runbook.md#Regressions are the point"]}
{"q": "what to do with citations that don't match retrieved passages", "expected": ["retrieval-augmented-generation.md#Grounding"]}
{"q": "questions about the main themes across all reports", "expected": ["retrieval-augmented-generation.md#When retrieval is the wrong"], "type": "global"}
{"q": "should language filters run before or after retrieval", "expected": ["hybrid-search.md#Filters belong before fusion"]}
{"q": "perché non posso cambiare il modello di embedding senza reindicizzare", "expected": ["embedding-models.md#Why the embedder"]}
{"q": "quanti candidati passare al reranker su un computer senza GPU", "expected": ["reranking.md#Candidate count"]}
{"q": "cosa succede quando nessun passaggio è abbastanza rilevante", "expected": ["reranking.md#The minimum-score gate"]}
{"q": "dividere i documenti seguendo titoli e struttura invece di token fissi", "expected": ["chunking-strategies.md#Structure first"]}
{"q": "quali metriche usare per valutare il recupero dei passaggi", "expected": ["evaluation-runbook.md#Metrics that earn"]}
{"q": "come si scartano le citazioni non valide nelle risposte", "expected": ["retrieval-augmented-generation.md#Grounding"]}
```

- [ ] **Step 2: Write the failing harness tests** (`core/ragcore/tests/test_eval_harness.py`)

```python
"""The eval harness: golden-set parsing, hit matching and metric math."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from ragcore.api.schemas import EvalQuestionResult, RetrievedChunk
from ragcore.eval import GOLDEN_DIR, first_hit, load_golden, matches, summarize
from ragcore.ingest.parse import parse

FIXTURE_DOCS = Path(__file__).resolve().parents[3] / "fixtures" / "docs"


def chunk(doc_id: str, section_path: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{doc_id}:1", doc_id=doc_id, doc_title="T", page_start=1, page_end=1,
        section_path=section_path, text="x",
    )


def test_an_expected_section_matches_by_file_name_and_heading_prefix() -> None:
    c = chunk("d1", "Reranking with Cross-Encoders > The minimum-score gate")

    assert matches(c, "/any/where/reranking.md", "reranking.md#The minimum-score")
    assert matches(c, "/any/where/reranking.md", "reranking.md")
    assert not matches(c, "/any/where/reranking.md", "reranking.md#Candidate count")
    assert not matches(c, "/any/where/hybrid-search.md", "reranking.md")
    assert not matches(c, None, "reranking.md")


def test_first_hit_is_the_rank_of_the_first_matching_chunk() -> None:
    chunks = [chunk("a", "A > One"), chunk("b", "B > Two"), chunk("b", "B > Three")]
    paths = {"a": "/x/a.md", "b": "/x/b.md"}

    assert first_hit(chunks, paths, ("b.md#Three",)) == 3
    assert first_hit(chunks, paths, ("b.md",)) == 2
    assert first_hit(chunks, paths, ("missing.md",)) is None


def test_a_malformed_golden_line_names_its_line_number(tmp_path: Path) -> None:
    golden = tmp_path / "bad.jsonl"
    golden.write_text('{"q": "fine", "expected": ["a.md"]}\n{"q": "no expected"}\n')

    with pytest.raises(ValueError, match="bad.jsonl:2"):
        load_golden(golden)


def test_the_metrics_follow_their_definitions() -> None:
    results = [
        EvalQuestionResult(q="a", type="local", hit=True, rank=1, ndcg=1.0),
        EvalQuestionResult(q="b", type="local", hit=True, rank=3, ndcg=1 / math.log2(4)),
        EvalQuestionResult(q="c", type="local", hit=False, rank=None, ndcg=0.0),
    ]

    metrics = summarize("t", results, baseline_recall=0.9)

    assert metrics.recall_at_1 == round(1 / 3, 4)
    assert metrics.recall_at_6 == round(2 / 3, 4)
    assert metrics.mrr == round((1 + 1 / 3) / 3, 4)
    assert metrics.ndcg_at_6 == round(1.5 / 3, 4)
    assert metrics.baseline_recall_at_6 == 0.9


def test_every_base_question_points_at_a_real_fixture_section() -> None:
    """Guards against golden rot: renaming a fixture heading must fail here, not in CI eval."""
    headings = {
        path.name: [page.section_path.split(" > ")[-1] for page in parse(path).pages]
        for path in FIXTURE_DOCS.glob("*.md")
    }
    questions = load_golden(GOLDEN_DIR / "base.jsonl")

    assert len(questions) == 20
    for question in questions:
        for expected in question.expected:
            name, _, section = expected.partition("#")
            assert name in headings, expected
            assert any(h.startswith(section) for h in headings[name]), expected
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest core/ragcore/tests/test_eval_harness.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ragcore.eval'`

- [ ] **Step 4: Create `eval.py`**

```python
"""Golden-set retrieval evaluation, shared by ``ragcore eval`` and the dev-only route.

A golden line is ``{"q": ..., "expected": ["file.md#Section prefix", ...], "type": "local"}``.
An expected entry matches a retrieved chunk from a file with that name whose section
heading starts with the text after ``#`` (no ``#``: any chunk of that file). File names,
not doc ids: ids hash absolute paths and differ on every machine.
"""

from __future__ import annotations

import json
import math
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from pathlib import Path

from ragcore.api.schemas import (
    EvalMetrics,
    EvalQuestionResult,
    QueryFilters,
    RetrievalSettings,
    RetrievedChunk,
    StageLatency,
)

K = 6
# The CI gate's tolerance: one question of the 20-question base set. (The roadmap's
# "2 points" assumed a 50-question set, where one question is 2 points.)
MAX_RECALL_DROP = 0.05
# The committed sets; exists in a dev checkout only, never in the packaged sidecar.
GOLDEN_DIR = next(
    (
        parent / "eval" / "golden"
        for parent in Path(__file__).resolve().parents
        if (parent / "eval" / "golden").is_dir()
    ),
    None,
)


@dataclass(frozen=True, slots=True)
class GoldenQuestion:
    q: str
    expected: tuple[str, ...]
    type: str = "local"


def load_golden(path: Path) -> list[GoldenQuestion]:
    questions: list[GoldenQuestion] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            questions.append(
                GoldenQuestion(
                    q=row["q"], expected=tuple(row["expected"]), type=row.get("type", "local")
                )
            )
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"{path.name}:{number}: {exc!r}") from exc
    return questions


def golden_sets(data_dir: Path) -> list[Path]:
    """The committed sets (dev checkout) plus private ones the user put in ``<data dir>/eval``."""
    dirs = [d for d in (GOLDEN_DIR, data_dir / "eval") if d is not None and d.is_dir()]
    return [path for d in dirs for path in sorted(d.glob("*.jsonl"))]


def matches(chunk: RetrievedChunk, path: str | None, expected: str) -> bool:
    name, _, section = expected.partition("#")
    if path is None or Path(path).name != name:
        return False
    return (chunk.section_path or "").split(" > ")[-1].startswith(section)


def first_hit(
    chunks: list[RetrievedChunk], paths: dict[str, str], expected: Iterable[str]
) -> int | None:
    expected = tuple(expected)
    return next(
        (
            rank
            for rank, chunk in enumerate(chunks[:K], start=1)
            if any(matches(chunk, paths.get(chunk.doc_id), e) for e in expected)
        ),
        None,
    )


async def score_questions(
    retriever,
    questions: list[GoldenQuestion],
    *,
    settings: RetrievalSettings,
    doc_meta: dict[str, dict],
    paths: dict[str, str],
) -> AsyncIterator[EvalQuestionResult]:
    for question in questions:
        chunks, latency, _ = await retriever.asearch(
            question.q, settings=settings, filters=QueryFilters(), doc_meta=doc_meta
        )
        rank = first_hit(chunks, paths, question.expected)
        yield EvalQuestionResult(
            q=question.q,
            type=question.type,
            hit=rank is not None,
            rank=rank,
            # One relevant target per question, so the ideal DCG is 1.
            ndcg=1 / math.log2(rank + 1) if rank else 0.0,
            latency_ms=round(latency.total_ms, 2),
        )


def summarize(
    set_name: str, results: list[EvalQuestionResult], baseline_recall: float | None = None
) -> EvalMetrics:
    n = max(len(results), 1)

    def percentile(p: float) -> StageLatency:
        values = sorted(r.latency_ms for r in results) or [0.0]
        return StageLatency(total_ms=round(values[min(int(len(values) * p), len(values) - 1)], 2))

    return EvalMetrics(
        set_name=set_name,
        n_questions=len(results),
        recall_at_1=round(sum(r.rank == 1 for r in results) / n, 4),
        recall_at_6=round(sum(r.hit for r in results) / n, 4),
        mrr=round(sum(1 / r.rank for r in results if r.rank) / n, 4),
        ndcg_at_6=round(sum(r.ndcg for r in results) / n, 4),
        latency_p50=percentile(0.5),
        latency_p95=percentile(0.95),
        baseline_recall_at_6=baseline_recall,
    )


def baseline_recall(path: Path) -> float | None:
    """recall@6 from a report ``ragcore eval --report`` wrote; None if there is none."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))["metrics"]["recall_at_6"]
    except (FileNotFoundError, KeyError, TypeError, json.JSONDecodeError):
        return None
```

- [ ] **Step 5: Run the harness tests**

Run: `uv run pytest core/ragcore/tests/test_eval_harness.py -v`
Expected: PASS

- [ ] **Step 6: Rewrite the route tests** (`tests/test_evals.py`). Replace `test_the_shipped_eval_sets_are_listed` and `test_an_empty_index_still_produces_a_result` with the versions below, and add the private-set test. Keep `test_an_unknown_set_is_a_404`, `test_eval_is_refused_in_a_production_build`, `test_progress_counts_up_to_the_question_total`, `test_the_metrics_are_internally_consistent`, `test_a_question_that_hit_carries_the_rank_it_hit_at` and `test_the_baseline_comparison_is_opt_out` unchanged.

```python
def test_the_committed_golden_sets_are_listed(client: TestClient) -> None:
    names = {s["name"]: s for s in client.get("/eval/sets").json()}

    assert names["base"]["n_questions"] == 20


def test_a_private_set_in_the_data_dir_is_listed_too(client: TestClient, tmp_path) -> None:
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "mine.jsonl").write_text('{"q": "x", "expected": ["a.md"]}\n')

    names = {s["name"] for s in client.get("/eval/sets").json()}

    assert {"base", "mine"} <= names


def test_an_empty_index_scores_zero(client: TestClient, read_events) -> None:
    client.post("/settings/wipe", json={"confirm": "DELETE"})

    with client.stream("POST", "/eval/run", json={"set_name": "base"}) as response:
        events = read_events(response)

    assert events[-1][0] == "result"
    metrics = events[-1][1]["metrics"]
    assert metrics["n_questions"] == 20
    assert metrics["recall_at_6"] == 0.0
```

- [ ] **Step 7: Rewrite `api/routes/evals.py`**

```python
"""Golden-set evaluation over whatever the index currently holds. Dev builds only."""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import APIRouter

from ragcore.api.deps import ConfigDep, StoreDep
from ragcore.api.errors import api_error
from ragcore.api.schemas import EvalResult, EvalRunRequest, EvalSet
from ragcore.api.sse import frame, sse_response
from ragcore.eval import baseline_recall, golden_sets, load_golden, score_questions, summarize

router = APIRouter(prefix="/eval", tags=["eval"])


@router.get("/sets", response_model=list[EvalSet])
def list_sets(config: ConfigDep) -> list[EvalSet]:
    return [
        EvalSet(name=path.stem, n_questions=len(load_golden(path)), description=str(path.parent))
        for path in golden_sets(config.data_dir)
    ]


@router.post("/run")
async def run_eval(payload: EvalRunRequest, store: StoreDep, config: ConfigDep):
    if not config.dev_mode:
        raise api_error(403, "eval_dev_only", "eval runs in dev builds only")
    path = next((p for p in golden_sets(config.data_dir) if p.stem == payload.set_name), None)
    if path is None:
        raise api_error(
            404, "eval_set_unknown", f"unknown eval set {payload.set_name!r}", name=payload.set_name
        )
    questions = load_golden(path)
    paths = {document.id: document.path for document in store.documents.values()}

    async def events() -> AsyncIterator[str]:
        results = []
        async for result in score_questions(
            store.retriever,
            questions,
            settings=store.settings.retrieval,
            doc_meta=store.doc_meta(),
            paths=paths,
        ):
            results.append(result)
            yield frame(
                "progress",
                {"completed": len(results), "total": len(questions), "question": result.q},
            )
        baseline = (
            baseline_recall(path.with_suffix(".baseline.json"))
            if payload.compare_baseline
            else None
        )
        metrics = summarize(path.stem, results, baseline)
        yield frame("result", EvalResult(metrics=metrics, questions=results))

    return sse_response(events())
```

Before replacing the file, check the current `router = APIRouter(...)` line at the top of `evals.py` and keep its exact `prefix`/`tags`. The UI calls `/eval/sets` and `/eval/run`.

- [ ] **Step 8: Delete the fake sets**: remove `eval_sets: list[EvalSet]` from `StorePort` in `ports.py` (and `EvalSet` from its import if now unused), and the `self.eval_sets = [...]` block in `stub/store.py` (and its `EvalSet` import if unused). Run `git grep -n "eval_sets" core` and expect no hits.

- [ ] **Step 9: Run the route tests**

Run: `uv run pytest core/ragcore/tests/test_evals.py core/ragcore/tests/test_eval_harness.py -v`
Expected: PASS

- [ ] **Step 10: Write the failing CLI tests** (append to `tests/test_cli.py`, reusing its `fake_models` fixture and `corpus()` helper)

```python
def _golden(tmp_path: Path) -> Path:
    golden = tmp_path / "set.jsonl"
    golden.write_text(
        '{"q": "cross encoders score query passage pairs", '
        '"expected": ["reranking.md#Cross encoders"]}\n'
    )
    return golden


def test_eval_scores_a_golden_set_against_a_corpus(
    tmp_path: Path, fake_models, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["eval", str(_golden(tmp_path)), "--corpus", str(corpus(tmp_path))]) == 0

    out = capsys.readouterr().out
    assert "hit " in out  # rank under the fake embedder is arbitrary; that it's in the top 6 isn't
    assert "recall@6 1.00" in out


def test_the_eval_gate_fails_below_the_baseline(tmp_path: Path, fake_models) -> None:
    golden = _golden(tmp_path)
    docs = corpus(tmp_path)
    baseline = golden.with_suffix(".baseline.json")

    baseline.write_text('{"metrics": {"recall_at_6": 2.0}}')
    assert main(["eval", str(golden), "--corpus", str(docs), "--gate"]) == 1

    baseline.write_text('{"metrics": {"recall_at_6": 0.0}}')
    assert main(["eval", str(golden), "--corpus", str(docs), "--gate"]) == 0


def test_the_eval_gate_needs_a_baseline(tmp_path: Path, fake_models) -> None:
    assert main(
        ["eval", str(_golden(tmp_path)), "--corpus", str(corpus(tmp_path)), "--gate"]
    ) == 1


def test_eval_report_is_a_baseline_the_gate_can_read(tmp_path: Path, fake_models) -> None:
    from ragcore.eval import baseline_recall

    golden = _golden(tmp_path)
    report = golden.with_suffix(".baseline.json")

    assert main(["eval", str(golden), "--corpus", str(corpus(tmp_path)), "--report", str(report)]) == 0
    assert baseline_recall(report) == 1.0


def test_eval_never_touches_the_users_library(
    tmp_path: Path, fake_models, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    main(["eval", str(_golden(tmp_path)), "--corpus", str(corpus(tmp_path))])

    assert not (tmp_path / "home" / ".ibid").exists()
```

- [ ] **Step 11: Run to verify failure**

Run: `uv run pytest core/ragcore/tests/test_cli.py -k eval -v`
Expected: FAIL. argparse exits with `invalid choice: 'eval'`.

- [ ] **Step 12: Add the `eval` command to `cli.py`**

Add `import tempfile` to the imports and `EvalResult` to the `ragcore.api.schemas` import. In `build_parser`, after the `ask` parser:

```python
    ev = sub.add_parser(
        "eval", parents=[common], help="Score retrieval on a golden set (real backend)"
    )
    ev.add_argument("set", type=Path, help="Golden set .jsonl")
    ev.add_argument("--corpus", type=Path, required=True, help="Folder the set's files live in")
    ev.add_argument("--report", type=Path, help="Write the full result as JSON (a baseline)")
    ev.add_argument(
        "--gate", action="store_true", help="Exit 1 if recall@6 falls below <set>.baseline.json"
    )
    # A scratch index, never the user's library, unless --data-dir is given explicitly.
    ev.set_defaults(data_dir=None)
```

Add the command:

```python
async def _eval(args: argparse.Namespace) -> int:
    from ragcore.eval import (
        MAX_RECALL_DROP,
        baseline_recall,
        load_golden,
        score_questions,
        summarize,
    )

    questions = load_golden(args.set)
    with tempfile.TemporaryDirectory(prefix="ragcore-eval-") as scratch:
        if args.data_dir is None:
            args.data_dir = Path(scratch)
        backend = _real_backend(args)
        store = backend.store
        corpus = str(args.corpus.resolve())
        try:
            source = next((s for s in store.sources.values() if s.path == corpus), None)
            if source is None:
                source = store.add_source(SourceCreate(path=corpus))
            await store.ingest_source_async(source.id)
            paths = {d.id: d.path for d in store.documents.values() if d.source_id == source.id}
            doc_meta = {k: v for k, v in store.doc_meta().items() if k in paths}
            results = [
                r
                async for r in score_questions(
                    store.retriever,
                    questions,
                    settings=store.settings.retrieval,
                    doc_meta=doc_meta,
                    paths=paths,
                )
            ]
        except httpx.HTTPError as exc:
            print(f"error: embedder at {args.embed_url} failed: {exc}", file=sys.stderr)
            return 1
        except RuntimeError as exc:  # the index was built by another embedder
            print(f"error: {exc}", file=sys.stderr)
            return 1
        finally:
            store.close()

    baseline = baseline_recall(args.set.with_suffix(".baseline.json"))
    metrics = summarize(args.set.stem, results, baseline)
    for r in results:
        print(f"{f'hit {r.rank}' if r.hit else 'MISS '}  {r.q}")
    print(
        f"\nrecall@1 {metrics.recall_at_1:.2f}  recall@6 {metrics.recall_at_6:.2f}  "
        f"MRR {metrics.mrr:.2f}  nDCG@6 {metrics.ndcg_at_6:.2f}  "
        f"p50 {metrics.latency_p50.total_ms:.0f} ms"
    )
    if args.report:
        result = EvalResult(metrics=metrics, questions=results)
        args.report.write_text(result.model_dump_json(indent=2) + "\n")
    if not args.gate:
        return 0
    if baseline is None:
        print(f"error: no baseline at {args.set.with_suffix('.baseline.json')}", file=sys.stderr)
        return 1
    if metrics.recall_at_6 < baseline - MAX_RECALL_DROP:
        print(
            f"FAIL: recall@6 {metrics.recall_at_6:.2f} is more than {MAX_RECALL_DROP} below "
            f"the baseline {baseline:.2f}",
            file=sys.stderr,
        )
        return 1
    print(f"gate: recall@6 within {MAX_RECALL_DROP} of the baseline {baseline:.2f}")
    return 0
```

In `main`, dispatch `eval` **before** `_adopt_legacy_data_dir`/`mkdir`, because its `data_dir` may be `None`:

```python
    if args.command == "eval":
        return asyncio.run(_eval(args))
```

Add one line to the module docstring: ``eval`` scores retrieval on a golden set in a scratch index (``--gate`` for CI).

- [ ] **Step 13: Run all tests**

Run: `uv run pytest -q && uv run ruff check .`
Expected: PASS

- [ ] **Step 14: Commit**

```bash
git add eval/golden/base.jsonl core/ragcore/src/ragcore/{eval.py,cli.py,ports.py,stub/store.py,api/routes/evals.py} core/ragcore/tests/{test_eval_harness.py,test_evals.py,test_cli.py}
git commit -m "feat(eval): real golden-set harness shared by ragcore eval and /eval/run"
```

---

### Task 9: Eval baseline and CI gate

**Files:**
- Create: `eval/golden/base.baseline.json` (generated, then committed)
- Create: `.github/workflows/eval.yml`

**Interfaces:**
- Consumes: `ragcore eval SET --corpus DIR [--report FILE] [--gate]` from Task 8.

- [ ] **Step 1: Generate the baseline on CPU, embedder only.** This is the default user config: the reranker is opt-in, and with it down the retriever keeps fused order. CI runs on CPU too, so the baseline must come from CPU.

```bash
scripts/fetch-models.sh
llama-server -m ~/.ibid/models/embeddinggemma-300M-qat-Q4_0.gguf \
  --embedding -c 2048 -b 2048 -ub 2048 -ngl 0 --port 8770 --host 127.0.0.1 &
uv run ragcore eval eval/golden/base.jsonl --corpus fixtures/docs \
  --rerank-url http://127.0.0.1:1 --report eval/golden/base.baseline.json
kill %1
```

Expected: about 20 `hit N` lines, `recall@6` near 1.00 (`scripts/eval/results.md` measured R@6 1.00 for this embedder), and `eval/golden/base.baseline.json` written. If recall@6 is below 0.85 (the spec's Phase 2 exit bar), stop and report: that's a retrieval regression, not a baseline to commit.

- [ ] **Step 2: Create `.github/workflows/eval.yml`**

```yaml
name: eval

# Retrieval quality gate: recall@6 on eval/golden/base.jsonl may not drop more than one
# question below the committed baseline. Runs only when retrieval-relevant code changes.
on:
  pull_request:
    paths:
      - core/ragcore/src/ragcore/ingest/**
      - core/ragcore/src/ragcore/retrieve/**
      - core/ragcore/src/ragcore/models/**
      - core/ragcore/src/ragcore/store/**
      - core/ragcore/src/ragcore/eval.py
      - eval/**
      - fixtures/docs/**
  push:
    branches: [master]
    paths:
      - core/ragcore/src/ragcore/ingest/**
      - core/ragcore/src/ragcore/retrieve/**
      - core/ragcore/src/ragcore/models/**
      - core/ragcore/src/ragcore/store/**
      - core/ragcore/src/ragcore/eval.py
      - eval/**
      - fixtures/docs/**

jobs:
  recall:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v5
      - uses: astral-sh/setup-uv@v6
        with:
          enable-cache: true
      - uses: actions/cache@v4
        with:
          path: ~/.ibid/models
          key: embedder-embeddinggemma-300M-qat-Q4_0
      - name: Fetch the embedder
        run: |
          mkdir -p ~/.ibid/models
          f=~/.ibid/models/embeddinggemma-300M-qat-Q4_0.gguf
          [ -f "$f" ] || curl -fL -o "$f" \
            https://huggingface.co/ggml-org/embeddinggemma-300M-qat-q4_0-GGUF/resolve/main/embeddinggemma-300M-qat-Q4_0.gguf
      - run: brew install llama.cpp
      - name: Start the embedder (CPU, like the baseline)
        run: |
          llama-server -m ~/.ibid/models/embeddinggemma-300M-qat-Q4_0.gguf \
            --embedding -c 2048 -b 2048 -ub 2048 -ngl 0 --port 8770 --host 127.0.0.1 \
            > "$RUNNER_TEMP/embed.log" 2>&1 &
          for _ in $(seq 60); do curl -sf http://127.0.0.1:8770/health && exit 0; sleep 2; done
          cat "$RUNNER_TEMP/embed.log"
          exit 1
      - run: uv sync
      - run: >-
          uv run ragcore eval eval/golden/base.jsonl --corpus fixtures/docs
          --rerank-url http://127.0.0.1:1 --gate
```

- [ ] **Step 3: Check the gate locally both ways.** With the embedder running as in Step 1:
  - Run `uv run ragcore eval eval/golden/base.jsonl --corpus fixtures/docs --rerank-url http://127.0.0.1:1 --gate`. Expected: exit 0, `gate: recall@6 within 0.05 ...`.
  - Temporarily edit the baseline's `recall_at_6` to `2.0` and rerun. Expected: exit 1 with `FAIL: ...`. Then `git checkout eval/golden/base.baseline.json`.

- [ ] **Step 4: Commit**

```bash
git add eval/golden/base.baseline.json .github/workflows/eval.yml
git commit -m "ci: gate retrieval recall@6 against a committed baseline"
```

---

### Task 10: License scan in CI

The spec asks for `pip-licenses` + `cargo-deny`. Python gets a ~40-line stdlib script instead of `pip-licenses`. It walks only ragcore's *runtime* requirements, so dev/build tools are out of scope: PyInstaller is GPL-with-exception and must not trip it. It prefers `License-Expression`, then classifiers, then the first line of `License`, because numpy's full license text mentions GPL for a bundled runtime. A dry run on the current tree found 32 runtime packages and 0 flagged. Rust uses `cargo-deny` with an allow list taken from the current tree's 467 crates.

**Files:**
- Create: `scripts/check-licenses.py`
- Create: `apps/desktop/src-tauri/deny.toml`
- Modify: `apps/desktop/src-tauri/Cargo.toml` (`publish = false`, so cargo-deny ignores the unlicensed app crate)
- Modify: `.github/workflows/test.yml`

- [ ] **Step 1: Create `scripts/check-licenses.py`**

```python
#!/usr/bin/env python3
"""Fail if a runtime dependency of ragcore is under a strong copyleft license.

The sidecar is distributed as a frozen binary, so GPL/AGPL/SSPL code in it is a
problem (see the license guard in plan/ibid-build-plan.md). LGPL is allowed.
Walks ragcore's installed requirements transitively; dev and build tools are
not runtime and are not checked. Run: uv run python scripts/check-licenses.py
"""

from __future__ import annotations

import re
import sys
from importlib.metadata import PackageNotFoundError, distribution

COPYLEFT = re.compile(r"(?<!L)A?GPL|(?<!Lesser )General Public License|SSPL", re.IGNORECASE)
# Package -> why it is fine despite matching. Keep each entry justified.
ALLOW: dict[str, str] = {}


def licenses(dist) -> list[str]:
    """Most specific first: an SPDX expression, else classifiers, else the License field's
    first line (some packages paste whole license texts there, bundled ones included)."""
    meta = dist.metadata
    if meta.get("License-Expression"):
        return [meta["License-Expression"]]
    classifiers = [c for c in meta.get_all("Classifier") or [] if c.startswith("License ::")]
    if classifiers:
        return classifiers
    text = (meta.get("License") or "").strip()
    return [text.splitlines()[0]] if text else []


def main() -> int:
    seen: set[str] = set()
    todo = ["ragcore"]
    flagged: list[tuple[str, list[str]]] = []
    while todo:
        name = todo.pop()
        key = re.sub(r"[-_.]+", "-", name).lower()
        if key in seen:
            continue
        seen.add(key)
        try:
            dist = distribution(name)
        except PackageNotFoundError:  # a requirement for another platform
            continue
        for requirement in dist.requires or []:
            if "extra ==" not in requirement:
                todo.append(re.match(r"[A-Za-z0-9._-]+", requirement).group())
        found = licenses(dist)
        if key == "ragcore" or key in ALLOW:
            continue
        if not found or any(COPYLEFT.search(text) for text in found):
            flagged.append((key, found))
    for key, found in flagged:
        print(f"{key}: {found or 'no license metadata'}", file=sys.stderr)
    print(f"{len(seen)} runtime packages checked, {len(flagged)} flagged")
    return 1 if flagged else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run it**

Run: `uv run python scripts/check-licenses.py`
Expected: `N runtime packages checked, 0 flagged` (N ≈ 45 now that Tasks 2/6/7 added rapidocr, opencv-python-headless, onnxruntime, shapely, py3langid, watchdog…). If something is flagged, look up its real license. Add a false positive to `ALLOW` with a one-line reason. A genuine GPL dependency: stop and report.

- [ ] **Step 3: Self-check the matcher** with a quick one-off (don't commit it):

```bash
uv run python -c "
import importlib.util as u; s=u.spec_from_file_location('c','scripts/check-licenses.py'); m=u.module_from_spec(s); s.loader.exec_module(m)
hit=lambda t: bool(m.COPYLEFT.search(t))
assert hit('GPL-3.0-or-later') and hit('AGPL-3.0') and hit('License :: OSI Approved :: GNU General Public License v3 (GPLv3)')
assert not hit('LGPL-2.1-or-later') and not hit('License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)') and not hit('MIT')
print('ok')"
```

Expected: `ok`

- [ ] **Step 4: Create `apps/desktop/src-tauri/deny.toml`**

```toml
# cargo deny check licenses — the Rust half of the license guard (plan/ibid-build-plan.md §6).
# Allow list = what the current dependency tree actually uses; all permissive except
# MPL-2.0 (file-level copyleft: fine for unmodified crates).

[licenses]
allow = [
  "MIT",
  "MIT-0",
  "Apache-2.0",
  "Apache-2.0 WITH LLVM-exception",
  "BSD-3-Clause",
  "ISC",
  "Zlib",
  "0BSD",
  "Unlicense",
  "Unicode-3.0",
  "CC0-1.0",
  "CDLA-Permissive-2.0",
  "MPL-2.0",
]
confidence-threshold = 0.93

[licenses.private]
# ibid-desktop itself has no license field; publish = false marks it private.
ignore = true
```

In `apps/desktop/src-tauri/Cargo.toml` `[package]`, add `publish = false`.

- [ ] **Step 5: Run cargo-deny locally**

Run: `cargo install --locked cargo-deny && cargo deny --manifest-path apps/desktop/src-tauri/Cargo.toml check licenses`
Expected: `licenses ok`. If it rejects a crate, check that crate's license. Add the SPDX id to `allow` only if it is permissive, and say which crate needed it in the commit body.

- [ ] **Step 6: Wire both into `.github/workflows/test.yml`**

In the `python` job, add after `uv run ruff check .`:

```yaml
      - run: uv run python scripts/check-licenses.py
```

Add a job:

```yaml
  licenses-rust:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5
      - uses: EmbarkStudios/cargo-deny-action@v2
        with:
          manifest-path: apps/desktop/src-tauri/Cargo.toml
          command: check licenses
```

- [ ] **Step 7: Commit**

```bash
git add scripts/check-licenses.py apps/desktop/src-tauri/{deny.toml,Cargo.toml} .github/workflows/test.yml
git commit -m "ci: fail on copyleft licenses in Python runtime and Rust dependencies"
```

---

### Task 11: Bundle, docs, install

**Files:**
- Modify: `CLAUDE.md`
- Verify: `scripts/build-sidecar.sh` (flags added in Tasks 2, 6, 7), `scripts/install-local.sh`

- [ ] **Step 1: Build the frozen sidecar**

Run: `scripts/build-sidecar.sh`
Expected: ends with the ragcore version line. Note the binary size against the previous one (`ls -lh apps/desktop/src-tauri/binaries/ragcore`). Expect roughly +100 MB from onnxruntime + OpenCV + the OCR models; report the real number.

- [ ] **Step 2: Smoke-test the frozen binary's new code paths** (`--data-dir` keeps it off `~/.ibid`):

```bash
mkdir -p /tmp/rc-smoke/docs
cp fixtures/docs/reranking.md /tmp/rc-smoke/docs/
printf 'name;dim\nEmbeddingGemma;768\n' > /tmp/rc-smoke/docs/models.csv
scripts/dev/serve-models.sh &   # needs scripts/fetch-models.sh once
apps/desktop/src-tauri/binaries/ragcore index /tmp/rc-smoke/docs --data-dir /tmp/rc-smoke/data
```

Expected: `2 document(s) ...` with no `failed:` lines. Also check OCR in the frozen build: put a scanned PDF (any image-only PDF, e.g. a phone scan) into `/tmp/rc-smoke/docs`, re-run `index`, and expect it to be indexed, not `failed:`. A `ModuleNotFoundError`/missing-model error here means a PyInstaller `--collect-*` flag is wrong.

- [ ] **Step 3: Update `CLAUDE.md`**
  - In "Sidecar architecture", change `parse.py` (Markdown, TXT, PDF, PPTX) to `parse.py` (Markdown, TXT, PDF with RapidOCR for pages without a text layer, PPTX, DOCX, HTML, CSV; sets `lang` via py3langid). Add a bullet: `watch.py`: watchdog observer; rescans a watched source 2 s after its files settle (real backend only).
  - In "Commands", under the Python block, add: `uv run ragcore eval eval/golden/base.jsonl --corpus fixtures/docs   # recall@k/MRR/nDCG in a scratch index; --gate fails below eval/golden/base.baseline.json`.
  - Add one line: `scripts/check-licenses.py` + `apps/desktop/src-tauri/deny.toml` are the license guard (CI fails on GPL/AGPL/SSPL).

- [ ] **Step 4: Install locally** (CLAUDE.md)

Run: `scripts/install-local.sh`
Expected: the app is replaced in `/Applications`. Launch it and check by hand:
  - Two-turn chat: ask "How does reranking work?", then "And on a laptop without a GPU?". The second answer should be about reranker candidate counts.
  - Library: add a folder with a `.docx`, `.html` and `.csv`. All three should index.
  - Drop a new `.md` into a watched folder. It should appear in the library within ~5 s without a manual rescan.

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: CLAUDE.md for OCR, new formats, watcher, eval and license guard"
```
