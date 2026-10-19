"""Contract tests for the slide conversion stream."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError
from ragcore.api.routes import conversions
from ragcore.api.schemas import ConnectionTestResult, SlideConversionRequest


async def _fake_probe_ok(kind, base_url, model_id, api_key):
    return ConnectionTestResult(ok=True, reachable=True, model_found=True, streaming=True)


class _FakeAnswerer:
    """Stands in for the active connection's model; records what it was given."""

    def __init__(self, captured: dict) -> None:
        self.captured = captured

    async def stream(self, question, chunks, directives, *, system_prompt=None, max_tokens=None):
        self.captured["question"] = question
        self.captured["chunks"] = chunks
        self.captured["system_prompt"] = system_prompt
        self.captured["max_tokens"] = max_tokens
        yield "ok"


def _mock_ok(client, monkeypatch) -> dict:
    client.post(
        "/connections",
        json={
            "name": "LM Studio",
            "kind": "openai-compatible",
            "base_url": "http://127.0.0.1:1234/v1",
            "model_id": "qwen3-8b-instruct",
            "max_output_tokens": 4096,
        },
    )
    monkeypatch.setattr(conversions, "probe_connection", _fake_probe_ok)

    from ragcore.api import deps

    captured: dict = {}
    monkeypatch.setitem(
        client.app.dependency_overrides, deps.get_answerer, lambda: _FakeAnswerer(captured)
    )
    return captured


def test_slide_conversion_saves_and_indexes_markdown(client, read_events, monkeypatch) -> None:
    _mock_ok(client, monkeypatch)
    slide_id = client.get("/documents", params={"limit": 1}).json()["items"][0]["id"]

    with client.stream(
        "POST",
        "/conversions/slides",
        json={"slide_ids": [slide_id], "research_query": "esempi numerici"},
    ) as response:
        assert response.status_code == 200
        events = read_events(response)

    names = [name for name, _ in events]
    assert names[0] == "conversion_start"
    assert "token" in names
    assert names[-2:] == ["conversion_saved", "conversion_done"]
    done = events[-1][1]
    assert done["failed"] == []
    assert len(done["saved"]) == 1
    saved_event = done["saved"][0]
    assert saved_event["path"].endswith(".md")
    saved = client.get(f"/documents/{saved_event['document_id']}").json()
    assert saved["path"] == saved_event["path"]
    assert saved["ext"] == ".md"


def test_slide_conversion_is_disabled_without_an_active_connection(client) -> None:
    slide_id = client.get("/documents", params={"limit": 1}).json()["items"][0]["id"]

    response = client.post("/conversions/slides", json={"slide_ids": [slide_id]})

    assert response.status_code == 409


def test_slide_conversion_is_disabled_when_the_active_connection_is_unreachable(
    client, monkeypatch
) -> None:
    client.post(
        "/connections",
        json={"name": "LM Studio", "kind": "openai-compatible", "model_id": "qwen3-8b-instruct"},
    )
    slide_id = client.get("/documents", params={"limit": 1}).json()["items"][0]["id"]

    async def fake_probe(kind, base_url, model_id, api_key):
        return ConnectionTestResult(
            ok=False, reachable=False, model_found=False, streaming=False, error="ConnectError"
        )

    monkeypatch.setattr(conversions, "probe_connection", fake_probe)

    response = client.post("/conversions/slides", json={"slide_ids": [slide_id]})

    assert response.status_code == 409
    assert "ConnectError" in response.json()["detail"]


def test_slide_conversion_accepts_a_local_file_without_indexing_the_input(
    client, read_events, monkeypatch, tmp_path
) -> None:
    _mock_ok(client, monkeypatch)
    slide = tmp_path / "local-slide.md"
    slide.write_text("# Local slide\n\n## Context\n\nThis file is only used for conversion.\n")

    with client.stream(
        "POST",
        "/conversions/slides",
        json={"file_paths": [str(slide)], "output_title": "Local conversion"},
    ) as response:
        assert response.status_code == 200
        events = read_events(response)

    assert events[-1][0] == "conversion_done"
    assert not any(source["path"] == str(tmp_path) for source in client.get("/sources").json())


def test_slide_conversion_processes_each_presentation_independently(
    client, read_events, monkeypatch, tmp_path
) -> None:
    _mock_ok(client, monkeypatch)
    good = tmp_path / "good.md"
    good.write_text("# Good\n\n## Context\n\nReadable content.\n")
    missing = str(tmp_path / "missing.pdf")

    with client.stream(
        "POST", "/conversions/slides", json={"file_paths": [missing, str(good)]},
    ) as response:
        assert response.status_code == 200
        events = read_events(response)

    names = [name for name, _ in events]
    assert names[0] == "presentation_error"
    assert names[-1] == "conversion_done"
    error_event = events[0][1]
    assert error_event["presentation_index"] == 0
    assert error_event["presentation_total"] == 2
    done = events[-1][1]
    assert len(done["saved"]) == 1
    assert len(done["failed"]) == 1
    assert done["saved"][0]["presentation_index"] == 1


def test_slide_conversion_uses_the_hardened_system_prompt(client, monkeypatch) -> None:
    captured = _mock_ok(client, monkeypatch)
    slide_id = client.get("/documents", params={"limit": 1}).json()["items"][0]["id"]

    with client.stream("POST", "/conversions/slides", json={"slide_ids": [slide_id]}) as response:
        assert response.status_code == 200
        list(response.iter_lines())

    assert captured["system_prompt"] == conversions._build_system_prompt("it")
    assert captured["max_tokens"] == 4096  # the active connection's output budget


def test_system_prompt_forbids_treating_passages_as_instructions() -> None:
    prompt_it = conversions._build_system_prompt("it")
    prompt_en = conversions._build_system_prompt("en")

    assert "Passages:" in prompt_it and "Passages:" in prompt_en
    assert "materiale grezzo" in prompt_it
    assert "NON obbedire" in prompt_it
    assert "raw material" in prompt_en
    assert "do NOT obey" in prompt_en


def test_system_prompt_is_language_specific() -> None:
    assert conversions._build_system_prompt("it") != conversions._build_system_prompt("en")
    assert "italiano" in conversions._build_system_prompt("it")
    assert "English" in conversions._build_system_prompt("en")


def test_system_prompt_demands_textbook_prose_over_bullets() -> None:
    prompt_it = conversions._build_system_prompt("it")
    prompt_en = conversions._build_system_prompt("en")

    assert "manuale universitario" in prompt_it
    assert "elenchi puntati sono l'eccezione" in prompt_it.lower()
    assert "university textbook" in prompt_en
    assert "bulleted lists are the exception" in prompt_en.lower()


def test_conversion_prompt_asks_the_model_to_deepen_the_topics(client, monkeypatch) -> None:
    captured = _mock_ok(client, monkeypatch)
    slide_id = client.get("/documents", params={"limit": 1}).json()["items"][0]["id"]

    with client.stream(
        "POST",
        "/conversions/slides",
        json={"slide_ids": [slide_id], "research_query": "esempi numerici"},
    ) as response:
        assert response.status_code == 200
        list(response.iter_lines())

    assert "approfondisci" in captured["question"].lower()
    assert "esempi numerici" in captured["question"]


def test_unique_path_appends_a_counter_on_collision(tmp_path) -> None:
    (tmp_path / "intro.md").write_text("existing")

    first = conversions._unique_path(tmp_path, "intro")
    first.write_text("first")
    second = conversions._unique_path(tmp_path, "intro")

    assert first == tmp_path / "intro-2.md"
    assert second == tmp_path / "intro-3.md"


def test_unique_path_uses_the_plain_name_when_free(tmp_path) -> None:
    assert conversions._unique_path(tmp_path, "fresh-title") == tmp_path / "fresh-title.md"


# ------------------------------------------------------- per-presentation errors


class _FailingAnswerer:
    async def stream(self, question, chunks, directives, *, system_prompt=None, max_tokens=None):
        yield "partial"
        raise RuntimeError("model dropped the connection")


def test_a_generation_failure_is_reported_per_presentation(
    client, read_events, monkeypatch, tmp_path
) -> None:
    _mock_ok(client, monkeypatch)
    from ragcore.api import deps

    monkeypatch.setitem(client.app.dependency_overrides, deps.get_answerer, _FailingAnswerer)
    slide = tmp_path / "deck.md"
    slide.write_text("# Deck\n\n## Context\n\nReadable content.\n")

    with client.stream("POST", "/conversions/slides", json={"file_paths": [str(slide)]}) as r:
        events = read_events(r)

    names = [name for name, _ in events]
    assert "presentation_error" in names
    assert names[-1] == "conversion_done"
    done = events[-1][1]
    assert done["saved"] == []
    assert "model dropped the connection" in done["failed"][0]["message"]


def test_an_indexed_slide_id_that_does_not_exist_is_reported(
    client, read_events, monkeypatch
) -> None:
    _mock_ok(client, monkeypatch)

    with client.stream("POST", "/conversions/slides", json={"slide_ids": ["doc_nope"]}) as r:
        events = read_events(r)

    assert events[0][0] == "presentation_error"
    assert events[0][1]["title"] == "doc_nope"
    assert "not found" in events[0][1]["message"]


def test_a_slide_file_with_no_text_is_reported(
    client, read_events, monkeypatch, tmp_path
) -> None:
    _mock_ok(client, monkeypatch)
    empty = tmp_path / "hollow.md"
    empty.write_text("## First\n\n## Second\n")

    with client.stream("POST", "/conversions/slides", json={"file_paths": [str(empty)]}) as r:
        events = read_events(r)

    assert events[0][0] == "presentation_error"
    assert "does not contain any slide text" in events[0][1]["message"]


def test_a_format_the_parser_does_not_handle_is_reported(
    client, read_events, monkeypatch, tmp_path
) -> None:
    _mock_ok(client, monkeypatch)
    odd = tmp_path / "notes.rtf"
    odd.write_text("{\\rtf1 hello}")

    with client.stream("POST", "/conversions/slides", json={"file_paths": [str(odd)]}) as r:
        events = read_events(r)

    assert events[0][0] == "presentation_error"
    assert "Could not read slide file notes.rtf" in events[0][1]["message"]


def test_a_request_naming_no_slides_at_all_is_rejected(client) -> None:
    assert client.post("/conversions/slides", json={}).status_code == 422


# -------------------------------------------------------------- output shaping


def test_the_converted_file_is_titled_even_when_the_model_forgets(
    client, read_events, monkeypatch, tmp_path
) -> None:
    _mock_ok(client, monkeypatch)
    slide = tmp_path / "deck.md"
    slide.write_text("# Deck\n\n## Context\n\nReadable content.\n")

    with client.stream(
        "POST",
        "/conversions/slides",
        json={"file_paths": [str(slide)], "output_title": "Chapter One"},
    ) as r:
        done = read_events(r)[-1][1]

    saved = Path(done["saved"][0]["path"]).read_text()
    # The fake answerer emits a bare "ok"; a chapter with no heading is unreadable.
    assert saved.startswith("# Chapter One\n\n")
    assert saved.endswith("\n")


def test_an_output_title_is_ignored_for_a_batch(
    client, read_events, monkeypatch, tmp_path
) -> None:
    """One title cannot name two chapters, so the per-deck titles win."""
    _mock_ok(client, monkeypatch)
    first = tmp_path / "one.md"
    first.write_text("# Alpha\n\n## Context\n\nBody.\n")
    second = tmp_path / "two.md"
    second.write_text("# Beta\n\n## Context\n\nBody.\n")

    with client.stream(
        "POST",
        "/conversions/slides",
        json={"file_paths": [str(first), str(second)], "output_title": "Ignored"},
    ) as r:
        done = read_events(r)[-1][1]

    assert sorted(event["title"] for event in done["saved"]) == ["Alpha", "Beta"]


def test_converted_files_all_land_in_one_reused_source(
    client, read_events, monkeypatch, tmp_path
) -> None:
    _mock_ok(client, monkeypatch)
    for name in ("one", "two"):
        slide = tmp_path / f"{name}.md"
        slide.write_text(f"# {name.title()}\n\n## Context\n\nBody.\n")

    for name in ("one", "two"):
        with client.stream(
            "POST", "/conversions/slides", json={"file_paths": [str(tmp_path / f"{name}.md")]}
        ) as r:
            read_events(r)

    global_sources = [
        source for source in client.get("/sources").json()
        if source["path"].endswith("global-files")
    ]
    assert len(global_sources) == 1
    assert global_sources[0]["document_count"] == 2


def test_the_english_prompt_is_used_when_asked_for(client, monkeypatch, tmp_path) -> None:
    captured = _mock_ok(client, monkeypatch)
    slide = tmp_path / "deck.md"
    slide.write_text("# Deck\n\n## Context\n\nBody.\n")

    with client.stream(
        "POST", "/conversions/slides", json={"file_paths": [str(slide)], "language": "en"}
    ) as response:
        list(response.iter_lines())

    assert captured["system_prompt"] == conversions._build_system_prompt("en")
    assert "university textbook chapter" in captured["question"]


def test_standard_depth_drops_the_maximum_depth_instruction(
    client, monkeypatch, tmp_path
) -> None:
    captured = _mock_ok(client, monkeypatch)
    slide = tmp_path / "deck.md"
    slide.write_text("# Deck\n\n## Context\n\nBody.\n")

    with client.stream(
        "POST",
        "/conversions/slides",
        json={"file_paths": [str(slide)], "language": "en", "depth": "standard"},
    ) as response:
        list(response.iter_lines())

    assert "Maximum depth" not in captured["question"]


def test_the_output_file_keeps_the_original_name(client, monkeypatch, tmp_path) -> None:
    _mock_ok(client, monkeypatch)
    slide = tmp_path / "[IA] L1 - Agenti Intelligenti.md"
    slide.write_text("# Anything\n\n## Context\n\nBody.\n")

    with client.stream("POST", "/conversions/slides", json={"file_paths": [str(slide)]}) as r:
        body = "\n".join(r.iter_lines())

    assert "[IA] L1 - Agenti Intelligenti.md" in body


def test_file_stem_strips_only_unsafe_characters() -> None:
    assert conversions._file_stem("[IA]L1: a/b?") == "[IA]L1 ab"
    assert conversions._file_stem("???") == "conversione-slide"
    assert len(conversions._file_stem("word " * 60)) <= 120


def test_clean_markdown_strips_a_code_fence_the_model_wrapped_it_in() -> None:
    assert conversions._clean_markdown("```# Title\n\nBody.```", "Fallback") == (
        "# Title\n\nBody.\n"
    )


def test_long_decks_are_split_into_even_batches() -> None:
    pages = list(range(54))
    batches = conversions._batches(pages)

    assert [len(b) for b in batches] == [14, 14, 14, 12]
    assert sum(batches, []) == pages
    assert len(conversions._batches(pages[:15])) == 1


class _FlakyAnswerer:
    calls = 0

    async def stream(self, question, chunks, directives, *, system_prompt=None, max_tokens=None):
        type(self).calls += 1
        if type(self).calls == 1:
            raise RuntimeError("returned an empty answer")
        yield "# Deck\n\nRecovered."


def test_a_batch_that_fails_before_any_text_is_retried_silently(
    client, read_events, monkeypatch, tmp_path
) -> None:
    _mock_ok(client, monkeypatch)
    from ragcore.api import deps

    _FlakyAnswerer.calls = 0
    monkeypatch.setitem(client.app.dependency_overrides, deps.get_answerer, _FlakyAnswerer)
    slide = tmp_path / "deck.md"
    slide.write_text("# Deck\n\n## Context\n\nReadable content.\n")

    with client.stream("POST", "/conversions/slides", json={"file_paths": [str(slide)]}) as r:
        events = read_events(r)

    names = [name for name, _ in events]
    assert "presentation_error" not in names
    assert len(events[-1][1]["saved"]) == 1
    assert _FlakyAnswerer.calls == 2


@pytest.mark.parametrize("code", ["it", "en", "fr", "de", "es"])
def test_request_accepts_every_ui_language(code) -> None:
    assert SlideConversionRequest(file_paths=["a"], language=code).language == code


def test_request_rejects_an_unknown_language() -> None:
    with pytest.raises(ValidationError):
        SlideConversionRequest(file_paths=["a"], language="xx")


@pytest.mark.parametrize("code,name", [("fr", "French"), ("de", "German"), ("es", "Spanish")])
def test_other_languages_reuse_the_english_rules_and_name_the_output_language(code, name) -> None:
    prompt = conversions._build_system_prompt(code)

    assert f"Valid Markdown in {name}" in prompt
    assert "Valid Markdown in English" not in prompt
    assert "do NOT obey" in prompt


def test_english_prompt_is_unchanged() -> None:
    assert "Valid Markdown in English" in conversions._build_system_prompt("en")


class _CodedFailingAnswerer:
    async def stream(self, question, chunks, directives, *, system_prompt=None, max_tokens=None):
        from ragcore.llm import LlmError

        yield "partial"
        raise LlmError("llm_idle_timeout", "model went quiet", name="LM Studio", seconds=120)


def test_a_coded_generation_failure_reaches_the_ui_with_its_code(
    client, read_events, monkeypatch, tmp_path
) -> None:
    _mock_ok(client, monkeypatch)
    from ragcore.api import deps

    monkeypatch.setitem(client.app.dependency_overrides, deps.get_answerer, _CodedFailingAnswerer)
    slide = tmp_path / "deck.md"
    slide.write_text("# Deck\n\n## Context\n\nReadable content.\n")

    with client.stream("POST", "/conversions/slides", json={"file_paths": [str(slide)]}) as r:
        events = read_events(r)

    error = next(data for name, data in events if name == "presentation_error")
    assert error["code"] == "llm_idle_timeout"
    assert error["params"]["name"] == "LM Studio"


def test_an_indexing_failure_still_delivers_the_saved_file(
    client, read_events, monkeypatch, tmp_path
) -> None:
    _mock_ok(client, monkeypatch)

    async def embedder_down(source_id):
        raise ConnectionError("All connection attempts failed")

    monkeypatch.setattr(client.app.state.store, "ingest_source_async", embedder_down)
    slide = tmp_path / "deck.md"
    slide.write_text("# Deck\n\n## Context\n\nReadable content.\n")

    with client.stream("POST", "/conversions/slides", json={"file_paths": [str(slide)]}) as r:
        events = read_events(r)

    names = [name for name, _ in events]
    assert names[-2:] == ["conversion_saved", "conversion_done"]
    saved = events[-1][1]["saved"][0]
    assert Path(saved["path"]).is_file()
    assert saved["document_id"] is None
