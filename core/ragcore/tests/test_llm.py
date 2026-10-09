"""llm_stream() must talk to whatever connection the user activated: its URL,
its model, its key, its provider routing — and never the environment."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest
from ragcore import llm
from ragcore.api.schemas import Connection


class _FakeStreamResponse:
    status_code = 200

    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _FakeStreamContext:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    async def __aenter__(self) -> _FakeStreamResponse:
        return _FakeStreamResponse(self._lines)

    async def __aexit__(self, *exc) -> None:
        return None


class _FakeClient:
    def __init__(self, captured: dict, lines: list[str]) -> None:
        self._captured = captured
        self._lines = lines

    async def __aenter__(self) -> _FakeClient:
        return self

    async def __aexit__(self, *exc) -> None:
        return None

    def stream(self, method: str, url: str, *, json: dict, headers: dict):
        self._captured["url"] = url
        self._captured["payload"] = json
        self._captured["headers"] = headers
        return _FakeStreamContext(self._lines)


OPENAI_LINES = ['data: {"choices": [{"delta": {"content": "hello"}}]}', "data: [DONE]"]
ANTHROPIC_LINES = [
    'data: {"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": "hm"}}',
    'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "hello"}}',
    'data: {"type": "message_stop"}',
]


def connection(**overrides) -> Connection:
    fields = {
        "id": "conn_1",
        "name": "Test",
        "kind": "openai-compatible",
        "base_url": "http://x/v1",
        "model_id": "gemma-3-27b",
        "max_output_tokens": 4096,
        "thinking": "off",
        "is_remote": True,
        "has_api_key": True,
        "active": True,
        "created_at": datetime.now(tz=UTC),
    }
    return Connection(**{**fields, **overrides})


def run(conn: Connection, api_key: str | None = "sk-test", lines=OPENAI_LINES, **kwargs):
    captured: dict = {}

    async def collect() -> list[str]:
        return [
            piece
            async for piece in llm.llm_stream(
                conn, api_key, "What is reranking?", [], **kwargs
            )
        ]

    original = llm.httpx.AsyncClient
    llm.httpx.AsyncClient = lambda **_: _FakeClient(captured, lines)
    try:
        return asyncio.run(collect()), captured
    finally:
        llm.httpx.AsyncClient = original


def test_openai_compatible_uses_the_connection_url_model_and_key() -> None:
    pieces, captured = run(connection())

    assert pieces == ["hello"]
    assert captured["url"] == "http://x/v1/chat/completions"
    assert captured["payload"]["model"] == "gemma-3-27b"
    assert captured["headers"]["Authorization"] == "Bearer sk-test"


def test_max_tokens_defaults_to_the_connections_output_budget() -> None:
    _, captured = run(connection(max_output_tokens=32000))
    assert captured["payload"]["max_tokens"] == 32000

    _, captured = run(connection(max_output_tokens=32000), max_tokens=512)
    assert captured["payload"]["max_tokens"] == 512


def test_the_given_system_prompt_wins_over_the_chat_default() -> None:
    _, captured = run(connection(), system_prompt="CUSTOM PROMPT")
    assert captured["payload"]["messages"][0] == {"role": "system", "content": "CUSTOM PROMPT"}

    _, captured = run(connection())
    assert captured["payload"]["messages"][0] == {
        "role": "system",
        "content": llm.SYSTEM_PROMPT,
    }


def test_openrouter_provider_routing_is_sent_only_when_the_user_set_it() -> None:
    _, captured = run(connection(provider_sort="price", provider_order=["together"]))
    assert captured["payload"]["provider"] == {"sort": "price", "order": ["together"]}

    _, captured = run(connection())
    assert "provider" not in captured["payload"]


def test_anthropic_uses_its_native_messages_endpoint_and_headers() -> None:
    pieces, captured = run(
        connection(kind="anthropic", base_url=None), lines=ANTHROPIC_LINES
    )

    assert pieces == ["hello"]  # the thinking_delta never reaches the UI
    assert captured["url"] == "https://api.anthropic.com/v1/messages"
    assert captured["headers"]["x-api-key"] == "sk-test"
    assert "Authorization" not in captured["headers"]
    assert captured["payload"]["system"] == llm.SYSTEM_PROMPT


def test_a_connection_without_a_base_url_fails_loudly_instead_of_scripting() -> None:
    with pytest.raises(RuntimeError):
        run(connection(base_url=None))


# ------------------------------------------------------- what reaches the UI


def test_only_text_deltas_reach_the_ui() -> None:
    """Anthropic reasoning arrives as thinking_delta and must be dropped."""
    pieces, _ = run(connection(kind="anthropic"), lines=ANTHROPIC_LINES)

    assert pieces == ["hello"]


def test_an_openai_reasoning_channel_is_dropped_too() -> None:
    pieces, _ = run(
        connection(),
        lines=[
            'data: {"choices": [{"delta": {"reasoning": "thinking out loud"}}]}',
            'data: {"choices": [{"delta": {"content": "answer"}}]}',
            "data: [DONE]",
        ],
    )

    assert pieces == ["answer"]


def test_a_line_that_is_not_json_is_skipped_rather_than_crashing() -> None:
    pieces, _ = run(
        connection(),
        lines=["data: not-json", 'data: {"choices": [{"delta": {"content": "ok"}}]}'],
    )

    assert pieces == ["ok"]


def test_lines_outside_the_data_channel_are_ignored() -> None:
    pieces, _ = run(
        connection(),
        lines=[": keep-alive", "", 'data: {"choices": [{"delta": {"content": "ok"}}]}'],
    )

    assert pieces == ["ok"]


def test_a_chunk_with_no_choices_is_skipped() -> None:
    pieces, _ = run(
        connection(),
        lines=['data: {"choices": []}', 'data: {"choices": [{"delta": {"content": "ok"}}]}'],
    )

    assert pieces == ["ok"]


def test_a_null_delta_is_skipped() -> None:
    pieces, _ = run(
        connection(),
        lines=[
            'data: {"choices": [{"delta": null}]}',
            'data: {"choices": [{"delta": {"content": "ok"}}]}',
        ],
    )

    assert pieces == ["ok"]


def test_the_done_sentinel_ends_the_stream() -> None:
    pieces, _ = run(
        connection(),
        lines=[
            'data: {"choices": [{"delta": {"content": "first"}}]}',
            "data: [DONE]",
            'data: {"choices": [{"delta": {"content": "never"}}]}',
        ],
    )

    assert pieces == ["first"]


def test_an_http_error_names_the_connection_and_the_status() -> None:
    class _ErrorResponse:
        status_code = 429

        async def aread(self) -> bytes:
            return b'{"error": "rate limited"}'

        async def aiter_lines(self):  # pragma: no cover - never reached
            yield ""

    class _ErrorContext:
        async def __aenter__(self):
            return _ErrorResponse()

        async def __aexit__(self, *exc) -> None:
            return None

    class _ErrorClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc) -> None:
            return None

        def stream(self, *args, **kwargs):
            return _ErrorContext()

    async def collect() -> list[str]:
        return [
            piece
            async for piece in llm.llm_stream(
                connection(name="OpenRouter"), "sk", "q", []
            )
        ]

    original = llm.httpx.AsyncClient
    llm.httpx.AsyncClient = lambda **_: _ErrorClient()
    try:
        with pytest.raises(RuntimeError, match="OpenRouter returned HTTP 429"):
            asyncio.run(collect())
    finally:
        llm.httpx.AsyncClient = original


def test_a_stream_cut_off_while_reasoning_says_so() -> None:
    lines = ['data: {"choices": [{"delta": {"reasoning": "thinking hard"}}]}']

    with pytest.raises(RuntimeError, match="13 chars of reasoning.*no finish reason"):
        run(connection(name="OpenRouter"), lines=lines)


def test_a_stalled_stream_gives_up_despite_keep_alives(monkeypatch) -> None:
    monkeypatch.setattr(llm, "IDLE_TIMEOUT", 0.2)

    async def keep_alives(self):
        while True:
            await asyncio.sleep(0.05)
            yield ": OPENROUTER PROCESSING"

    monkeypatch.setattr(_FakeStreamResponse, "aiter_lines", keep_alives)

    with pytest.raises(RuntimeError, match="sent nothing"):
        run(connection(name="OpenRouter"))


def test_a_connection_with_no_key_sends_no_auth_header() -> None:
    _, captured = run(connection(), api_key=None)

    assert "Authorization" not in captured["headers"]


def test_anthropic_without_a_key_sends_no_api_key_header() -> None:
    _, captured = run(connection(kind="anthropic"), api_key=None, lines=ANTHROPIC_LINES)

    assert "x-api-key" not in captured["headers"]
    assert captured["headers"]["anthropic-version"] == "2023-06-01"


def test_the_passages_are_handed_over_as_labelled_context() -> None:
    from ragcore.api.schemas import RetrievedChunk

    chunk = RetrievedChunk(
        chunk_id="c1", doc_id="rerank", doc_title="Reranking", page_start=2, page_end=2,
        section_path="Reranking > Why", text="A cross encoder rescores candidates.",
    )

    captured: dict = {}

    async def collect() -> None:
        async for _ in llm.llm_stream(connection(), "sk", "Why rerank?", [chunk]):
            pass

    original = llm.httpx.AsyncClient
    llm.httpx.AsyncClient = lambda **_: _FakeClient(captured, OPENAI_LINES)
    try:
        asyncio.run(collect())
    finally:
        llm.httpx.AsyncClient = original

    user = captured["payload"]["messages"][-1]["content"]
    assert "[rerank:2]" in user, "the marker the model must cite with"
    assert "A cross encoder rescores candidates." in user
    assert user.endswith("Question: Why rerank?")


def _failure(lines: list[str]) -> str:
    with pytest.raises(RuntimeError) as caught:
        run(connection(), lines=lines)
    return str(caught.value)


def test_output_spent_entirely_on_reasoning_is_an_error_not_an_empty_answer() -> None:
    message = _failure(
        [
            'data: {"choices": [{"delta": {"reasoning": "long thoughts"}}]}',
            'data: {"choices": [{"delta": {}, "finish_reason": "length"}]}',
            "data: [DONE]",
        ]
    )

    assert "4096 tokens" in message
    assert "Max output tokens" in message


def test_no_output_cap_leaves_max_tokens_to_the_model() -> None:
    _, captured = run(connection(max_output_tokens=None))
    assert "max_tokens" not in captured["payload"]

    # Anthropic rejects a request without one, so it gets a generous default.
    _, captured = run(connection(kind="anthropic", max_output_tokens=None), lines=ANTHROPIC_LINES)
    assert captured["payload"]["max_tokens"] == llm.ANTHROPIC_DEFAULT_MAX_TOKENS


def test_an_error_inside_a_200_stream_is_raised() -> None:
    message = _failure(['data: {"error": {"code": 502, "message": "Provider returned error"}}'])

    assert "Provider returned error" in message


def test_a_stream_with_no_text_at_all_is_an_error() -> None:
    assert "empty answer" in _failure(["data: [DONE]"])


def test_english_and_missing_language_leave_the_system_prompt_unchanged() -> None:
    from ragcore.llm import SYSTEM_PROMPT, system_prompt_for

    assert system_prompt_for(None) == SYSTEM_PROMPT
    assert system_prompt_for("en") == SYSTEM_PROMPT
    assert system_prompt_for("xx") == SYSTEM_PROMPT


def test_other_languages_add_one_instruction_and_keep_the_citation_format() -> None:
    from ragcore.llm import SYSTEM_PROMPT, system_prompt_for

    prompt = system_prompt_for("it")

    assert prompt.startswith(SYSTEM_PROMPT)
    assert prompt.endswith(
        "Write the answer in Italian. Keep the [document_id:page] markers unchanged."
    )
    assert "Write the answer in German." in system_prompt_for("de")


import contextlib  # noqa: E402

from ragcore.local_connection import local_connection  # noqa: E402


class _FakeLocal:
    def __init__(self, state: str = "ready") -> None:
        self.state = state
        self.leases = 0
        self.released = 0

    def status(self) -> str:
        return self.state

    @contextlib.asynccontextmanager
    async def lease(self):
        self.leases += 1
        try:
            yield "http://127.0.0.1:9"
        finally:
            self.released += 1


def test_local_streams_from_the_leased_server_without_auth() -> None:
    fake = _FakeLocal()

    pieces, captured = run(local_connection(active=True), api_key=None, local=fake)

    assert pieces == ["hello"]
    assert captured["url"] == "http://127.0.0.1:9/v1/chat/completions"
    assert "Authorization" not in captured["headers"]
    assert captured["payload"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert captured["payload"]["max_tokens"] == 1024
    assert fake.leases == 1 and fake.released == 1


def test_local_without_the_model_installed_says_so() -> None:
    with pytest.raises(llm.LlmError) as raised:
        run(local_connection(active=True), api_key=None, local=_FakeLocal("missing"))

    assert raised.value.code == "local_model_missing"


def test_closing_a_local_stream_early_releases_the_lease() -> None:
    fake = _FakeLocal()
    lines = [
        'data: {"choices": [{"delta": {"content": "a"}}]}',
        'data: {"choices": [{"delta": {"content": "b"}}]}',
        "data: [DONE]",
    ]

    async def scenario() -> None:
        stream = llm.llm_stream(local_connection(active=True), None, "q", [], local=fake)
        assert await anext(stream) == "a"
        await stream.aclose()

    original = llm.httpx.AsyncClient
    llm.httpx.AsyncClient = lambda **_: _FakeClient({}, lines)
    try:
        asyncio.run(scenario())
    finally:
        llm.httpx.AsyncClient = original

    assert fake.released == 1


def test_a_local_server_that_fails_to_start_is_reported_as_a_coded_error() -> None:
    class _Broken(_FakeLocal):
        @contextlib.asynccontextmanager
        async def lease(self):
            raise RuntimeError("llama-server did not become healthy")
            yield  # pragma: no cover

    with pytest.raises(llm.LlmError) as raised:
        run(local_connection(active=True), api_key=None, local=_Broken())

    assert raised.value.code == "local_model_failed"
    assert "did not become healthy" in raised.value.params["reason"]


def test_extra_instructions_follow_the_grounding_rules() -> None:
    from ragcore.llm import SYSTEM_PROMPT, system_prompt_for

    prompt = system_prompt_for(None, "Keep it short {x}\nUse bullets.")
    assert prompt.startswith(SYSTEM_PROMPT)
    assert prompt.endswith("Keep it short {x}\nUse bullets.")


def test_blank_extra_instructions_change_nothing() -> None:
    from ragcore.llm import SYSTEM_PROMPT, system_prompt_for

    assert system_prompt_for(None, "   \n") == SYSTEM_PROMPT
    assert system_prompt_for("fr", "") == system_prompt_for("fr")
