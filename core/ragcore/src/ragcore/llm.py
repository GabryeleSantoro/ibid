"""Generation through the connection the user activated.

OpenAI-compatible connections stream from ``/chat/completions``; Anthropic uses
its native ``/v1/messages``. Nothing here reads the environment: which model
answers, where it lives and with which key comes from the connection alone.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import AsyncIterator

import httpx

from ragcore.api.schemas import Connection, RetrievedChunk

SYSTEM_PROMPT = """You answer questions using only the passages provided.
Cite every claim with a marker of the form [document_id:page] taken from the
passage headers. If the passages do not contain the answer, say so plainly and
cite nothing. Do not invent document ids."""

LANGUAGE_NAMES = {"it": "Italian", "fr": "French", "de": "German", "es": "Spanish"}


def system_prompt_for(lang: str | None, extra: str = "") -> str:
    """The answer prompt, told to write in the UI language, plus the user's own notes.

    The user's text goes last and after the grounding rules, which are not editable.
    English with no notes adds nothing.
    """
    prompt = SYSTEM_PROMPT
    name = LANGUAGE_NAMES.get(lang or "")
    if name is not None:
        prompt = (
            f"{prompt}\nWrite the answer in {name}. "
            "Keep the [document_id:page] markers unchanged."
        )
    extra = extra.strip()
    return f"{prompt}\n{extra}" if extra else prompt



logger = logging.getLogger("ragcore.llm")


class LlmError(RuntimeError):
    """A generation failure with a stable code the UI can translate."""

    def __init__(self, code: str, message: str, **params: object) -> None:
        super().__init__(message)
        self.code = code
        self.params = params

ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1"
ANTHROPIC_VERSION = "2023-06-01"
# Anthropic requires max_tokens. Used when the connection sets no cap.
# ponytail: one value for every Claude model; per-model limits if one rejects it.
ANTHROPIC_DEFAULT_MAX_TOKENS = 32000


def _user_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    context = "\n\n".join(
        f"[{c.doc_id}:{c.page_start}] {c.doc_title} — {c.section_path or ''}\n{c.text}"
        for c in chunks
    )
    return f"Passages:\n\n{context}\n\nQuestion: {question}"


def _request(
    connection: Connection,
    api_key: str | None,
    question: str,
    chunks: list[RetrievedChunk],
    system_prompt: str,
    max_tokens: int | None,
) -> tuple[str, dict, dict]:
    """The URL, JSON body and headers for one connection's streaming call."""
    user = _user_prompt(question, chunks)
    if connection.kind == "anthropic":
        base = (connection.base_url or ANTHROPIC_BASE_URL).rstrip("/")
        headers = {"Content-Type": "application/json", "anthropic-version": ANTHROPIC_VERSION}
        if api_key:
            headers["x-api-key"] = api_key
        body = {
            "model": connection.model_id,
            "stream": True,
            "max_tokens": max_tokens or ANTHROPIC_DEFAULT_MAX_TOKENS,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user}],
        }
        return f"{base}/messages", body, headers

    base = (connection.base_url or "").rstrip("/")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body = {
        "model": connection.model_id,
        "stream": True,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user},
        ],
    }
    if max_tokens:
        body["max_tokens"] = max_tokens
    # OpenRouter reads this; every other OpenAI-compatible server ignores it.
    routing = {
        key: value
        for key, value in (
            ("sort", connection.provider_sort),
            ("order", connection.provider_order),
        )
        if value
    }
    if routing:
        body["provider"] = routing
    return f"{base}/chat/completions", body, headers


def _raise_on_error(connection: Connection, data: str) -> None:
    """Providers (OpenRouter, Anthropic) report some failures inside a 200 stream."""
    try:
        event = json.loads(data)
    except json.JSONDecodeError:
        return
    error = event.get("error") if isinstance(event, dict) else None
    if error:
        message = error.get("message", error) if isinstance(error, dict) else error
        logger.error("%s: error inside the stream: %s", connection.name, message)
        raise LlmError(
            "llm_failed_mid_answer",
            f"{connection.name} failed mid-answer: {message}",
            name=connection.name,
            reason=str(message),
        )


def _finish_reason(kind: str, data: str) -> str | None:
    try:
        event = json.loads(data)
        if kind == "anthropic":
            return (event.get("delta") or {}).get("stop_reason")
        return event["choices"][0].get("finish_reason")
    except (json.JSONDecodeError, KeyError, IndexError, AttributeError, TypeError):
        return None


def _piece(kind: str, data: str) -> str | None:
    """One text fragment out of one SSE data line, or None if it carries none."""
    try:
        event = json.loads(data)
    except json.JSONDecodeError:
        return None
    if kind == "anthropic":
        # Reasoning blocks arrive as thinking_delta; only text reaches the UI.
        delta = event.get("delta") or {}
        return delta.get("text") if delta.get("type") == "text_delta" else None
    try:
        # Reasoning models emit a separate channel; it never reaches the UI.
        return (event["choices"][0]["delta"] or {}).get("content")
    except (KeyError, IndexError):
        return None


def _reasoning_len(kind: str, data: str) -> int:
    """Size of the reasoning fragment in one SSE data line (never shown, only counted)."""
    try:
        event = json.loads(data)
        if kind == "anthropic":
            return len((event.get("delta") or {}).get("thinking") or "")
        delta = event["choices"][0]["delta"] or {}
        return len(delta.get("reasoning") or delta.get("reasoning_content") or "")
    except (json.JSONDecodeError, KeyError, IndexError, AttributeError, TypeError):
        return 0


LOCAL_MAX_TOKENS = 1024
IDLE_TIMEOUT = 120.0  # seconds without a data line before a call is abandoned


async def _stream(
    connection: Connection,
    api_key: str | None,
    question: str,
    chunks: list[RetrievedChunk],
    *,
    system_prompt: str | None = None,
    max_tokens: int | None = None,
    extra_body: dict | None = None,
) -> AsyncIterator[str]:
    if connection.kind != "anthropic" and not connection.base_url:
        raise LlmError(
            "connection_no_url",
            f"Connection {connection.name!r} has no base URL",
            name=connection.name,
        )

    url, body, headers = _request(
        connection,
        api_key,
        question,
        chunks,
        system_prompt or SYSTEM_PROMPT,
        max_tokens or connection.max_output_tokens,
    )
    body.update(extra_body or {})
    cap = max_tokens or connection.max_output_tokens
    logger.info(
        "%s: calling %s at %s (%d passages, output cap %s)",
        connection.name, connection.model_id, url, len(chunks), cap or "none",
    )
    started = time.perf_counter()
    chars = 0
    reasoning = 0
    async with (
        httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client,
        client.stream("POST", url, json=body, headers=headers) as response,
    ):
        if response.status_code >= 400:
            detail = (await response.aread()).decode(errors="replace").strip()[:500]
            logger.error("%s: HTTP %s: %s", connection.name, response.status_code, detail)
            raise LlmError(
                "llm_http_error",
                f"{connection.name} returned HTTP {response.status_code}: {detail}",
                name=connection.name,
                status=response.status_code,
                detail=detail,
            )
        produced = False
        finish: str | None = None
        done = False
        lines = response.aiter_lines()
        last_data = time.perf_counter()
        while True:
            # Keep-alive comments reset httpx's read timeout forever, so a model that
            # stalls silently would hang the run; only real data lines count as life.
            remaining = IDLE_TIMEOUT - (time.perf_counter() - last_data)
            try:
                line = await asyncio.wait_for(anext(lines), max(remaining, 0.01))
            except StopAsyncIteration:
                break
            except TimeoutError:
                logger.error("%s: no data for %.0fs, giving up", connection.name, IDLE_TIMEOUT)
                raise LlmError(
                    "llm_idle_timeout",
                    f"{connection.name} sent nothing for {IDLE_TIMEOUT:.0f}s "
                    f"({chars} chars of text, {reasoning} of reasoning so far). Lower "
                    "Thinking, use a non-reasoning model, or try again.",
                    name=connection.name,
                    seconds=round(IDLE_TIMEOUT),
                    chars=chars,
                    reasoning=reasoning,
                ) from None
            if not line.startswith("data:"):
                continue
            last_data = time.perf_counter()
            data = line[5:].strip()
            if data == "[DONE]":
                done = True
                break
            _raise_on_error(connection, data)
            finish = _finish_reason(connection.kind, data) or finish
            reasoning += _reasoning_len(connection.kind, data)
            piece = _piece(connection.kind, data)
            if piece:
                produced = True
                chars += len(piece)
                yield piece

    logger.info(
        "%s: finished in %.1fs, %d chars (+%d reasoning chars), finish reason %s",
        connection.name, time.perf_counter() - started, chars, reasoning, finish or "none",
    )

    # A 200 with no text is still a failure: saying so beats saving an empty answer.
    if not produced:
        if finish in ("length", "max_tokens"):
            budget = max_tokens or connection.max_output_tokens
            spent = f"its whole output budget ({budget} tokens)" if budget else "its output limit"
            raise LlmError(
                "llm_budget_spent",
                f"{connection.name} used {spent} without writing any text, most likely on "
                "reasoning. Raise or clear Max output tokens, or lower Thinking.",
                name=connection.name,
                budget=budget,
            )
        if (finish is None and not done) or (finish == "error" and reasoning):
            detail = f"{reasoning} chars of reasoning, no text" if reasoning else "no text"
            raise LlmError(
                "llm_stream_cut",
                f"{connection.name} closed the stream after "
                f"{time.perf_counter() - started:.0f}s with {detail} and "
                f"{'finish reason error' if finish else 'no finish reason'}, "
                "most likely still reasoning when the provider cut it off. Lower Thinking, "
                "use a non-reasoning model, or convert fewer slides at a time.",
                name=connection.name,
                seconds=round(time.perf_counter() - started),
            )
        raise LlmError(
            "llm_empty_answer",
            f"{connection.name} returned an empty answer (finish reason: {finish or 'none'})",
            name=connection.name,
            finish=finish or "none",
        )


async def llm_stream(
    connection: Connection,
    api_key: str | None,
    question: str,
    chunks: list[RetrievedChunk],
    *,
    system_prompt: str | None = None,
    max_tokens: int | None = None,
    local=None,
) -> AsyncIterator[str]:
    """Stream from the connection the user activated, and from nothing else."""
    if connection.kind != "local":
        async for piece in _stream(
            connection,
            api_key,
            question,
            chunks,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
        ):
            yield piece
        return
    if local is None or local.status() != "ready":
        raise LlmError("local_model_missing", "The built-in model is not installed")
    async with contextlib.AsyncExitStack() as stack:
        try:
            base_url = await stack.enter_async_context(local.lease())
        except (RuntimeError, OSError) as exc:
            raise LlmError(
                "local_model_failed",
                f"The built-in model could not start: {exc}",
                reason=str(exc),
            ) from exc
        served = connection.model_copy(
            update={"kind": "openai-compatible", "base_url": f"{base_url}/v1"}
        )
        async for piece in _stream(
            served,
            None,
            question,
            chunks,
            system_prompt=system_prompt,
            max_tokens=max_tokens or connection.max_output_tokens or LOCAL_MAX_TOKENS,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        ):
            yield piece
