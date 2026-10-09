# Built-in Local Answer Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A user with no API key gets cited answers from a small model running on their own computer, loaded in RAM only while answering, with a prompt to switch to OpenRouter when generation is too slow.

**Architecture:** A synthetic always-listed connection `local` (kind `"local"`) routes `/query` through the existing `llm_stream` to a lazily started llama-server owned by `LocalLLM`. A `lease()` context manager keeps the server alive during an answer and arms a 60 s idle stop afterwards. Generation speed is measured in `query.py` and sent in the `done` event; the chat UI counts slow answers and shows a banner.

**Tech Stack:** Python 3.13 / FastAPI / pytest (core/ragcore), React 19 + TypeScript + vitest + i18next (apps/desktop). Rust shell untouched.

**Spec:** `docs/superpowers/specs/2026-10-01-local-answer-model-design.md`

## Global Constraints

- Model stays Qwen3-1.7B Q4 (`MODEL_FILE` in `local_llm.py`); no tiers, no other models.
- Idle stop after `IDLE_S = 60`; context `-c 8192`; local output cap default 1024 tokens; thinking disabled via `chat_template_kwargs: {enable_thinking: false}`.
- Server never starts at app launch or on activation, only on the first generate/answer.
- Speed: `tokens_per_s` is `None` for non-local connections and for answers under 30 pieces; slow threshold 5 tok/s; banner after 2 consecutive slow answers; dismissal is permanent.
- Citation validation (`citations.py`) and the stub fallback are untouched.
- Python: ruff line-length 100. Frontend strings go through i18n in all five locales (en, it, fr, de, es); no hard-coded text (`no-hardcoded-text.test.ts`).
- After the last task run `scripts/install-local.sh` (project rule). Do not cut a release.
- Deviation from spec §4: the banner prefills ONE suggested OpenRouter model (constant), not a list (YAGNI; widen later if wanted).

## Review Focus

- Cancelling or closing an answer mid-stream must release the lease so the idle timer can free RAM (Task 1 + Task 3 tests).
- `local` active but the model files deleted: the chat must show `local_model_missing`, not hang (Task 3 test).
- Creating the first remote connection while `local` is active must not steal "active" (Task 2 test).
- An answer whose pieces all arrive at one instant (zero duration) must give `tokens_per_s = None`, not `ZeroDivisionError` (Task 4 test).
- `localStorage` unavailable/corrupt must not break chat (Task 6 test).

## File Structure

- `core/ragcore/src/ragcore/local_llm.py` — modify: `lease()`, idle 60 s, ctx 8192, `generate` uses lease.
- `core/ragcore/src/ragcore/local_connection.py` — create: the synthetic `Connection` factory.
- `core/ragcore/src/ragcore/api/schemas.py` — modify: `ConnectionKind` += `"local"`, `DoneEvent.tokens_per_s`.
- `core/ragcore/src/ragcore/backend.py` — modify: `Backend.local_llm`, `ConnectionAnswerEngine(store, local_llm)`.
- `core/ragcore/src/ragcore/llm.py` — modify: `local` branch around the existing stream.
- `core/ragcore/src/ragcore/stub/store.py` — modify: `active_connection()` resolves `"local"`.
- `core/ragcore/src/ragcore/api/routes/connections.py` — modify: list/activate/create.
- `core/ragcore/src/ragcore/api/routes/query.py` — modify: speed + close stream.
- `core/ragcore/src/ragcore/api/app.py` — modify: use `backend.local_llm`.
- `core/ragcore/tests/test_local_llm.py`, `tests/test_local_connection.py` — create; `tests/test_llm.py`, `tests/test_query.py` — modify.
- `apps/desktop/src/features/settings/local-model.tsx` — create: install/progress/activate control.
- `apps/desktop/src/lib/slow-speed.ts` (+ `.test.ts`) — create: pure streak logic.
- `apps/desktop/src/features/chat/slow-speed-banner.tsx` — create.
- Modify: `lib/ipc.ts`, `lib/api-types.ts`, `lib/queries.ts`, `features/chat/use-chat.ts`, `features/chat/chat-view.tsx`, `features/chat/message.tsx`, `features/settings/settings-panel.tsx`, `features/settings/connection-dialog.tsx`, `features/onboarding/onboarding-view.tsx`, `locales/{en,it,fr,de,es}.json`.

---

### Task 1: `LocalLLM.lease()` and RAM-only-when-needed

**Files:**
- Modify: `core/ragcore/src/ragcore/local_llm.py`
- Test: `core/ragcore/tests/test_local_llm.py` (create)

**Interfaces:**
- Produces: `LocalLLM.lease() -> AsyncContextManager[str]` yielding the server base URL (e.g. `http://127.0.0.1:51234`, no `/v1`). Constants `IDLE_S = 60`, `CONTEXT = 8192`.

- [ ] **Step 1: Write the failing tests**

```python
"""LocalLLM keeps the model in RAM only while an answer needs it."""

from __future__ import annotations

import asyncio

from ragcore import local_llm
from ragcore.local_llm import LocalLLM


def _llm(tmp_path, monkeypatch):
    monkeypatch.setattr(local_llm, "IDLE_S", 0.05)
    llm = LocalLLM(tmp_path)
    started: list[int] = []
    stops: list[int] = []

    async def fake_ensure() -> str:
        started.append(1)
        return "http://127.0.0.1:1"

    llm._ensure_server = fake_ensure
    llm.stop = lambda: stops.append(1)
    return llm, started, stops


def test_nothing_starts_until_a_lease_is_taken(tmp_path, monkeypatch) -> None:
    _, started, _ = _llm(tmp_path, monkeypatch)
    assert started == []


def test_the_server_is_not_stopped_while_a_lease_is_held(tmp_path, monkeypatch) -> None:
    llm, started, stops = _llm(tmp_path, monkeypatch)

    async def scenario() -> None:
        async with llm.lease() as url:
            assert url == "http://127.0.0.1:1"
            await asyncio.sleep(0.15)
            assert stops == []
        await asyncio.sleep(0.15)
        assert stops == [1]

    asyncio.run(scenario())
    assert started == [1]


def test_overlapping_leases_keep_it_alive_until_the_last_one_ends(tmp_path, monkeypatch) -> None:
    llm, _, stops = _llm(tmp_path, monkeypatch)

    async def scenario() -> None:
        async with llm.lease():
            async with llm.lease():
                pass
            await asyncio.sleep(0.15)
            assert stops == []
        await asyncio.sleep(0.15)
        assert stops == [1]

    asyncio.run(scenario())


def test_a_lease_is_released_when_its_body_raises(tmp_path, monkeypatch) -> None:
    llm, _, stops = _llm(tmp_path, monkeypatch)

    async def scenario() -> None:
        try:
            async with llm.lease():
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        await asyncio.sleep(0.15)
        assert stops == [1]

    asyncio.run(scenario())
```

- [ ] **Step 2: Run to verify failure**

Run (repo root): `uv run pytest core/ragcore/tests/test_local_llm.py -v`
Expected: FAIL, `AttributeError: 'LocalLLM' object has no attribute 'lease'`.

- [ ] **Step 3: Implement**

In `local_llm.py`: change `IDLE_S = 180` to `IDLE_S = 60`; add `CONTEXT = 8192` next to it; in `__init__` add `self._leases = 0`; in `_ensure_server` replace `"-c", "4096"` with `"-c", str(CONTEXT)`. Replace `_touch` with:

```python
    def _arm(self) -> None:
        if self._idle:
            self._idle.cancel()
        self._idle = asyncio.get_running_loop().call_later(IDLE_S, self.stop)

    @contextlib.asynccontextmanager
    async def lease(self):
        """The server's base URL, kept running until the last holder lets go."""
        self._leases += 1
        if self._idle:
            self._idle.cancel()
            self._idle = None
        try:
            yield await self._ensure_server()
        finally:
            self._leases -= 1
            if self._leases == 0:
                self._arm()
```

Rewrite the top of `generate` so it holds a lease for the whole call (delete the `_touch()` calls):

```python
    async def generate(self, system: str, user: str, max_tokens: int = 200) -> str:
        async with self.lease() as url, httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{url}/v1/chat/completions",
                json={
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "max_tokens": max_tokens,
                    "temperature": 0.6,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
        response.raise_for_status()
        text = response.json()["choices"][0]["message"]["content"]
        return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
```

Also update the module docstring: "stops after IDLE_S without use" stays true; change nothing else.

- [ ] **Step 4: Run tests**

Run: `uv run pytest core/ragcore/tests/test_local_llm.py core/ragcore/tests/test_suggestions.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add core/ragcore/src/ragcore/local_llm.py core/ragcore/tests/test_local_llm.py
git commit -m "feat: LocalLLM lease keeps the model in RAM only while in use"
```

---

### Task 2: The synthetic `local` connection

**Files:**
- Create: `core/ragcore/src/ragcore/local_connection.py`
- Modify: `schemas.py`, `backend.py`, `stub/store.py`, `api/routes/connections.py`, `api/app.py`
- Test: `core/ragcore/tests/test_local_connection.py` (create)

**Interfaces:**
- Produces: `local_connection(active: bool) -> Connection` (id `"local"`, kind `"local"`, `is_remote=False`, `has_api_key=False`, `model_id="Qwen3-1.7B"`); `LOCAL_ID = "local"`. `Backend.local_llm: LocalLLM`; `ConnectionAnswerEngine(store, local_llm)`.

- [ ] **Step 1: Write the failing tests**

```python
"""The built-in model shows up as a connection the user can activate once it is installed."""

from __future__ import annotations

_REMOTE = {
    "name": "OpenRouter",
    "kind": "openai-compatible",
    "base_url": "https://openrouter.ai/api/v1",
    "model_id": "x/y",
}


class _FakeLocal:
    def __init__(self, state: str) -> None:
        self._state = state

    def status(self) -> str:
        return self._state

    def stop(self) -> None:
        pass


def test_local_is_listed_first_and_inactive(client) -> None:
    first = client.get("/connections").json()[0]

    assert first["id"] == "local"
    assert first["kind"] == "local"
    assert first["is_remote"] is False
    assert first["has_api_key"] is False
    assert first["active"] is False


def test_activation_is_refused_until_the_model_is_installed(client) -> None:
    client.app.state.local_llm = _FakeLocal("missing")

    response = client.post("/connections/local/activate")

    assert response.status_code == 409
    assert response.json()["code"] == "local_model_missing"
    assert client.get("/connections").json()[0]["active"] is False


def test_an_installed_model_can_be_activated_and_answers(client) -> None:
    client.app.state.local_llm = _FakeLocal("ready")

    response = client.post("/connections/local/activate")

    assert response.status_code == 200
    assert client.get("/connections").json()[0]["active"] is True
    assert client.app.state.store.active_connection().kind == "local"


def test_it_cannot_be_edited_or_deleted(client) -> None:
    assert client.delete("/connections/local").status_code == 404
    body = {**_REMOTE, "name": "x"}
    assert client.patch("/connections/local", json=body).status_code == 404


def test_the_first_remote_connection_does_not_steal_active_from_local(client) -> None:
    client.app.state.local_llm = _FakeLocal("ready")
    client.post("/connections/local/activate")

    created = client.post("/connections", json=_REMOTE).json()

    assert created["active"] is False
    assert client.get("/connections").json()[0]["active"] is True


def test_activating_a_remote_connection_deactivates_local(client) -> None:
    client.app.state.local_llm = _FakeLocal("ready")
    client.post("/connections/local/activate")
    remote = client.post("/connections", json=_REMOTE).json()

    client.post(f"/connections/{remote['id']}/activate")

    listed = client.get("/connections").json()
    assert listed[0]["active"] is False
    assert [c["active"] for c in listed[1:]] == [True]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest core/ragcore/tests/test_local_connection.py -v`
Expected: FAIL (`first["id"] == "local"` assertion).

- [ ] **Step 3: Implement**

`schemas.py`: `ConnectionKind = Literal["openai-compatible", "anthropic", "local"]`.

`local_connection.py`:

```python
"""The built-in model as a connection: always listed, never stored, no key."""

from __future__ import annotations

from datetime import UTC, datetime

from ragcore.api.schemas import Connection

LOCAL_ID = "local"


def local_connection(active: bool) -> Connection:
    return Connection(
        id=LOCAL_ID,
        name="This computer",
        kind="local",
        base_url=None,
        model_id="Qwen3-1.7B",
        max_output_tokens=None,
        thinking="off",
        is_remote=False,
        has_api_key=False,
        active=active,
        created_at=datetime.fromtimestamp(0, tz=UTC),
    )
```

`stub/store.py` (`active_connection`, line ~496):

```python
    def active_connection(self) -> Connection | None:
        """The one connection that answers, chosen by the user in the app."""
        active_id = self.settings.active_connection_id or ""
        if active_id == LOCAL_ID:
            return local_connection(active=True)
        return self.connections.get(active_id)
```
with `from ragcore.local_connection import LOCAL_ID, local_connection` at the top (check no import cycle; `local_connection` imports only `api.schemas`).

`connections.py`:

```python
from fastapi import APIRouter, Request

from ragcore.local_connection import LOCAL_ID, local_connection

@router.get("", response_model=list[Connection])
def list_connections(store: StoreDep) -> list[Connection]:
    local = local_connection(active=store.settings.active_connection_id == LOCAL_ID)
    return [local, *store.connections.values()]
```
In `create_connection` change `active=not store.connections,` to
`active=not store.connections and store.settings.active_connection_id is None,`.
In `activate` add `request: Request` and, before the lookup:

```python
    if connection_id == LOCAL_ID:
        if request.app.state.local_llm.status() != "ready":
            raise api_error(409, "local_model_missing", "install the built-in model first")
        for other in store.connections.values():
            other.active = False
        store.settings.active_connection_id = LOCAL_ID
        store.save_connections()
        return local_connection(active=True)
```
(`delete`/`update` already 404 for `local` because it is not in `store.connections`.)

`backend.py`: add `local_llm: object` to `Backend`; `ConnectionAnswerEngine.__init__(self, store, local_llm=None)` storing `self.local_llm`; `build_backend` creates `local_llm = LocalLLM(config.data_dir)` (import from `ragcore.local_llm`), passes it to both `Backend(local_llm=local_llm, ...)` and `ConnectionAnswerEngine(store, local_llm)`. In `stream()` pass `local=self.local_llm` to `llm_stream` (the keyword is added in Task 3). `app.py`: replace `app.state.local_llm = LocalLLM(config.data_dir)` with `app.state.local_llm = backend.local_llm` and drop the now-unused import if ruff flags it.

- [ ] **Step 4: Run tests**

Run: `uv run pytest core/ragcore/tests/test_local_connection.py core/ragcore/tests/test_connections.py core/ragcore/tests/test_backend.py -v && uv run ruff check .`
Expected: PASS, ruff clean (the `local=` kwarg is not yet accepted, so `test_backend.py` may need Task 3; if it fails only on that kwarg, proceed and re-run after Task 3).

- [ ] **Step 5: Commit**

```bash
git add core/ragcore
git commit -m "feat: built-in local model as an always-listed connection"
```

---

### Task 3: `llm_stream` for kind `local`

**Files:**
- Modify: `core/ragcore/src/ragcore/llm.py`, `core/ragcore/src/ragcore/api/routes/query.py`
- Test: `core/ragcore/tests/test_llm.py`

**Interfaces:**
- Consumes: `LocalLLM.lease()` (Task 1), `local_connection` (Task 2).
- Produces: `llm_stream(..., local: LocalLLM | None = None)`; error code `local_model_missing` (message "The built-in model is not installed").

- [ ] **Step 1: Write the failing tests** (append to `tests/test_llm.py`)

```python
import contextlib

from ragcore.local_connection import local_connection


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
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest core/ragcore/tests/test_llm.py -v`
Expected: FAIL (`unexpected keyword argument 'local'`).

- [ ] **Step 3: Implement**

In `llm.py` add `LOCAL_MAX_TOKENS = 1024`. Rename the existing `async def llm_stream(` to `async def _stream(` and add one keyword parameter `extra_body: dict | None = None`; right after `url, body, headers = _request(...)` add `body.update(extra_body or {})`. Then add the public wrapper below it:

```python
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
            connection, api_key, question, chunks, system_prompt=system_prompt, max_tokens=max_tokens
        ):
            yield piece
        return
    if local is None or local.status() != "ready":
        raise LlmError("local_model_missing", "The built-in model is not installed")
    async with local.lease() as base_url:
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
```

`query.py`: in the `finally:` of the answer loop (the one with `_cancelled.discard(query_id)`) add before the discard:

```python
            close = getattr(stream, "aclose", None)
            if close is not None:
                await close()
```
so a cancelled or disconnected answer releases the lease immediately.

- [ ] **Step 4: Run tests**

Run: `uv run pytest core/ragcore -v && uv run ruff check .`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add core/ragcore
git commit -m "feat: answer through the built-in model when the local connection is active"
```

---

### Task 4: Generation speed in the `done` event

**Files:**
- Modify: `schemas.py`, `api/routes/query.py`, `apps/desktop/src/lib/api-types.ts` (regenerated), `apps/desktop/src/lib/ipc.ts`
- Test: `core/ragcore/tests/test_query.py`

**Interfaces:**
- Produces: `tokens_per_second(pieces: int, first_at: float | None, last_at: float | None) -> float | None` in `query.py`; `DoneEvent.tokens_per_s: float | None = None`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_query.py`)

```python
from ragcore.api.routes.query import tokens_per_second


def test_speed_counts_from_the_first_piece() -> None:
    assert tokens_per_second(101, 5.0, 15.0) == 10.0


def test_short_answers_have_no_speed() -> None:
    assert tokens_per_second(29, 0.0, 10.0) is None


def test_an_answer_that_arrived_at_once_has_no_speed() -> None:
    assert tokens_per_second(100, 3.0, 3.0) is None
    assert tokens_per_second(0, None, None) is None


def test_remote_or_stub_answers_report_no_speed(client, read_events) -> None:
    response = client.post("/query", json={"q": "what does reranking do"})
    done = [data for event, data in read_events(response.text) if event == "done"][0]
    assert done["tokens_per_s"] is None
```
(Use the same `read_events` call style the neighbouring tests in `test_query.py` use; adjust the first lines of the last test to match them.)

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest core/ragcore/tests/test_query.py -v`
Expected: FAIL (`ImportError: tokens_per_second`).

- [ ] **Step 3: Implement**

`schemas.py` `DoneEvent`: add `tokens_per_s: float | None = None`.

`query.py`: above the `query` route add

```python
MIN_PIECES_FOR_SPEED = 30


def tokens_per_second(pieces: int, first_at: float | None, last_at: float | None) -> float | None:
    """Pieces per second from the first to the last one; load and prompt time excluded."""
    if pieces < MIN_PIECES_FOR_SPEED or first_at is None or last_at is None:
        return None
    span = last_at - first_at
    return (pieces - 1) / span if span > 0 else None
```
In the stream loop keep `pieces = 0` and `last_piece_at: float | None = None` next to `first_token_at`, and inside the loop after the `first_token_at` assignment add `pieces += 1; last_piece_at = time.perf_counter()` (declare both before the `try`). In the `DoneEvent(...)` call add:

```python
                tokens_per_s=(
                    tokens_per_second(pieces, first_token_at, last_piece_at)
                    if connection and connection.kind == "local"
                    else None
                ),
```

Frontend types: from `apps/desktop`, with `uv run --directory ../../core/ragcore ragcore serve --port 8765` running in another shell, run `bun run gen:types`. In `ipc.ts` add `tokens_per_s: number | null;` to the `done` event data type (after `remote: boolean;`).

- [ ] **Step 4: Run tests**

Run: `uv run pytest core/ragcore -v && (cd apps/desktop && bun run typecheck)`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add core/ragcore apps/desktop/src/lib
git commit -m "feat: report generation speed of local answers in the done event"
```

---

### Task 5: Install / activate UI, onboarding and settings

**Files:**
- Create: `apps/desktop/src/features/settings/local-model.tsx`
- Modify: `lib/queries.ts`, `features/settings/settings-panel.tsx`, `features/onboarding/onboarding-view.tsx`, `features/chat/message.tsx`, `locales/*.json`

**Interfaces:**
- Consumes: `api.suggestionModel`, `api.installSuggestionModel`, `api.activateConnection("local")`, generated `Connection` type (`kind` now includes `"local"`).
- Produces: `<LocalModelControl connection={Connection} />`; query key `keys.localModel`; i18n keys `localModel.*`, `chat.waitingModel`, `errors.local_model_missing`.

- [ ] **Step 1: Locale keys.** Add to each of the five locale files a top-level `"localModel"` object, `chat.waitingModel`, and `errors.local_model_missing`:

en:
```json
"localModel": {
  "name": "This computer",
  "blurb": "Private: nothing leaves this computer",
  "install": "Install (~1.1 GB)",
  "downloading": "Downloading {{percent}}%",
  "failed": "Download failed: {{error}}"
}
```
`"chat.waitingModel": "Waiting for the model…"`, `"errors.local_model_missing": "The built-in model is not installed. Install it in Settings → Connections."`

it: `"name": "Questo computer"`, `"blurb": "Privato: nulla lascia questo computer"`, `"install": "Installa (~1,1 GB)"`, `"downloading": "Download {{percent}}%"`, `"failed": "Download non riuscito: {{error}}"`; `waitingModel: "In attesa del modello…"`; `local_model_missing: "Il modello integrato non è installato. Installalo in Impostazioni → Connessioni."`

fr: `"name": "Cet ordinateur"`, `"blurb": "Privé : rien ne quitte cet ordinateur"`, `"install": "Installer (~1,1 Go)"`, `"downloading": "Téléchargement {{percent}} %"`, `"failed": "Échec du téléchargement : {{error}}"`; `waitingModel: "En attente du modèle…"`; `local_model_missing: "Le modèle intégré n'est pas installé. Installez-le dans Réglages → Connexions."`

de: `"name": "Dieser Computer"`, `"blurb": "Privat: nichts verlässt diesen Computer"`, `"install": "Installieren (~1,1 GB)"`, `"downloading": "Download {{percent}} %"`, `"failed": "Download fehlgeschlagen: {{error}}"`; `waitingModel: "Warte auf das Modell…"`; `local_model_missing: "Das integrierte Modell ist nicht installiert. Installieren Sie es unter Einstellungen → Verbindungen."`

es: `"name": "Este ordenador"`, `"blurb": "Privado: nada sale de este ordenador"`, `"install": "Instalar (~1,1 GB)"`, `"downloading": "Descargando {{percent}} %"`, `"failed": "Error de descarga: {{error}}"`; `waitingModel: "Esperando al modelo…"`; `local_model_missing: "El modelo integrado no está instalado. Instálalo en Ajustes → Conexiones."`

(Use the exact settings/connection wording each locale already uses for "Settings → Connections"; flag all new strings for native review like the i18n spec.)

- [ ] **Step 2: Run locale tests to confirm parity holds**

Run (apps/desktop): `bunx vitest run src/locales`
Expected: PASS.

- [ ] **Step 3: Implement.** `queries.ts` keys: add `localModel: ["local-model"] as const,`.

`local-model.tsx`:

```tsx
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { errorText } from "@/lib/errors";
import { type Connection, api } from "@/lib/ipc";
import { keys } from "@/lib/queries";

/** Install, download progress, then Use: the built-in model's whole lifecycle in one slot. */
export function LocalModelControl({ connection }: { connection: Connection }) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const status = useQuery({
    queryKey: keys.localModel,
    queryFn: api.suggestionModel,
    refetchInterval: (query) => (query.state.data?.state === "downloading" ? 1000 : false),
  });
  const install = useMutation({
    mutationFn: api.installSuggestionModel,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: keys.localModel }),
  });
  const activate = useMutation({
    mutationFn: () => api.activateConnection("local"),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: keys.connections }),
    onError: (error: Error) => toast.error(errorText(error)),
  });

  const data = status.data;
  if (!data) return <Skeleton className="h-7 w-24" />;
  if (data.state === "downloading") {
    return (
      <span className="text-xs text-muted-foreground">
        {t("localModel.downloading", { percent: Math.round(data.progress * 100) })}
      </span>
    );
  }
  if (data.state === "missing") {
    return (
      <div className="flex items-center gap-2">
        {data.error ? (
          <span className="text-xs text-status-error">{t("localModel.failed", { error: data.error })}</span>
        ) : null}
        <Button size="sm" className="h-7" onClick={() => install.mutate()}>
          {t("localModel.install")}
        </Button>
      </div>
    );
  }
  return connection.active ? null : (
    <Button variant="secondary" size="sm" className="h-7" onClick={() => activate.mutate()}>
      {t("models.use")}
    </Button>
  );
}
```
(Import `Connection`/`api` from the same module `connection-dialog.tsx` imports them from; adjust the path if it differs.)

`settings-panel.tsx` in `ConnectionsSection`: show the localized name and blurb for local, and swap the right-hand buttons. Replace `{connection.name}` text with `{connection.kind === "local" ? t("localModel.name") : connection.name}`; replace the model/base_url line and the output-limit line for local with `t("localModel.blurb")` (wrap both `<p>`s in `connection.kind === "local" ? <p className=…>{t("localModel.blurb")}</p> : <>…existing…</>`); and replace the `<div className="flex shrink-0 …">` buttons block with:

```tsx
{connection.kind === "local" ? (
  <LocalModelControl connection={connection} />
) : (
  /* existing activate / edit / remove buttons, unchanged */
)}
```
Import `LocalModelControl`.

`onboarding-view.tsx` `ConnectionStep`: in the row for each connection use the same name swap and set `aside={connection.kind === "local" ? <LocalModelControl connection={connection} /> : <ConnectionDialog …existing… />}`, body `connection.kind === "local" ? t("localModel.blurb") : (existing mono model line)`. The always-present local row breaks the "no connection yet" logic, so fix both places that use `list.length`: the add-button label uses `list.filter((c) => c.kind !== "local").length === 0`, and the Next-gating (`step === "connection" && (connections.data ?? []).length === 0`, line ~514) becomes `step === "connection" && !(connections.data ?? []).some((c) => c.active)`.

`message.tsx`: after the `retrieving` span add

```tsx
        {streaming && !text ? (
          <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
            <Loader2Icon className="size-3 animate-spin" />
            {t("chat.waitingModel")}
          </span>
        ) : null}
```

- [ ] **Step 4: Verify**

Run (apps/desktop): `bun run typecheck && bun run test`
Expected: PASS. Manual (`bun tauri dev`): onboarding step 3 shows "This computer" with Install; Next is disabled until a connection is active; Install shows progress then Use; Settings → Connections shows the same row without Edit/Remove.

- [ ] **Step 5: Commit**

```bash
git add apps/desktop/src
git commit -m "feat: install and activate the built-in model from onboarding and settings"
```

---

### Task 6: Slow-speed banner

**Files:**
- Create: `apps/desktop/src/lib/slow-speed.ts`, `apps/desktop/src/lib/slow-speed.test.ts`, `apps/desktop/src/features/chat/slow-speed-banner.tsx`
- Modify: `features/chat/use-chat.ts`, `features/chat/chat-view.tsx`, `features/settings/connection-dialog.tsx`, `locales/*.json`

**Interfaces:**
- Produces: `SLOW_TOKENS_PER_S = 5`, `SLOW_STREAK = 2`, `type SlowState = { streak: number; dismissed: boolean }`, `recordAnswer(state, tokensPerS: number | null): SlowState`, `shouldSuggest(state): boolean`, `loadSlowState(): SlowState`, `saveSlowState(state): void`; `PendingAnswer.tokensPerS: number | null`; `ConnectionDialog` prop `preset?: { model_id: string }` that prefills OpenRouter.

- [ ] **Step 1: Write the failing tests** (`slow-speed.test.ts`)

```ts
import { afterEach, describe, expect, it, vi } from "vitest";

import { loadSlowState, recordAnswer, saveSlowState, shouldSuggest } from "./slow-speed";

const fresh = { streak: 0, dismissed: false };

describe("recordAnswer", () => {
  it("two slow answers in a row trigger the suggestion", () => {
    const one = recordAnswer(fresh, 3);
    expect(shouldSuggest(one)).toBe(false);
    expect(shouldSuggest(recordAnswer(one, 4))).toBe(true);
  });
  it("a fast answer resets the streak", () => {
    const slow = recordAnswer(recordAnswer(fresh, 2), 2);
    expect(recordAnswer(slow, 20).streak).toBe(0);
  });
  it("answers without a speed change nothing", () => {
    const one = recordAnswer(fresh, 3);
    expect(recordAnswer(one, null)).toEqual(one);
  });
  it("a dismissed suggestion never comes back", () => {
    const state = { streak: 5, dismissed: true };
    expect(shouldSuggest(state)).toBe(false);
    expect(shouldSuggest(recordAnswer(state, 1))).toBe(false);
  });
});

describe("storage", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("round-trips", () => {
    const store = new Map<string, string>();
    vi.stubGlobal("localStorage", {
      getItem: (k: string) => store.get(k) ?? null,
      setItem: (k: string, v: string) => void store.set(k, v),
    });
    saveSlowState({ streak: 1, dismissed: true });
    expect(loadSlowState()).toEqual({ streak: 1, dismissed: true });
  });
  it("survives unavailable or corrupt storage", () => {
    vi.stubGlobal("localStorage", {
      getItem: () => {
        throw new Error("denied");
      },
      setItem: () => {
        throw new Error("denied");
      },
    });
    expect(loadSlowState()).toEqual(fresh);
    expect(() => saveSlowState(fresh)).not.toThrow();
    vi.stubGlobal("localStorage", { getItem: () => "{nope", setItem: () => {} });
    expect(loadSlowState()).toEqual(fresh);
  });
});
```

- [ ] **Step 2: Run to verify failure**

Run (apps/desktop): `bunx vitest run src/lib/slow-speed.test.ts`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement `slow-speed.ts`**

```ts
export const SLOW_TOKENS_PER_S = 5;
export const SLOW_STREAK = 2;
const KEY = "localSlowState";

export type SlowState = { streak: number; dismissed: boolean };

export function recordAnswer(state: SlowState, tokensPerS: number | null): SlowState {
  if (tokensPerS === null) return state;
  return { ...state, streak: tokensPerS < SLOW_TOKENS_PER_S ? state.streak + 1 : 0 };
}

export const shouldSuggest = (state: SlowState) => !state.dismissed && state.streak >= SLOW_STREAK;

export function loadSlowState(): SlowState {
  try {
    const raw = JSON.parse(localStorage.getItem(KEY) ?? "null");
    if (raw && typeof raw.streak === "number" && typeof raw.dismissed === "boolean") {
      return { streak: raw.streak, dismissed: raw.dismissed };
    }
  } catch {
    // unavailable or corrupt storage: start fresh
  }
  return { streak: 0, dismissed: false };
}

export function saveSlowState(state: SlowState): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(state));
  } catch {
    // per-viewer convenience only
  }
}
```

- [ ] **Step 4: Run tests**

Run: `bunx vitest run src/lib/slow-speed.test.ts` → PASS.

- [ ] **Step 5: Wire the chat and the OpenRouter shortcut.**

Pick the suggested model: `curl -s https://openrouter.ai/api/v1/models | head -c 3000` and choose one currently listed, cheap, fast, non-reasoning model; put its exact ID in `export const OPENROUTER_SUGGESTED_MODEL = "<id>";` at the top of `slow-speed-banner.tsx`.

`use-chat.ts`: add `tokensPerS: number | null;` to `PendingAnswer`, `tokensPerS: null,` to `EMPTY`, and in `case "done":` add `tokensPerS: frame.data.tokens_per_s,`.

`connection-dialog.tsx`: add prop `preset?: { model_id: string }` to `ConnectionDialog`; change the form init to
`useState(() => ({ ...BLANK, ...(connection ?? {}), ...(preset ? { kind: "openai-compatible" as ConnectionKind, base_url: "https://openrouter.ai/api/v1", is_remote: true, model_id: preset.model_id, name: "OpenRouter" } : {}) }))` (`matchProvider` then selects OpenRouter from the URL).

`slow-speed-banner.tsx`:

```tsx
import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";

import { Button } from "@/components/ui/button";
import { ConnectionDialog } from "@/features/settings/connection-dialog";
import {
  loadSlowState,
  recordAnswer,
  saveSlowState,
  shouldSuggest,
  type SlowState,
} from "@/lib/slow-speed";

export const OPENROUTER_SUGGESTED_MODEL = "REPLACE_WITH_ID_FROM_CURL";

export function SlowSpeedBanner({ done, tokensPerS }: { done: boolean; tokensPerS: number | null }) {
  const { t } = useTranslation();
  const [state, setState] = useState<SlowState>(loadSlowState);

  useEffect(() => {
    if (!done) return;
    setState((current) => {
      const next = recordAnswer(current, tokensPerS);
      saveSlowState(next);
      return next;
    });
    // once per finished answer
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [done]);

  if (!done || !shouldSuggest(state)) return null;
  return (
    <div className="space-y-2 rounded-md border border-status-warn/30 bg-status-warn/8 px-3 py-2 text-xs text-status-warn">
      <p>{t("slowSpeed.text", { speed: Math.round(tokensPerS ?? 0) })}</p>
      <p className="opacity-80">{t("slowSpeed.privacy")}</p>
      <div className="flex gap-2">
        <ConnectionDialog
          preset={{ model_id: OPENROUTER_SUGGESTED_MODEL }}
          trigger={<Button size="sm" className="h-7">{t("slowSpeed.openrouter")}</Button>}
        />
        <Button
          variant="ghost"
          size="sm"
          className="h-7"
          onClick={() => {
            const next = { ...state, dismissed: true };
            saveSlowState(next);
            setState(next);
          }}
        >
          {t("slowSpeed.keep")}
        </Button>
      </div>
    </div>
  );
}
```
Replace `REPLACE_WITH_ID_FROM_CURL` with the ID chosen above; the plan fails if it is left in (grep for it in Step 6).

`chat-view.tsx`: import the banner and render it right after the pending `<Message … />` (inside the `{pending ? (` block, line ~231-250): `<SlowSpeedBanner done={pending.status === "done"} tokensPerS={pending.tokensPerS} />`.

Locales: add top-level `"slowSpeed"` to all five:

en: `{"text": "Answers are slow on this computer (~{{speed}} tok/s). A cloud model would be faster.", "privacy": "With OpenRouter, the retrieved passages are sent to the provider.", "openrouter": "Set up OpenRouter", "keep": "Keep using this computer"}`
it: `{"text": "Le risposte sono lente su questo computer (~{{speed}} tok/s). Un modello cloud sarebbe più veloce.", "privacy": "Con OpenRouter, i passaggi recuperati vengono inviati al provider.", "openrouter": "Configura OpenRouter", "keep": "Continua a usare questo computer"}`
fr: `{"text": "Les réponses sont lentes sur cet ordinateur (~{{speed}} tok/s). Un modèle cloud serait plus rapide.", "privacy": "Avec OpenRouter, les passages récupérés sont envoyés au fournisseur.", "openrouter": "Configurer OpenRouter", "keep": "Continuer sur cet ordinateur"}`
de: `{"text": "Antworten sind auf diesem Computer langsam (~{{speed}} Tok/s). Ein Cloud-Modell wäre schneller.", "privacy": "Mit OpenRouter werden die gefundenen Passagen an den Anbieter gesendet.", "openrouter": "OpenRouter einrichten", "keep": "Auf diesem Computer bleiben"}`
es: `{"text": "Las respuestas son lentas en este ordenador (~{{speed}} tok/s). Un modelo en la nube sería más rápido.", "privacy": "Con OpenRouter, los pasajes recuperados se envían al proveedor.", "openrouter": "Configurar OpenRouter", "keep": "Seguir usando este ordenador"}`

- [ ] **Step 6: Verify**

Run (apps/desktop): `grep -rn REPLACE_WITH_ID src && echo "STILL A PLACEHOLDER" || bun run typecheck && bun run test`
Expected: no placeholder output, typecheck and tests PASS.

- [ ] **Step 7: Commit**

```bash
git add apps/desktop/src
git commit -m "feat: suggest OpenRouter when local answers are slow"
```

---

### Task 7: Quality gate and install

**Files:**
- Create: `docs/superpowers/notes/2026-10-01-local-model-eval.md`

- [ ] **Step 1: Full test run**

Run: `uv run pytest -v && uv run ruff check . && (cd apps/desktop && bun run typecheck && bun run test)`
Expected: all PASS.

- [ ] **Step 2: Quality gate.** Install the model through the app (or `curl -X POST` the `/suggestions/model` endpoint), activate "This computer", run the eval harness in `scripts/eval` over `fixtures/docs` (read `scripts/eval/` for its invocation), and write results to the notes file: citation validity rate, answer quality on 5–10 questions (copy the question and a one-line verdict each), tokens/s on this machine. If answers are unusable, STOP and report to the user before shipping; the model (e.g. 4B) is then a spec change.

- [ ] **Step 3: Manual RAM and orphan check.** In Activity Monitor note llama-server memory right after an answer, again 70 s later (process gone), and after force-quitting the app mid-answer (no orphan llama-server).

- [ ] **Step 4: Install and commit**

```bash
scripts/install-local.sh
git add docs/superpowers/notes/2026-10-01-local-model-eval.md
git commit -m "docs: local answer model eval notes"
```
