# Built-in local answer model — design

## Goal
A user with no API key can install Ibid, index a folder and get cited answers, with the answer model running on their own computer. The model occupies RAM only while an answer is being written. If this computer generates too slowly, the app suggests a cloud model on OpenRouter.

## Non-goals
- Hardware tiers and models other than the small one (Qwen3-1.7B Q4, already pinned in `local_llm.py`). The model is one constant, so tiers can come later.
- Windows/Linux builds (separate project), OCR/DOCX (separate sub-project).
- Replacing connections. OpenRouter/OpenAI/Anthropic/OpenAI-compatible stay exactly as they are.
- Changes to the Rust shell. ragcore already owns the llama-server process.

## Architecture

### 1. A built-in `local` connection (ragcore)
- `ConnectionKind` gains `"local"`.
- A synthetic `Connection(id="local", kind="local", name="This computer", is_remote=False, has_api_key=False)` is always first in `GET /connections`. It is not stored in `connections.json`, cannot be edited or deleted (404/409 via `api_error`), and activating it sets `settings.active_connection_id = "local"`.
- `active_connection()` resolves `"local"` to the synthetic object. `activate` refuses (409 `local_model_missing`) unless `LocalLLM.status() == "ready"`.
- `llm_stream`: when `connection.kind == "local"`, the URL comes from `LocalLLM` (starting the server if needed) instead of `base_url`, no auth header, request body reuses the OpenAI-compatible path plus `chat_template_kwargs: {enable_thinking: false}`. Output cap defaults to 1024 tokens when the connection has none. Everything downstream (SSE framing, `extract_citations`, error mapping) is unchanged, so citations to passages that were not retrieved are still dropped.
- The existing `/suggestions/model` endpoints (status, install) are reused by the new UI; no new install route.

### 2. RAM only when needed (`local_llm.py`)
- The server is never started at app start or on activation. It starts on the first `generate()` or answer.
- New `lease()` async context manager: increments an in-flight counter and cancels the idle timer; on exit, decrements and, at zero, arms the timer. `llm_stream` holds a lease for the whole stream, so the server cannot be stopped mid-answer (today `_touch()` only runs around `generate()`).
- `IDLE_S` 180 → 60. `stop()` terminates the process, which frees the model memory. The process is also stopped on app shutdown (already wired in `app.py`).
- Starter questions and answers share the one instance, so the model is never loaded twice.
- Context `-c 4096` → `8192` so retrieved passages fit. If the prompt still exceeds it, llama-server's 400 is surfaced as an `LlmError` with a code (`llm_http_error`, existing path).
- The first query after idle takes a few seconds. The query stream emits the existing `mode`/`sources` frames first, so the UI shows "Searching…"; the UI additionally shows "Loading the model…" while no token has arrived and the connection is `local`.

### 3. Speed detection
- Measured on real answers, in `query.py` (works without llama-server `timings`): count streamed pieces (one piece ≈ one token for llama-server) and time from the first to the last piece. `tokens_per_s = (n - 1) / (t_last - t_first)`. Model load and prompt processing are excluded.
- `DoneEvent` gains `tokens_per_s: float | None`. It is `None` for non-local connections and for answers under 30 pieces (too noisy).
- `api-types.ts` regenerated with `gen:types`.

### 4. Suggestion banner (frontend)
- Chat keeps a counter in `localStorage` (`localSlowStreak`, `localSlowDismissed`): each local answer with `tokens_per_s < 5` increments the streak, any answer at or above resets it. Streak ≥ 2 and not dismissed shows a banner under the answer:
  "Answers are slow on this computer (~N tok/s). A cloud model would be faster." Buttons: **Set up OpenRouter** and **Keep using this computer**.
- Set up OpenRouter navigates to Settings → Connections with the OpenRouter form preselected and a short list of suggested models (constant `OPENROUTER_SUGGESTED` in the frontend, IDs verified against openrouter.ai when implementing). The form shows the existing remote warning: retrieved passages are sent to the provider.
- Keep using sets `localSlowDismissed` permanently (reset by a "Show speed tips again" toggle is out of scope).
- Threshold 5 tok/s is a constant, tuned during manual testing.

### 5. Onboarding and settings
- Onboarding step 3 shows two cards: **Run on this computer** (private, no key, ~1.1 GB download, progress from `/suggestions/model`; activating happens when ready) and **Use a service** (today's flow, unchanged).
- Settings → Connections lists "This computer" with its state (missing / downloading / ready) and an Install button when missing.
- All new strings go through the existing i18n catalogs (en/it/fr/de/es), with error codes `local_model_missing` added.

## Data flow
`/query` → retrieve → `ConnectionAnswerEngine` → `llm_stream` (kind `local`) → `LocalLLM.lease()` starts llama-server if stopped → SSE tokens → citations → `done{tokens_per_s}` → UI updates streak → maybe banner. Lease released → 60 s idle → server stops, RAM freed.

## Error handling
- Model not installed but `local` active (files deleted): `/query` raises `LlmError("local_model_missing")`, UI offers Install.
- Server fails to start or becomes unhealthy within `START_TIMEOUT_S`: existing `RuntimeError` is wrapped in an `LlmError`; the chat shows it with a retry.
- Download failure: existing `LocalLLM.error`, shown in onboarding/settings, `install()` retries.
- No active connection: unchanged (scripted stub).

## Testing
- pytest: synthetic connection listed first, not deletable, activation refused when missing and allowed when ready (with a fake `LocalLLM`).
- pytest: `lease()` — server not stopped while a lease is held, stopped after `IDLE_S` once released, never started by `activate`.
- pytest: `tokens_per_s` computed from a fake stream with controlled timing; `None` under 30 pieces and for remote connections.
- pytest: `llm_stream` with kind `local` uses the lease URL, sends no auth header and sets `enable_thinking: false`.
- Vitest: slow-streak logic (2 slow answers show the banner, a fast one resets, dismissal sticks), catalog parity for new keys.
- Quality gate (manual, before merge): run `scripts/eval` over `fixtures/docs` with Qwen3-1.7B as the active connection; record citation validity and answer quality in `docs/superpowers/notes/`. If answers are unusable, stop and revisit the model (e.g. 4B) before shipping.
- Manual: confirm RAM drops after the idle window (Activity Monitor), kill the app mid-answer and check no orphan llama-server remains.

## Rollout
Commits: (1) `lease()` + idle/context changes, (2) `local` connection + `llm_stream` branch, (3) `tokens_per_s` in `done` + gen:types, (4) onboarding/settings UI, (5) speed banner, (6) i18n strings. After changes run `scripts/install-local.sh` (project rule). No release.

## Risks
- A 1.7B model may answer poorly. Mitigated by the eval gate and by keeping connections available.
- 8192 context raises RAM use; measure it and lower to 6144 if needed.
- Piece count is an approximation of token count; fine for a slow/not-slow decision.
- A stale `localSlowDismissed` in `localStorage` can hide the banner forever; acceptable for v1.
