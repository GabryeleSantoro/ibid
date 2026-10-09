# Settings Additions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Appearance, Chat, Shortcuts settings sections and a "Launch at login" toggle to Ibid.

**Architecture:** Appearance and Shortcuts are frontend-only (theme provider already exists; shortcuts are a fixed list matched by a pure function). Chat adds one `chat_extra_instructions` field to `AppSettings`, appended to the system prompt in `llm.py`. Launch at login uses `tauri-plugin-autostart`; state is OS-owned and read through the JS plugin API.

**Tech Stack:** React 19 + TanStack Router/Query, vitest (pure-logic tests, no DOM), FastAPI/pydantic, pytest, Tauri 2 plugins, react-i18next (5 locales).

**Spec:** `docs/superpowers/specs/2026-10-02-settings-additions-design.md`

## Global Constraints

- Sidebar order: general, connections, appearance, chat, retrieval, performance, shortcuts, storage, updates.
- `chat_extra_instructions: str = ""`, max 1000 chars; empty = prompt unchanged.
- Grounding/citation rules in `SYSTEM_PROMPT` are not editable; extra text goes after them (`citations.py` invariant untouched).
- Shortcuts: ⌘N new chat, ⌘K focus composer, ⌘, open Settings, ⌘1/2/3 chat/library/convert; Ctrl instead of ⌘ off macOS. No rebinding UI.
- Font size S/M/L, default M, stored in localStorage inside try/catch.
- Autostart state is not stored in `AppSettings`.
- All new UI text via i18n keys in en, it, de, es, fr (locale tests enforce parity and no hardcoded text).
- Python line length 100; `uv run ruff check .` clean.
- After the final task run `scripts/install-local.sh`. No release.

## Review Focus

- Whitespace-only extra instructions: treated as empty (prompt identical to default). Pinned in Task 1.
- Extra instructions with `{}`/braces or newlines: appended verbatim, no formatting crash. Pinned in Task 1.
- Over-length (>1000) PATCH: rejected with 422, stored value unchanged. Pinned in Task 1.
- Older persisted `settings.json`/SQLite row lacking the new field: loads with `""`. Pinned in Task 1.
- Shortcut pressed while typing in the composer/inputs: ⌘N/⌘, still work, but bare keys never fire; ⌘K inside the composer is a no-op focus. Pinned in Task 4 (matcher ignores events with no modifier).
- `localStorage` throwing or holding garbage font size: falls back to M. Pinned in Task 2.

---

### Task 1: Chat extra instructions (backend)

**Files:**
- Modify: `core/ragcore/src/ragcore/api/schemas.py:517-533`
- Modify: `core/ragcore/src/ragcore/llm.py:29-36`
- Modify: `core/ragcore/src/ragcore/api/routes/query.py:155-157`
- Test: `core/ragcore/tests/test_llm.py`, `core/ragcore/tests/test_settings.py`, `core/ragcore/tests/test_query.py`

**Interfaces:**
- Produces: `AppSettings.chat_extra_instructions: str`, `AppSettingsPatch.chat_extra_instructions: str | None`, `system_prompt_for(lang: str | None, extra: str = "") -> str`.

- [ ] **Step 1: Write failing tests**

Append to `core/ragcore/tests/test_llm.py`:

```python
def test_extra_instructions_follow_the_grounding_rules() -> None:
    from ragcore.llm import SYSTEM_PROMPT, system_prompt_for

    prompt = system_prompt_for(None, "Keep it short {x}\nUse bullets.")
    assert prompt.startswith(SYSTEM_PROMPT)
    assert prompt.endswith("Keep it short {x}\nUse bullets.")


def test_blank_extra_instructions_change_nothing() -> None:
    from ragcore.llm import SYSTEM_PROMPT, system_prompt_for

    assert system_prompt_for(None, "   \n") == SYSTEM_PROMPT
    assert system_prompt_for("fr", "") == system_prompt_for("fr")
```

Append to `core/ragcore/tests/test_settings.py`:

```python
def test_extra_instructions_default_empty_and_round_trip(client: TestClient) -> None:
    assert client.get("/settings").json()["chat_extra_instructions"] == ""
    patched = client.patch("/settings", json={"chat_extra_instructions": "Be brief"}).json()
    assert patched["chat_extra_instructions"] == "Be brief"
    assert client.get("/settings").json()["chat_extra_instructions"] == "Be brief"


def test_over_long_extra_instructions_are_rejected(client: TestClient) -> None:
    response = client.patch("/settings", json={"chat_extra_instructions": "x" * 1001})
    assert response.status_code == 422
    assert client.get("/settings").json()["chat_extra_instructions"] == ""


def test_settings_saved_before_the_field_existed_still_load() -> None:
    from ragcore.api.schemas import AppSettings

    old = AppSettings(storage_path="/tmp/x").model_dump_json(exclude={"chat_extra_instructions"})
    assert AppSettings.model_validate_json(old).chat_extra_instructions == ""
```

Append to `core/ragcore/tests/test_query.py`:

```python
def test_extra_instructions_reach_the_answer_engine(client: TestClient, read_events) -> None:
    from ragcore.api import deps

    captured: dict = {}

    class Recorder:
        async def stream(
            self, question, chunks, directives, *, system_prompt=None, max_tokens=None
        ):
            captured["system_prompt"] = system_prompt
            yield "ok"

    client.patch("/settings", json={"chat_extra_instructions": "Answer in bullets"})
    client.app.dependency_overrides[deps.get_answerer] = Recorder
    try:
        with client.stream("POST", "/query", json={"q": "Why rerank?"}) as response:
            read_events(response)
    finally:
        client.app.dependency_overrides.pop(deps.get_answerer, None)

    assert captured["system_prompt"].endswith("Answer in bullets")
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest core/ragcore/tests/test_llm.py core/ragcore/tests/test_settings.py core/ragcore/tests/test_query.py -v -k "extra"`
Expected: FAIL (`TypeError` on `system_prompt_for`, `KeyError: 'chat_extra_instructions'`).

- [ ] **Step 3: Implement**

`schemas.py`, add to both classes:

```python
class AppSettings(BaseModel):
    ...
    performance: PerformanceSettings = Field(default_factory=PerformanceSettings)
    chat_extra_instructions: str = Field(default="", max_length=1000)


class AppSettingsPatch(BaseModel):
    ...
    performance: PerformanceSettings | None = None
    chat_extra_instructions: str | None = Field(default=None, max_length=1000)
```

`llm.py`, replace `system_prompt_for`:

```python
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
```

`query.py`:

```python
        stream = answerer.stream(
            question,
            chunks,
            directives,
            system_prompt=system_prompt_for(payload.lang, store.settings.chat_extra_instructions),
        )
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest -v && uv run ruff check .`
Expected: all PASS, ruff clean (existing `endswith("Write the answer in French...")` test still passes because extra is empty).

- [ ] **Step 5: Regenerate types and commit**

Run (sidecar on :8765 in another shell: `uv run --directory core/ragcore ragcore serve --port 8765`): `cd apps/desktop && bun run gen:types`
Expected: `src/lib/api-types.ts` gains `chat_extra_instructions`.

```bash
git add core apps/desktop/src/lib/api-types.ts
git commit -m "feat: extra chat instructions appended to the answer prompt"
```

---

### Task 2: Appearance (font size + theme select)

**Files:**
- Create: `apps/desktop/src/lib/font-size.ts`, `apps/desktop/src/lib/font-size.test.ts`
- Modify: `apps/desktop/src/main.tsx` (apply on boot), `apps/desktop/src/features/settings/settings-panel.tsx`, `settings-sidebar.tsx`

**Interfaces:**
- Produces: `type FontSize = "s" | "m" | "l"`, `getFontSize(): FontSize`, `setFontSize(size: FontSize): void` (persists and applies), `applyFontSize(size: FontSize): void`.
- Consumes: `useTheme()` / `Theme` from `components/shell/theme-provider.tsx`.

- [ ] **Step 1: Write failing test** `font-size.test.ts`

```ts
import { afterEach, describe, expect, it, vi } from "vitest";
import { getFontSize, parseFontSize } from "./font-size";

describe("font size", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("accepts s, m, l and falls back to m for anything else", () => {
    expect(parseFontSize("s")).toBe("s");
    expect(parseFontSize("l")).toBe("l");
    expect(parseFontSize("huge")).toBe("m");
    expect(parseFontSize(null)).toBe("m");
  });

  it("falls back to m when storage throws", () => {
    vi.stubGlobal("localStorage", {
      getItem: () => {
        throw new Error("blocked");
      },
    });
    expect(getFontSize()).toBe("m");
  });
});
```

- [ ] **Step 2:** Run `cd apps/desktop && bunx vitest run src/lib/font-size.test.ts` — Expected: FAIL (module missing).

- [ ] **Step 3: Implement** `font-size.ts`

```ts
export type FontSize = "s" | "m" | "l";

const STORAGE_KEY = "ibid.fontSize";
const PX: Record<FontSize, string> = { s: "14px", m: "16px", l: "18px" };

export function parseFontSize(value: string | null): FontSize {
  return value === "s" || value === "l" ? value : "m";
}

export function getFontSize(): FontSize {
  try {
    return parseFontSize(localStorage.getItem(STORAGE_KEY));
  } catch {
    // Private windows and blocked site data both throw here.
    return "m";
  }
}

export function applyFontSize(size: FontSize) {
  document.documentElement.style.fontSize = PX[size];
}

export function setFontSize(size: FontSize) {
  applyFontSize(size);
  try {
    localStorage.setItem(STORAGE_KEY, size);
  } catch {
    // Not being able to remember the choice is not worth failing over.
  }
}
```

In `main.tsx`, before render: `import { applyFontSize, getFontSize } from "@/lib/font-size";` and `applyFontSize(getFontSize());`.

In `settings-sidebar.tsx` add `{ slug: "appearance" }` after connections. In `settings-panel.tsx` add:

```tsx
function AppearanceSection() {
  const { t } = useTranslation();
  const { theme, setTheme } = useTheme();
  const [size, setSize] = useState<FontSize>(getFontSize());

  return (
    <div className="space-y-6">
      <Field label={t("settings.appearance.theme")}>
        <Select value={theme} onValueChange={(value) => setTheme(value as Theme)}>
          <SelectTrigger><SelectValue /></SelectTrigger>
          <SelectContent>
            <SelectItem value="system">{t("nav.themeSystem")}</SelectItem>
            <SelectItem value="light">{t("nav.themeLight")}</SelectItem>
            <SelectItem value="dark">{t("nav.themeDark")}</SelectItem>
          </SelectContent>
        </Select>
      </Field>
      <Field label={t("settings.appearance.fontSize")} hint={t("settings.appearance.fontSizeHint")}>
        <Select
          value={size}
          onValueChange={(value) => {
            setSize(value as FontSize);
            setFontSize(value as FontSize);
          }}
        >
          <SelectTrigger><SelectValue /></SelectTrigger>
          <SelectContent>
            <SelectItem value="s">{t("settings.appearance.small")}</SelectItem>
            <SelectItem value="m">{t("settings.appearance.medium")}</SelectItem>
            <SelectItem value="l">{t("settings.appearance.large")}</SelectItem>
          </SelectContent>
        </Select>
      </Field>
    </div>
  );
}
```

Render `{section === "appearance" ? <AppearanceSection /> : null}`; import `useTheme`, `type Theme` from the theme provider and the font-size helpers.

i18n keys (add under `settings` in all 5 locales; en shown): `sections.appearance.label` "Appearance", `sections.appearance.blurb` "Theme and text size.", `appearance.theme` "Theme", `appearance.fontSize` "Text size", `appearance.fontSizeHint` "Scales the whole interface.", `appearance.small` "Small", `appearance.medium` "Medium", `appearance.large` "Large". Translate for it/de/es/fr.

- [ ] **Step 4:** Run `bunx vitest run && bun run typecheck` — Expected: PASS (locale parity tests green).

- [ ] **Step 5: Commit**

```bash
git add apps/desktop/src
git commit -m "feat: appearance settings with theme select and text size"
```

---

### Task 3: Chat settings section (frontend)

**Files:**
- Modify: `settings-sidebar.tsx`, `settings-panel.tsx`, the 5 locale files

**Interfaces:**
- Consumes: `useSettingsDraft()` (existing in `settings-panel.tsx`), `Textarea` from `@/components/ui/textarea`, `AppSettings.chat_extra_instructions` (Task 1).

- [ ] **Step 1:** Add `{ slug: "chat" }` after appearance in `SETTINGS_SECTIONS`. Add:

```tsx
function ChatSection() {
  const { t } = useTranslation();
  const { draft, setDraft, save } = useSettingsDraft();
  if (!draft) return <Skeleton className="h-40 w-full" />;

  return (
    <div className="space-y-6">
      <Field label={t("settings.chat.extra")} hint={t("settings.chat.extraHint")}>
        <Textarea
          rows={5}
          maxLength={1000}
          placeholder={t("settings.chat.extraPlaceholder")}
          value={draft.chat_extra_instructions}
          onChange={(event) => setDraft({ ...draft, chat_extra_instructions: event.target.value })}
        />
      </Field>
      <Button
        disabled={save.isPending}
        onClick={() => save.mutate({ chat_extra_instructions: draft.chat_extra_instructions })}
      >
        {t("common.save")}
      </Button>
    </div>
  );
}
```

Field's right column is `max-w-xs`; that is narrow for a textarea, so change `Field`'s wrapper to accept an optional `wide` prop (`className={cn("max-w-xs", wide && "max-w-xl")}`) and pass `wide` here.

Render `{section === "chat" ? <ChatSection /> : null}`.

i18n (en): `sections.chat.label` "Chat", `sections.chat.blurb` "Defaults for how answers are written.", `chat.extra` "Extra instructions", `chat.extraHint` "Added to every question. Answers still only use your documents and keep their citations.", `chat.extraPlaceholder` "e.g. Keep answers short and use bullet points." Translate for it/de/es/fr.

- [ ] **Step 2:** Run `cd apps/desktop && bunx vitest run && bun run typecheck` — Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add apps/desktop/src
git commit -m "feat: chat extra instructions setting"
```

---

### Task 4: Shortcuts

**Files:**
- Create: `apps/desktop/src/lib/shortcuts.ts`, `apps/desktop/src/lib/shortcuts.test.ts`
- Modify: `apps/desktop/src/routes/_shell.tsx`, `apps/desktop/src/features/chat/composer.tsx`, `settings-sidebar.tsx`, `settings-panel.tsx`, locales

**Interfaces:**
- Produces:
  - `type ShortcutId = "newChat" | "focusComposer" | "openSettings" | "goChat" | "goLibrary" | "goConvert"`
  - `SHORTCUTS: { id: ShortcutId; key: string }[]` (key lowercase, e.g. `"n"`, `","`, `"1"`)
  - `matchShortcut(event: { key: string; metaKey: boolean; ctrlKey: boolean; altKey: boolean; shiftKey: boolean }, mac: boolean): ShortcutId | null`
  - `COMPOSER_FOCUS_EVENT = "ibid:focus-composer"` (window CustomEvent name)

- [ ] **Step 1: Write failing test** `shortcuts.test.ts`

```ts
import { describe, expect, it } from "vitest";
import { matchShortcut } from "./shortcuts";

const key = (k: string, extra: Partial<Parameters<typeof matchShortcut>[0]> = {}) => ({
  key: k, metaKey: false, ctrlKey: false, altKey: false, shiftKey: false, ...extra,
});

describe("matchShortcut", () => {
  it("uses cmd on mac", () => {
    expect(matchShortcut(key("n", { metaKey: true }), true)).toBe("newChat");
    expect(matchShortcut(key(",", { metaKey: true }), true)).toBe("openSettings");
    expect(matchShortcut(key("2", { metaKey: true }), true)).toBe("goLibrary");
  });
  it("uses ctrl elsewhere and ignores cmd", () => {
    expect(matchShortcut(key("k", { ctrlKey: true }), false)).toBe("focusComposer");
    expect(matchShortcut(key("k", { metaKey: true }), false)).toBeNull();
  });
  it("never fires on a bare key (typing in inputs)", () => {
    expect(matchShortcut(key("n"), true)).toBeNull();
    expect(matchShortcut(key("1"), false)).toBeNull();
  });
  it("ignores extra modifiers and unknown keys", () => {
    expect(matchShortcut(key("n", { metaKey: true, shiftKey: true }), true)).toBeNull();
    expect(matchShortcut(key("n", { metaKey: true, altKey: true }), true)).toBeNull();
    expect(matchShortcut(key("z", { metaKey: true }), true)).toBeNull();
  });
});
```

- [ ] **Step 2:** Run `bunx vitest run src/lib/shortcuts.test.ts` — Expected: FAIL (module missing).

- [ ] **Step 3: Implement** `shortcuts.ts`

```ts
export type ShortcutId =
  | "newChat"
  | "focusComposer"
  | "openSettings"
  | "goChat"
  | "goLibrary"
  | "goConvert";

export const SHORTCUTS: { id: ShortcutId; key: string }[] = [
  { id: "newChat", key: "n" },
  { id: "focusComposer", key: "k" },
  { id: "openSettings", key: "," },
  { id: "goChat", key: "1" },
  { id: "goLibrary", key: "2" },
  { id: "goConvert", key: "3" },
];

export const COMPOSER_FOCUS_EVENT = "ibid:focus-composer";

type KeyLike = { key: string; metaKey: boolean; ctrlKey: boolean; altKey: boolean; shiftKey: boolean };

export function matchShortcut(event: KeyLike, mac: boolean): ShortcutId | null {
  const mod = mac ? event.metaKey && !event.ctrlKey : event.ctrlKey && !event.metaKey;
  if (!mod || event.altKey || event.shiftKey) return null;
  return SHORTCUTS.find((s) => s.key === event.key.toLowerCase())?.id ?? null;
}
```

In `_shell.tsx` `Shell`, add an effect (imports: `isMac` from `@/lib/platform`, helpers above):

```tsx
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const id = matchShortcut(event, isMac);
      if (!id) return;
      event.preventDefault();
      if (id === "focusComposer") {
        window.dispatchEvent(new Event(COMPOSER_FOCUS_EVENT));
        return;
      }
      const to = { newChat: "/chat", openSettings: "/settings", goChat: "/chat", goLibrary: "/library", goConvert: "/convert" }[id];
      void navigate({ to });
      if (id === "newChat") window.dispatchEvent(new Event(COMPOSER_FOCUS_EVENT));
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [navigate]);
```

In `composer.tsx` add next to the existing effect (focus after route change may happen before mount, so also defer one tick):

```tsx
  useEffect(() => {
    const focus = () => setTimeout(() => ref.current?.focus(), 0);
    window.addEventListener(COMPOSER_FOCUS_EVENT, focus);
    return () => window.removeEventListener(COMPOSER_FOCUS_EVENT, focus);
  }, []);
```

Settings: add `{ slug: "shortcuts" }` after performance. Add:

```tsx
function ShortcutsSection() {
  const { t } = useTranslation();
  const mod = isMac ? "⌘" : "Ctrl+";
  return (
    <div className="max-w-md divide-y rounded-lg border">
      {SHORTCUTS.map((s) => (
        <div key={s.id} className="flex items-center justify-between px-3 py-2 text-[0.8125rem]">
          <span>{t(`settings.shortcuts.${s.id}`)}</span>
          <kbd className="rounded bg-muted px-1.5 py-0.5 font-mono text-[0.6875rem]">
            {mod}{s.key.toUpperCase()}
          </kbd>
        </div>
      ))}
    </div>
  );
}
```

Render `{section === "shortcuts" ? <ShortcutsSection /> : null}`.

i18n (en) under `settings`: `sections.shortcuts.label` "Shortcuts", `sections.shortcuts.blurb` "Keyboard shortcuts that work anywhere in the app.", `shortcuts.newChat` "New chat", `shortcuts.focusComposer` "Focus the question box", `shortcuts.openSettings` "Open Settings", `shortcuts.goChat` "Go to Chat", `shortcuts.goLibrary` "Go to Library", `shortcuts.goConvert` "Go to Slides to text". Translate for it/de/es/fr. (`keys-exist.test.ts` checks dynamic keys; if it cannot see template keys, list the ids explicitly.)

- [ ] **Step 4:** Run `bunx vitest run && bun run typecheck` — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/desktop/src
git commit -m "feat: app-wide keyboard shortcuts with a read-only reference"
```

---

### Task 5: Launch at login

**Files:**
- Modify: `apps/desktop/src-tauri/Cargo.toml`, `apps/desktop/src-tauri/src/lib.rs:97-101`, `apps/desktop/src-tauri/capabilities/default.json`, `apps/desktop/package.json`, `settings-panel.tsx`, locales

**Interfaces:**
- Consumes: JS `isEnabled()`, `enable()`, `disable()` from `@tauri-apps/plugin-autostart`.

- [ ] **Step 1: Add the plugin**

Run: `cd apps/desktop && bun add @tauri-apps/plugin-autostart && cd src-tauri && cargo add tauri-plugin-autostart@2`

Because autostart is desktop-only, in `Cargo.toml` move it under `[target.'cfg(not(any(target_os = "android", target_os = "ios")))'.dependencies]` if `cargo add` placed it in `[dependencies]` and mobile is a target (this app is desktop-only, so keeping it in `[dependencies]` is fine).

In `lib.rs` add after the process plugin:

```rust
        .plugin(tauri_plugin_autostart::init(
            tauri_plugin_autostart::MacosLauncher::LaunchAgent,
            None::<Vec<&str>>,
        ))
```

In `capabilities/default.json` permissions add: `"autostart:allow-enable"`, `"autostart:allow-disable"`, `"autostart:allow-is-enabled"`.

- [ ] **Step 2: Toggle in General** (`settings-panel.tsx`)

```tsx
function LaunchAtLogin() {
  const { t } = useTranslation();
  const [enabled, setEnabled] = useState<boolean | null>(null);

  useEffect(() => {
    isEnabled().then(setEnabled).catch(() => setEnabled(null));
  }, []);

  const toggle = async (next: boolean) => {
    try {
      await (next ? enable() : disable());
      setEnabled(next);
    } catch (error) {
      toast.error(t("settings.startup.failed"), { description: errorText(error as Error) });
    }
  };

  // Browser/dev without the shell: the plugin call rejects and the row stays hidden.
  if (enabled === null) return null;
  return (
    <Field label={t("settings.startup.label")} hint={t("settings.startup.hint")}>
      <Switch checked={enabled} onCheckedChange={toggle} />
    </Field>
  );
}
```

Render `<LaunchAtLogin />` inside `GeneralSection` after the language field. Import `{ disable, enable, isEnabled } from "@tauri-apps/plugin-autostart"`.

i18n (en): `startup.label` "Launch at login", `startup.hint` "Open Ibid when you sign in to your computer.", `startup.failed` "Could not change the login setting". Translate for it/de/es/fr.

- [ ] **Step 3: Verify**

Run: `cd apps/desktop && bunx vitest run && bun run typecheck && cd src-tauri && cargo test --lib`
Expected: PASS, cargo compiles with the plugin.

- [ ] **Step 4: Commit**

```bash
git add apps/desktop
git commit -m "feat: launch at login toggle"
```

---

### Task 6: Full verification and local install

- [ ] **Step 1:** Run `uv run pytest -v && uv run ruff check . && cd apps/desktop && bun run test && bun run typecheck && (cd src-tauri && cargo test --lib)` — Expected: all PASS.
- [ ] **Step 2:** Run `scripts/install-local.sh` from the repo root (project rule: after every change).
- [ ] **Step 3: Manual check in the installed app:** ⌘N/⌘K/⌘,/⌘1-3 each act and the webview doesn't swallow ⌘N or ⌘,; theme and text size apply and survive restart; set "Answer in bullets" under Chat and confirm the next answer follows it and still cites; toggle Launch at login and see Ibid appear/disappear in System Settings > General > Login Items.
- [ ] **Step 4:** Report results to the user; do not tag a release.
