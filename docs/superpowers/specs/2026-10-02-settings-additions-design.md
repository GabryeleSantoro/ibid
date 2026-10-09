# Settings additions — design

## Intent

Fill daily-use gaps in Settings. Today: General (language), Connections, Retrieval,
Performance, Storage, Updates. Chosen scope: Appearance, Chat defaults, Shortcuts,
Launch at login. Out of scope: indexing options, data export/import, notifications,
chunk-size tuning, shortcut rebinding, "restore last session".

## Final sidebar

general, connections, **appearance**, **chat**, retrieval, performance, **shortcuts**,
storage, updates. (`SETTINGS_SECTIONS` in `settings-sidebar.tsx`; render branches in
`settings-panel.tsx`.)

## 1. Appearance (frontend only)

- New section. Theme select (light / dark / system) wired to the existing
  `components/shell/theme-provider.tsx`; the icon-rail toggle stays.
- Font size (S / M / L) stored in localStorage (wrapped in try/catch), applied as
  root `font-size`. Default M.
- No backend change.

## 2. Chat defaults

- New section with one textarea, "Extra instructions" (e.g. "keep answers short").
- Backend: add `chat_extra_instructions: str = ""` (max 1000 chars) to `AppSettings`
  and `AppSettingsPatch` in `api/schemas.py`. `routes/settings.py` already patches
  generically. The persisted `settings.json` loads with the default when the field
  is absent.
- `routes/query.py` appends it to `system_prompt_for(lang)` after the grounding and
  citation rules. Those rules are not editable, so `citations.py` validation is
  unaffected. Empty value = no change.
- Regenerate `src/lib/api-types.ts` (`bun run gen:types`).
- Tests (pytest): extra text reaches the system prompt; empty leaves it identical;
  over-length is rejected.

## 3. Shortcuts (frontend only)

- Fixed app-wide set, no rebinding: ⌘N new chat, ⌘K focus search/composer,
  ⌘, open Settings, ⌘1/2/3 switch screens (Ctrl on non-macOS).
- One `useAppShortcuts` hook mounted in the root layout. The Shortcuts section renders
  a read-only table of the same list (single source constant).
- Test (vitest): keydown triggers the expected navigation.

## 4. Launch at login (Rust shell)

- Add `tauri-plugin-autostart` to `Cargo.toml`, register it in `run()`, add the
  permission to `capabilities/default.json`.
- Toggle in General. State is read from and written to the plugin (OS-owned); not
  stored in `AppSettings`.
- Verify: `cd src-tauri && cargo test --lib`, plus a manual check in macOS Login Items.

## Cross-cutting

- New i18n keys in all five locales (en, it, de, es, fr).
- `bun run typecheck`, `bun run test`, `uv run pytest`, `uv run ruff check .` pass.
- After the change run `scripts/install-local.sh`. No release.

## Risks

- Shortcuts may collide with webview defaults (⌘N, ⌘,); `preventDefault` in the hook,
  confirm in the built app.
- Autostart plugin behaviour in dev vs. bundled app differs; verify on the installed build.
