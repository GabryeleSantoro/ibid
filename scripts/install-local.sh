#!/usr/bin/env bash
# Builds the app from the working tree and swaps it into /Applications, like an
# update would: data in ~/.ibid and keys in the keychain are untouched.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
app_name="Ibid.app"
bundle="$root/apps/desktop/src-tauri/target/release/bundle/macos/$app_name"
sidecar="$root/apps/desktop/src-tauri/binaries/ragcore/ragcore"

# PyInstaller is the slow part: only rerun it when the Python side changed.
if [ ! -x "$sidecar" ] || [ -n "$(find "$root/core/ragcore/src" "$root/fixtures" -newer "$sidecar" -print -quit)" ]; then
  "$root/scripts/build-sidecar.sh"
fi

# Updater artifacts need the CI signing key; a local install has no use for them.
# Stamp the next release's version: Cargo.toml lags the tags, and a lower version
# gets replaced by the latest release as soon as the auto-updater runs.
version="$("$root/scripts/next-version.sh")"
(cd "$root/apps/desktop" && bun tauri build --bundles app \
  --config "{\"version\":\"$version\",\"bundle\":{\"createUpdaterArtifacts\":false}}")

# Quit through AppKit, not kill: the shell's exit hook takes the sidecar down with it.
if pgrep -f "/Applications/$app_name/Contents/MacOS/" >/dev/null; then
  osascript -e 'tell application id "com.ibid.desktop" to quit' || true
  for _ in $(seq 1 50); do
    pgrep -f "/Applications/$app_name/Contents/" >/dev/null || break
    sleep 0.2
  done
  pkill -f "/Applications/$app_name/Contents/" || true
fi

# Before the rename the app was "Custom RAG": remove it so only Ibid remains.
if pgrep -f "/Applications/Custom RAG.app/Contents/MacOS/" >/dev/null; then
  osascript -e 'tell application id "com.customrag.desktop" to quit' || true
  sleep 2
  pkill -f "/Applications/Custom RAG.app/Contents/" || true
fi
rm -rf "/Applications/$app_name" "/Applications/Custom RAG.app"
ditto "$bundle" "/Applications/$app_name"
open "/Applications/$app_name"
echo "installed $(defaults read "/Applications/$app_name/Contents/Info" CFBundleShortVersionString) from $(git -C "$root" rev-parse --short HEAD)$(git -C "$root" diff --quiet || echo '+dirty')"
