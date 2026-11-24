#!/usr/bin/env bash
# Freezes ragcore into the folder (onedir) the packaged app spawns: binaries/ragcore/ragcore.
# Not --onefile: that unpacks ~230 MB to a temp dir on every launch (12+ s).
# Dev builds run it from the workspace through uv instead (see sidecars.rs).
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

rm -rf apps/desktop/src-tauri/binaries/ragcore  # drops a stale onefile binary too
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

uv run --group build pyinstaller \
  --name ragcore --onedir --noconfirm --clean \
  --distpath apps/desktop/src-tauri/binaries \
  --workpath "$work/build" --specpath "$work" \
  --collect-all lancedb \
  --collect-all pyarrow \
  --collect-all pypdfium2 \
  --collect-all rapidocr \
  --collect-all onnxruntime \
  --collect-data py3langid \
  --collect-submodules watchdog \
  --collect-all uvicorn \
  --add-data "$root/fixtures/docs:fixtures/docs" \
  core/ragcore/src/ragcore/__main__.py

apps/desktop/src-tauri/binaries/ragcore/ragcore --version
