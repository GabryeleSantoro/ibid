#!/usr/bin/env bash
# Freezes ragcore into the single binary the packaged app spawns.
# Dev builds run it from the workspace through uv instead (see sidecars.rs).
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

uv run --group build pyinstaller \
  --name ragcore --onefile --noconfirm --clean \
  --distpath apps/desktop/src-tauri/binaries \
  --workpath "$work/build" --specpath "$work" \
  --collect-all lancedb \
  --collect-all pyarrow \
  --collect-all pypdfium2 \
  --collect-all rapidocr \
  --collect-all onnxruntime \
  --collect-all uvicorn \
  --add-data "$root/fixtures/docs:fixtures/docs" \
  core/ragcore/src/ragcore/__main__.py

apps/desktop/src-tauri/binaries/ragcore --version
