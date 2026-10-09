#!/usr/bin/env bash
# Next patch version after the highest v* tag. No tags yet -> 0.0.1 (pre-alpha baseline).
set -euo pipefail

next_after() {
  local last="${1#v}"
  [ -z "$last" ] && { echo "0.0.1"; return; }
  IFS=. read -r major minor patch <<< "$last"
  patch="${patch%%-*}"  # v0.0.9-alpha -> 9
  echo "$major.$minor.$((patch + 1))"
}

if [ "${1:-}" = "--self-test" ]; then
  [ "$(next_after "")" = "0.0.1" ] || { echo "FAIL: empty"; exit 1; }
  [ "$(next_after "v0.0.1")" = "0.0.2" ] || { echo "FAIL: v0.0.1"; exit 1; }
  [ "$(next_after "v0.0.9")" = "0.0.10" ] || { echo "FAIL: v0.0.9"; exit 1; }
  [ "$(next_after "v1.4.0")" = "1.4.1" ] || { echo "FAIL: v1.4.0"; exit 1; }
  [ "$(next_after "v0.0.9-alpha")" = "0.0.10" ] || { echo "FAIL: v0.0.9-alpha"; exit 1; }
  echo "ok"
  exit 0
fi

next_after "$(git tag -l 'v[0-9]*' --sort=-v:refname | head -1)"
