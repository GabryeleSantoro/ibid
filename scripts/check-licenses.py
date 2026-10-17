#!/usr/bin/env python3
"""Fail if a runtime dependency of ragcore is under a strong copyleft license.

The sidecar is distributed as a frozen binary, so GPL/AGPL/SSPL code in it is a
problem (see the license guard in plan/ibid-build-plan.md). LGPL is allowed.
Walks ragcore's installed requirements transitively; dev and build tools are
not runtime and are not checked. Run: uv run python scripts/check-licenses.py
"""

from __future__ import annotations

import re
import sys
from importlib.metadata import PackageNotFoundError, distribution

COPYLEFT = re.compile(r"(?<!L)A?GPL|(?<!Lesser )General Public License|SSPL", re.IGNORECASE)
# Package -> why it is fine despite matching. Keep each entry justified.
ALLOW: dict[str, str] = {}


def licenses(dist) -> list[str]:
    """Most specific first: an SPDX expression, else classifiers, else the License field's
    first line (some packages paste whole license texts there, bundled ones included)."""
    meta = dist.metadata
    if meta.get("License-Expression"):
        return [meta["License-Expression"]]
    classifiers = [c for c in meta.get_all("Classifier") or [] if c.startswith("License ::")]
    if classifiers:
        return classifiers
    text = (meta.get("License") or "").strip()
    return [text.splitlines()[0]] if text else []


def main() -> int:
    seen: set[str] = set()
    todo = ["ragcore"]
    flagged: list[tuple[str, list[str]]] = []
    while todo:
        name = todo.pop()
        key = re.sub(r"[-_.]+", "-", name).lower()
        if key in seen:
            continue
        seen.add(key)
        try:
            dist = distribution(name)
        except PackageNotFoundError:  # a requirement for another platform
            continue
        for requirement in dist.requires or []:
            if "extra ==" not in requirement:
                todo.append(re.match(r"[A-Za-z0-9._-]+", requirement).group())
        found = licenses(dist)
        if key == "ragcore" or key in ALLOW:
            continue
        if not found or any(COPYLEFT.search(text) for text in found):
            flagged.append((key, found))
    for key, found in flagged:
        print(f"{key}: {found or 'no license metadata'}", file=sys.stderr)
    print(f"{len(seen)} runtime packages checked, {len(flagged)} flagged")
    return 1 if flagged else 0


if __name__ == "__main__":
    raise SystemExit(main())
