"""Fixtures for the stub core's contract tests.

Every test runs against a real app instance with auth on, because the token
middleware is part of the contract the Rust shell depends on.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from ragcore.api.app import create_app
from ragcore.config import Config

TOKEN = "test-token"


FIXTURE_DOCS = Path(__file__).resolve().parents[3] / "fixtures" / "docs"


def _config(tmp_path: Path, backend: str = "stub") -> Config:
    return Config(
        host="127.0.0.1",
        port=0,
        token=TOKEN,
        data_dir=tmp_path,
        dev_mode=True,
        ram_mb=16384,
        vram_mb=0,
        gpu_backend="cpu",
        backend=backend,
    )


@pytest.fixture
def client(request, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """The stub app, or either backend under ``@both_backends``.

    The real backend indexes the documents the stub seeds itself with, through
    fake models, so both answer the same corpus.
    """
    backend = getattr(request, "param", "stub")
    monkeypatch.setenv("RAGCORE_FAKE_MODELS", "1")
    with TestClient(create_app(_config(tmp_path, backend))) as test_client:
        test_client.headers["Authorization"] = f"Bearer {TOKEN}"
        if backend == "real":
            response = test_client.post(
                "/sources", json={"path": str(FIXTURE_DOCS), "include_globs": ["**/*.md"]}
            )
            assert response.status_code == 201, response.text
        yield test_client


@pytest.fixture
def read_events() -> Callable[..., list[tuple[str, dict]]]:
    """Collect an SSE body into (event, payload) pairs, in arrival order."""
    return _read_events


def _read_events(response) -> list[tuple[str, dict]]:
    """Collect an SSE body into (event, payload) pairs, in arrival order."""
    events: list[tuple[str, dict]] = []
    name: str | None = None
    for raw in response.iter_lines():
        line = raw.decode() if isinstance(raw, bytes) else raw
        if line.startswith("event:"):
            name = line.removeprefix("event:").strip()
        elif line.startswith("data:") and name is not None:
            events.append((name, json.loads(line.removeprefix("data:").strip())))
            name = None
    return events


@pytest.fixture
def text_pdf() -> Callable[..., bytes]:
    """Build a tiny valid PDF, one page per argument; "" is a page with no text layer."""
    return _text_pdf


def _text_pdf(*pages: str) -> bytes:
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(len(pages)))
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode("ascii"),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for i, text in enumerate(pages):
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET\n".encode("ascii") if text else b""
        objects.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 3 0 R >> >> /Contents "
            + f"{5 + 2 * i} 0 R >>".encode("ascii")
        )
        objects.append(
            f"<< /Length {len(stream)} >>\nstream\n".encode("ascii") + stream + b"endstream"
        )
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{number} 0 obj\n".encode("ascii") + body + b"\nendobj\n")
    xref = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("ascii"))
    pdf.extend(b"".join(f"{offset:010d} 00000 n \n".encode("ascii") for offset in offsets))
    pdf.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode(
            "ascii"
        )
    )
    return bytes(pdf)


@pytest.fixture
def wait_for_topics() -> Callable:
    """Block until the app's brain has a tree; the ingest in ``client`` scheduled it."""

    def wait(client: TestClient):
        brain = client.app.state.store.brain
        deadline = time.monotonic() + 15
        while (tree := brain.tree()).status != "ready":
            assert time.monotonic() < deadline, tree
            time.sleep(0.02)
        return tree

    return wait
