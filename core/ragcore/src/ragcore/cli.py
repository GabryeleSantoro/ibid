"""Command line entry point.

The Rust shell spawns ``serve`` with an explicit port, session token and the
hardware budget it measured. Running it by hand works too: every flag has a
sensible default and a random session token is generated (and logged) when none is given.

``index`` and ``ask`` drive the real backend headless, against the llama-servers
``scripts/dev/serve-models.sh`` starts.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import secrets
import sys
from pathlib import Path

import httpx
import uvicorn

from ragcore import __version__
from ragcore.api.app import create_app
from ragcore.api.schemas import QueryFilters, SourceCreate
from ragcore.backend import build_backend
from ragcore.citations import extract_citations
from ragcore.config import Config
from ragcore.llm import LlmError


def _adopt_legacy_data_dir(data_dir: Path) -> None:
    """Before the rename to Ibid, data lived in ~/.custom-rag: move it over once."""
    legacy = Path.home() / ".custom-rag"
    if data_dir == Path.home() / ".ibid" and not data_dir.exists() and legacy.is_dir():
        legacy.rename(data_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ragcore", description="RAG core sidecar")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-dir", type=Path, default=Path.home() / ".ibid")
    common.add_argument("--embed-url", default="http://127.0.0.1:8770")
    common.add_argument("--rerank-url", default="http://127.0.0.1:8771")

    serve = sub.add_parser("serve", parents=[common], help="Run the HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765, help="0 picks a free port")
    serve.add_argument("--token", default="", help="Session token; a random one is logged if empty")
    serve.add_argument("--ram-mb", type=int, default=None)
    serve.add_argument("--vram-mb", type=int, default=0)
    serve.add_argument("--gpu-backend", default="cpu", choices=["metal", "cuda", "vulkan", "cpu"])
    # The shell passes --backend real; ragcore supervises the llama-servers itself
    # (model_servers.py). The stub stays the default for bare `ragcore serve` dev runs.
    serve.add_argument("--backend", default="stub", choices=["stub", "real"])
    serve.add_argument("--prod", action="store_true", help="Disable dev-only routes")
    serve.add_argument("--log-level", default="info")

    index = sub.add_parser("index", parents=[common], help="Index a folder with the real backend")
    index.add_argument("path", type=Path)

    ask = sub.add_parser("ask", parents=[common], help="Ask the real backend's index a question")
    ask.add_argument("question")
    ask.add_argument("--top-k", type=int, default=None, help="Passages to answer from")
    return parser


def _real_backend(args: argparse.Namespace):
    config = Config(
        data_dir=args.data_dir,
        backend="real",
        embed_url=args.embed_url,
        rerank_url=args.rerank_url,
    )
    return build_backend(config)


async def _index(args: argparse.Namespace) -> int:
    backend = _real_backend(args)
    store = backend.store
    path = str(args.path.resolve())
    try:
        source = next((s for s in store.sources.values() if s.path == path), None)
        if source is None:
            source = store.add_source(SourceCreate(path=path))
        documents = await store.ingest_source_async(source.id)
    except httpx.HTTPError as exc:
        print(f"error: embedder at {args.embed_url} failed: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:  # the index was built by another embedder
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        store.close()

    for document in documents:
        if document.status == "error":
            print(f"failed: {document.path}: {document.error}", file=sys.stderr)
    unchanged = sum(d.status == "skipped" for d in documents)
    chunks = sum(d.n_chunks for d in documents)
    print(f"{len(documents)} document(s) ({unchanged} unchanged), {chunks} chunk(s) indexed")
    return 0


async def _ask(args: argparse.Namespace) -> int:
    backend = _real_backend(args)
    store = backend.store
    try:
        if store.index_stats().documents == 0:
            print("No documents in the index: run `ragcore index <folder>` first.")
            return 1
        settings = store.settings.retrieval
        if args.top_k:
            settings = settings.model_copy(update={"top_k": args.top_k})
        chunks, _, _ = await store.retriever.asearch(
            args.question, settings=settings, filters=QueryFilters(), doc_meta=store.doc_meta()
        )
        answer = "".join([p async for p in backend.answerer.stream(args.question, chunks, set())])
    except (httpx.HTTPError, LlmError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        backend.local_llm.stop()
        store.close()

    print(answer)
    citations, _, _ = extract_citations(answer, chunks)
    if citations:
        print()
    for citation in citations:
        print(f"{citation.marker} {citation.doc_title}, p. {citation.page}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 1

    _adopt_legacy_data_dir(args.data_dir)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    if args.command == "index":
        return asyncio.run(_index(args))
    if args.command == "ask":
        return asyncio.run(_ask(args))

    # No timestamp: the Rust shell stamps every line it collects.
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(name)s: %(message)s")

    token = args.token
    if not token:
        token = secrets.token_urlsafe(24)
        logging.getLogger("ragcore").warning("no --token given; generated one: %s", token)

    config = Config(
        host=args.host,
        port=args.port,
        token=token,
        data_dir=args.data_dir,
        dev_mode=not args.prod,
        vram_mb=args.vram_mb,
        gpu_backend=args.gpu_backend,
        backend=args.backend,
        embed_url=args.embed_url,
        rerank_url=args.rerank_url,
    )
    if args.ram_mb:
        config.ram_mb = args.ram_mb

    uvicorn.run(
        create_app(config),
        host=config.host,
        port=config.port,
        log_level=args.log_level,
        access_log=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
