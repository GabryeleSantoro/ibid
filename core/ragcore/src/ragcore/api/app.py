"""FastAPI application factory."""

from __future__ import annotations

import secrets
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ragcore import __version__
from ragcore.api.errors import ApiError, api_error_handler
from ragcore.api.routes import (
    chats,
    connections,
    conversions,
    documents,
    evals,
    folders,
    health,
    jobs,
    models,
    query,
    settings,
    sources,
    suggestions,
)
from ragcore.backend import build_backend
from ragcore.config import Config

# Everything else needs the session token the Rust shell generated at spawn.
PUBLIC_PATHS = {"/health", "/openapi.json", "/docs", "/redoc", "/docs/oauth2-redirect"}


def create_app(config: Config) -> FastAPI:
    if not config.token:
        raise ValueError("ragcore refuses to serve without a session token")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.config = config
        app.state.started_at = time.monotonic()
        backend = build_backend(config)
        app.state.backend = backend
        app.state.store = backend.store
        app.state.jobs = backend.jobs
        app.state.hub = backend.hub
        app.state.answerer = backend.answerer
        app.state.local_llm = backend.local_llm
        app.state.servers = backend.servers
        if backend.servers:
            await backend.servers.start()
        try:
            yield
        finally:
            if backend.servers:
                backend.servers.stop()
            app.state.local_llm.stop()
            await backend.jobs.shutdown()
            await backend.hub.aclose()

    app = FastAPI(
        title="ragcore",
        version=__version__,
        summary="Local RAG core: ingestion, retrieval and grounded generation.",
        lifespan=lifespan,
    )

    app.add_exception_handler(ApiError, api_error_handler)

    @app.middleware("http")
    async def require_token(request: Request, call_next):
        if request.url.path not in PUBLIC_PATHS:
            header = request.headers.get("authorization", "")
            presented = header.removeprefix("Bearer ").strip()
            if not secrets.compare_digest(presented.encode(), config.token.encode()):
                return JSONResponse({"detail": "unauthorized"}, status_code=401)
        return await call_next(request)

    for router in (
        health.router,
        sources.router,
        documents.router,
        jobs.router,
        query.router,
        chats.router,
        connections.router,
        conversions.router,
        models.router,
        settings.router,
        evals.router,
        folders.router,
        suggestions.router,
    ):
        app.include_router(router)

    return app
