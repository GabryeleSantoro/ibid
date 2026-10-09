"""Chooses and assembles a backend for one process lifetime."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from dataclasses import dataclass

from ragcore.api.schemas import RetrievedChunk
from ragcore.config import Config
from ragcore.ports import AnswerEngine, StorePort


@dataclass(frozen=True, slots=True)
class Backend:
    store: StorePort
    jobs: object
    hub: object
    answerer: AnswerEngine
    local_llm: object
    servers: object | None = None


class ConnectionAnswerEngine:
    """Scripted until the user activates a connection; then that connection answers."""

    def __init__(self, store: StorePort, local_llm: object | None = None) -> None:
        self.store = store
        self.local_llm = local_llm

    def stream(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        directives: set[str],
        *,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        from ragcore.llm import llm_stream
        from ragcore.stub.answers import scripted_stream

        connection = self.store.active_connection()
        if connection is None:
            return scripted_stream(question, chunks, directives)
        return llm_stream(
            connection,
            self.store.secrets.get(connection.id),
            question,
            chunks,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            local=self.local_llm,
        )


def build_backend(config: Config) -> Backend:
    from ragcore.stub.hub import HubClient
    from ragcore.stub.jobs import JobManager

    if config.backend == "stub":
        from ragcore.stub.store import Store

        store = Store(config)
    elif config.backend == "real":
        from ragcore.models.embed import EmbedClient
        from ragcore.models.fakes import FakeEmbedClient
        from ragcore.models.rerank import RerankClient
        from ragcore.store.real import RealStore

        # A test affordance; the CLI never sets it. No reranker under fakes: the
        # fake's overlap scores sit under min_score and would empty contract answers.
        if os.getenv("RAGCORE_FAKE_MODELS"):
            store = RealStore(config, embedder=FakeEmbedClient())
        else:
            store = RealStore(
                config,
                embedder=EmbedClient(config.embed_url),
                reranker=RerankClient(config.rerank_url),
            )
    else:
        raise ValueError(f"unknown backend: {config.backend!r}")
    from ragcore.local_llm import LocalLLM

    local_llm = LocalLLM(config.data_dir)
    servers = None
    if config.backend == "real" and not os.getenv("RAGCORE_FAKE_MODELS"):
        from ragcore.model_servers import ModelServers

        servers = ModelServers(config.data_dir, local_llm, config.embed_url, config.rerank_url)
    return Backend(
        store=store,
        jobs=JobManager(),
        hub=HubClient(config),
        answerer=ConnectionAnswerEngine(store, local_llm),
        local_llm=local_llm,
        servers=servers,
    )
