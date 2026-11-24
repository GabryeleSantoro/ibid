"""Reranking over llama-server's /v1/rerank endpoint.

With a correct GGUF conversion (docs/superpowers/notes/2026-09-17-s0-reranker-gate.md),
Qwen3-Reranker's `relevance_score` is already P(yes) in 0..1, the scale
`min_score` is expressed on, so scores pass through untouched.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

import httpx

from ragcore.models.embed import MAX_INPUT_CHARS


class RerankClient:
    def __init__(
        self, base_url: str, *, model: str = "qwen3-reranker", timeout: float = 120.0
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = httpx.AsyncClient(timeout=timeout)
        # Set by build_backend: awaits the model servers' startup so early calls wait, not fail.
        self.gate: Callable[[], Awaitable[None]] | None = None

    async def _post(self, query: str, documents: list[str], top_n: int) -> list[dict]:
        if self.gate:
            await self.gate()
        response = await self._client.post(
            f"{self.base_url}/v1/rerank",
            json={"query": query, "documents": documents, "top_n": top_n, "model": self.model},
        )
        response.raise_for_status()
        return response.json()["results"]

    async def rerank(
        self, query: str, documents: list[str], *, top_n: int
    ) -> list[tuple[int, float]]:
        results = await self._post(query, [d[:MAX_INPUT_CHARS] for d in documents], top_n)
        scored = [(int(r["index"]), float(r["relevance_score"])) for r in results]
        return sorted(scored, key=lambda pair: -pair[1])[:top_n]

    async def aclose(self) -> None:
        await self._client.aclose()
