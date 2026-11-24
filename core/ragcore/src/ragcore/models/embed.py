"""Embedding over llama-server's OpenAI-compatible endpoint.

The server is started with `--embedding` (EmbeddingGemma's pooling comes from
the GGUF metadata). EmbeddingGemma expects a task prefix on every input, one
for queries and one for documents; `embed` adds it. Normalisation is done here
regardless of what the server returns, because the index stores unit vectors
and cosine similarity on unit vectors is a dot product.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

import httpx
import numpy as np

EMBED_DIM = 768
QUERY_PREFIX = "task: search result | query: "
DOC_PREFIX = "title: none | text: "
# Both model servers need a whole input inside one micro-batch (`-ub 2048` in
# scripts/dev/serve-models.sh); past it llama-server answers 500 and the batch is
# lost. 2000 characters stays under 2048 tokens even for CJK at ~1 char/token.
# ponytail: a character cap, not a token count; rare byte-fallback text can still
# overflow. Token-aware chunking removes the need for it.
MAX_INPUT_CHARS = 2000


class EmbedClient:
    dim = EMBED_DIM

    def __init__(
        self,
        base_url: str,
        *,
        model: str = "embeddinggemma",
        timeout: float = 120.0,
        normalize: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.normalize = normalize
        self._client = httpx.AsyncClient(timeout=timeout)
        # Set by build_backend: awaits the model servers' startup so early calls wait, not fail.
        self.gate: Callable[[], Awaitable[None]] | None = None

    async def _post(self, texts: list[str]) -> list[list[float]]:
        if self.gate:
            await self.gate()
        response = await self._client.post(
            f"{self.base_url}/v1/embeddings",
            json={"input": texts, "model": self.model},
        )
        response.raise_for_status()
        rows = sorted(response.json()["data"], key=lambda row: row["index"])
        return [row["embedding"] for row in rows]

    async def embed(
        self, texts: list[str], *, batch_size: int = 32, query: bool = False
    ) -> list[list[float]]:
        if not texts:
            return []
        prefix = QUERY_PREFIX if query else DOC_PREFIX
        texts = [prefix + text[:MAX_INPUT_CHARS] for text in texts]
        out: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            out.extend(await self._post(texts[start : start + batch_size]))
        if not self.normalize:
            return out
        matrix = np.asarray(out, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (matrix / norms).tolist()

    async def aclose(self) -> None:
        await self._client.aclose()
