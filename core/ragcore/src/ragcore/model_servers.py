"""The embedder and reranker llama-servers the real backend depends on.

Reuses LocalLLM's llama.cpp install. Models download only when the user asks
(`install()`); once present, `start()` launches both servers and a watchdog
restarts either one that dies. Flags mirror scripts/dev/serve-models.sh.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import httpx

from ragcore.local_llm import LocalLLM

logger = logging.getLogger("ragcore.model_servers")

WATCH_S = 5
START_TIMEOUT_S = 60


@dataclass(frozen=True)
class _Spec:
    name: str
    file: str
    url: str
    mode: str  # llama-server flag selecting the pooling mode
    context: int


EMBED = _Spec(
    "embedder",
    "embeddinggemma-300M-qat-Q4_0.gguf",
    "https://huggingface.co/ggml-org/embeddinggemma-300M-qat-q4_0-GGUF/resolve/main/embeddinggemma-300M-qat-Q4_0.gguf",
    "--embedding",
    2048,
)
RERANK = _Spec(
    "reranker",
    "Qwen3-Reranker-0.6B-Q8_0.gguf",
    "https://huggingface.co/ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/resolve/main/qwen3-reranker-0.6b-q8_0.gguf",
    "--reranking",
    8192,
)


class ModelServers:
    def __init__(self, data_dir: Path, llm: LocalLLM, embed_url: str, rerank_url: str) -> None:
        self.dir = data_dir / "models"
        self.llm = llm
        self.progress = 0.0
        self.error: str | None = None
        self._urls = {EMBED.name: embed_url, RERANK.name: rerank_url}
        self._procs: dict[str, asyncio.subprocess.Process] = {}
        self._download: asyncio.Task | None = None
        self._watchdog: asyncio.Task | None = None

    def status(self) -> str:
        if self._download and not self._download.done():
            return "downloading"
        have = all((self.dir / s.file).exists() for s in (EMBED, RERANK))
        return "ready" if have and self.llm._binary() else "missing"

    def install(self) -> None:
        if self.status() == "missing":
            self.error = None
            self._download = asyncio.create_task(self._install())

    async def _install(self) -> None:
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            self.llm.dir.mkdir(parents=True, exist_ok=True)
            self.progress = 0.0
            await self.llm.ensure_binary()
            for i, spec in enumerate((EMBED, RERANK)):
                if not (self.dir / spec.file).exists():
                    await self.llm._fetch(spec.url, self.dir / spec.file, 0.45, 0.1 + 0.45 * i)
            self.progress = 1.0
            await self.start()
        except Exception as exc:  # noqa: BLE001 - surfaced through status, retried on next install()
            logger.error("model install failed: %s", exc)
            self.error = str(exc)

    async def _spawn(self, spec: _Spec) -> None:
        port = urlparse(self._urls[spec.name]).port
        self._procs[spec.name] = await asyncio.create_subprocess_exec(
            str(self.llm._binary()), "-m", str(self.dir / spec.file), spec.mode,
            "-c", str(spec.context), "-b", "2048", "-ub", "2048", "-ngl", "999",
            "--host", "127.0.0.1", "--port", str(port),
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        )  # fmt: skip

    async def _healthy(self, spec: _Spec) -> bool:
        async with httpx.AsyncClient() as client:
            for _ in range(START_TIMEOUT_S * 2):
                proc = self._procs[spec.name]
                if proc.returncode is not None:
                    return False
                with contextlib.suppress(httpx.HTTPError):
                    if (await client.get(f"{self._urls[spec.name]}/health")).status_code == 200:
                        return True
                await asyncio.sleep(0.5)
        return False

    async def start(self) -> None:
        """Launch whatever is not running. Idempotent; no-op until models are installed."""
        if self.status() != "ready":
            return
        for spec in (EMBED, RERANK):
            proc = self._procs.get(spec.name)
            if proc is None or proc.returncode is not None:
                await self._spawn(spec)
                if not await self._healthy(spec):
                    self.error = f"{spec.name} did not become healthy"
                    logger.error(self.error)
        if self._watchdog is None:
            self._watchdog = asyncio.create_task(self._watch())

    async def _watch(self) -> None:
        # ponytail: fixed 5 s retry, no backoff cap; a model that crashes on load loops quietly.
        while True:
            await asyncio.sleep(WATCH_S)
            if any(p.returncode is not None for p in self._procs.values()):
                await self.start()

    def stop(self) -> None:
        if self._watchdog:
            self._watchdog.cancel()
            self._watchdog = None
        for proc in self._procs.values():
            if proc.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    proc.terminate()
