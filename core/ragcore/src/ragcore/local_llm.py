"""A small local model (Qwen3-1.7B, Q4) behind llama-server, for chat starter questions.

Nothing is bundled or downloaded until the user asks: `install()` fetches a pinned
llama.cpp build plus the GGUF into the data dir. The server starts on the first
`generate()` and stops after IDLE_S without use, so it costs nothing when idle.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import platform
import re
import shutil
import socket
from pathlib import Path

import httpx

logger = logging.getLogger("ragcore.local_llm")

LLAMA_TAG = "b11261"
MODEL_FILE = "Qwen3-1.7B-Q4_K_M.gguf"
MODEL_URL = f"https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/main/{MODEL_FILE}"
IDLE_S = 60
CONTEXT = 8192
START_TIMEOUT_S = 90
# ponytail: three prebuilt CPU targets; add a row when another platform needs it.
_ASSETS = {
    ("Darwin", "arm64"): "macos-arm64.tar.gz",
    ("Darwin", "x64"): "macos-x64.tar.gz",
    ("Linux", "x64"): "ubuntu-x64.tar.gz",
    ("Windows", "x64"): "win-cpu-x64.zip",
    ("Windows", "arm64"): "win-cpu-arm64.zip",
}


def _asset() -> str | None:
    machine = platform.machine().lower()
    arch = "x64" if machine in ("x86_64", "amd64") else machine
    suffix = _ASSETS.get((platform.system(), arch))
    return f"llama-{LLAMA_TAG}-bin-{suffix}" if suffix else None


class LocalLLM:
    def __init__(self, data_dir: Path) -> None:
        self.dir = data_dir / "local-llm"
        self.progress = 0.0
        self.error: str | None = None
        self._download: asyncio.Task | None = None
        self._proc: asyncio.subprocess.Process | None = None
        self._url = ""
        self._lock = asyncio.Lock()
        self._idle: asyncio.TimerHandle | None = None
        self._leases = 0

    # ------------------------------------------------------------------ install

    def _binary(self) -> Path | None:
        name = "llama-server.exe" if platform.system() == "Windows" else "llama-server"
        if override := os.environ.get("LLAMA_SERVER"):
            return Path(override)
        found = next(self.dir.glob(f"llama-*/**/{name}"), None) or next(
            self.dir.glob(f"**/{name}"), None
        )
        if found:
            return found
        on_path = shutil.which("llama-server")
        return Path(on_path) if on_path else None

    def status(self) -> str:
        if self._download and not self._download.done():
            return "downloading"
        model = self.dir / MODEL_FILE
        return "ready" if model.exists() and self._binary() else "missing"

    def install(self) -> None:
        if self.status() == "missing":
            self.error = None
            self._download = asyncio.create_task(self._install())

    async def _fetch(self, url: str, dest: Path, weight: float, base: float) -> None:
        part = dest.with_suffix(dest.suffix + ".part")
        async with (
            httpx.AsyncClient(follow_redirects=True, timeout=None) as client,
            client.stream("GET", url) as response,
        ):
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            done = 0
            with part.open("wb") as out:
                async for block in response.aiter_bytes(1 << 20):
                    out.write(block)
                    done += len(block)
                    if total:
                        self.progress = base + weight * done / total
        part.replace(dest)

    async def ensure_binary(self) -> Path:
        if (binary := self._binary()) is not None:
            return binary
        asset = _asset()
        if asset is None:
            raise RuntimeError(f"no llama.cpp build for {platform.system()}")
        archive = self.dir / asset
        url = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_TAG}/{asset}"
        await self._fetch(url, archive, 0.05, 0.0)
        await asyncio.to_thread(shutil.unpack_archive, archive, self.dir)
        archive.unlink()
        binary = self._binary()
        if binary is None:
            raise RuntimeError("llama-server missing from the downloaded build")
        binary.chmod(0o755)
        return binary

    async def _install(self) -> None:
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            self.progress = 0.0
            await self.ensure_binary()
            if not (self.dir / MODEL_FILE).exists():
                await self._fetch(MODEL_URL, self.dir / MODEL_FILE, 0.95, 0.05)
            self.progress = 1.0
        except Exception as exc:  # noqa: BLE001 - surfaced through status, retried on next install()
            logger.error("local model install failed: %s", exc)
            self.error = str(exc)

    # ------------------------------------------------------------------ serving

    async def _ensure_server(self) -> str:
        async with self._lock:
            if self._proc and self._proc.returncode is None:
                return self._url
            binary = self._binary()
            if binary is None:
                raise RuntimeError("llama-server not installed")
            with contextlib.closing(socket.socket()) as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            self._proc = await asyncio.create_subprocess_exec(
                str(binary), "-m", str(self.dir / MODEL_FILE), "--host", "127.0.0.1",
                "--port", str(port), "-c", str(CONTEXT),
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
            )  # fmt: skip
            self._url = f"http://127.0.0.1:{port}"
            async with httpx.AsyncClient() as client:
                for _ in range(START_TIMEOUT_S * 3):
                    if self._proc.returncode is not None:
                        raise RuntimeError("llama-server exited during startup")
                    with contextlib.suppress(httpx.HTTPError):
                        if (await client.get(f"{self._url}/health")).status_code == 200:
                            return self._url
                    await asyncio.sleep(1 / 3)
            self.stop()
            raise RuntimeError("llama-server did not become healthy")

    def _arm(self) -> None:
        if self._idle:
            self._idle.cancel()
        self._idle = asyncio.get_running_loop().call_later(IDLE_S, self.stop)

    @contextlib.asynccontextmanager
    async def lease(self):
        """The server's base URL, kept running until the last holder lets go."""
        self._leases += 1
        if self._idle:
            self._idle.cancel()
            self._idle = None
        try:
            yield await self._ensure_server()
        finally:
            self._leases -= 1
            if self._leases == 0:
                self._arm()

    def stop(self) -> None:
        if self._idle:
            self._idle.cancel()
            self._idle = None
        if self._proc and self._proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                self._proc.terminate()

    async def generate(self, system: str, user: str, max_tokens: int = 200) -> str:
        async with self.lease() as url, httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{url}/v1/chat/completions",
                json={
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "max_tokens": max_tokens,
                    "temperature": 0.6,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
        response.raise_for_status()
        text = response.json()["choices"][0]["message"]["content"]
        return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
