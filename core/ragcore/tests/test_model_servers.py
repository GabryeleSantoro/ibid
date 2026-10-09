"""ModelServers launches both llama-servers from installed files and restarts a dead one."""

from __future__ import annotations

import asyncio
import socket
import stat
import sys

from ragcore.local_llm import LocalLLM
from ragcore.model_servers import EMBED, RERANK, ModelServers

FAKE = f"""#!{sys.executable}
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

port = int(sys.argv[sys.argv.index("--port") + 1])


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *a):
        pass


HTTPServer(("127.0.0.1", port), H).serve_forever()
"""


def _port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_start_and_restart(tmp_path, monkeypatch):
    fake = tmp_path / "llama-server"
    fake.write_text(FAKE)
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("LLAMA_SERVER", str(fake))
    ports = (_port(), _port())

    async def run() -> None:
        servers = ModelServers(
            tmp_path,
            LocalLLM(tmp_path),
            f"http://127.0.0.1:{ports[0]}",
            f"http://127.0.0.1:{ports[1]}",
        )
        assert servers.status() == "missing"
        await servers.start()  # no models yet: no-op
        assert not servers._procs

        servers.dir.mkdir()
        for spec in (EMBED, RERANK):
            (servers.dir / spec.file).write_bytes(b"x")
        assert servers.status() == "ready"
        await servers.start()
        assert servers.error is None
        assert all(p.returncode is None for p in servers._procs.values())

        servers._procs[EMBED.name].kill()
        await servers._procs[EMBED.name].wait()
        await servers.start()
        assert servers._procs[EMBED.name].returncode is None
        servers.stop()

    asyncio.run(run())
