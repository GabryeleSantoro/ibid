import asyncio
from pathlib import Path

from ragcore.model_servers import ModelServers
from ragcore.models.embed import EmbedClient


def test_client_waits_for_servers_boot():
    async def run():
        servers = ModelServers(Path("/nonexistent"), None, "http://x", "http://y")
        order = []

        async def boot():
            await asyncio.sleep(0.05)
            order.append("booted")

        servers._boot = asyncio.create_task(boot())
        client = EmbedClient("http://127.0.0.1:1")
        client.gate = servers.wait_ready
        try:
            await client.embed(["a"])
        except Exception:  # noqa: BLE001 - connection refused is expected; only ordering matters
            order.append("posted")
        assert order == ["booted", "posted"]

    asyncio.run(run())
