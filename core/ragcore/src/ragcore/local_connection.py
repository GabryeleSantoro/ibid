"""The built-in model as a connection: always listed, never stored, no key."""

from __future__ import annotations

from datetime import UTC, datetime

from ragcore.api.schemas import Connection

LOCAL_ID = "local"


def local_connection(active: bool) -> Connection:
    return Connection(
        id=LOCAL_ID,
        name="This computer",
        kind="local",
        base_url=None,
        model_id="Qwen3-1.7B",
        max_output_tokens=None,
        thinking="off",
        is_remote=False,
        has_api_key=False,
        active=active,
        created_at=datetime.fromtimestamp(0, tz=UTC),
    )
