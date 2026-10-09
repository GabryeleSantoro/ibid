from __future__ import annotations

import contextlib
import time
from datetime import UTC, datetime

import httpx
from fastapi import APIRouter, Request

from ragcore.api.deps import StoreDep
from ragcore.api.errors import api_error
from ragcore.api.schemas import (
    Connection,
    ConnectionInput,
    ConnectionSecret,
    ConnectionTestRequest,
    ConnectionTestResult,
    Ok,
)
from ragcore.local_connection import LOCAL_ID, local_connection

router = APIRouter(prefix="/connections", tags=["connections"])

LOCAL_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0", "::1")


def _is_remote(kind: str, base_url: str | None) -> bool:
    if kind == "anthropic":
        return True
    return not any(host in (base_url or "") for host in LOCAL_HOSTS)


@router.get("", response_model=list[Connection])
def list_connections(store: StoreDep) -> list[Connection]:
    local = local_connection(active=store.settings.active_connection_id == LOCAL_ID)
    return [local, *store.connections.values()]


@router.post("", response_model=Connection, status_code=201)
def create_connection(payload: ConnectionInput, store: StoreDep) -> Connection:
    data = payload.model_dump(exclude={"api_key"})
    data["is_remote"] = _is_remote(payload.kind, payload.base_url)
    connection = Connection(
        id=store.new_id("conn"),
        has_api_key=bool(payload.api_key),
        active=not store.connections and store.settings.active_connection_id is None,
        created_at=datetime.now(tz=UTC),
        **data,
    )
    store.connections[connection.id] = connection
    if payload.api_key:
        store.secrets[connection.id] = payload.api_key
    if connection.active:
        store.settings.active_connection_id = connection.id
    store.save_connections()
    return connection


@router.patch("/{connection_id}", response_model=Connection)
def update_connection(
    connection_id: str, payload: ConnectionInput, store: StoreDep
) -> Connection:
    connection = store.connections.get(connection_id)
    if connection is None:
        raise api_error(404, "connection_not_found", "connection not found")
    for field, value in payload.model_dump(exclude={"api_key"}).items():
        setattr(connection, field, value)
    connection.is_remote = _is_remote(payload.kind, payload.base_url)
    if payload.api_key is not None:
        connection.has_api_key = bool(payload.api_key)
        if payload.api_key:
            store.secrets[connection_id] = payload.api_key
        else:
            store.secrets.pop(connection_id, None)
    store.save_connections()
    return connection


@router.post("/{connection_id}/activate", response_model=Connection)
def activate(connection_id: str, store: StoreDep, request: Request) -> Connection:
    if connection_id == LOCAL_ID:
        if request.app.state.local_llm.status() != "ready":
            raise api_error(409, "local_model_missing", "install the built-in model first")
        for other in store.connections.values():
            other.active = False
        store.settings.active_connection_id = LOCAL_ID
        store.save_connections()
        return local_connection(active=True)
    connection = store.connections.get(connection_id)
    if connection is None:
        raise api_error(404, "connection_not_found", "connection not found")
    for other in store.connections.values():
        other.active = other.id == connection_id
    store.settings.active_connection_id = connection_id
    store.save_connections()
    return connection


@router.delete("/{connection_id}", response_model=Ok)
def delete_connection(connection_id: str, store: StoreDep) -> Ok:
    if connection_id not in store.connections:
        raise api_error(404, "connection_not_found", "connection not found")
    store.connections.pop(connection_id)
    store.secrets.pop(connection_id, None)
    if store.settings.active_connection_id == connection_id:
        store.settings.active_connection_id = next(iter(store.connections), None)
    store.save_connections()
    return Ok()


@router.put("/{connection_id}/secret", response_model=Ok)
def restore_secret(connection_id: str, payload: ConnectionSecret, store: StoreDep) -> Ok:
    """The shell hands back a key from the OS keychain after a restart."""
    if connection_id not in store.connections:
        raise api_error(404, "connection_not_found", "connection not found")
    store.secrets[connection_id] = payload.api_key
    return Ok()


async def probe_connection(
    kind: str,
    base_url: str | None,
    model_id: str | None,
    api_key: str | None,
) -> ConnectionTestResult:
    """A real probe. Reachability is the thing users actually get wrong."""
    if kind == "anthropic":
        url, headers = "https://api.anthropic.com/v1/models", {
            "anthropic-version": "2023-06-01",
        }
        if api_key:
            headers["x-api-key"] = api_key
    else:
        if not base_url:
            return ConnectionTestResult(
                ok=False, reachable=False, model_found=False, streaming=False,
                error="No base URL set",
            )
        url, headers = f"{base_url.rstrip('/')}/models", {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

    started = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(8.0, connect=4.0)) as client:
            response = await client.get(url, headers=headers)
    except httpx.HTTPError as exc:
        return ConnectionTestResult(
            ok=False, reachable=False, model_found=False, streaming=False,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
            error=f"{type(exc).__name__}: {exc}",
        )

    latency_ms = round((time.perf_counter() - started) * 1000, 1)
    if response.status_code >= 400:
        return ConnectionTestResult(
            ok=False, reachable=True, model_found=False, streaming=False,
            latency_ms=latency_ms, error=f"HTTP {response.status_code}",
        )

    ids: list[str] = []
    with contextlib.suppress(Exception):
        body = response.json()
        ids = [entry.get("id", "") for entry in body.get("data", [])]

    found = bool(model_id) and any(model_id == i or model_id in i for i in ids)
    return ConnectionTestResult(
        ok=found,
        reachable=True,
        model_found=found,
        streaming=True,
        latency_ms=latency_ms,
        error=None if found else f"Model {model_id!r} not offered by this endpoint",
    )


@router.post("/test", response_model=ConnectionTestResult)
async def test_connection(payload: ConnectionTestRequest, store: StoreDep) -> ConnectionTestResult:
    kind = payload.kind
    base_url = payload.base_url
    model_id = payload.model_id
    api_key = payload.api_key

    if payload.connection_id:
        connection = store.connections.get(payload.connection_id)
        if connection is None:
            raise api_error(404, "connection_not_found", "connection not found")
        kind, base_url, model_id = connection.kind, connection.base_url, connection.model_id
        # Testing a saved connection must not require retyping its key.
        api_key = api_key or store.secrets.get(connection.id)

    return await probe_connection(kind, base_url, model_id, api_key)
