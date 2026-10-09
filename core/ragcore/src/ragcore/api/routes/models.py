from __future__ import annotations

import os
import platform
from datetime import UTC, datetime

import httpx
from fastapi import APIRouter, Query, Request
from pydantic import BaseModel

from ragcore.api.deps import ConfigDep, JobsDep, StoreDep
from ragcore.api.errors import api_error
from ragcore.api.schemas import (
    DownloadRequest,
    HardwareInfo,
    HubModelDetail,
    HubSearchResult,
    InstalledModel,
    Job,
    ModelInventory,
    ModelRole,
    Ok,
)

router = APIRouter(prefix="/models", tags=["models"])


def _hub(request: Request):
    return request.app.state.hub


def _active(store: StoreDep, role: ModelRole) -> str | None:
    return next((m.id for m in store.models.values() if m.role == role and m.active), None)


@router.get("", response_model=ModelInventory)
def inventory(store: StoreDep) -> ModelInventory:
    return ModelInventory(
        installed=sorted(store.models.values(), key=lambda m: (m.role, m.name)),
        active_embedding=_active(store, "embedding"),
        active_reranking=_active(store, "reranking"),
    )


class RuntimeStatus(BaseModel):
    state: str  # missing | downloading | ready | unavailable (stub backend)
    progress: float
    error: str | None = None


@router.get("/runtime", response_model=RuntimeStatus)
def runtime_status(request: Request) -> RuntimeStatus:
    servers = request.app.state.servers
    if servers is None:
        return RuntimeStatus(state="unavailable", progress=0.0)
    return RuntimeStatus(state=servers.status(), progress=servers.progress, error=servers.error)


@router.post("/runtime/install", response_model=RuntimeStatus)
async def runtime_install(request: Request) -> RuntimeStatus:
    if request.app.state.servers:
        request.app.state.servers.install()
    return runtime_status(request)


@router.get("/hardware", response_model=HardwareInfo)
def hardware(config: ConfigDep) -> HardwareInfo:
    """What the Rust shell measured and handed over at spawn time."""
    if config.vram_mb >= 8192:
        profile = "gpu"
    elif config.gpu_backend != "cpu":
        profile = "balanced"
    else:
        profile = "cpu"

    return HardwareInfo(
        os=platform.system().lower(),
        arch=platform.machine(),
        cpu_count=os.cpu_count() or 1,
        ram_mb=config.ram_mb,
        vram_mb=config.vram_mb,
        gpu_backend=config.gpu_backend,  # type: ignore[arg-type]
        profile=profile,
    )


@router.get("/hub/search", response_model=HubSearchResult)
async def hub_search(
    request: Request,
    role: ModelRole | None = None,
    q: str | None = None,
    sort: str = Query(
        default="downloads",
        pattern="^(downloads|likes|trendingScore|lastModified)$",
    ),
    limit: int = Query(default=24, le=60),
) -> HubSearchResult:
    try:
        items = await _hub(request).search(role=role, query=q, sort=sort, limit=limit)
    except httpx.HTTPError as exc:
        raise api_error(
            502, "hub_unreachable", f"Hugging Face unreachable: {exc}", reason=str(exc)
        ) from exc
    return HubSearchResult(items=items, role=role, query=q)


@router.get("/hub/recommended", response_model=HubSearchResult)
async def hub_recommended(request: Request, role: ModelRole) -> HubSearchResult:
    try:
        items = await _hub(request).recommended(role)
    except httpx.HTTPError as exc:
        raise api_error(
            502, "hub_unreachable", f"Hugging Face unreachable: {exc}", reason=str(exc)
        ) from exc
    return HubSearchResult(items=items, role=role)


@router.get("/hub/{repo_id:path}", response_model=HubModelDetail)
async def hub_detail(repo_id: str, request: Request, role: ModelRole | None = None):
    try:
        return await _hub(request).detail(repo_id, role)
    except httpx.HTTPStatusError as exc:
        raise api_error(
            exc.response.status_code,
            "hub_error",
            f"{repo_id}: {exc}",
            repo_id=repo_id,
            reason=str(exc),
        ) from exc
    except httpx.HTTPError as exc:
        raise api_error(
            502, "hub_unreachable", f"Hugging Face unreachable: {exc}", reason=str(exc)
        ) from exc


@router.post("/download", response_model=Job)
async def download(
    payload: DownloadRequest, request: Request, store: StoreDep, jobs: JobsDep
) -> Job:
    if payload.role == "embedding" and payload.activate and not payload.accept_reindex:
        # Vectors from two embedders are not comparable; activating one without
        # a re-index would silently destroy retrieval quality.
        raise api_error(
            409,
            "reindex_required",
            "Activating a different embedder invalidates the index. "
            "Resend with accept_reindex=true.",
        )

    detail = await _hub(request).detail(payload.repo_id, payload.role)
    file = next((f for f in detail.files if f.path == payload.filename), None)
    if file is None:
        raise api_error(
            404,
            "hub_file_missing",
            f"{payload.filename} not found in {payload.repo_id}",
            filename=payload.filename,
            repo_id=payload.repo_id,
        )

    model_id = f"{payload.repo_id}/{payload.filename}".replace("/", "_").lower()
    job = jobs.create(
        "download",
        f"Downloading {payload.filename}",
        total=file.size_bytes,
        model_id=model_id,
        detail=f"from {payload.repo_id}",
    )

    async def finish(_job: Job, report) -> None:
        await jobs.staged_runner([("Downloading", 3.0), ("Verifying sha256", 0.8)])(_job, report)
        store.models[model_id] = InstalledModel(
            id=model_id,
            role=payload.role,
            name=payload.filename.removesuffix(".gguf"),
            repo_id=payload.repo_id,
            filename=payload.filename,
            quant=file.quant,
            size_bytes=file.size_bytes,
            sha256=None,
            active=payload.activate,
            shipped=False,
            downloaded_at=datetime.now(tz=UTC),
        )
        if payload.activate:
            for other in store.models.values():
                if other.role == payload.role and other.id != model_id:
                    other.active = False

    return jobs.start(job, finish)


@router.post("/{model_id}/activate", response_model=InstalledModel)
def activate_model(model_id: str, store: StoreDep, accept_reindex: bool = False) -> InstalledModel:
    model = store.models.get(model_id)
    if model is None:
        raise api_error(404, "model_not_installed", "model not installed")
    if model.role == "embedding" and not accept_reindex:
        raise api_error(
            409, "reindex_required", "Switching the embedder requires accept_reindex=true"
        )
    for other in store.models.values():
        if other.role == model.role:
            other.active = other.id == model_id
    return model


@router.delete("/{model_id}", response_model=Ok)
def remove_model(model_id: str, store: StoreDep) -> Ok:
    model = store.models.get(model_id)
    if model is None:
        raise api_error(404, "model_not_installed", "model not installed")
    if model.shipped:
        raise api_error(409, "model_shipped", "Shipped models cannot be removed")
    if model.active:
        raise api_error(409, "model_role_active", "Activate another model for this role first")
    store.models.pop(model_id)
    return Ok()
