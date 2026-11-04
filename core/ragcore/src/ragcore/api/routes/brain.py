"""The topic map: read it, rebuild it."""

from __future__ import annotations

from fastapi import APIRouter

from ragcore.api.deps import JobsDep, StoreDep
from ragcore.api.errors import api_error
from ragcore.api.schemas import Job, TopicTree

router = APIRouter(prefix="/brain", tags=["brain"])


@router.get("/topics", response_model=TopicTree)
def topics(store: StoreDep) -> TopicTree:
    return store.brain.tree()


@router.post("/recluster", response_model=Job)
async def recluster(store: StoreDep, jobs: JobsDep) -> Job:
    """The same rebuild the brain schedules itself, as a Job so the jobs stream shows it."""
    if store.index_blocked:
        raise api_error(409, "index_blocked", store.index_blocked)
    job = jobs.create("recluster", "Re-clustering topics")

    async def run(job: Job, report) -> None:
        await store.brain.recluster()

    return jobs.start(job, run)
