"""Golden-set evaluation over whatever the index currently holds. Dev builds only."""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import APIRouter

from ragcore.api.deps import ConfigDep, StoreDep
from ragcore.api.errors import api_error
from ragcore.api.schemas import EvalResult, EvalRunRequest, EvalSet
from ragcore.api.sse import frame, sse_response
from ragcore.eval import baseline_recall, golden_sets, load_golden, score_questions, summarize

router = APIRouter(prefix="/eval", tags=["eval"])


@router.get("/sets", response_model=list[EvalSet])
def list_sets(config: ConfigDep) -> list[EvalSet]:
    return [
        EvalSet(name=path.stem, n_questions=len(load_golden(path)), description=str(path.parent))
        for path in golden_sets(config.data_dir)
    ]


@router.post("/run")
async def run_eval(payload: EvalRunRequest, store: StoreDep, config: ConfigDep):
    if not config.dev_mode:
        raise api_error(403, "eval_dev_only", "eval runs in dev builds only")
    path = next((p for p in golden_sets(config.data_dir) if p.stem == payload.set_name), None)
    if path is None:
        raise api_error(
            404, "eval_set_unknown", f"unknown eval set {payload.set_name!r}", name=payload.set_name
        )
    questions = load_golden(path)
    paths = {document.id: document.path for document in store.documents.values()}

    async def events() -> AsyncIterator[str]:
        results = []
        async for result in score_questions(
            store.retriever,
            questions,
            settings=store.settings.retrieval,
            doc_meta=store.doc_meta(),
            paths=paths,
        ):
            results.append(result)
            yield frame(
                "progress",
                {"completed": len(results), "total": len(questions), "question": result.q},
            )
        baseline = (
            baseline_recall(path.with_suffix(".baseline.json"))
            if payload.compare_baseline
            else None
        )
        metrics = summarize(path.stem, results, baseline)
        yield frame("result", EvalResult(metrics=metrics, questions=results))

    return sse_response(events())
