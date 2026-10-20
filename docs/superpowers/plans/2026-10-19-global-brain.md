# Global Brain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Questions about the library as a whole get an answer built from every topic (doc cards → numpy topic tree → embedding router → topic-packed global context), plus a browsable Topics screen with "Ask about this topic".

**Architecture:** Each indexed document gets a *card* (mean vector, 3 summary chunks, term counts) written in the same SQLite transaction as the document. A `Brain` per `RealStore` clusters cards into a ≤3-level tree (PCA + spherical k-means + silhouette k + c-TF-IDF labels), keeps it current by nearest-leaf assignment, and reclusters in a thread when 15% has churned. `/query` routes with example-similarity; global questions get 1 representative + 2 dense chunks per top topic, interleaved and packed, answered with a global system prompt through the unchanged `AnswerEngine`.

**Tech Stack:** Python 3.13, FastAPI, numpy, LanceDB, SQLite; React 19 + TanStack Router/Query, Tailwind 4; vitest; pytest (asyncio_mode auto).

**Spec:** `docs/superpowers/specs/2026-10-19-global-brain-design.md`

---

## Context

Today "global" is a badge: `route_mode` (`core/ragcore/src/ragcore/api/routes/query.py:138`) matches English keywords and both modes run the same 6-passage retrieval, so "what are the main themes of my library" is answered from the 6 passages nearest the question. This builds build-plan phase 5 (adapted): documents (not passages) clustered with numpy only, one LLM call per global answer, a Topics screen, and an eval gate for coverage and routing.

First executor step: copy this file to `docs/superpowers/plans/2026-10-19-global-brain.md` and commit it with Task 1.

## Global Constraints

- Python 3.13; numpy only for clustering. No scikit-learn, HDBSCAN, UMAP; no new Python or JS dependency.
- `uv run ruff check .` clean (line-length 100); `bun run typecheck` clean.
- SSE frame order unchanged: `start → mode → sources → token* → citations → done`; no new event types.
- Citations: `extract_citations(text, chunks)` with exactly the packed chunks; unbacked markers dropped (`citations.py` invariant).
- Routes depend only on ports: `StorePort.brain`. Stub brain is a null object.
- UI calls go `lib/ipc.ts` → Tauri → proxy; never the sidecar directly.
- Every new UI string in en/it/fr/de/es; `locales/*.test.ts` stay green.
- Generative model never from env vars (unchanged).
- Constants, verbatim from the spec: under 6 cards → no tree; PCA to `min(50, n - 1, dim)`; k from 2..`min(20, n // 3)`; silhouette sample ≤ 2,000; fixed seed; 50 k-means iterations max; recurse into clusters with more than 30 docs; at most 3 levels; 3 summary chunks; 300 terms (3+ chars, stopwords removed); c-TF-IDF `tf(t, c) × log(1 + A / f(t))`, top 5 keywords, top 3 joined with ` · ` as label; 3 representative docs; recluster at `changed_since ≥ 15%` of cards; ≤ 8 topics per answer; 1 representative + 2 dense hits per topic; `global_token_budget` default 6144; router margin `> +0.05` global, `< −0.05` local; reason `"Global (router 0.12)"`; targets coverage ≥ 0.70, router accuracy ≥ 0.90.
- After every task: `scripts/install-local.sh` (project rule). No release, no tag.

## Deliberate deviations from the spec (flag in review)

1. **Card `keywords` (top-8 TF-IDF) not built.** Nothing reads them: doc cards in the reader are a non-goal. Add with that feature.
2. **Brain gate tolerance 0.04, not 0.02.** With 30 rows one router flip is 0.033; a 0.02 gate would fail on any single flip. Same reasoning as the existing `MAX_RECALL_DROP` comment in `eval.py`.
3. **Router reason / topic scope shown on the live answer only.** Stored `ChatMessage` keeps `mode` only, as today.
4. **`GlobalContext` also carries `candidates`** (chunks before packing), for `SourcesEvent.candidates`.
5. **`ragcore ask` goes through the same `retrieve()` as `/query`**, so the spec's manual check ("`ragcore ask` on global questions") exercises the global path; `ragcore index` waits for the recluster it triggers so `ask` finds a tree.
6. **Golden `expected` entries become `fnmatch` patterns** (`garden-*.md`). Literal names still match; `base.jsonl` is unaffected.

## Review Focus

1. **A document is added or deleted while a recluster runs** → the new tree must hold the new doc and not the deleted one. Pinned by `test_a_doc_added_during_a_recluster_joins_the_new_tree` (Task 3).
2. **A project-scoped chat asks a global question** → only that scope's documents reach the packed context. Pinned by `test_global_context_respects_the_document_filter` (Task 5).
3. **A library under 6 documents asks a global question** → answer from cards as pseudo-topics, not a local fallback. Pinned by `test_a_small_library_answers_global_from_cards` (Task 5).
4. **The embedder is down** → keyword routing, representative chunks only, no crash. Pinned by `test_an_embedder_outage_routes_by_keywords` (Task 4) and `test_global_context_without_a_vector_uses_representatives` (Task 5).
5. **A stale topic id after a recluster** → 404 before the stream, no stray chat; UI shows the translated "Topic not found". Pinned by `test_an_unknown_topic_is_a_404_before_the_stream` (Task 5) and `errors.topic_not_found` in every catalog (Task 7).

## File map

Create (Python, under `core/ragcore/src/ragcore/`):
- `stopwords.py` — shared stopword lists (moved from `api/routes/suggestions.py`).
- `brain/__init__.py` — empty.
- `brain/cluster.py` — pure numpy: `Node`, `pca`, `kmeans`, `silhouette`, `split`, `ctfidf`, `build_tree`.
- `brain/cards.py` — `DocCard`, `terms_of`, `build_card`.
- `brain/topics.py` — `Brain` (persistence, assignment, scheduling, recluster, route, for_query).
- `brain/null.py` — `NullBrain` for the stub (kept apart so the stub never imports LanceDB through it).
- `brain/route.py` — `GLOBAL_HINTS`, `EXAMPLES`, `MARGIN`, `Routed`, `keyword_route`, `embed_examples`, `score_route`.
- `brain/answer.py` — `GlobalContext`, `for_query`.
- `api/routes/brain.py` — `GET /brain/topics`, `POST /brain/recluster`.

Tests (under `core/ragcore/tests/`): `test_brain_cluster.py`, `test_brain_topics.py`, `test_brain_route.py`, `test_brain_api.py`; modify `test_query.py`, `test_eval_harness.py`, `conftest.py`.

Modify (Python): `api/routes/suggestions.py`, `store/meta.py`, `store/lance.py`, `store/real.py`, `stub/store.py`, `ports.py`, `api/schemas.py`, `llm.py`, `retrieve/hybrid.py` (rename `_to_chunk` → `to_chunk`), `api/routes/query.py`, `api/app.py`, `cli.py`, `eval.py`.

Frontend (under `apps/desktop/src/`): create `features/topics/tree.ts`, `features/topics/tree.test.ts`, `features/topics/topics-view.tsx`, `routes/_shell.topics.tsx`, `features/chat/use-chat.test.ts`; modify `lib/api-types.ts` (generated), `lib/ipc.ts`, `lib/ipc.test.ts`, `lib/queries.ts`, `lib/jobs-context.tsx`, `components/shell/icon-rail.tsx`, `routes/_shell.chat.tsx`, `features/chat/chat-view.tsx`, `features/chat/use-chat.ts`, `features/chat/message.tsx`, `locales/{en,it,fr,de,es}.json`, `routeTree.gen.ts` (generated).

Eval: `fixtures/brain/*.md` (40), `eval/golden/brain.jsonl`, `eval/golden/brain.baseline.json`, `.github/workflows/eval.yml`. Docs: `CLAUDE.md`.

---

### Task 1: Shared stopwords + numpy clustering

**Files:**
- Create: `core/ragcore/src/ragcore/stopwords.py`, `core/ragcore/src/ragcore/brain/__init__.py` (empty), `core/ragcore/src/ragcore/brain/cluster.py`
- Modify: `core/ragcore/src/ragcore/api/routes/suggestions.py:26-36`
- Test: `core/ragcore/tests/test_brain_cluster.py`

**Interfaces:**
- Produces: `stopwords.STOPWORDS: dict[str, frozenset[str]]`, `stopwords.ALL_STOPWORDS: frozenset[str]`.
- Produces: `cluster.Node` (dataclass: `id, parent_id, level, centroid: np.ndarray, doc_ids: list[str], keywords: list[str], label: str, rep_chunk_ids: list[str]`), `cluster.build_tree(doc_ids, vectors, terms, first_chunks, *, seed=0) -> list[Node]` (parents before children, siblings largest first, ids `t1`, `t1.2`, `t1.2.3`), `cluster.split(x, rng) -> np.ndarray`, `cluster.pca(x)`, `cluster.ctfidf(groups) -> list[list[str]]`.

- [ ] **Step 0: Save the plan into the repo**

```bash
mkdir -p docs/superpowers/plans
cp /Users/gabrielesantoro/.claude/plans/wobbly-leaping-trinket.md docs/superpowers/plans/2026-10-19-global-brain.md
```

- [ ] **Step 1: Write the failing tests** — `core/ragcore/tests/test_brain_cluster.py`

```python
"""Topic clustering: pure numpy, so these feed it arrays directly."""

from __future__ import annotations

import time
from collections import Counter

import numpy as np
from ragcore.brain.cluster import build_tree, ctfidf, pca, split


def blobs(k: int, per: int, *, dim: int = 768, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """k well-separated groups of unit vectors, and each row's true group."""
    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((k, dim))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    truth = np.repeat(np.arange(k), per)
    x = centres[truth] + rng.standard_normal((k * per, dim)) * 0.02
    return x / np.linalg.norm(x, axis=1, keepdims=True), truth


def test_separable_blobs_are_recovered() -> None:
    x, truth = blobs(3, 30)

    labels = split(x, np.random.default_rng(0))

    assert len(set(labels)) == 3
    for group in range(3):
        assert len(set(labels[truth == group])) == 1


def test_silhouette_picks_the_true_number_of_clusters() -> None:
    x, _ = blobs(5, 12)

    assert len(set(split(x, np.random.default_rng(0)))) == 5


def test_the_same_seed_builds_the_same_tree() -> None:
    x, _ = blobs(4, 20)
    ids = [f"doc_{i}" for i in range(len(x))]
    terms = [{"word": 1}] * len(x)
    chunks = [f"{i}:1:0" for i in ids]

    first = build_tree(ids, x, terms, chunks)
    second = build_tree(ids, x, terms, chunks)

    assert [(n.id, n.doc_ids) for n in first] == [(n.id, n.doc_ids) for n in second]


def test_pca_with_fewer_rows_than_dimensions_keeps_n_minus_one() -> None:
    x, _ = blobs(2, 5)

    z = pca(x)

    assert z.shape == (10, 9)
    assert np.allclose(np.linalg.norm(z, axis=1), 1.0)


def test_ctfidf_picks_each_clusters_distinguishing_terms() -> None:
    garden = Counter(tomato=5, soil=4, water=3, shared=10)
    sky = Counter(star=5, telescope=4, water=3, shared=10)

    words = ctfidf([garden, sky])

    assert words[0][:2] == ["tomato", "soil"]
    assert words[1][:2] == ["star", "telescope"]


def test_every_document_lands_in_exactly_one_top_level_topic() -> None:
    x, _ = blobs(3, 40)  # 40 per blob: each top topic gets subtopics
    ids = [f"doc_{i}" for i in range(len(x))]

    nodes = build_tree(ids, x, [{}] * len(x), [f"{i}:1:0" for i in ids])

    top = [n for n in nodes if n.level == 0]
    assert sorted(d for n in top for d in n.doc_ids) == sorted(ids)
    assert max(n.level for n in nodes) <= 2
    by_id = {n.id: n for n in nodes}
    for node in nodes:
        if node.parent_id:
            assert set(node.doc_ids) <= set(by_id[node.parent_id].doc_ids)
        assert len(node.rep_chunk_ids) <= 3
        assert abs(float(np.linalg.norm(node.centroid)) - 1) < 1e-5


def test_identical_vectors_do_not_crash() -> None:
    x = np.ones((10, 8)) / np.sqrt(8)
    ids = [f"doc_{i}" for i in range(10)]

    nodes = build_tree(ids, x, [{}] * 10, [f"{i}:1:0" for i in ids])

    assert [n.doc_ids for n in nodes if n.level == 0] == [ids]


def test_five_thousand_documents_cluster_in_under_thirty_seconds() -> None:
    rng = np.random.default_rng(3)
    x = rng.standard_normal((5000, 768)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    ids = [f"doc_{i}" for i in range(5000)]

    started = time.perf_counter()
    nodes = build_tree(ids, x, [{}] * 5000, [f"{i}:1:0" for i in ids])

    assert time.perf_counter() - started < 30
    assert sorted(d for n in nodes if n.level == 0 for d in n.doc_ids) == sorted(ids)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest core/ragcore/tests/test_brain_cluster.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'ragcore.brain'`.

- [ ] **Step 3: Shared stopwords** — create `core/ragcore/src/ragcore/stopwords.py`

```python
"""Function words per shipped UI language, shared by language detection and topic terms."""

from __future__ import annotations

# ponytail: stopword vote; swap for a real detector if short or mixed docs misfire.
_WORDS = {
    "en": "the and of to in is that for with as are this by on it from be or an",
    "it": "il lo la le di che e un una per con non sono del della dei nel si come da al è",
    "fr": "le la les de des et un une que pour dans est du au ce pas sur avec qui par",
    "de": "der die das und ist nicht ein eine zu mit den von auf für im dem sich auch",
    "es": "el la los las de que y en un una es por con para del se no al como más",
}

STOPWORDS = {k: frozenset(v.split()) for k, v in _WORDS.items()}
ALL_STOPWORDS = frozenset().union(*STOPWORDS.values())
```

In `api/routes/suggestions.py` delete lines 26-36 (`# ponytail: stopword vote…`, `_WORDS = {…}`, `STOPWORDS = …`) and add `from ragcore.stopwords import STOPWORDS` to the imports. `detect_language` is unchanged.

- [ ] **Step 4: Write `core/ragcore/src/ragcore/brain/cluster.py`** (and an empty `brain/__init__.py`)

```python
"""Documents into a topic tree: PCA, spherical k-means, silhouette-picked k, c-TF-IDF labels.

Pure numpy and no I/O, so it runs in a worker thread and tests feed it arrays.

ponytail: k-means has no outlier bucket, so a document that mixes subjects lands in its
strongest one; HDBSCAN is the upgrade if eval shows mixed documents mislabelled.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

SEED = 0
MAX_DIMS = 50
MAX_K = 20
MIN_PER_CLUSTER = 3  # k ≤ n // 3
SPLIT_ABOVE = 30  # a topic with more documents than this gets subtopics
MAX_LEVELS = 3
ITERATIONS = 50
SILHOUETTE_SAMPLE = 2000
KEYWORDS = 5
LABEL_TERMS = 3
REPRESENTATIVES = 3


@dataclass(slots=True)
class Node:
    id: str
    parent_id: str | None
    level: int
    centroid: np.ndarray  # unit vector in the embedding space: assignment and ranking use it
    doc_ids: list[str]  # every document in this subtree
    keywords: list[str] = field(default_factory=list)
    label: str = ""
    rep_chunk_ids: list[str] = field(default_factory=list)


def _unit(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.where(norms == 0, 1.0, norms)


def pca(x: np.ndarray) -> np.ndarray:
    """Project onto min(50, n - 1, dim) principal axes; rows unit length again."""
    n, dim = x.shape
    centred = x - x.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    return _unit(centred @ vt[: max(1, min(MAX_DIMS, n - 1, dim))].T)


def kmeans(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Spherical k-means (cosine) from a k-means++ start; labels 0..k'-1 with k' ≤ k."""
    n = len(x)
    centres = [x[rng.integers(n)]]
    for _ in range(1, k):
        weights = np.clip(1.0 - np.max(x @ np.stack(centres).T, axis=1), 0, None) ** 2
        total = weights.sum()
        centres.append(x[rng.choice(n, p=weights / total) if total > 0 else rng.integers(n)])
    c = np.stack(centres)
    labels = np.full(n, -1)
    for _ in range(ITERATIONS):
        assigned = np.argmax(x @ c.T, axis=1)
        if np.array_equal(assigned, labels):
            break
        labels = assigned
        for j in range(k):
            members = x[labels == j]
            if len(members):
                c[j] = _unit(members.sum(axis=0))
    return np.unique(labels, return_inverse=True)[1]


def silhouette(dist: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette from a precomputed cosine-distance matrix; -1 for a single cluster."""
    k = int(labels.max()) + 1
    if len(np.unique(labels)) < 2:
        return -1.0
    rows = np.arange(len(labels))
    onehot = np.eye(k)[labels]
    sums = dist @ onehot  # each row's summed distance to each cluster
    sizes = onehot.sum(axis=0)
    own = sizes[labels]
    a = sums[rows, labels] / np.maximum(own - 1, 1)
    other = np.where(sizes > 0, sums / np.maximum(sizes, 1), np.inf)
    other[rows, labels] = np.inf
    b = other.min(axis=1)
    s = np.where(own > 1, (b - a) / np.maximum(np.maximum(a, b), 1e-12), 0.0)
    return float(s.mean())


def split(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Labels for the k in 2..min(20, n // 3) with the best silhouette; all zeros if none splits."""
    n = len(x)
    best = np.zeros(n, dtype=int)
    if n < 2 * MIN_PER_CLUSTER:
        return best
    z = pca(x)
    sample = (
        rng.choice(n, SILHOUETTE_SAMPLE, replace=False) if n > SILHOUETTE_SAMPLE else np.arange(n)
    )
    dist = 1.0 - z[sample] @ z[sample].T
    best_score = -1.0
    for k in range(2, min(MAX_K, n // MIN_PER_CLUSTER) + 1):
        labels = kmeans(z, k, rng)
        score = silhouette(dist, labels[sample])
        if score > best_score:
            best, best_score = labels, score
    return best


def ctfidf(groups: list[Counter[str]]) -> list[list[str]]:
    """Top terms per cluster by tf(t, c) × log(1 + A / f(t)) over sibling clusters."""
    totals: Counter[str] = Counter()
    for group in groups:
        totals.update(group)
    average = sum(totals.values()) / max(len(groups), 1)
    out: list[list[str]] = []
    for group in groups:
        weight = {t: tf * math.log(1 + average / totals[t]) for t, tf in group.items()}
        out.append(sorted(weight, key=lambda t: (-weight[t], t))[:KEYWORDS])
    return out


def build_tree(
    doc_ids: list[str],
    vectors: np.ndarray,
    terms: list[dict[str, int]],
    first_chunks: list[str],
    *,
    seed: int = SEED,
) -> list[Node]:
    """The whole tree, parents before their children, siblings largest first."""
    rng = np.random.default_rng(seed)
    x = _unit(np.asarray(vectors, dtype=np.float32))
    nodes: list[Node] = []

    def summed(rows: np.ndarray) -> Counter[str]:
        total: Counter[str] = Counter()
        for i in rows:
            total.update(terms[i])
        return total

    def grow(rows: np.ndarray, parent: Node | None, level: int) -> None:
        labels = split(x[rows], rng)
        groups = sorted((rows[labels == j] for j in range(labels.max() + 1)), key=len, reverse=True)
        if parent is not None and len(groups) < 2:
            return  # one child identical to its parent says nothing
        for number, (group, keywords) in enumerate(
            zip(groups, ctfidf([summed(g) for g in groups]), strict=True), start=1
        ):
            centroid = _unit(x[group].sum(axis=0))
            closest = group[np.argsort(-(x[group] @ centroid), kind="stable")[:REPRESENTATIVES]]
            node = Node(
                id=f"{parent.id}.{number}" if parent else f"t{number}",
                parent_id=parent.id if parent else None,
                level=level,
                centroid=centroid,
                doc_ids=[doc_ids[i] for i in group],
                keywords=keywords,
                label=" · ".join(keywords[:LABEL_TERMS]) or "untitled",
                rep_chunk_ids=[first_chunks[i] for i in closest],
            )
            nodes.append(node)
            if len(group) > SPLIT_ABOVE and level + 1 < MAX_LEVELS:
                grow(group, node, level + 1)

    grow(np.arange(len(doc_ids)), None, 0)
    return nodes
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest core/ragcore/tests/test_brain_cluster.py core/ragcore/tests/test_suggestions.py -v && uv run ruff check .`
Expected: all PASS, ruff clean. If `test_silhouette_picks_the_true_number_of_clusters` picks a neighbouring k, check `silhouette` against a hand computation on 4 points before touching constants.

- [ ] **Step 6: Install and commit**

```bash
scripts/install-local.sh
git add docs/superpowers/plans/2026-10-19-global-brain.md core/ragcore/src/ragcore/stopwords.py \
  core/ragcore/src/ragcore/brain core/ragcore/src/ragcore/api/routes/suggestions.py \
  core/ragcore/tests/test_brain_cluster.py
git commit -m "feat(brain): shared stopwords and numpy topic clustering"
```

---

### Task 2: Doc cards in SQLite + backfill

**Files:**
- Create: `core/ragcore/src/ragcore/brain/cards.py`
- Modify: `core/ragcore/src/ragcore/store/meta.py` (tables, `save_documents`, `delete_documents`, `remove_source`, `wipe`, new card methods), `core/ragcore/src/ragcore/store/lance.py` (`rows_for_docs`), `core/ragcore/src/ragcore/store/real.py` (`_write`, `backfill_cards`)
- Test: `core/ragcore/tests/test_brain_topics.py` (created here, grown in Tasks 3–5)

**Interfaces:**
- Consumes: `stopwords.ALL_STOPWORDS`.
- Produces: `cards.DocCard(doc_id: str, vector: np.ndarray, summary_chunk_ids: list[str], terms: dict[str, int])`, `cards.build_card(doc_id, chunk_ids, texts, vectors) -> DocCard | None`, `cards.terms_of(texts) -> dict[str, int]`.
- Produces on `MetaStore`: `save_documents(docs, shas, cards=())`, `save_cards(cards)`, `load_cards(doc_ids: list[str] | None = None) -> dict[str, DocCard]`, `card_ids() -> set[str]`.
- Produces on `VectorStore`: `rows_for_docs(doc_ids) -> list[ChunkRow]`.
- Produces on `RealStore`: `backfill_cards() -> list[DocCard]` (sync; Task 3 runs it in a thread).

- [ ] **Step 1: Write the failing tests** — `core/ragcore/tests/test_brain_topics.py`

```python
"""The brain over a real store with fake models: cards, topics, assignment, global context."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ragcore.api.schemas import SourceCreate
from ragcore.config import Config
from ragcore.models.embed import EMBED_DIM
from ragcore.models.fakes import FakeEmbedClient
from ragcore.store.real import RealStore, doc_id_for

WORDS = [
    "tomato soil compost",
    "telescope orbit planet",
    "budget savings interest",
    "flour yeast dough",
    "chain saddle gears",
    "embedding chunk rerank",
    "violin chord melody",
    "glacier valley summit",
]


def library(root: Path, n: int) -> Path:
    folder = root / "library"
    folder.mkdir(exist_ok=True)
    for i in range(n):
        words = WORDS[i % len(WORDS)]
        (folder / f"note{i}.md").write_text(f"# Note {i}\n\n## Body\n\n{words} {words} detail{i}.\n")
    return folder


def build(tmp_path: Path, embedder: FakeEmbedClient | None = None) -> RealStore:
    return RealStore(Config(data_dir=tmp_path, backend="real"), embedder=embedder or FakeEmbedClient())


async def ingested(tmp_path: Path, n: int):
    store = build(tmp_path)
    source = store.add_source(SourceCreate(path=str(library(tmp_path, n))))
    await store.ingest_source_async(source.id)
    return store, source


# -------------------------------------------------------------------- cards


async def test_ingest_writes_a_card_per_document(tmp_path: Path) -> None:
    store, _ = await ingested(tmp_path, 3)

    cards = store.meta.load_cards()

    assert set(cards) == set(store.documents)
    card = cards[sorted(cards)[0]]
    assert card.vector.shape == (EMBED_DIM,)
    assert abs(float(np.linalg.norm(card.vector)) - 1) < 1e-5
    assert 1 <= len(card.summary_chunk_ids) <= 3
    assert card.terms and all(len(t) >= 3 for t in card.terms)
    assert "the" not in card.terms


async def test_a_file_that_loses_its_text_loses_its_card(tmp_path: Path) -> None:
    store, source = await ingested(tmp_path, 3)
    path = tmp_path / "library" / "note0.md"
    path.write_text("")

    await store.ingest_source_async(source.id)

    assert doc_id_for(str(path)) not in store.meta.card_ids()


async def test_removing_a_document_deletes_its_card(tmp_path: Path) -> None:
    store, _ = await ingested(tmp_path, 3)
    doc_id = sorted(store.documents)[0]

    store.remove_document(doc_id)

    assert doc_id not in store.meta.card_ids()


async def test_removing_a_source_deletes_its_cards(tmp_path: Path) -> None:
    store, source = await ingested(tmp_path, 3)

    store.remove_source(source.id)

    assert store.meta.card_ids() == set()


async def test_backfill_builds_cards_for_documents_indexed_before_cards(tmp_path: Path) -> None:
    store, _ = await ingested(tmp_path, 3)
    before = store.meta.load_cards()
    with store.meta._conn:  # a library indexed before doc cards existed
        store.meta._conn.execute("DELETE FROM doc_cards")

    built = store.backfill_cards()

    after = store.meta.load_cards()
    assert {c.doc_id for c in built} == set(store.documents) == set(after)
    for doc_id, card in after.items():
        assert np.allclose(card.vector, before[doc_id].vector, atol=1e-5)
        assert set(card.summary_chunk_ids) == set(before[doc_id].summary_chunk_ids)


async def test_backfill_skips_a_blocked_index(tmp_path: Path) -> None:
    store, _ = await ingested(tmp_path, 3)
    with store.meta._conn:
        store.meta._conn.execute("DELETE FROM doc_cards")
    store.index_blocked = "another embedder"

    assert store.backfill_cards() == []


async def test_wipe_deletes_the_cards(tmp_path: Path) -> None:
    store, _ = await ingested(tmp_path, 3)

    store.wipe(keep_connections=True)

    assert store.meta.card_ids() == set()
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest core/ragcore/tests/test_brain_topics.py -v`
Expected: FAIL, `AttributeError: 'MetaStore' object has no attribute 'load_cards'`.

- [ ] **Step 3: Write `core/ragcore/src/ragcore/brain/cards.py`**

```python
"""A document in one record: where it sits in embedding space, its best passages, its words.

Built where the chunk vectors are already in memory (``RealStore._write``), so clustering and
corpus term statistics never read the chunks back.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from ragcore.stopwords import ALL_STOPWORDS

SUMMARY_CHUNKS = 3
MAX_TERMS = 300
_TOKEN = re.compile(r"[^\W\d_]{3,}")


@dataclass(frozen=True, slots=True)
class DocCard:
    doc_id: str
    vector: np.ndarray  # float32, unit length: the mean of the document's chunk vectors
    summary_chunk_ids: list[str]  # the chunks closest to that mean, best first
    terms: dict[str, int]  # the most frequent words with their counts


def terms_of(texts: Iterable[str]) -> dict[str, int]:
    counts = Counter(
        token
        for text in texts
        for token in _TOKEN.findall(text.lower())
        if token not in ALL_STOPWORDS
    )
    return dict(counts.most_common(MAX_TERMS))


def build_card(
    doc_id: str,
    chunk_ids: Sequence[str],
    texts: Sequence[str],
    vectors: Sequence[Sequence[float]],
) -> DocCard | None:
    """None for a document with no chunks (parse error, empty file, scan without text)."""
    if not chunk_ids:
        return None
    matrix = np.asarray(vectors, dtype=np.float32)
    mean = matrix.mean(axis=0)
    vector = mean / (np.linalg.norm(mean) or 1.0)
    best = np.argsort(-(matrix @ vector), kind="stable")[:SUMMARY_CHUNKS]
    return DocCard(doc_id, vector, [chunk_ids[i] for i in best], terms_of(texts))
```

- [ ] **Step 4: Cards in `store/meta.py`**

Add imports `import json`, `import numpy as np`, `from ragcore.brain.cards import DocCard`. Add module constants and helpers below `_UPSERT_SHA`:

```python
_INSERT_CARD = "INSERT OR REPLACE INTO doc_cards (doc_id, vector, json) VALUES (?, ?, ?)"


def _placeholders(values: list) -> str:
    return ",".join("?" * len(values))


def _card_row(card: DocCard) -> tuple[str, bytes, str]:
    payload = {"summary_chunk_ids": card.summary_chunk_ids, "terms": card.terms}
    return card.doc_id, card.vector.astype(np.float32).tobytes(), json.dumps(payload)


def _card(doc_id: str, blob: bytes, raw: str) -> DocCard:
    data = json.loads(raw)
    return DocCard(
        doc_id, np.frombuffer(blob, dtype=np.float32), data["summary_chunk_ids"], data["terms"]
    )
```

In `__init__`'s `with self._conn:` block add:

```python
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS doc_cards ("
                "doc_id TEXT PRIMARY KEY, vector BLOB NOT NULL, json TEXT NOT NULL)"
            )
```

Replace `save_documents` and add the card methods in the documents section:

```python
    def save_documents(
        self, docs: list[Document], shas: list[tuple[str, str]], cards: Iterable[DocCard] = ()
    ) -> None:
        """Upsert documents, ``(path, sha)`` digests and doc cards in one transaction.

        Every saved document drops its old card first, so one that no longer has chunks
        keeps none.
        """
        ids = [d.id for d in docs]
        with self._lock, self._conn:
            self._conn.executemany(
                _UPSERT_DOCUMENT, [(d.id, d.source_id, d.path, d.model_dump_json()) for d in docs]
            )
            self._conn.executemany(_UPSERT_SHA, shas)
            self._delete_cards(ids)
            self._conn.executemany(_INSERT_CARD, [_card_row(c) for c in cards])

    def _delete_cards(self, doc_ids: list[str]) -> None:
        """Caller holds the lock and the transaction."""
        if doc_ids:
            self._conn.execute(
                f"DELETE FROM doc_cards WHERE doc_id IN ({_placeholders(doc_ids)})", doc_ids
            )

    def save_cards(self, cards: Iterable[DocCard]) -> None:
        with self._lock, self._conn:
            self._conn.executemany(_INSERT_CARD, [_card_row(c) for c in cards])

    def load_cards(self, doc_ids: list[str] | None = None) -> dict[str, DocCard]:
        query, params = "SELECT doc_id, vector, json FROM doc_cards", []
        if doc_ids is not None:
            if not doc_ids:
                return {}
            query, params = f"{query} WHERE doc_id IN ({_placeholders(doc_ids)})", doc_ids
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return {row[0]: _card(*row) for row in rows}

    def card_ids(self) -> set[str]:
        with self._lock:
            return {row[0] for row in self._conn.execute("SELECT doc_id FROM doc_cards")}
```

In `delete_documents`, inside the `with` block after the documents delete, add `self._delete_cards(doc_ids)`. In `remove_source`, after `self._conn.execute("DELETE FROM documents WHERE source_id = ?", ...)`, add `self._delete_cards(doc_ids)`. In `wipe`, add `self._conn.execute("DELETE FROM doc_cards")`.

- [ ] **Step 5: `rows_for_docs` in `store/lance.py`** (after `get`)

```python
    def rows_for_docs(self, doc_ids: list[str]) -> list[ChunkRow]:
        """Every chunk row, vector included, of these documents; any order."""
        total = self.chunks.count_rows()
        if not doc_ids or total == 0:
            return []
        return (
            self.chunks.search().where(f"doc_id IN ({_sql_list(doc_ids)})").limit(total).to_list()
        )
```

- [ ] **Step 6: Cards in `store/real.py`**

Imports: `from collections import defaultdict`, `from ragcore.brain.cards import DocCard, build_card`. Constant under `MAX_IN_FLIGHT`: `BACKFILL_BATCH = 64  # documents per LanceDB read when filling missing cards`.

In `_write`, replace the `self.meta.save_documents(...)` call with:

```python
        cards = [
            card
            for p in files
            if (
                card := build_card(
                    p.doc_id, [c.chunk_id for c in p.chunks], [c.text for c in p.chunks], p.vectors
                )
            )
        ]
        self.meta.save_documents(
            documents,
            [(str(p.found.path), p.found.sha256) for p in files if p.error is None],
            cards,
        )
```

Add to the documents section:

```python
    def backfill_cards(self) -> list[DocCard]:
        """Cards for indexed documents that predate them, built from their stored chunks.

        A batch that fails stays without cards (logged) and is retried on the next start.
        """
        if self.index_blocked:
            return []
        have = self.meta.card_ids()
        missing = sorted(
            d.id for d in self.documents.values() if d.status == "indexed" and d.id not in have
        )
        built: list[DocCard] = []
        for start in range(0, len(missing), BACKFILL_BATCH):
            batch = missing[start : start + BACKFILL_BATCH]
            try:
                rows = self.vectors.rows_for_docs(batch)
            except Exception as exc:  # noqa: BLE001 - those documents retry on the next start
                logger.warning("doc cards for %d documents not built: %s", len(batch), exc)
                continue
            by_doc: dict[str, list[ChunkRow]] = defaultdict(list)
            for row in rows:
                by_doc[row["doc_id"]].append(row)
            cards = [
                card
                for doc_id, doc_rows in by_doc.items()
                if (
                    card := build_card(
                        doc_id,
                        [r["chunk_id"] for r in doc_rows],
                        [r["text"] for r in doc_rows],
                        [r["vector"] for r in doc_rows],
                    )
                )
            ]
            self.meta.save_cards(cards)
            built.extend(cards)
        if built:
            logger.info("built doc cards for %d documents indexed before cards", len(built))
        return built
```

- [ ] **Step 7: Run the tests**

Run: `uv run pytest core/ragcore/tests/test_brain_topics.py core/ragcore/tests/test_real_store.py core/ragcore/tests/test_meta_store.py -v && uv run ruff check .`
Expected: all PASS.

- [ ] **Step 8: Install and commit**

```bash
scripts/install-local.sh
git add core/ragcore/src/ragcore/brain/cards.py core/ragcore/src/ragcore/store \
  core/ragcore/tests/test_brain_topics.py
git commit -m "feat(brain): doc cards written with documents, backfilled from chunks"
```

---

### Task 3: `Brain` — topic tree, assignment, recluster

**Files:**
- Create: `core/ragcore/src/ragcore/brain/topics.py`, `core/ragcore/src/ragcore/brain/null.py`
- Modify: `core/ragcore/src/ragcore/api/schemas.py` (Topic, TopicTree), `core/ragcore/src/ragcore/store/meta.py` (topics table, brain state), `core/ragcore/src/ragcore/store/real.py` (wire the brain), `core/ragcore/src/ragcore/stub/store.py` (`self.brain = NullBrain()`), `core/ragcore/src/ragcore/ports.py` (`BrainPort`, `StorePort.brain`), `core/ragcore/src/ragcore/api/app.py` (start task)
- Test: `core/ragcore/tests/test_brain_topics.py`

**Interfaces:**
- Consumes: `build_tree`, `Node` (Task 1); `DocCard`, `MetaStore.load_cards/card_ids`, `RealStore.backfill_cards` (Task 2).
- Produces: schemas `Topic {id, parent_id, level, label, keywords, doc_ids, rep_chunk_ids}`, `TopicTree {status: "none"|"building"|"ready", built_at, changed_since, topics}`.
- Produces on `MetaStore`: `load_topics() -> dict[str, Node]`, `replace_topics(nodes, state)`, `save_topics(nodes, state)`, `brain_state() -> dict`, `clear_brain()`.
- Produces `Brain(store)`: attributes `nodes: dict[str, Node]`, `state: dict`, `building: bool`; methods `tree() -> TopicTree`, `topic(id) -> Topic | None`, `async start()`, `async changed(added: list[str], removed: list[str])`, `removed(doc_ids)` (sync, thread-safe), `clear()`, `async recluster()`, `async settle()`. Constants `MIN_CARDS = 6`, `RECLUSTER_SHARE = 0.15`.
- Produces `NullBrain` with `building`, `tree`, `topic`, `start`, `recluster` (Tasks 4–5 add `route`, `for_query`).

- [ ] **Step 1: Append the failing tests** to `core/ragcore/tests/test_brain_topics.py`

Add imports at the top: `import pytest`, `from ragcore.brain import topics as topics_module`, `from ragcore.brain.cards import build_card`.

```python
# ------------------------------------------------------------------- topics


def leaves(store: RealStore) -> list:
    parents = {n.parent_id for n in store.brain.nodes.values()}
    return [n for n in store.brain.nodes.values() if n.id not in parents]


async def test_six_cards_build_a_tree_over_every_document(tmp_path: Path) -> None:
    store, _ = await ingested(tmp_path, 8)

    await store.brain.settle()  # the ingest scheduled it: no tree yet, 8 cards

    tree = store.brain.tree()
    assert tree.status == "ready"
    assert tree.built_at is not None and tree.changed_since == 0
    top = [t for t in tree.topics if t.level == 0]
    assert sorted(d for t in top for d in t.doc_ids) == sorted(store.documents)
    assert all(t.label and t.rep_chunk_ids for t in top)
    assert store.index_stats().topics == len(tree.topics)


async def test_under_six_cards_there_is_no_tree(tmp_path: Path) -> None:
    store, _ = await ingested(tmp_path, 5)

    await store.brain.recluster()

    assert store.brain.tree().status == "none"
    assert store.brain.tree().topics == []


async def test_the_tree_survives_a_restart(tmp_path: Path) -> None:
    store, _ = await ingested(tmp_path, 8)
    await store.brain.settle()
    before = store.brain.tree()
    store.close()

    after = build(tmp_path).brain.tree()

    assert after.topics == before.topics
    assert after.built_at == before.built_at


async def test_a_new_document_joins_its_nearest_leaf(tmp_path: Path) -> None:
    store, source = await ingested(tmp_path, 8)
    await store.brain.settle()
    path = tmp_path / "library" / "fresh.md"
    path.write_text("# Fresh\n\nviolin chord melody tomato.\n")

    await store.ingest_source_async(source.id)

    doc_id = doc_id_for(str(path))
    card = store.meta.load_cards([doc_id])[doc_id]
    nearest = max(leaves(store), key=lambda n: float(n.centroid @ card.vector))
    holders = {n.id for n in store.brain.nodes.values() if doc_id in n.doc_ids}
    lineage, node_id = set(), nearest.id
    while node_id is not None:
        lineage.add(node_id)
        node_id = store.brain.nodes[node_id].parent_id
    assert holders == lineage
    assert store.brain.tree().changed_since == 1


async def test_a_removed_document_leaves_every_topic(tmp_path: Path) -> None:
    store, _ = await ingested(tmp_path, 8)
    await store.brain.settle()
    doc_id = sorted(store.documents)[0]

    store.remove_document(doc_id)

    assert all(doc_id not in t.doc_ids for t in store.brain.tree().topics)
    assert store.brain.tree().changed_since == 1


async def test_churn_past_fifteen_percent_schedules_a_recluster(tmp_path: Path) -> None:
    store, source = await ingested(tmp_path, 10)
    await store.brain.settle()
    built = store.brain.tree().built_at
    for i in (0, 1):  # 2 of 10 documents: 20%
        (tmp_path / "library" / f"note{i}.md").write_text(f"# Note {i}\n\nrewritten {i} text.\n")

    await store.ingest_source_async(source.id)
    await store.brain.settle()

    tree = store.brain.tree()
    assert tree.built_at > built
    assert tree.changed_since == 0


async def test_rebuilding_the_index_clears_the_brain(tmp_path: Path) -> None:
    store, _ = await ingested(tmp_path, 8)
    await store.brain.settle()

    store.rebuild_index()

    assert store.meta.card_ids() == set()
    assert store.brain.tree().topics == []
    store.close()
    assert build(tmp_path).brain.tree().topics == []


async def test_wipe_clears_the_brain(tmp_path: Path) -> None:
    store, _ = await ingested(tmp_path, 8)
    await store.brain.settle()

    store.wipe(keep_connections=True)

    assert store.brain.tree().status == "none"
    store.close()
    reopened = build(tmp_path)
    assert reopened.brain.tree().topics == []
    assert reopened.brain.tree().changed_since == 0


async def test_a_doc_added_during_a_recluster_joins_the_new_tree(
    tmp_path: Path, monkeypatch
) -> None:
    store, _ = await ingested(tmp_path, 8)
    await store.brain.settle()
    gone = sorted(store.documents)[0]
    [vector] = await store.embedder.embed(["late arrival words"])
    late = build_card("doc_late", ["doc_late:1:0"], ["late arrival words"], [vector])
    real_build = topics_module.build_tree

    def racing(*args, **kwargs):
        nodes = real_build(*args, **kwargs)
        store.meta.save_cards([late])  # an ingest that landed while the tree was building
        store.meta.delete_documents([gone])  # and a removal
        return nodes

    monkeypatch.setattr(topics_module, "build_tree", racing)
    await store.brain.recluster()

    held = {d for t in store.brain.tree().topics for d in t.doc_ids}
    assert "doc_late" in held
    assert gone not in held


async def test_a_recluster_that_fails_keeps_the_previous_tree(tmp_path: Path, monkeypatch) -> None:
    store, _ = await ingested(tmp_path, 8)
    await store.brain.settle()
    before = store.brain.tree().topics

    def broken(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(topics_module, "build_tree", broken)
    with pytest.raises(RuntimeError, match="boom"):
        await store.brain.recluster()

    assert store.brain.tree().topics == before
    assert store.brain.tree().status == "ready"


async def test_start_backfills_and_builds_a_tree(tmp_path: Path) -> None:
    store, _ = await ingested(tmp_path, 8)
    await store.brain.settle()
    store.brain.clear()  # an older library: documents indexed, no cards, no tree
    store.close()
    reopened = build(tmp_path)

    await reopened.brain.start()
    await reopened.brain.settle()

    assert reopened.meta.card_ids() == set(reopened.documents)
    assert reopened.brain.tree().status == "ready"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest core/ragcore/tests/test_brain_topics.py -v`
Expected: new tests FAIL with `AttributeError: 'RealStore' object has no attribute 'brain'`.

- [ ] **Step 3: Schemas** — in `api/schemas.py`, add after the `query` section's `ErrorEvent` (before `# ---- slide to text`):

```python
# ---------------------------------------------------------------------------- brain


class Topic(BaseModel):
    id: str
    parent_id: str | None = None
    level: int
    label: str
    keywords: list[str]
    doc_ids: list[str] = Field(description="Every document in this topic's subtree.")
    rep_chunk_ids: list[str]


class TopicTree(BaseModel):
    """The topic map, flat; parents come before their children."""

    status: Literal["none", "building", "ready"]
    built_at: datetime | None = None
    changed_since: int = 0
    topics: list[Topic]
```

- [ ] **Step 4: Topics and brain state in `store/meta.py`**

Add `from ragcore.brain.cluster import Node` and constants:

```python
_BRAIN_KEY = "brain"
EMPTY_BRAIN_STATE = {"built_at": None, "changed_since": 0}
_UPSERT_TOPIC = (
    "INSERT INTO topics (id, parent_id, level, centroid, json) VALUES (?, ?, ?, ?, ?) "
    "ON CONFLICT(id) DO UPDATE SET parent_id = excluded.parent_id, level = excluded.level, "
    "centroid = excluded.centroid, json = excluded.json"
)


def _topic_row(node: Node) -> tuple:
    payload = {
        "label": node.label,
        "keywords": node.keywords,
        "doc_ids": node.doc_ids,
        "rep_chunk_ids": node.rep_chunk_ids,
    }
    return (
        node.id,
        node.parent_id,
        node.level,
        node.centroid.astype(np.float32).tobytes(),
        json.dumps(payload),
    )
```

Table in `__init__`:

```python
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS topics (id TEXT PRIMARY KEY, parent_id TEXT, "
                "level INTEGER NOT NULL, centroid BLOB NOT NULL, json TEXT NOT NULL)"
            )
```

Methods (a new `# ---- brain` section):

```python
    # -------------------------------------------------------------------- brain

    def load_topics(self) -> dict[str, Node]:
        """In the order they were written: parents first, siblings largest first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, parent_id, level, centroid, json FROM topics ORDER BY rowid"
            ).fetchall()
        nodes: dict[str, Node] = {}
        for node_id, parent_id, level, blob, raw in rows:
            data = json.loads(raw)
            nodes[node_id] = Node(
                id=node_id,
                parent_id=parent_id,
                level=level,
                centroid=np.frombuffer(blob, dtype=np.float32),
                doc_ids=data["doc_ids"],
                keywords=data["keywords"],
                label=data["label"],
                rep_chunk_ids=data["rep_chunk_ids"],
            )
        return nodes

    def replace_topics(self, nodes: list[Node], state: dict) -> None:
        """A recluster: the whole tree and its state in one transaction."""
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM topics")
            self._conn.executemany(_UPSERT_TOPIC, [_topic_row(n) for n in nodes])
            self._save_brain_state(state)

    def save_topics(self, nodes: list[Node], state: dict) -> None:
        """An assignment: only the topics whose documents changed."""
        with self._lock, self._conn:
            self._conn.executemany(_UPSERT_TOPIC, [_topic_row(n) for n in nodes])
            self._save_brain_state(state)

    def _save_brain_state(self, state: dict) -> None:
        self._conn.execute(
            "INSERT INTO settings (key, json) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET json = excluded.json",
            (_BRAIN_KEY, json.dumps(state)),
        )

    def brain_state(self) -> dict:
        with self._lock:
            row = self._conn.execute(
                "SELECT json FROM settings WHERE key = ?", (_BRAIN_KEY,)
            ).fetchone()
        return json.loads(row[0]) if row else dict(EMPTY_BRAIN_STATE)

    def clear_brain(self) -> None:
        with self._lock, self._conn:
            self._clear_brain()

    def _clear_brain(self) -> None:
        self._conn.execute("DELETE FROM doc_cards")
        self._conn.execute("DELETE FROM topics")
        self._conn.execute("DELETE FROM settings WHERE key = ?", (_BRAIN_KEY,))
```

In `wipe`, replace the `DELETE FROM doc_cards` line from Task 2 with `self._clear_brain()`.

- [ ] **Step 5: Write `core/ragcore/src/ragcore/brain/topics.py`**

```python
"""The topic map over doc cards: rebuilt by reclustering, kept current by assignment between.

One per RealStore. Between reclusters a new or changed document joins the leaf whose
centroid is nearest and a removed one leaves; each counts toward ``changed_since`` until
15% of the cards have moved and the brain schedules its own recluster.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, datetime

import numpy as np

from ragcore.api.schemas import Topic, TopicTree
from ragcore.brain.cards import DocCard
from ragcore.brain.cluster import Node, build_tree
from ragcore.store.meta import EMPTY_BRAIN_STATE

logger = logging.getLogger("ragcore.brain")

MIN_CARDS = 6
RECLUSTER_SHARE = 0.15


def to_topic(node: Node) -> Topic:
    return Topic(
        id=node.id,
        parent_id=node.parent_id,
        level=node.level,
        label=node.label,
        keywords=node.keywords,
        doc_ids=node.doc_ids,
        rep_chunk_ids=node.rep_chunk_ids,
    )


def _lineage(nodes: dict[str, Node], node_id: str | None) -> Iterator[str]:
    while node_id is not None:
        yield node_id
        node_id = nodes[node_id].parent_id


class Brain:
    def __init__(self, store) -> None:
        self.store = store
        self.nodes: dict[str, Node] = store.meta.load_topics()
        self.state: dict = store.meta.brain_state()
        self.building = False
        # nodes and state are swapped, never mutated: readers keep the dict they started with.
        # Ingest (event loop), the sync remove routes (threadpool) and a recluster (thread)
        # all write, so writes take this lock.
        self._lock = threading.Lock()
        self._recluster = asyncio.Lock()
        self._generation = 0  # bumped by clear(); a recluster started before it is discarded
        self._task: asyncio.Task[None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    # ------------------------------------------------------------------- reads

    def tree(self) -> TopicTree:
        nodes, state = self.nodes, self.state
        status = "building" if self.building else "ready" if nodes else "none"
        return TopicTree(
            status=status,
            built_at=state["built_at"],
            changed_since=state["changed_since"],
            topics=[to_topic(n) for n in nodes.values()],
        )

    def topic(self, topic_id: str) -> Topic | None:
        node = self.nodes.get(topic_id)
        return to_topic(node) if node else None

    # ----------------------------------------------------------------- changes

    async def start(self) -> None:
        """At startup: cards for documents indexed before cards existed, then a recluster if due."""
        self._loop = asyncio.get_running_loop()
        built = {c.doc_id: c for c in await asyncio.to_thread(self.store.backfill_cards)}
        if built:
            self._apply(built, set(built), len(built))
        self._kick()

    async def changed(self, added: list[str], removed: list[str]) -> None:
        """End of an ingest: re-place what was (re)indexed, drop what left the folder."""
        self._loop = asyncio.get_running_loop()
        cards = self.store.meta.load_cards(added) if added else {}
        self._apply(cards, {*added, *removed}, len(added) + len(removed))
        self._kick()

    def removed(self, doc_ids: list[str]) -> None:
        """From the sync remove routes, which run in a worker thread."""
        self._apply({}, set(doc_ids), len(doc_ids))
        if self._loop is not None:
            with contextlib.suppress(RuntimeError):  # the loop is closed: nothing to schedule on
                self._loop.call_soon_threadsafe(self._kick)

    def clear(self) -> None:
        """The index was rebuilt or the library wiped: cards and topics go with the vectors."""
        with self._lock:
            self._generation += 1
            self.store.meta.clear_brain()
            self.nodes, self.state = {}, dict(EMPTY_BRAIN_STATE)

    def _apply(self, cards: dict[str, DocCard], drop: set[str], count: int) -> None:
        """Take ``drop`` out of every topic, put each card in its nearest leaf, persist."""
        with self._lock:
            nodes = dict(self.nodes)
            dirty: set[str] = set()
            for node in self.nodes.values():
                if drop.intersection(node.doc_ids):
                    kept = [d for d in node.doc_ids if d not in drop]
                    nodes[node.id] = replace(node, doc_ids=kept)
                    dirty.add(node.id)
            parents = {n.parent_id for n in nodes.values()}
            leaves = [n.id for n in nodes.values() if n.id not in parents]
            if leaves and cards:
                centroids = np.stack([nodes[i].centroid for i in leaves])
                for card in cards.values():
                    leaf = leaves[int(np.argmax(centroids @ card.vector))]
                    for node_id in _lineage(nodes, leaf):
                        node = nodes[node_id]
                        nodes[node_id] = replace(node, doc_ids=[*node.doc_ids, card.doc_id])
                        dirty.add(node_id)
            state = {**self.state, "changed_since": self.state["changed_since"] + count}
            self.store.meta.save_topics([nodes[i] for i in dirty], state)
            self.nodes, self.state = nodes, state

    # --------------------------------------------------------------- recluster

    def _due(self) -> bool:
        if self.store.index_blocked:
            return False
        cards = len(self.store.meta.card_ids())
        if not self.nodes:
            return cards >= MIN_CARDS
        return self.state["changed_since"] >= RECLUSTER_SHARE * cards

    def _kick(self) -> None:
        """Schedule a background recluster if one is due and none is pending. Loop thread only."""
        if (self._task is None or self._task.done()) and self._due():
            self._task = asyncio.get_running_loop().create_task(self._auto())

    async def _auto(self) -> None:
        try:
            await self.recluster()
        except Exception:  # noqa: BLE001 - the previous tree keeps serving
            logger.exception("recluster failed; keeping the previous topic map")

    async def settle(self) -> None:
        """Wait for a scheduled recluster: the CLI exits right after its ingest."""
        if self._task is not None:
            await self._task

    async def recluster(self) -> None:
        """Rebuild the tree from every card. One at a time: a second caller waits, then rebuilds."""
        async with self._recluster:
            self.building = True
            try:
                await asyncio.to_thread(self._rebuild)
            finally:
                self.building = False

    def _rebuild(self) -> None:
        meta = self.store.meta
        generation = self._generation
        cards = meta.load_cards()
        ids = sorted(cards)
        nodes: list[Node] = []
        if len(ids) >= MIN_CARDS:
            nodes = build_tree(
                ids,
                np.stack([cards[i].vector for i in ids]),
                [cards[i].terms for i in ids],
                [cards[i].summary_chunk_ids[0] for i in ids],
            )
        state = {"built_at": datetime.now(tz=UTC).isoformat(), "changed_since": 0}
        with self._lock:
            if generation != self._generation:
                return  # cleared while building: these cards are gone
            meta.replace_topics(nodes, state)
            self.nodes, self.state = {n.id: n for n in nodes}, state
        logger.info("topic map rebuilt: %d topics over %d documents", len(nodes), len(ids))
        # Documents indexed or removed while the tree was being built.
        current = meta.card_ids()
        late, gone = current - set(ids), set(ids) - current
        if late or gone:
            self._apply(meta.load_cards(sorted(late)), late | gone, len(late) + len(gone))
```

- [ ] **Step 6: Write `core/ragcore/src/ragcore/brain/null.py`**

```python
"""The stub backend's brain: no cards, no topics. Kept apart so the stub never loads LanceDB."""

from __future__ import annotations

from ragcore.api.schemas import Topic, TopicTree


class NullBrain:
    building = False

    def tree(self) -> TopicTree:
        return TopicTree(status="none", topics=[])

    def topic(self, topic_id: str) -> Topic | None:
        return None

    async def start(self) -> None:
        return None

    async def recluster(self) -> None:
        return None
```

- [ ] **Step 7: Wire it**

`stub/store.py`: import `from ragcore.brain.null import NullBrain`; in `Store.__init__` after `self.retriever = Retriever([])` add `self.brain = NullBrain()`.

`store/real.py`: import `from ragcore.brain.topics import Brain`. In `__init__`, after `self.retriever = HybridRetriever(...)` add `self.brain = Brain(self)`. Then:

```python
    # rebuild_index: first line
        self.brain.clear()
    # remove_source: after self._forget(doc_ids)
        self.brain.removed(doc_ids)
    # remove_document: after self._forget([doc_id])
        self.brain.removed([doc_id])
    # wipe: after self.meta.wipe(...)
        self.brain.clear()
    # index_stats: topics=0 becomes
            topics=len(self.brain.nodes),
```

In `ingest_source_async`, replace

```python
            if todo or gone:
                await asyncio.to_thread(self.vectors.optimize)
```

with

```python
            if todo or gone:
                await asyncio.to_thread(self.vectors.optimize)
                await self.brain.changed([doc_id for doc_id, _ in todo], gone)
```

`ports.py`: add (imports `Topic`, `TopicTree` from schemas):

```python
@runtime_checkable
class BrainPort(Protocol):
    """The topic map. ``brain.topics.Brain`` on the real backend, ``brain.null.NullBrain`` on the stub."""

    building: bool

    def tree(self) -> TopicTree: ...
    def topic(self, topic_id: str) -> Topic | None: ...
    async def start(self) -> None: ...
    async def recluster(self) -> None: ...
```

and in `StorePort` add `brain: BrainPort` after `retriever: RetrieverPort`.

`api/app.py` lifespan: after the `watch_task = ...` line add
`brain_task = asyncio.create_task(backend.store.brain.start())`, and in `finally`, before the watch-task block:

```python
            brain_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await brain_task
```

- [ ] **Step 8: Run the tests**

Run: `uv run pytest core/ragcore/tests -v && uv run ruff check .`
Expected: all PASS (the existing suite included: the stub has a `NullBrain`, and app startup runs `brain.start()` on both backends).

- [ ] **Step 9: Install and commit**

```bash
scripts/install-local.sh
git add core/ragcore/src/ragcore core/ragcore/tests/test_brain_topics.py
git commit -m "feat(brain): topic tree with nearest-leaf assignment and background recluster"
```

---

### Task 4: Router

**Files:**
- Create: `core/ragcore/src/ragcore/brain/route.py`
- Modify: `core/ragcore/src/ragcore/brain/topics.py` (`route`), `core/ragcore/src/ragcore/brain/null.py` (`route`), `core/ragcore/src/ragcore/ports.py` (`BrainPort.route`), `core/ragcore/src/ragcore/api/routes/query.py` (drop `GLOBAL_HINTS`, `route_mode`)
- Test: `core/ragcore/tests/test_brain_route.py`; modify `core/ragcore/tests/test_query.py` (router tests move out)

**Interfaces:**
- Produces: `route.Routed(mode: Literal["local","global"], reason: str, vector: list[float] | None = None)`, `route.keyword_route(question, vector=None) -> Routed`, `route.score_route(question, vector, examples) -> Routed`, `route.embed_examples(embedder) -> tuple[np.ndarray, np.ndarray]`, `route.EXAMPLES`, `route.MARGIN = 0.05`, `route.GLOBAL_HINTS`.
- Produces: `Brain.route(question) -> Routed` (async), `NullBrain.route` (keywords only).

- [ ] **Step 1: Write the failing tests** — `core/ragcore/tests/test_brain_route.py`

```python
"""Local or global: example similarity, keyword hints in the dead zone, keywords alone offline."""

from __future__ import annotations

from pathlib import Path

import httpx
import numpy as np
import pytest
from ragcore.brain.null import NullBrain
from ragcore.brain.route import EXAMPLES, keyword_route, score_route
from ragcore.config import Config
from ragcore.models.fakes import FakeEmbedClient
from ragcore.store.real import RealStore

# One global example along e0, one local along e1: the score is q[0] - q[1].
EXAMPLES_2D = (np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]]))


def test_a_clear_global_score_routes_global_and_reports_it() -> None:
    routed = score_route("what is in here", [0.53, 0.47], EXAMPLES_2D)

    assert routed.mode == "global"
    assert routed.reason == "Global (router 0.06)"
    assert routed.vector == [0.53, 0.47]


def test_a_clear_local_score_routes_local() -> None:
    routed = score_route("what is in here", [0.47, 0.53], EXAMPLES_2D)

    assert routed.mode == "local"
    assert routed.reason == "Local (router -0.06)"


@pytest.mark.parametrize(
    ("question", "mode"),
    [("what are the main themes across my notes", "global"), ("what is in here", "local")],
)
def test_the_dead_zone_falls_back_to_keyword_hints(question: str, mode: str) -> None:
    routed = score_route(question, [0.51, 0.49], EXAMPLES_2D)

    assert routed.mode == mode
    assert routed.reason.endswith("(router 0.02)")


@pytest.mark.parametrize(
    "question",
    [
        "What are the main themes across the corpus?",
        "Summarize all documents",
        "Compare the two approaches",
        "What topics does this library cover?",
    ],
)
def test_a_corpus_wide_phrase_routes_global(question: str) -> None:
    routed = keyword_route(question)

    assert routed.mode == "global"
    assert "asks about the corpus" in routed.reason


def test_a_passage_question_routes_local() -> None:
    assert keyword_route("How does reciprocal rank fusion combine two rankings?").mode == "local"


def test_keyword_hints_ignore_case() -> None:
    assert keyword_route("SUMMARIZE ALL the notes").mode == "global"


def test_the_examples_cover_both_modes_and_five_languages() -> None:
    kinds = [kind for kind, _ in EXAMPLES]

    assert len(EXAMPLES) >= 28
    assert kinds.count("global") >= 14 and kinds.count("local") >= 14


class DownEmbedClient(FakeEmbedClient):
    async def embed(self, texts, *, batch_size=32, query=False):
        raise httpx.ConnectError("embedder down")


class CountingEmbedClient(FakeEmbedClient):
    def __init__(self) -> None:
        self.calls: list[int] = []

    async def embed(self, texts, *, batch_size=32, query=False):
        self.calls.append(len(texts))
        return await super().embed(texts, batch_size=batch_size, query=query)


def store_with(tmp_path: Path, embedder) -> RealStore:
    return RealStore(Config(data_dir=tmp_path, backend="real"), embedder=embedder)


async def test_an_embedder_outage_routes_by_keywords(tmp_path: Path) -> None:
    store = store_with(tmp_path, DownEmbedClient())

    routed = await store.brain.route("main themes across my notes")

    assert routed.mode == "global"
    assert routed.vector is None


async def test_the_examples_are_embedded_once(tmp_path: Path) -> None:
    embedder = CountingEmbedClient()
    store = store_with(tmp_path, embedder)

    await store.brain.route("first question")
    routed = await store.brain.route("second question")

    assert embedder.calls == [len(EXAMPLES), 1, 1]
    assert routed.vector is not None


async def test_the_stub_brain_routes_by_keywords() -> None:
    assert (await NullBrain().route("Summarize all documents")).mode == "global"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest core/ragcore/tests/test_brain_route.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'ragcore.brain.route'`.

- [ ] **Step 3: Write `core/ragcore/src/ragcore/brain/route.py`**

```python
"""Local or global: is the question about passages or about the library as a whole?

Cosine to a few labelled example questions decides; inside the dead zone the keyword hints
do. The manual mode in the request still wins (query route).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

# Questions about the corpus as a whole cannot be answered from six passages.
GLOBAL_HINTS = (
    "across",
    "overall",
    "in general",
    "main themes",
    "summarize all",
    "summarise all",
    "every document",
    "all documents",
    "what topics",
    "compare the",
)

# Tuned on eval/golden/brain.jsonl; keep its exact phrasings out of here.
MARGIN = 0.05

EXAMPLES: tuple[tuple[str, str], ...] = (
    ("global", "What are the recurring themes across these files?"),
    ("global", "Summarise the collection as a whole"),
    ("global", "Which subjects does this library cover?"),
    ("global", "Give me a bird's-eye view of all the material"),
    ("global", "What is the big picture across every document?"),
    ("global", "Which ideas come up again and again in my notes?"),
    ("global", "How do the different parts of my notes relate to each other?"),
    ("global", "What would you say this archive is mostly about?"),
    ("global", "What are the main threads running through my library?"),
    ("global", "Quali temi ricorrono in tutti questi file?"),
    ("global", "Riassumi la raccolta nel suo insieme"),
    ("global", "Di cosa parla complessivamente questo archivio?"),
    ("global", "Quels sont les grands thèmes de l'ensemble de mes documents ?"),
    ("global", "Worum geht es in meinen Notizen insgesamt?"),
    ("global", "¿De qué tratan mis documentos en conjunto?"),
    ("local", "How does reciprocal rank fusion combine two rankings?"),
    ("local", "What is the deadline mentioned in the contract?"),
    ("local", "Which page explains how to reset the device?"),
    ("local", "What did the report say about revenue in March?"),
    ("local", "Define chunk overlap"),
    ("local", "Who wrote the onboarding guide?"),
    ("local", "What error code does the upload return when the file is too big?"),
    ("local", "How many grams of salt go into the brine?"),
    ("local", "What is the warranty period for the laptop?"),
    ("local", "Come si imposta il timeout del server?"),
    ("local", "Cosa dice il documento sulla garanzia?"),
    ("local", "Qual è la scadenza indicata nella fattura?"),
    ("local", "Comment configurer la sauvegarde automatique ?"),
    ("local", "Wie lange dauert die Lieferung laut Vertrag?"),
    ("local", "¿Qué dice el contrato sobre la cancelación?"),
)

Examples = tuple[np.ndarray, np.ndarray]  # (global rows, local rows), unit vectors


@dataclass(frozen=True, slots=True)
class Routed:
    mode: Literal["local", "global"]
    reason: str
    vector: list[float] | None = None  # the question's embedding, reused by the global path


def keyword_route(question: str, vector: list[float] | None = None) -> Routed:
    lowered = question.lower()
    hit = next((h for h in GLOBAL_HINTS if h in lowered), None)
    if hit:
        return Routed("global", f'Phrase "{hit}" asks about the corpus, not a passage', vector)
    return Routed("local", "Answerable from individual passages", vector)


async def embed_examples(embedder) -> Examples:
    matrix = np.asarray(
        await embedder.embed([text for _, text in EXAMPLES], query=True), dtype=np.float32
    )
    kinds = np.array([kind for kind, _ in EXAMPLES])
    return matrix[kinds == "global"], matrix[kinds == "local"]


def score_route(question: str, vector: list[float], examples: Examples) -> Routed:
    q = np.asarray(vector, dtype=np.float32)
    score = float(np.max(examples[0] @ q) - np.max(examples[1] @ q))
    if score > MARGIN:
        return Routed("global", f"Global (router {score:.2f})", vector)
    if score < -MARGIN:
        return Routed("local", f"Local (router {score:.2f})", vector)
    hint = keyword_route(question, vector)
    return Routed(hint.mode, f"{hint.reason} (router {score:.2f})", vector)
```

- [ ] **Step 4: `route` on both brains**

`brain/topics.py`: imports `import httpx`, `from ragcore.brain.route import Examples, Routed, embed_examples, keyword_route, score_route`. In `__init__` add `self._examples: Examples | None = None`. Add to the reads section:

```python
    async def route(self, question: str) -> Routed:
        """Embed the question (and, once, the examples); keywords alone if the embedder is down."""
        embedder = self.store.embedder
        try:
            if self._examples is None:
                self._examples = await embed_examples(embedder)
            [vector] = await embedder.embed([question], query=True)
        except httpx.HTTPError as exc:
            logger.warning("router could not embed, routing by keywords: %s", exc)
            return keyword_route(question)
        return score_route(question, vector, self._examples)
```

`brain/null.py`: import `from ragcore.brain.route import Routed, keyword_route` and add

```python
    async def route(self, question: str) -> Routed:
        return keyword_route(question)
```

`ports.py` `BrainPort`: add `async def route(self, question: str) -> Routed: ...` with `from ragcore.brain.route import Routed` under `if TYPE_CHECKING:` (add `from typing import TYPE_CHECKING`).

- [ ] **Step 5: Query route uses the brain**

In `api/routes/query.py` delete `GLOBAL_HINTS` (lines 40-52) and `route_mode` (lines 138-143). In `events()` replace `mode, reason = route_mode(question)` with:

```python
        routed = await store.brain.route(question)
        mode, reason = routed.mode, routed.reason
```

In `tests/test_query.py` remove `route_mode` from the import list and delete the three router tests (`test_a_corpus_wide_question_routes_to_global`, `test_a_passage_question_routes_to_local`, `test_the_router_ignores_case`); they live on in `test_brain_route.py`.

- [ ] **Step 6: Run the tests**

Run: `uv run pytest core/ragcore/tests -v && uv run ruff check .`
Expected: all PASS.

- [ ] **Step 7: Install and commit**

```bash
scripts/install-local.sh
git add core/ragcore/src/ragcore core/ragcore/tests/test_brain_route.py core/ragcore/tests/test_query.py
git commit -m "feat(brain): embedding router with keyword fallback"
```

---

### Task 5: Global answer path + query route

**Files:**
- Create: `core/ragcore/src/ragcore/brain/answer.py`
- Modify: `core/ragcore/src/ragcore/api/schemas.py` (`QueryRequest.topic_id`, `RetrievalSettings.global_token_budget`), `core/ragcore/src/ragcore/llm.py` (`GLOBAL_SYSTEM_PROMPT`, `global_prompt`, `system_prompt_for(base=)`), `core/ragcore/src/ragcore/retrieve/hybrid.py` (`_to_chunk` → `to_chunk`), `core/ragcore/src/ragcore/brain/topics.py` + `null.py` (`for_query`), `core/ragcore/src/ragcore/ports.py`, `core/ragcore/src/ragcore/api/routes/query.py` (`retrieve`, 404), `core/ragcore/src/ragcore/cli.py` (`_ask`, `_index`)
- Test: `core/ragcore/tests/test_brain_topics.py`, `core/ragcore/tests/test_query.py`, `core/ragcore/tests/conftest.py`

**Interfaces:**
- Consumes: `Brain.nodes`, `Brain.store`, `MetaStore.load_cards`, `Routed.vector`, `pack()` (`retrieve/pack.py`), `allowed()` (`stub/retrieval.py`), `to_chunk()` (`retrieve/hybrid.py`), `retrieval_query()` (`query.py`).
- Produces: `answer.GlobalContext(chunks: list[RetrievedChunk], outline: str, scope_label: str | None, n_topics: int, candidates: int)`; `Brain.for_query(vector, *, search, question, filters, doc_meta, topic_id, budget) -> GlobalContext | None`; `llm.GLOBAL_SYSTEM_PROMPT`, `llm.global_prompt(outline) -> str`, `llm.system_prompt_for(lang, extra="", base=SYSTEM_PROMPT)`; `query.Retrieval` and `async query.retrieve(store, question, *, mode, topic_id, filters, history, settings, lang) -> Retrieval`; fixture `wait_for_topics(client) -> TopicTree`.

- [ ] **Step 1: Write the failing store-level tests** — append to `core/ragcore/tests/test_brain_topics.py` (add `from ragcore.api.schemas import QueryFilters`)

```python
# ----------------------------------------------------------- global context


async def global_context(store: RealStore, vector, **overrides):
    args = {
        "search": "what is here",
        "question": "what is here",
        "filters": QueryFilters(),
        "doc_meta": store.doc_meta(),
        "topic_id": None,
        "budget": 6144,
    }
    return await store.brain.for_query(vector, **(args | overrides))


async def question_vector(store: RealStore) -> list[float]:
    [vector] = await store.embedder.embed(["what is here"], query=True)
    return vector


async def test_a_small_library_answers_global_from_cards(tmp_path: Path) -> None:
    store, _ = await ingested(tmp_path, 3)

    context = await global_context(store, await question_vector(store))

    assert store.brain.tree().status == "none"
    assert context is not None and context.n_topics == 3
    assert {c.doc_id for c in context.chunks} == set(store.documents)
    assert len(context.outline.splitlines()) == 3


async def test_global_context_respects_the_document_filter(tmp_path: Path) -> None:
    store, _ = await ingested(tmp_path, 8)
    await store.brain.settle()
    keep = sorted(store.documents)[:2]

    context = await global_context(
        store, await question_vector(store), filters=QueryFilters(doc_ids=keep)
    )

    assert context is not None and context.chunks
    assert {c.doc_id for c in context.chunks} <= set(keep)


async def test_global_context_without_a_vector_uses_representatives(tmp_path: Path) -> None:
    store, _ = await ingested(tmp_path, 8)
    await store.brain.settle()
    top = [n for n in store.brain.nodes.values() if n.parent_id is None]

    context = await global_context(store, None)

    assert context is not None and context.n_topics == len(top)
    assert len(context.chunks) == len(top)  # one representative each, no dense hits
    assert {c.chunk_id for c in context.chunks} <= {cid for n in top for cid in n.rep_chunk_ids}


async def test_global_passages_interleave_across_topics(tmp_path: Path) -> None:
    store, _ = await ingested(tmp_path, 8)
    await store.brain.settle()
    top = [n for n in store.brain.nodes.values() if n.parent_id is None]
    owner = {d: n.id for n in top for d in n.doc_ids}

    context = await global_context(store, await question_vector(store))

    first_round = [owner[c.doc_id] for c in context.chunks[: len(top)]]
    assert len(set(first_round)) == len(top)
    assert all(len(c.text.split()) for c in context.chunks)


async def test_a_blocked_index_has_no_global_context(tmp_path: Path) -> None:
    store, _ = await ingested(tmp_path, 3)
    store.index_blocked = "another embedder"

    assert await global_context(store, await question_vector(store)) is None
```

- [ ] **Step 2: Write the failing route tests**

`core/ragcore/tests/conftest.py`: add `import time` and

```python
@pytest.fixture
def wait_for_topics() -> Callable:
    """Block until the app's brain has a tree; the ingest in ``client`` scheduled it."""

    def wait(client: TestClient):
        brain = client.app.state.store.brain
        deadline = time.monotonic() + 15
        while (tree := brain.tree()).status != "ready":
            assert time.monotonic() < deadline, tree
            time.sleep(0.02)
        return tree

    return wait
```

`core/ragcore/tests/test_query.py`: add `real = pytest.mark.parametrize("client", ["real"], indirect=True)` under the imports; change `test_the_route_reports_a_manual_mode_as_manual` to send `"mode": "local"` and assert `mode["mode"] == "local"`; in `test_auto_is_the_default_mode` drop the `mode["mode"] == "global"` assertion (the stub has no topic map, so the keyword hint now falls back to local) and keep `mode["reason"] != "Set manually"`. Append:

```python
# ----------------------------------------------------------------- global mode


def ask(client: TestClient, read_events, **body) -> dict:
    with client.stream("POST", "/query", json={"q": "What is in here?"} | body) as response:
        return read_events(response)


def test_without_cards_a_global_question_runs_locally(client: TestClient, read_events) -> None:
    mode = dict(ask(client, read_events, mode="global"))["mode"]

    assert mode == {"mode": "local", "reason": "No topic map yet", "scope": None}


def test_an_unknown_topic_is_a_404_before_the_stream(client: TestClient) -> None:
    before = len(client.get("/chats").json())

    response = client.post("/query", json={"q": "x", "topic_id": "t404"})

    assert response.status_code == 404
    assert response.json()["code"] == "topic_not_found"
    assert len(client.get("/chats").json()) == before


@real
def test_a_global_answer_keeps_the_frame_order(client, read_events, wait_for_topics) -> None:
    wait_for_topics(client)

    events = ask(client, read_events, mode="global")

    names = [name for name, _ in events]
    assert names[:3] == ["start", "mode", "sources"]
    assert names[-2:] == ["citations", "done"]
    assert set(names[3:-2]) <= {"token"}
    assert dict(events)["mode"]["mode"] == "global"


@real
def test_global_sources_span_several_topics(client, read_events, wait_for_topics) -> None:
    tree = wait_for_topics(client)
    owner = {d: t.id for t in tree.topics if t.level == 0 for d in t.doc_ids}

    chunks = dict(ask(client, read_events, mode="global"))["sources"]["chunks"]

    assert len({owner[c["doc_id"]] for c in chunks}) >= 2


@real
def test_global_citations_are_checked_against_the_packed_passages(
    client, read_events, wait_for_topics
) -> None:
    from ragcore.api import deps

    wait_for_topics(client)

    class Cites:
        async def stream(
            self, question, chunks, directives, *, system_prompt=None, max_tokens=None, history=()
        ):
            yield f"Real [{chunks[0].doc_id}:{chunks[0].page_start}] and made up [doc_nope:1]."

    client.app.dependency_overrides[deps.get_answerer] = Cites
    try:
        events = dict(ask(client, read_events, mode="global"))
    finally:
        client.app.dependency_overrides.pop(deps.get_answerer, None)

    first = events["sources"]["chunks"][0]
    assert [c["doc_id"] for c in events["citations"]["citations"]] == [first["doc_id"]]
    assert events["citations"]["dropped"] == 1


@real
def test_a_topic_id_restricts_passages_to_its_subtree(client, read_events, wait_for_topics) -> None:
    topic = next(t for t in wait_for_topics(client).topics if t.level == 0)

    events = dict(ask(client, read_events, topic_id=topic.id))

    assert events["mode"] == {"mode": "global", "reason": "Asked about one topic", "scope": topic.label}
    assert events["sources"]["chunks"]
    assert {c["doc_id"] for c in events["sources"]["chunks"]} <= set(topic.doc_ids)


@real
def test_the_global_prompt_carries_the_topic_outline(client, read_events, wait_for_topics) -> None:
    from ragcore.api import deps
    from ragcore.llm import GLOBAL_SYSTEM_PROMPT

    tree = wait_for_topics(client)
    captured: dict = {}

    class Recorder:
        async def stream(
            self, question, chunks, directives, *, system_prompt=None, max_tokens=None, history=()
        ):
            captured["system_prompt"] = system_prompt
            yield "ok"

    client.app.dependency_overrides[deps.get_answerer] = Recorder
    try:
        ask(client, read_events, mode="global")
    finally:
        client.app.dependency_overrides.pop(deps.get_answerer, None)

    assert captured["system_prompt"].startswith(GLOBAL_SYSTEM_PROMPT)
    for topic in (t for t in tree.topics if t.level == 0):
        assert topic.label in captured["system_prompt"]
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest core/ragcore/tests/test_brain_topics.py core/ragcore/tests/test_query.py -v`
Expected: new tests FAIL (`AttributeError: ... 'for_query'`, 422 on `topic_id`, `ImportError: GLOBAL_SYSTEM_PROMPT`).

- [ ] **Step 4: Schemas, prompt, helper rename**

`api/schemas.py`: in `QueryRequest` add

```python
    topic_id: str | None = Field(
        default=None, description="Ask about one topic of the map: global mode, its subtree only."
    )
```

and in `RetrievalSettings` add `global_token_budget: int = 6144`.

`llm.py`: below `SYSTEM_PROMPT` add

```python
GLOBAL_SYSTEM_PROMPT = """You answer questions about a whole library of documents, using only
the passages provided. The passages were picked a few per topic so that every topic listed
below is represented. Synthesise across the topics rather than summarising one passage, and
say plainly where the passages leave a topic thin. Cite every claim with a marker of the form
[document_id:page] taken from the passage headers. If the passages do not contain the answer,
say so plainly and cite nothing. Do not invent document ids."""


def global_prompt(outline: str) -> str:
    return f"{GLOBAL_SYSTEM_PROMPT}\n\nTopics in the library:\n{outline}"
```

and change `system_prompt_for` to take a base (body otherwise unchanged; `prompt = SYSTEM_PROMPT` becomes `prompt = base`):

```python
def system_prompt_for(lang: str | None, extra: str = "", base: str = SYSTEM_PROMPT) -> str:
```

`retrieve/hybrid.py`: rename `_to_chunk` to `to_chunk` (definition and its one call site).

- [ ] **Step 5: Write `core/ragcore/src/ragcore/brain/answer.py`**

```python
"""Context for a question about the whole library: a few passages from every topic.

Topics are ranked by how close their centroid sits to the question; each gives its first
allowed representative passage plus its best dense hits. Passages interleave across topics
before packing, so a tight budget trims every topic a little instead of dropping the last
ones whole.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from itertools import zip_longest

import httpx
import numpy as np

from ragcore.api.schemas import QueryFilters, RetrievedChunk
from ragcore.retrieve.hybrid import to_chunk
from ragcore.retrieve.pack import pack
from ragcore.stub.retrieval import allowed

MAX_TOPICS = 8
PER_TOPIC = 3  # 1 representative + 2 dense hits


@dataclass(frozen=True, slots=True)
class GlobalContext:
    chunks: list[RetrievedChunk]
    outline: str
    scope_label: str | None
    n_topics: int
    candidates: int


@dataclass(frozen=True, slots=True)
class _Candidate:
    label: str
    centroid: np.ndarray
    doc_ids: frozenset[str]
    rep_chunk_ids: list[str]
    sublabels: list[str]


def _candidates(brain, docs: set[str], topic_id: str | None):
    """The scope's children; a leaf topic or no tree at all: one pseudo-topic per card."""
    nodes = brain.nodes
    scope = nodes.get(topic_id) if topic_id else None
    children = [n for n in nodes.values() if n.parent_id == (scope.id if scope else None)]
    if children:
        return scope, [
            _Candidate(
                n.label,
                n.centroid,
                frozenset(n.doc_ids) & docs,
                n.rep_chunk_ids,
                [c.label for c in nodes.values() if c.parent_id == n.id],
            )
            for n in children
        ]
    pool = docs & set(scope.doc_ids) if scope else docs
    documents = brain.store.documents
    return scope, [
        _Candidate(
            documents[d].title if d in documents else d,
            card.vector,
            frozenset([d]),
            card.summary_chunk_ids[:1],
            [],
        )
        for d, card in brain.store.meta.load_cards(sorted(pool)).items()
    ]


async def for_query(
    brain,
    vector: list[float] | None,
    *,
    search: str,
    question: str,
    filters: QueryFilters,
    doc_meta: dict[str, dict],
    topic_id: str | None,
    budget: int,
) -> GlobalContext | None:
    """None when there is nothing to stand on: no cards, no allowed documents, a blocked index."""
    store = brain.store
    if store.index_blocked:
        return None
    docs = {d for d in doc_meta if allowed(d, filters, doc_meta)}
    if not docs:
        return None
    scope, candidates = _candidates(brain, docs, topic_id)
    candidates = [c for c in candidates if c.doc_ids]
    if not candidates:
        return None
    if vector is not None:
        q = np.asarray(vector, dtype=np.float32)
        candidates.sort(key=lambda c: -float(c.centroid @ q))
    else:  # the embedder is down: the biggest topics first
        candidates.sort(key=lambda c: -len(c.doc_ids))
    kept = candidates[:MAX_TOPICS]

    # A follow-up searches with the previous question too, as in local mode.
    dense_vector = vector
    if vector is not None and search != question:
        with contextlib.suppress(httpx.HTTPError):  # the question's own vector still works
            [dense_vector] = await store.embedder.embed([search], query=True)

    rep_doc = {
        row["chunk_id"]: row["doc_id"]
        for row in store.vectors.get([cid for c in kept for cid in c.rep_chunk_ids])
    }
    per_topic: list[list[str]] = []
    for c in kept:
        rep = [cid for cid in c.rep_chunk_ids if rep_doc.get(cid) in c.doc_ids][:1]
        hits = (
            [cid for cid, _ in store.vectors.dense(dense_vector, PER_TOPIC, doc_ids=sorted(c.doc_ids))]
            if dense_vector is not None
            else []
        )
        per_topic.append(list(dict.fromkeys(rep + hits))[:PER_TOPIC])

    order = [cid for round_ in zip_longest(*per_topic) for cid in round_ if cid is not None]
    rows = {row["chunk_id"]: row for row in store.vectors.get(order)}
    ranked = [cid for cid in order if cid in rows]
    # Scores fall with interleave position, so pack()'s score order is the round-robin order.
    chunks = [
        to_chunk(rows[cid], dense_rank=None, bm25_rank=None, score=round(1 - i / len(ranked), 4))
        for i, cid in enumerate(ranked)
    ]
    outline = "\n".join(
        f"- {c.label} ({len(c.doc_ids)} documents)"
        + (f"; subtopics: {', '.join(c.sublabels)}" if c.sublabels else "")
        for c in kept
    )
    return GlobalContext(
        chunks=pack(chunks, budget),
        outline=outline,
        scope_label=scope.label if scope else None,
        n_topics=len(kept),
        candidates=len(chunks),
    )
```

Note: `pack()` with `budget` keeps at least one chunk even if it alone exceeds the budget (existing behaviour).

- [ ] **Step 6: `for_query` on both brains, port**

`brain/topics.py`: `from ragcore.brain import answer` and

```python
    async def for_query(self, vector: list[float] | None, **kwargs) -> answer.GlobalContext | None:
        return await answer.for_query(self, vector, **kwargs)
```

`brain/null.py`:

```python
    async def for_query(self, vector, **kwargs) -> None:
        return None
```

`ports.py` `BrainPort` (types under `TYPE_CHECKING`, `QueryFilters` already imported):

```python
    async def for_query(
        self,
        vector: list[float] | None,
        *,
        search: str,
        question: str,
        filters: QueryFilters,
        doc_meta: dict[str, dict],
        topic_id: str | None,
        budget: int,
    ) -> GlobalContext | None: ...
```

- [ ] **Step 7: `retrieve()` and the 404 in `api/routes/query.py`**

Imports: `from dataclasses import dataclass`, `from typing import Literal`, `from ragcore.api.errors import api_error`, add `QueryMode, RetrievalSettings, RetrievedChunk, StageLatency` to the schemas import, `from ragcore.llm import LlmError, global_prompt, system_prompt_for`. Add above the route:

```python
@dataclass(slots=True)
class Retrieval:
    mode: Literal["local", "global"]
    reason: str
    scope: str | None
    chunks: list[RetrievedChunk]
    latency: StageLatency
    candidates: int
    system_prompt: str


async def retrieve(
    store,
    question: str,
    *,
    mode: QueryMode,
    topic_id: str | None,
    filters: QueryFilters,
    history: list[tuple[str, str]],
    settings: RetrievalSettings,
    lang: str | None,
) -> Retrieval:
    """Route, then gather passages: a few per topic when global, by search when local."""
    routed = await store.brain.route(question)
    chosen, reason = routed.mode, routed.reason
    if topic_id is not None:
        chosen, reason = "global", "Asked about one topic"
    elif mode != "auto":
        chosen, reason = mode, "Set manually"
    extra = store.settings.chat_extra_instructions
    search = retrieval_query(question, history)
    if chosen == "global":
        started = time.perf_counter()
        context = await store.brain.for_query(
            routed.vector,
            search=search,
            question=question,
            filters=filters,
            doc_meta=store.doc_meta(),
            topic_id=topic_id,
            budget=settings.global_token_budget,
        )
        if context is not None:
            return Retrieval(
                "global",
                reason,
                context.scope_label,
                context.chunks,
                StageLatency(dense_ms=(time.perf_counter() - started) * 1000),
                context.candidates,
                system_prompt_for(lang, extra, base=global_prompt(context.outline)),
            )
        chosen, reason = "local", "No topic map yet"
    chunks, latency, candidates = await store.retriever.asearch(
        search, settings=settings, filters=filters, doc_meta=store.doc_meta()
    )
    return Retrieval(chosen, reason, None, chunks, latency, candidates, system_prompt_for(lang, extra))
```

In `query()`, first statements:

```python
    if payload.topic_id is not None and store.brain.topic(payload.topic_id) is None:
        raise api_error(404, "topic_not_found", "topic not found")
```

In `events()`, replace everything from the router lines (Task 4) through the `answerer.stream(...)` call with:

```python
        filters = project_filters(store, session_id, payload.filters)
        history = chat_history(
            store.messages.get(session_id, []), store.settings.retrieval.history_token_budget
        )
        found = await retrieve(
            store,
            question,
            mode=payload.mode,
            topic_id=payload.topic_id,
            filters=filters,
            history=history,
            settings=store.settings.retrieval,
            lang=payload.lang,
        )
        chunks, latency = found.chunks, found.latency
        yield frame("mode", ModeEvent(mode=found.mode, reason=found.reason, scope=found.scope))
        yield frame(
            "sources",
            SourcesEvent(chunks=chunks, candidates=found.candidates, kept=len(chunks)),
        )

        store.append_message(...)  # unchanged user-turn block

        connection = store.active_connection()
        stream = answerer.stream(
            question, chunks, directives, system_prompt=found.system_prompt, history=history
        )
```

and in the assistant `ChatMessage` use `mode=found.mode`. `extract_citations(text, chunks)` stays as is (the packed chunks).

- [ ] **Step 8: CLI uses the same path**

`cli.py` `_index`: after `documents = await store.ingest_source_async(source.id)` add `await store.brain.settle()  # ask finds the tree the ingest scheduled`.

`_ask`: replace the `store.retriever.asearch(...)` + `answer = ...` lines with

```python
        from ragcore.api.routes.query import retrieve

        found = await retrieve(
            store,
            args.question,
            mode="auto",
            topic_id=None,
            filters=QueryFilters(),
            history=[],
            settings=settings,
            lang=None,
        )
        chunks = found.chunks
        print(f"[{found.mode}] {found.reason}", file=sys.stderr)
        answer = "".join(
            [
                p
                async for p in backend.answerer.stream(
                    args.question, chunks, set(), system_prompt=found.system_prompt
                )
            ]
        )
```

(`extract_citations(answer, chunks)` below is unchanged.)

- [ ] **Step 9: Run the tests**

Run: `uv run pytest core/ragcore/tests -v && uv run ruff check .`
Expected: all PASS, including `test_cli.py` (`ask` prints the cited passage on either path).

- [ ] **Step 10: Install and commit**

```bash
scripts/install-local.sh
git add core/ragcore/src/ragcore core/ragcore/tests
git commit -m "feat(brain): topic-packed global answers in /query and ragcore ask"
```

---

### Task 6: `/brain` routes, generated types, ipc

**Files:**
- Create: `core/ragcore/src/ragcore/api/routes/brain.py`, `core/ragcore/tests/test_brain_api.py`
- Modify: `core/ragcore/src/ragcore/api/app.py`, `apps/desktop/src/lib/api-types.ts` (generated), `apps/desktop/src/lib/ipc.ts`, `apps/desktop/src/lib/ipc.test.ts`, `apps/desktop/src/lib/queries.ts`, `apps/desktop/src/lib/jobs-context.tsx`

**Interfaces:**
- Consumes: `store.brain.tree()`, `store.brain.recluster()`, `JobManager.create/start` (`stub/jobs.py`).
- Produces: `GET /brain/topics -> TopicTree`, `POST /brain/recluster -> Job` (409 `index_blocked`); TS `Topic`, `TopicTree`, `api.getTopics()`, `api.recluster()`, `keys.topics`, `topicsQuery`.

- [ ] **Step 1: Write the failing API tests** — `core/ragcore/tests/test_brain_api.py`

```python
"""The topic map over HTTP."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

real = pytest.mark.parametrize("client", ["real"], indirect=True)


def test_the_stub_has_no_topic_map(client: TestClient) -> None:
    assert client.get("/brain/topics").json() == {
        "status": "none",
        "built_at": None,
        "changed_since": 0,
        "topics": [],
    }


@real
def test_topics_come_flat_with_parents_first(client: TestClient, wait_for_topics) -> None:
    wait_for_topics(client)

    tree = client.get("/brain/topics").json()

    seen: set[str] = set()
    for topic in tree["topics"]:
        assert topic["parent_id"] is None or topic["parent_id"] in seen
        seen.add(topic["id"])
    docs = {d["id"] for d in client.get("/documents").json()["items"]}
    top = [t for t in tree["topics"] if t["level"] == 0]
    assert sorted(d for t in top for d in t["doc_ids"]) == sorted(docs)
    assert client.get("/health").json()["index"]["topics"] == len(tree["topics"])


@real
def test_recluster_returns_a_job_that_finishes(client: TestClient, wait_for_topics) -> None:
    wait_for_topics(client)

    job = client.post("/brain/recluster").json()

    assert job["kind"] == "recluster"
    deadline = time.monotonic() + 15
    while (state := client.get(f"/jobs/{job['id']}").json()["state"]) in ("queued", "running"):
        assert time.monotonic() < deadline
        time.sleep(0.02)
    assert state == "done"


def test_recluster_is_refused_while_the_index_is_blocked(client: TestClient) -> None:
    client.app.state.store.index_blocked = "built with another embedder"

    response = client.post("/brain/recluster")

    assert response.status_code == 409
    assert response.json()["code"] == "index_blocked"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest core/ragcore/tests/test_brain_api.py -v`
Expected: FAIL with 404 on `/brain/topics`.

- [ ] **Step 3: Write `core/ragcore/src/ragcore/api/routes/brain.py`**

```python
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
```

`api/app.py`: add `brain` to the routes import and `brain.router,` to the router tuple.

- [ ] **Step 4: Run the API tests**

Run: `uv run pytest core/ragcore/tests -v && uv run ruff check .`
Expected: all PASS.

- [ ] **Step 5: Regenerate the TS types**

```bash
uv run --directory core/ragcore ragcore serve --port 8765 --token dev &
SERVE=$!; sleep 4
(cd apps/desktop && bun run gen:types)
kill $SERVE
git diff --stat apps/desktop/src/lib/api-types.ts   # Topic, TopicTree, topic_id, global_token_budget
```

- [ ] **Step 6: Write the failing ipc test** — in `apps/desktop/src/lib/ipc.test.ts`, inside `describe("request building", …)`:

```ts
  it("reads the topic map and asks for a recluster", async () => {
    await api.getTopics();
    await api.recluster();

    expect(invoke).toHaveBeenNthCalledWith(1, "api_request", {
      req: { method: "GET", path: "/brain/topics", body: null },
    });
    expect(invoke).toHaveBeenNthCalledWith(2, "api_request", {
      req: { method: "POST", path: "/brain/recluster", body: null },
    });
  });
```

Run: `cd apps/desktop && bunx vitest run src/lib/ipc.test.ts` → FAIL (`api.getTopics is not a function`).

- [ ] **Step 7: ipc, queries, jobs**

`lib/ipc.ts`: types after `ConversionDoneEvent`:

```ts
export type Topic = Schemas["Topic"];
export type TopicTree = Schemas["TopicTree"];
```

in `api`, after `evalSets`:

```ts
  getTopics: () => request<TopicTree>("GET", "/brain/topics"),
  recluster: () => request<Job>("POST", "/brain/recluster"),
```

`lib/queries.ts`: `topics: ["topics"] as const,` in `keys`, and

```ts
export const topicsQuery = queryOptions({
  queryKey: keys.topics,
  queryFn: api.getTopics,
  refetchInterval: (query) => (query.state.data?.status === "building" ? 1500 : false),
});
```

`lib/jobs-context.tsx`: in the `else` branch of the settled-job invalidation add
`void queryClient.invalidateQueries({ queryKey: keys.topics });` (ingest and recluster jobs both change the map).

- [ ] **Step 8: Run frontend checks**

Run: `cd apps/desktop && bun run test && bun run typecheck`
Expected: PASS.

- [ ] **Step 9: Install and commit**

```bash
scripts/install-local.sh
git add core/ragcore/src/ragcore/api core/ragcore/tests/test_brain_api.py apps/desktop/src/lib
git commit -m "feat(brain): /brain/topics and /brain/recluster, typed in the UI"
```

---

### Task 7: Topics screen, chat topic chip, i18n

**Files:**
- Create: `apps/desktop/src/features/topics/tree.ts`, `apps/desktop/src/features/topics/tree.test.ts`, `apps/desktop/src/features/topics/topics-view.tsx`, `apps/desktop/src/routes/_shell.topics.tsx`, `apps/desktop/src/features/chat/use-chat.test.ts`
- Modify: `apps/desktop/src/components/shell/icon-rail.tsx`, `apps/desktop/src/routes/_shell.chat.tsx`, `apps/desktop/src/features/chat/chat-view.tsx`, `apps/desktop/src/features/chat/use-chat.ts`, `apps/desktop/src/features/chat/message.tsx`, `apps/desktop/src/locales/{en,it,fr,de,es}.json`, `apps/desktop/src/routeTree.gen.ts` (generated)

**Interfaces:**
- Consumes: `api.getTopics`, `api.recluster`, `topicsQuery`, `keys.topics`, `Topic` (Task 6); `useJobs()`, `healthQuery`, `runtimeQuery`, `relativeTime`, `errorText`.
- Produces: `nestTopics(topics) -> { roots: TopicNode[]; byId: Map<string, TopicNode> }`; route `/topics`; chat search param `?topic=<id>` on `/_shell/chat` (inherited by `/chat/` and `/chat/$sessionId`); `useChat().send(q, { mode, filters, topicId })`; `PendingAnswer.modeScope`.

- [ ] **Step 1: Write the failing tests**

`apps/desktop/src/features/topics/tree.test.ts`:

```ts
import { describe, expect, it } from "vitest";

import type { Topic } from "@/lib/ipc";

import { nestTopics } from "./tree";

const topic = (id: string, parent_id: string | null): Topic => ({
  id,
  parent_id,
  level: parent_id ? 1 : 0,
  label: id,
  keywords: [],
  doc_ids: [],
  rep_chunk_ids: [],
});

describe("nestTopics", () => {
  it("hangs children under their parent in server order", () => {
    const { roots } = nestTopics([
      topic("t1", null),
      topic("t1.1", "t1"),
      topic("t2", null),
      topic("t1.2", "t1"),
    ]);

    expect(roots.map((node) => node.id)).toEqual(["t1", "t2"]);
    expect(roots[0].children.map((node) => node.id)).toEqual(["t1.1", "t1.2"]);
    expect(roots[1].children).toEqual([]);
  });

  it("indexes every node by id", () => {
    const { byId } = nestTopics([topic("t1", null), topic("t1.1", "t1")]);

    expect(byId.get("t1.1")?.parent_id).toBe("t1");
    expect(byId.get("t1")?.children).toHaveLength(1);
  });

  it("keeps a node whose parent is missing as a root", () => {
    expect(nestTopics([topic("t3.1", "t3")]).roots.map((node) => node.id)).toEqual(["t3.1"]);
  });
});
```

`apps/desktop/src/features/chat/use-chat.test.ts`:

```ts
// @vitest-environment jsdom
import { act, renderHook } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { createElement, type ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const streamQuery = vi.fn((..._args: unknown[]) => ({
  streamId: "s",
  done: Promise.resolve(),
  cancel: vi.fn(),
}));
// queries.ts reads api.* at import time, so the mock carries empty api/shell objects.
vi.mock("@/lib/ipc", () => ({ streamQuery: (...args: unknown[]) => streamQuery(...args), api: {}, shell: {} }));

const { useChat } = await import("./use-chat");

function wrapper({ children }: { children: ReactNode }) {
  return createElement(QueryClientProvider, { client: new QueryClient() }, children);
}

type Handlers = { onEvent: (event: unknown) => void };

beforeEach(() => streamQuery.mockClear());

describe("useChat", () => {
  it("sends the topic id while one is set", () => {
    const { result } = renderHook(() => useChat(null), { wrapper });

    act(() => result.current.send("What is here?", { topicId: "t2" }));

    expect(streamQuery.mock.calls[0][0]).toMatchObject({ q: "What is here?", topic_id: "t2" });
  });

  it("sends no topic id without one", () => {
    const { result } = renderHook(() => useChat(null), { wrapper });

    act(() => result.current.send("Why?"));

    expect(streamQuery.mock.calls[0][0]).toMatchObject({ topic_id: null });
  });

  it("keeps the topic scope the mode frame reports", () => {
    const { result } = renderHook(() => useChat(null), { wrapper });
    act(() => result.current.send("What is here?", { topicId: "t2" }));
    const handlers = streamQuery.mock.calls[0][1] as Handlers;

    act(() =>
      handlers.onEvent({
        event: "mode",
        data: { mode: "global", reason: "Asked about one topic", scope: "keychain · tokens" },
      }),
    );

    expect(result.current.pending?.modeScope).toBe("keychain · tokens");
  });
});
```

Run: `cd apps/desktop && bunx vitest run src/features/topics/tree.test.ts src/features/chat/use-chat.test.ts`
Expected: FAIL (`Cannot find module './tree'`; `topic_id` missing from the payload).

- [ ] **Step 2: `features/topics/tree.ts`**

```ts
import type { Topic } from "@/lib/ipc";

export type TopicNode = Topic & { children: TopicNode[] };

/** The server sends the tree flat, parents first; the screen wants it nested. */
export function nestTopics(topics: Topic[]): { roots: TopicNode[]; byId: Map<string, TopicNode> } {
  const byId = new Map<string, TopicNode>(
    topics.map((topic) => [topic.id, { ...topic, children: [] }]),
  );
  const roots: TopicNode[] = [];
  for (const node of byId.values()) {
    const parent = node.parent_id ? byId.get(node.parent_id) : undefined;
    (parent ? parent.children : roots).push(node);
  }
  return { roots, byId };
}
```

- [ ] **Step 3: `use-chat.ts` sends `topic_id`, keeps the scope**

- `PendingAnswer`: add `modeScope: string | null;` after `modeReason`; `EMPTY`: add `modeScope: null,`.
- `send` signature: `options: { mode?: "auto" | "local" | "global"; filters?: QueryFilters; topicId?: string | null } = {}`; payload gains `topic_id: options.topicId ?? null,`.
- `case "mode":` returns `{ ...current, mode: frame.data.mode, modeReason: frame.data.reason, modeScope: frame.data.scope }`.

Re-run the two vitest files → PASS.

- [ ] **Step 4: `message.tsx` shows the scope**

```tsx
export function ModeBadge({
  mode,
  reason,
  scope,
}: {
  mode: "local" | "global";
  reason?: string | null;
  scope?: string | null;
}) {
  const { t } = useTranslation();
  const Icon = mode === "global" ? GlobeIcon : LayersIcon;
  return (
    <Badge variant="outline" className="gap-1 font-normal" title={reason ?? undefined}>
      <Icon className="size-3" />
      {mode === "global" ? t("chat.modeGlobal") : t("chat.modeLocal")}
      {scope ? <span className="max-w-48 truncate text-muted-foreground">· {scope}</span> : null}
    </Badge>
  );
}
```

`AssistantMessage`: add prop `modeScope?: string | null` (type and destructuring) and render `<ModeBadge mode={mode} reason={modeReason} scope={modeScope} />`.

- [ ] **Step 5: Chat search param + chip**

`routes/_shell.chat.tsx`:

```tsx
type ChatSearch = { topic?: string };

export const Route = createFileRoute("/_shell/chat")({
  validateSearch: (search: Record<string, unknown>): ChatSearch => ({
    topic: typeof search.topic === "string" ? search.topic : undefined,
  }),
  component: ChatSection,
});
```

`features/chat/chat-view.tsx`:
- imports: `useSearch` from `@tanstack/react-router`; `NetworkIcon, XIcon` from lucide; `topicsQuery` from `@/lib/queries`.
- after `const navigate = useNavigate();`:

```tsx
  const { topic: topicId } = useSearch({ from: "/_shell/chat" });
  const topics = useQuery({ ...topicsQuery, enabled: Boolean(topicId) });
  const topicLabel = topics.data?.topics.find((topic) => topic.id === topicId)?.label ?? topicId;
  const dismissTopic = () =>
    void navigate({ to: ".", search: (previous) => ({ ...previous, topic: undefined }) });
```

- the new-chat effect keeps the chip across the move to the session URL:
  `void navigate({ to: "/chat/$sessionId", params: { sessionId: activeSession }, search: true });`
- `onSend`: `send(text, { mode, filters: …unchanged…, topicId: topicId ?? null });`
- pending `<AssistantMessage …>` gains `modeScope={pending.modeScope}`.
- first child inside `<Composer …>`:

```tsx
          {topicId ? (
            <Badge variant="secondary" className="h-7 gap-1.5 font-normal">
              <NetworkIcon className="size-3" />
              {t("chat.topicChip", { label: topicLabel })}
              <button
                type="button"
                aria-label={t("chat.dismissTopic")}
                onClick={dismissTopic}
                className="-mr-1 grid size-4 place-items-center rounded-sm hover:bg-foreground/10"
              >
                <XIcon className="size-3" />
              </button>
            </Badge>
          ) : null}
```

- [ ] **Step 6: Topics screen** — `features/topics/topics-view.tsx`

```tsx
import { useMemo, useState } from "react";
import { Link } from "@tanstack/react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { ChevronRightIcon, Loader2Icon, MessageSquarePlusIcon, RefreshCwIcon } from "lucide-react";
import { toast } from "sonner";

import { Page, PageBody, PageHeader } from "@/components/shell/page";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { nestTopics, type TopicNode } from "@/features/topics/tree";
import { errorText } from "@/lib/errors";
import { relativeTime } from "@/lib/format";
import { api } from "@/lib/ipc";
import { useJobs } from "@/lib/jobs-context";
import { healthQuery, keys, runtimeQuery, topicsQuery } from "@/lib/queries";
import { cn } from "@/lib/utils";

const MIN_DOCUMENTS = 6;

export function TopicsView() {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const topics = useQuery(topicsQuery);
  const health = useQuery(healthQuery);
  const runtime = useQuery(runtimeQuery);
  const documents = useQuery({
    queryKey: keys.documents({ limit: 500 }),
    queryFn: () => api.listDocuments({ limit: 500 }),
  });
  const { active } = useJobs();
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const { roots, byId } = useMemo(() => nestTopics(topics.data?.topics ?? []), [topics.data]);
  const titles = new Map((documents.data?.items ?? []).map((d) => [d.id, d.title]));
  const selected = (selectedId ? byId.get(selectedId) : undefined) ?? roots[0] ?? null;
  const building = topics.data?.status === "building" || active.some((j) => j.kind === "recluster");

  const recluster = useMutation({
    mutationFn: api.recluster,
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: keys.topics }),
    onError: (error: Error) =>
      toast.error(t("topics.reclusterFailed"), { description: errorText(error) }),
  });

  let empty: React.ReactNode = null;
  if (topics.data && roots.length === 0) {
    if (building) {
      empty = (
        <span className="flex items-center gap-2">
          <Loader2Icon className="size-3.5 animate-spin" />
          {t("topics.building")}
        </span>
      );
    } else if (runtime.data && runtime.data.state !== "ready") {
      empty = (
        <span>
          {t("topics.noEmbedder")}{" "}
          <Link to="/models" className="underline underline-offset-2 hover:text-foreground">
            {t("topics.openModels")}
          </Link>
        </span>
      );
    } else if ((health.data?.index.documents ?? 0) < MIN_DOCUMENTS) {
      empty = t("topics.under6");
    } else {
      empty = t("topics.empty");
    }
  }

  return (
    <Page>
      <PageHeader
        title={t("topics.title")}
        description={
          topics.data?.built_at
            ? `${t("topics.built", { when: relativeTime(topics.data.built_at) })} · ${t(
                "topics.changed",
                { count: topics.data.changed_since },
              )}`
            : undefined
        }
        actions={
          <Button
            size="sm"
            variant="secondary"
            disabled={building || recluster.isPending}
            onClick={() => recluster.mutate()}
          >
            {building ? (
              <Loader2Icon className="size-3.5 animate-spin" />
            ) : (
              <RefreshCwIcon className="size-3.5" />
            )}
            {building ? t("topics.reclustering") : t("topics.recluster")}
          </Button>
        }
      />
      <PageBody>
        {empty ? (
          <p className="px-5 py-10 text-sm text-muted-foreground">{empty}</p>
        ) : (
          <div className="grid min-h-0 grid-cols-[minmax(14rem,18rem)_1fr]">
            <ul className="space-y-0.5 border-r border-border p-2">
              {roots.map((node) => (
                <TopicRow
                  key={node.id}
                  node={node}
                  depth={0}
                  selectedId={selected?.id ?? null}
                  onSelect={setSelectedId}
                />
              ))}
            </ul>
            {selected ? (
              <TopicDetail node={selected} titles={titles} onSelect={setSelectedId} />
            ) : null}
          </div>
        )}
      </PageBody>
    </Page>
  );
}

function TopicRow({
  node,
  depth,
  selectedId,
  onSelect,
}: {
  node: TopicNode;
  depth: number;
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);
  return (
    <li>
      <div
        className={cn(
          "flex items-center gap-1 rounded-md pr-2 text-[0.8125rem]",
          node.id === selectedId ? "bg-primary/12 text-primary" : "hover:bg-sidebar-accent",
        )}
        style={{ paddingLeft: 4 + depth * 14 }}
      >
        {node.children.length ? (
          <button
            type="button"
            aria-label={open ? t("topics.collapse") : t("topics.expand")}
            aria-expanded={open}
            onClick={() => setOpen(!open)}
            className="grid size-5 place-items-center text-muted-foreground"
          >
            <ChevronRightIcon className={cn("size-3.5 transition-transform", open && "rotate-90")} />
          </button>
        ) : (
          <span className="size-5" />
        )}
        <button
          type="button"
          onClick={() => onSelect(node.id)}
          className="min-w-0 flex-1 truncate py-1.5 text-left"
        >
          {node.label}
        </button>
        <span className="text-[0.6875rem] tabular-nums text-muted-foreground">
          {node.doc_ids.length}
        </span>
      </div>
      {open && node.children.length ? (
        <ul>
          {node.children.map((child) => (
            <TopicRow
              key={child.id}
              node={child}
              depth={depth + 1}
              selectedId={selectedId}
              onSelect={onSelect}
            />
          ))}
        </ul>
      ) : null}
    </li>
  );
}

function TopicDetail({
  node,
  titles,
  onSelect,
}: {
  node: TopicNode;
  titles: Map<string, string>;
  onSelect: (id: string) => void;
}) {
  const { t } = useTranslation();
  return (
    <div className="min-w-0 space-y-5 px-6 py-5">
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <h2 className="truncate text-lg font-semibold tracking-tight">{node.label}</h2>
          <p className="text-xs text-muted-foreground">
            {t("topics.documents", { count: node.doc_ids.length })}
          </p>
        </div>
        <Button asChild size="sm">
          <Link to="/chat" search={{ topic: node.id }}>
            <MessageSquarePlusIcon className="size-3.5" />
            {t("topics.ask")}
          </Link>
        </Button>
      </div>

      <section className="space-y-1.5">
        <h3 className="text-xs font-medium text-muted-foreground">{t("topics.keywords")}</h3>
        <div className="flex flex-wrap gap-1.5">
          {node.keywords.map((keyword) => (
            <Badge key={keyword} variant="secondary" className="font-normal">
              {keyword}
            </Badge>
          ))}
        </div>
      </section>

      {node.children.length ? (
        <section className="space-y-1.5">
          <h3 className="text-xs font-medium text-muted-foreground">{t("topics.subtopics")}</h3>
          <div className="flex flex-wrap gap-1.5">
            {node.children.map((child) => (
              <button
                key={child.id}
                type="button"
                onClick={() => onSelect(child.id)}
                className="rounded-md border border-border px-2 py-1 text-xs hover:border-muted-foreground/40"
              >
                {child.label} · {child.doc_ids.length}
              </button>
            ))}
          </div>
        </section>
      ) : null}

      <section className="space-y-1.5">
        <h3 className="text-xs font-medium text-muted-foreground">{t("topics.docs")}</h3>
        <ul className="space-y-0.5">
          {node.doc_ids.map((docId) => (
            <li key={docId}>
              <Link
                to="/reader/$docId"
                params={{ docId }}
                className="block truncate rounded-md px-2 py-1 text-[0.8125rem] hover:bg-sidebar-accent"
              >
                {titles.get(docId) ?? docId}
              </Link>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}
```

`PageHeader` renders `description` under the title (`components/shell/page.tsx:39-40`).

- [ ] **Step 7: Route + rail**

`routes/_shell.topics.tsx`:

```tsx
import { createFileRoute } from "@tanstack/react-router";

import { TopicsView } from "@/features/topics/topics-view";

export const Route = createFileRoute("/_shell/topics")({
  component: TopicsView,
});
```

`components/shell/icon-rail.tsx`: import `NetworkIcon`; insert into `PRIMARY` between Library and Convert:
`{ to: "/topics", labelKey: "nav.topics", icon: NetworkIcon, match: ["/topics"] },`

Regenerate the route tree (the Vite plugin writes it): `cd apps/desktop && bunx vite build`.

- [ ] **Step 8: Strings in every catalog**

Add to each `locales/*.json`: `nav.topics`; a top-level `topics` object; `chat.topicChip`, `chat.dismissTopic`; `errors.topic_not_found`, `errors.index_blocked`.

| key | en | it | fr | de | es |
|---|---|---|---|---|---|
| nav.topics | Topics | Argomenti | Sujets | Themen | Temas |
| topics.title | Topics | Argomenti | Sujets | Themen | Temas |
| topics.built | Built {{when}} | Creata {{when}} | Créée {{when}} | Erstellt {{when}} | Creado {{when}} |
| topics.changed_one | {{count}} document changed since | {{count}} documento modificato da allora | {{count}} document modifié depuis | {{count}} Dokument seitdem geändert | {{count}} documento cambiado desde entonces |
| topics.changed_other | {{count}} documents changed since | {{count}} documenti modificati da allora | {{count}} documents modifiés depuis | {{count}} Dokumente seitdem geändert | {{count}} documentos cambiados desde entonces |
| topics.recluster | Re-cluster | Ricalcola | Recalculer | Neu gruppieren | Reagrupar |
| topics.reclustering | Re-clustering… | Ricalcolo… | Recalcul… | Wird neu gruppiert… | Reagrupando… |
| topics.reclusterFailed | Could not re-cluster | Impossibile ricalcolare gli argomenti | Impossible de recalculer les sujets | Themen konnten nicht neu gruppiert werden | No se pudieron reagrupar los temas |
| topics.documents_one | {{count}} document | {{count}} documento | {{count}} document | {{count}} Dokument | {{count}} documento |
| topics.documents_other | {{count}} documents | {{count}} documenti | {{count}} documents | {{count}} Dokumente | {{count}} documentos |
| topics.keywords | Keywords | Parole chiave | Mots-clés | Schlüsselwörter | Palabras clave |
| topics.subtopics | Subtopics | Sottoargomenti | Sous-sujets | Unterthemen | Subtemas |
| topics.docs | Documents | Documenti | Documents | Dokumente | Documentos |
| topics.ask | Ask about this topic | Chiedi su questo argomento | Poser une question sur ce sujet | Zu diesem Thema fragen | Preguntar sobre este tema |
| topics.expand | Expand | Espandi | Déplier | Aufklappen | Expandir |
| topics.collapse | Collapse | Comprimi | Replier | Zuklappen | Contraer |
| topics.under6 | Topics appear once your library has 6+ documents. | Gli argomenti compaiono quando la libreria ha almeno 6 documenti. | Les sujets apparaissent dès que la bibliothèque compte au moins 6 documents. | Themen erscheinen, sobald die Bibliothek mindestens 6 Dokumente enthält. | Los temas aparecen cuando la biblioteca tiene al menos 6 documentos. |
| topics.building | Building the topic map… | Creazione della mappa degli argomenti… | Création de la carte des sujets… | Themenkarte wird erstellt… | Creando el mapa de temas… |
| topics.noEmbedder | Topics need the embedding model running. | Gli argomenti richiedono il modello di embedding attivo. | Les sujets nécessitent le modèle d'embedding en marche. | Themen brauchen das laufende Embedding-Modell. | Los temas necesitan el modelo de embedding en marcha. |
| topics.openModels | Open Models | Apri Modelli | Ouvrir Modèles | Modelle öffnen | Abrir Modelos |
| topics.empty | No topic map yet. Re-cluster to build one. | Nessuna mappa degli argomenti. Ricalcola per crearne una. | Pas encore de carte des sujets. Recalculez pour en créer une. | Noch keine Themenkarte. Neu gruppieren, um eine zu erstellen. | Aún no hay mapa de temas. Reagrupa para crear uno. |
| chat.topicChip | Topic: {{label}} | Argomento: {{label}} | Sujet : {{label}} | Thema: {{label}} | Tema: {{label}} |
| chat.dismissTopic | Stop asking about this topic | Smetti di chiedere su questo argomento | Ne plus cibler ce sujet | Nicht mehr zu diesem Thema fragen | Dejar de preguntar sobre este tema |
| errors.topic_not_found | Topic not found. It may have changed after a re-cluster. | Argomento non trovato. Potrebbe essere cambiato dopo un ricalcolo. | Sujet introuvable. Il a peut-être changé après un recalcul. | Thema nicht gefunden. Es hat sich nach einer Neugruppierung vielleicht geändert. | Tema no encontrado. Puede haber cambiado tras reagrupar. |
| errors.index_blocked | The index was built with another embedding model. Re-index to continue. | L'indice è stato creato con un altro modello di embedding. Reindicizza per continuare. | L'index a été créé avec un autre modèle d'embedding. Réindexez pour continuer. | Der Index wurde mit einem anderen Embedding-Modell erstellt. Neu indexieren, um fortzufahren. | El índice se creó con otro modelo de embedding. Reindexa para continuar. |

- [ ] **Step 9: Run frontend checks**

Run: `cd apps/desktop && bun run test && bun run typecheck && bun run build`
Expected: PASS (locale parity, keys-exist and no-hardcoded-text included).

- [ ] **Step 10: Install, look at it, commit**

```bash
scripts/install-local.sh
```

Open the installed app: Topics rail item between Library and Convert; tree expands; selecting a topic shows keywords, subtopics, documents (click opens the Reader); Re-cluster shows progress; **Ask about this topic** opens a new chat with the chip; the first answer is Global with the topic label beside the badge; the chip survives the move to `/chat/<id>` and the × removes it.

```bash
git add apps/desktop/src
git commit -m "feat(topics): topics screen and ask-about-this-topic chat chip"
```

---

### Task 8: Eval corpus, golden set, metrics, CI gate

**Files:**
- Create: `fixtures/brain/*.md` (40), `eval/golden/brain.jsonl`, `eval/golden/brain.baseline.json` (generated)
- Modify: `core/ragcore/src/ragcore/eval.py`, `core/ragcore/src/ragcore/api/schemas.py` (`EvalQuestionResult`, `EvalMetrics`), `core/ragcore/src/ragcore/cli.py` (`_eval`), `.github/workflows/eval.yml`, `CLAUDE.md`
- Test: `core/ragcore/tests/test_eval_harness.py`

**Interfaces:**
- Consumes: `Brain.route`, `Brain.for_query`, `Brain.settle` (Tasks 3–5).
- Produces: `eval.coverage(chunks, paths, expected) -> float`, `score_questions(..., brain=None)`, `summarize` fills `global_coverage` / `router_accuracy`, `eval.baseline_metrics(path) -> dict | None`, `eval.gate_failures(metrics, baseline: dict) -> list[str]`, `eval.MAX_BRAIN_DROP = 0.04`.

- [ ] **Step 1: Write the failing tests** — append to `core/ragcore/tests/test_eval_harness.py` (add imports `from fnmatch import fnmatch`, `coverage`, `gate_failures` from `ragcore.eval`)

```python
def test_coverage_is_the_share_of_expected_patterns_present() -> None:
    chunks = [chunk("doc_a", ""), chunk("doc_b", "")]
    paths = {"doc_a": "/x/garden-tomatoes.md", "doc_b": "/x/astro-moon.md"}

    assert coverage(chunks, paths, ["garden-*.md", "astro-*.md", "cook-*.md", "rag-*.md"]) == 0.5
    assert coverage([], paths, ["garden-*.md"]) == 0.0


def test_global_rows_report_coverage_and_routing_not_recall() -> None:
    results = [
        EvalQuestionResult(q="a", type="local", hit=True, rank=1, ndcg=1.0, routed="local"),
        EvalQuestionResult(q="b", type="global", hit=True, coverage=0.5, routed="global"),
        EvalQuestionResult(q="c", type="global", hit=True, coverage=1.0, routed="local"),
    ]

    metrics = summarize("t", results)

    assert metrics.recall_at_6 == 1.0  # the local row only
    assert metrics.global_coverage == 0.75
    assert metrics.router_accuracy == round(2 / 3, 4)


def test_the_gate_holds_each_metric_to_its_baseline() -> None:
    metrics = summarize(
        "t", [EvalQuestionResult(q="b", type="global", hit=True, coverage=0.7, routed="global")]
    )

    assert gate_failures(metrics, {"recall_at_6": 0.0, "global_coverage": 0.72, "router_accuracy": 1.0}) == []
    failed = gate_failures(metrics, {"recall_at_6": 0.0, "global_coverage": 0.8, "router_accuracy": 1.0})
    assert len(failed) == 1 and failed[0].startswith("global coverage")
    assert gate_failures(metrics, {"recall_at_6": 0.0}) == [
        "global coverage: no baseline value",
        "router accuracy: no baseline value",
    ]


def test_the_brain_set_points_at_files_that_exist() -> None:
    names = [path.name for path in BRAIN_DOCS.glob("*.md")]
    questions = load_golden(GOLDEN_DIR / "brain.jsonl")

    assert len(names) == 40
    assert sum(q.type == "global" for q in questions) == 15
    assert sum(q.type == "local" for q in questions) == 15
    for question in questions:
        for expected in question.expected:
            assert any(fnmatch(n, expected.partition("#")[0]) for n in names), expected
```

using the file's existing `chunk(doc_id, section_path)` helper (`test_eval_harness.py:16`), plus a constant near the top:

```python
BRAIN_DOCS = Path(__file__).resolve().parents[3] / "fixtures" / "brain"
```

Run: `uv run pytest core/ragcore/tests/test_eval_harness.py -v` → FAIL (`ImportError: coverage`).

- [ ] **Step 2: Schemas** — `EvalQuestionResult` gains

```python
    coverage: float | None = None  # global rows: share of expected docs in the packed context
    routed: Literal["local", "global"] | None = None
    route_reason: str | None = None
```

and `EvalMetrics` gains `global_coverage: float | None = None` and `router_accuracy: float | None = None`.

- [ ] **Step 3: `eval.py`**

- module docstring: append "Expected file names are ``fnmatch`` patterns (``garden-*.md``)."
- imports: `import time`, `from fnmatch import fnmatch`.
- constant: `MAX_BRAIN_DROP = 0.04  # one router question of 30; 2 points is under one question`.
- `matches`: `if path is None or not fnmatch(Path(path).name, name):`.
- new functions and the extended scorer:

```python
def coverage(chunks: list[RetrievedChunk], paths: dict[str, str], expected: Iterable[str]) -> float:
    """Share of expected entries that at least one packed passage matches."""
    expected = tuple(expected)
    if not expected:
        return 0.0
    found = sum(any(matches(c, paths.get(c.doc_id), e) for c in chunks) for e in expected)
    return found / len(expected)


async def score_questions(
    retriever,
    questions: list[GoldenQuestion],
    *,
    settings: RetrievalSettings,
    doc_meta: dict[str, dict],
    paths: dict[str, str],
    brain=None,
) -> AsyncIterator[EvalQuestionResult]:
    """With a brain, every row is routed too and global rows score the packed global context."""
    for question in questions:
        routed = await brain.route(question.q) if brain is not None else None
        route = {"routed": routed.mode, "route_reason": routed.reason} if routed else {}
        if routed is not None and question.type == "global":
            started = time.perf_counter()
            context = await brain.for_query(
                routed.vector,
                search=question.q,
                question=question.q,
                filters=QueryFilters(),
                doc_meta=doc_meta,
                topic_id=None,
                budget=settings.global_token_budget,
            )
            share = coverage(context.chunks if context else [], paths, question.expected)
            yield EvalQuestionResult(
                q=question.q,
                type="global",
                hit=share > 0,
                coverage=round(share, 4),
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
                **route,
            )
            continue
        chunks, latency, _ = await retriever.asearch(
            question.q, settings=settings, filters=QueryFilters(), doc_meta=doc_meta
        )
        rank = first_hit(chunks, paths, question.expected)
        yield EvalQuestionResult(
            q=question.q,
            type=question.type,
            hit=rank is not None,
            rank=rank,
            # One relevant target per question, so the ideal DCG is 1.
            ndcg=1 / math.log2(rank + 1) if rank else 0.0,
            latency_ms=round(latency.total_ms, 2),
            **route,
        )
```

- `summarize`: recall, MRR and nDCG over `local = [r for r in results if r.coverage is None]` (`n = max(len(local), 1)`; base sets have no global rows, so their numbers are unchanged); add

```python
    covered = [r.coverage for r in results if r.coverage is not None]
    routed = [r for r in results if r.routed is not None]
    ...
        global_coverage=round(sum(covered) / len(covered), 4) if covered else None,
        router_accuracy=(
            round(sum(r.routed == r.type for r in routed) / len(routed), 4) if routed else None
        ),
```

- baseline and gate:

```python
def baseline_metrics(path: Path) -> dict | None:
    """The metrics block of a report ``ragcore eval --report`` wrote; None if there is none."""
    try:
        metrics = json.loads(path.read_text(encoding="utf-8"))["metrics"]
    except (FileNotFoundError, KeyError, TypeError, json.JSONDecodeError):
        return None
    return metrics if isinstance(metrics, dict) else None


def baseline_recall(path: Path) -> float | None:
    metrics = baseline_metrics(path)
    return metrics.get("recall_at_6") if metrics else None


def gate_failures(metrics: EvalMetrics, baseline: dict) -> list[str]:
    """Every measured metric may fall at most its tolerance below the committed baseline."""
    checks = (
        ("recall@6", metrics.recall_at_6, "recall_at_6", MAX_RECALL_DROP),
        ("global coverage", metrics.global_coverage, "global_coverage", MAX_BRAIN_DROP),
        ("router accuracy", metrics.router_accuracy, "router_accuracy", MAX_BRAIN_DROP),
    )
    failures: list[str] = []
    for name, value, key, tolerance in checks:
        if value is None:
            continue
        base = baseline.get(key)
        if base is None:
            failures.append(f"{name}: no baseline value")
        elif value < base - tolerance:
            failures.append(
                f"{name} {value:.2f} is more than {tolerance} below the baseline {base:.2f}"
            )
    return failures
```

- [ ] **Step 4: `cli.py` `_eval`**

- imports from `ragcore.eval`: `baseline_metrics, gate_failures, load_golden, score_questions, summarize` (drop `MAX_RECALL_DROP`, `baseline_recall`).
- after `await store.ingest_source_async(source.id)`:

```python
            use_brain = any(q.type == "global" for q in questions)
            if use_brain:
                await store.brain.settle()  # the ingest scheduled the recluster
```

  and pass `brain=store.brain if use_brain else None` to `score_questions`.
- reporting and gate (replacing everything after the `finally:` block):

```python
    baseline = baseline_metrics(args.set.with_suffix(".baseline.json"))
    metrics = summarize(args.set.stem, results, baseline.get("recall_at_6") if baseline else None)
    for r in results:
        score = f"cov {r.coverage:.2f}" if r.coverage is not None else (
            f"hit {r.rank}" if r.hit else "MISS "
        )
        routed = f" [{r.routed}]" if r.routed else ""
        print(f"{score}{routed}  {r.q}")
        if r.routed and r.routed != r.type:
            print(f"    misrouted as {r.routed}: {r.route_reason}")
    extra = ""
    if metrics.global_coverage is not None:
        extra += f"  coverage {metrics.global_coverage:.2f}"
    if metrics.router_accuracy is not None:
        extra += f"  router {metrics.router_accuracy:.2f}"
    print(
        f"\nrecall@1 {metrics.recall_at_1:.2f}  recall@6 {metrics.recall_at_6:.2f}  "
        f"MRR {metrics.mrr:.2f}  nDCG@6 {metrics.ndcg_at_6:.2f}{extra}  "
        f"p50 {metrics.latency_p50.total_ms:.0f} ms"
    )
    if args.report:
        result = EvalResult(metrics=metrics, questions=results)
        args.report.write_text(result.model_dump_json(indent=2) + "\n")
    if not args.gate:
        return 0
    if baseline is None:
        print(f"error: no baseline at {args.set.with_suffix('.baseline.json')}", file=sys.stderr)
        return 1
    failures = gate_failures(metrics, baseline)
    for failure in failures:
        print(f"FAIL: {failure}", file=sys.stderr)
    if failures:
        return 1
    print("gate: every metric within tolerance of the baseline")
    return 0
```

Run: `uv run pytest core/ragcore/tests/test_eval_harness.py core/ragcore/tests/test_cli.py core/ragcore/tests/test_evals.py -v`
Expected: all PASS except `test_the_brain_set_points_at_files_that_exist` (corpus not written yet).

- [ ] **Step 5: Write the corpus** — `fixtures/brain/`, 40 Markdown files

Each file: `# Title`, two or three `##` sections, 150–250 words of plain, specific prose, in the language noted. Every file must state the facts listed (the local controls ask for them). Mixed English + one Italian file per theme.

| file | lang | must cover |
|---|---|---|
| garden-tomatoes.md | en | staking and pinching side shoots; water at the base; blossom-end rot from uneven watering |
| garden-composting.md | en | carbon-to-nitrogen ratio about 30:1 (browns vs greens); turn the heap every two weeks; ready in 3–6 months |
| garden-watering.md | en | water deeply once or twice a week; morning watering; mulch keeps soil moist |
| garden-pruning-roses.md | en | prune in late winter; cut above an outward bud at 45°; remove dead wood first |
| garden-raised-beds.md | en | 30 cm deep beds; soil mix of topsoil and compost; warms faster in spring |
| garden-pests.md | en | aphids and ladybirds; slug traps; companion planting with marigolds |
| garden-semina-primaverile.md | it | seminare i pomodori in semenzaio a marzo; temperatura 20–25 °C; trapiantare dopo le gelate |
| astro-choosing-a-telescope.md | en | aperture matters more than magnification (light gathering); 200 mm Dobsonian as a first scope |
| astro-moon-phases.md | en | 29.5-day cycle; terminator shows craters best; full moon washes out faint objects |
| astro-light-pollution.md | en | Bortle scale; dark-sky sites; red torches keep night vision |
| astro-planets-visible.md | en | Jupiter's four Galilean moons; Saturn's rings in a small scope; Venus phases |
| astro-meteor-showers.md | en | Perseids peak around 12 August, up to 100 meteors per hour; Geminids in December |
| astro-astrophotography.md | en | 500 rule for exposure; tracking mount; stacking frames |
| astro-costellazioni-invernali.md | it | Orione con Betelgeuse e Rigel; meglio visibile a gennaio; la cintura punta a Sirio |
| money-emergency-fund.md | en | three to six months of expenses; keep it in an easy-access savings account |
| money-index-funds.md | en | low fees; diversification across the market; time in the market beats timing |
| money-budget-50-30-20.md | en | 50% needs, 30% wants, 20% savings and debt repayment |
| money-credit-cards.md | en | pay the full balance monthly; interest compounds; utilisation under 30% |
| money-mortgage-basics.md | en | fixed vs variable rate; deposit size; total interest over the term |
| money-retirement-accounts.md | en | start early; employer match is free money; compound growth |
| money-spese-di-casa.md | it | bollette di luce e gas; un foglio di calcolo mensile per tenerne traccia; spese fisse e variabili |
| cook-sourdough.md | en | 75% hydration; bake at 250 °C covered for 20 minutes, then 230 °C uncovered for 20 minutes |
| cook-knife-skills.md | en | claw grip; keep the knife sharp; rock chop for herbs |
| cook-risotto.md | en | toast the rice; add hot stock a ladle at a time; finish with butter and parmesan |
| cook-stock.md | en | roast bones first; simmer, never boil; skim the foam |
| cook-meal-prep.md | en | cook grains and proteins in batches; airtight containers; three to four days in the fridge |
| cook-spices.md | en | toast whole spices in a dry pan; grind fresh; store away from light |
| cook-pasta-fresca.md | it | 100 g di farina per ogni uovo; impastare 10 minuti; riposo di 30 minuti coperta |
| cycle-tyre-pressure.md | en | 25 mm road tyres around 80 psi for a 75 kg rider; tubeless runs lower; check weekly |
| cycle-chain-care.md | en | replace the chain at 0.5% stretch; clean and lube every 300 km; a worn chain wears the cassette |
| cycle-bike-fit.md | en | saddle height with a slight knee bend; reach; cleat position |
| cycle-commuting.md | en | lights and visibility; panniers; mudguards |
| cycle-climbing.md | en | steady cadence; stay seated; low gears early |
| cycle-allenamento-invernale.md | it | sessioni sui rulli di 60–90 minuti; allenamento in zona 2; abbigliamento a strati fuori |
| rag-chunking.md | en | chunk size vs precision; overlap between windows; keep section headings |
| rag-embeddings.md | en | dense vectors capture meaning; one embedder for queries and documents; re-index when it changes |
| rag-reranking.md | en | a cross-encoder rescores the top 40 candidates and keeps 6; slower but more precise |
| rag-hybrid-search.md | en | BM25 for exact terms; dense for paraphrase; reciprocal rank fusion merges them |
| rag-citations.md | en | cite with document id and page; drop citations no retrieved passage backs |
| rag-valutazione.md | it | recall@6 e MRR; un golden set di domande con i documenti attesi; confronto con una baseline |

- [ ] **Step 6: Write `eval/golden/brain.jsonl`** (30 lines)

```jsonl
{"q": "what are the main themes in my library", "type": "global", "expected": ["garden-*.md", "astro-*.md", "money-*.md", "cook-*.md", "cycle-*.md", "rag-*.md"]}
{"q": "what topics do my notes cover", "type": "global", "expected": ["garden-*.md", "astro-*.md", "money-*.md", "cook-*.md", "cycle-*.md", "rag-*.md"]}
{"q": "give me an overview of everything I have written", "type": "global", "expected": ["garden-*.md", "astro-*.md", "money-*.md", "cook-*.md", "cycle-*.md", "rag-*.md"]}
{"q": "summarize all of my documents in a few lines", "type": "global", "expected": ["garden-*.md", "astro-*.md", "money-*.md", "cook-*.md", "cycle-*.md", "rag-*.md"]}
{"q": "across all my notes, what do I care about most", "type": "global", "expected": ["garden-*.md", "astro-*.md", "money-*.md", "cook-*.md", "cycle-*.md", "rag-*.md"]}
{"q": "how do my notes on cooking and gardening connect", "type": "global", "expected": ["garden-*.md", "cook-*.md"]}
{"q": "compare what I wrote about cycling with what I wrote about money", "type": "global", "expected": ["cycle-*.md", "money-*.md"]}
{"q": "overall, what do my notes say about the night sky", "type": "global", "expected": ["astro-*.md"]}
{"q": "in general, what have I learned about building search systems", "type": "global", "expected": ["rag-*.md"]}
{"q": "quali sono i temi principali della mia libreria", "type": "global", "expected": ["garden-*.md", "astro-*.md", "money-*.md", "cook-*.md", "cycle-*.md", "rag-*.md"]}
{"q": "di quali argomenti parlano i miei appunti", "type": "global", "expected": ["garden-*.md", "astro-*.md", "money-*.md", "cook-*.md", "cycle-*.md", "rag-*.md"]}
{"q": "fammi una panoramica di tutti i miei documenti", "type": "global", "expected": ["garden-*.md", "astro-*.md", "money-*.md", "cook-*.md", "cycle-*.md", "rag-*.md"]}
{"q": "riassumi tutto quello che ho scritto", "type": "global", "expected": ["garden-*.md", "astro-*.md", "money-*.md", "cook-*.md", "cycle-*.md", "rag-*.md"]}
{"q": "che legame c'è tra i miei appunti di cucina e quelli sull'orto", "type": "global", "expected": ["cook-*.md", "garden-*.md"]}
{"q": "in generale cosa dicono le mie note sulle finanze personali", "type": "global", "expected": ["money-*.md"]}
{"q": "what oven temperature does the sourdough recipe use", "type": "local", "expected": ["cook-sourdough.md"]}
{"q": "what tyre pressure for 25 mm road tyres", "type": "local", "expected": ["cycle-tyre-pressure.md"]}
{"q": "what does the 50/30/20 rule split my income into", "type": "local", "expected": ["money-budget-50-30-20.md"]}
{"q": "when do the Perseids peak", "type": "local", "expected": ["astro-meteor-showers.md"]}
{"q": "what carbon to nitrogen ratio should a compost heap have", "type": "local", "expected": ["garden-composting.md"]}
{"q": "how many candidates does the cross-encoder rescore", "type": "local", "expected": ["rag-reranking.md"]}
{"q": "how many months of expenses should an emergency fund cover", "type": "local", "expected": ["money-emergency-fund.md"]}
{"q": "why does aperture matter more than magnification", "type": "local", "expected": ["astro-choosing-a-telescope.md"]}
{"q": "how much chain stretch means it is time to replace it", "type": "local", "expected": ["cycle-chain-care.md"]}
{"q": "quanta farina per ogni uovo nella pasta fresca", "type": "local", "expected": ["cook-pasta-fresca.md"]}
{"q": "quando seminare i pomodori in semenzaio", "type": "local", "expected": ["garden-semina-primaverile.md"]}
{"q": "quanto dura una seduta sui rulli in inverno", "type": "local", "expected": ["cycle-allenamento-invernale.md"]}
{"q": "come tenere traccia delle bollette di casa", "type": "local", "expected": ["money-spese-di-casa.md"]}
{"q": "quali metriche usare per valutare il recupero dei passaggi", "type": "local", "expected": ["rag-valutazione.md"]}
{"q": "quali stelle si vedono nella costellazione di Orione", "type": "local", "expected": ["astro-costellazioni-invernali.md"]}
```

Run: `uv run pytest core/ragcore/tests -v && uv run ruff check .` → all PASS.

- [ ] **Step 7: Measure with the real embedder (CPU, like CI) and tune**

```bash
scripts/fetch-models.sh   # if ~/.ibid/models lacks the embedder
llama-server -m ~/.ibid/models/embeddinggemma-300M-qat-Q4_0.gguf --embedding \
  -c 2048 -b 2048 -ub 2048 -ngl 0 --port 8770 --host 127.0.0.1 &
uv run ragcore eval eval/golden/brain.jsonl --corpus fixtures/brain --rerank-url http://127.0.0.1:1
uv run ragcore eval eval/golden/base.jsonl --corpus fixtures/docs --rerank-url http://127.0.0.1:1 --gate
```

Targets: coverage ≥ 0.70, router ≥ 0.90; base gate still passes. If router accuracy is short, read the `misrouted as …` lines: adjust `MARGIN` (try 0.03 / 0.08) or add example phrasings to `EXAMPLES` that cover the miss without copying golden wording; re-run. If coverage is short, check how many top-level topics the 40 docs produced (`uv run ragcore index fixtures/brain --data-dir "$TMPDIR/brain"` then `sqlite3 "$TMPDIR/brain/app.db" "select id, json from topics where level = 0"`); report rather than tune clustering constants (they are spec values). Stop and tell the user if either target can't be met.

- [ ] **Step 8: Commit the baseline, wire CI**

```bash
uv run ragcore eval eval/golden/brain.jsonl --corpus fixtures/brain \
  --rerank-url http://127.0.0.1:1 --report eval/golden/brain.baseline.json
uv run ragcore eval eval/golden/brain.jsonl --corpus fixtures/brain --rerank-url http://127.0.0.1:1 --gate
```

`.github/workflows/eval.yml`: add `core/ragcore/src/ragcore/brain/**` and `fixtures/brain/**` to both `paths` lists, update the header comment to mention the brain set, and append a step:

```yaml
      - run: >-
          uv run ragcore eval eval/golden/brain.jsonl --corpus fixtures/brain
          --rerank-url http://127.0.0.1:1 --gate
```

`CLAUDE.md`: under Commands add `uv run ragcore eval eval/golden/brain.jsonl --corpus fixtures/brain   # global coverage + router accuracy; --gate vs brain.baseline.json`; under Sidecar architecture add a bullet: "`brain/`: doc cards (`cards.py`, written in `RealStore._write`), numpy topic tree (`cluster.py`), `Brain` per RealStore (`topics.py`: assignment, auto recluster at 15% churn), router (`route.py`), global context (`answer.py`). `/brain/topics`, `/brain/recluster`; `QueryRequest.topic_id` scopes a global answer."

- [ ] **Step 9: Full verification, install, commit**

```bash
uv run pytest -v && uv run ruff check .
(cd apps/desktop && bun run test && bun run typecheck && bun run build && cd src-tauri && cargo test --lib)
uv run python scripts/check-licenses.py
scripts/install-local.sh
git add fixtures/brain eval/golden core/ragcore .github/workflows/eval.yml CLAUDE.md
git commit -m "test(brain): eval corpus, golden set, coverage and router gate"
```

---

## Verification (end to end)

1. `uv run pytest -v` and `uv run ruff check .` — whole Python suite green.
2. `cd apps/desktop && bun run test && bun run typecheck && bun run build`; `cd src-tauri && cargo test --lib`.
3. Eval: both gates pass (`base`, `brain`) against the CPU embedder.
4. Manual, real models (`scripts/dev/serve-models.sh`) and a real connection active:
   `uv run ragcore index fixtures/brain --data-dir /tmp/rc-brain` then
   `uv run ragcore ask "what are the main themes in my library" --data-dir /tmp/rc-brain` → stderr `[global] Global (router …)`, answer spans several themes, every printed citation is a packed passage;
   `uv run ragcore ask "when do the Perseids peak" --data-dir /tmp/rc-brain` → `[local] …`.
5. Installed app (`scripts/install-local.sh`): Topics screen, re-cluster progress, ask-about-topic chip, global badge with router reason tooltip and topic scope; a library under 6 docs shows the empty state; Library → remove a doc → Topics count drops.

## Execution handoff

**Chosen: Native** (superpowers:executing-plans — one session implements every task, then one fresh reviewer on the whole branch). The tasks are a chain over shared interfaces (`Brain` grows in Tasks 3–5 and is consumed by query, routes, eval and UI), so a single context avoids re-deriving those names per task; a fresh per-task subagent would mostly re-read the same five files.
