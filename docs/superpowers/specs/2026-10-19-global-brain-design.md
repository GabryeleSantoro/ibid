# Global brain — design

## Goal
Questions about the library as a whole ("what are the main themes", "overall, what do my notes say about X", "what topics do I have") get an answer built from the whole corpus instead of the six passages closest to the question. Today `global` is only a badge: `route_mode` matches English keywords and both modes run the same retrieval. The library also gets a browsable topic map with "ask about this topic".

Build plan phase 5, adapted: documents (not passages) are clustered, numpy only, one LLM call per global answer.

## Non-goals
- Map-reduce over topic branches. One topic-packed call; map-reduce is the upgrade if eval shows coverage is the limit.
- Layer 5 abstractive summaries (phase 7). Map context is representative chunks.
- Router LLM fallback. Ambiguous scores fall back to keyword hints; the manual mode override already exists.
- scikit-learn / HDBSCAN / UMAP. The frozen sidecar is already 221 MB.
- Doc cards in the reader, editable topic labels, graph visualisations.
- An LLM-judged answer-correctness score in CI.
- Parent/child chunking, the SQLite job queue, the other phase 1–4 gaps (separate sub-projects).

## Architecture

### 1. Doc cards (`brain/cards.py`)
Built in `RealStore._write`, where chunk vectors are already in memory, so nothing is read back:
- `vector`: L2-normalised mean of the doc's chunk vectors.
- `summary_chunk_ids`: the 3 chunks with the highest cosine to that vector.
- `terms`: the doc's 300 most frequent terms with counts. Lowercased word tokens of 3+ characters, stopwords removed using the lists already in `api/routes/suggestions.py` (moved to a shared module so both use one copy). These make corpus TF-IDF and c-TF-IDF cheap without re-reading text.
- `keywords`: top 8 TF-IDF terms against the corpus, filled at cluster time.

Stored in SQLite (`store/meta.py`): `doc_cards(doc_id TEXT PRIMARY KEY, vector BLOB NOT NULL, json TEXT NOT NULL)`, vector as float32 bytes. Written in the same transaction as the documents in `_write`; deleted wherever documents are deleted (`delete_documents`, `remove_source`, `wipe`). A document with no chunks (parse error, empty, scan without text) gets no card.

Backfill: on startup, documents with status `indexed` and no card get one built from their LanceDB chunk rows (vector + text), as a background task after the store is up. No re-index is needed for existing libraries.

### 2. Topic tree (`brain/cluster.py`, `brain/topics.py`)
`cluster.py` is pure numpy, no I/O:
- PCA to `min(50, n - 1, dim)` dimensions via SVD of the centred matrix, rows re-normalised.
- Spherical k-means (cosine), k-means++ init, fixed seed, 50 iterations max.
- k chosen per level from 2..`min(20, n // 3)` by mean silhouette on a sample of at most 2,000 docs (fixed seed).
- Recurse into clusters with more than 30 docs, at most 3 levels.
- c-TF-IDF: each cluster is one "document" made of its docs' summed term counts; `weight(t, c) = tf(t, c) × log(1 + A / f(t))`, where `A` is the average total term count per cluster and `f(t)` the count of `t` across all clusters. Top 5 terms are the cluster's keywords, top 3 joined with ` · ` its label.
- Representative chunks: the first summary chunk of each of the 3 docs closest to the cluster centroid.

`topics.py` owns persistence, scheduling and assignment (`Brain` class, one per `RealStore`):
- Table `topics(id TEXT PRIMARY KEY, parent_id TEXT, level INTEGER, centroid BLOB NOT NULL, json TEXT NOT NULL)`; json holds `label`, `keywords`, `doc_ids`, `rep_chunk_ids`. A recluster replaces the whole table in one transaction. Plus `brain_state` in the existing settings table: `built_at`, `changed_since`.
- Corpus under 6 cards: no tree (`status: "none"`); the global path uses cards directly.
- Recluster runs in `asyncio.to_thread`; at most one at a time (lock). Status `building` while it runs.
- Incremental: at the end of `RealStore.ingest_source_async` (the one function every ingest path routes through: sources routes, conversions, the watcher, startup restore) and in `remove_document` / `remove_source`, the brain gets the changed doc ids. Added or changed docs are assigned to the nearest leaf centroid (in-place update of that topic's `doc_ids`); removed ones leave their topic. Each counts toward `changed_since`.
- Auto recluster: when `changed_since ≥ 15%` of cards, or there is no tree and there are 6+ cards, the brain schedules its own background recluster task (no Job). Manual `POST /brain/recluster` runs the same function inside a Job so the existing jobs stream shows progress.
- `rebuild_index` and `wipe` clear cards and topics.

Ceiling: k-means has no outlier bucket, so a doc that mixes subjects lands in its strongest one (`ponytail:` comment in `cluster.py`; HDBSCAN is the upgrade).

### 3. Router (`brain/route.py`)
Replaces `route_mode` in `query.py`.
- About 30 short labelled example questions (global and local; English, Italian, a few French/German/Spanish) as a module constant. Embedded once with `query=True` on first use, cached on the `Brain`.
- `score = max cos(q, global examples) − max cos(q, local examples)`. `> +0.05` → global, `< −0.05` → local, otherwise the existing `GLOBAL_HINTS` decide and the default is local. Thresholds are constants tuned on the eval set.
- `ModeEvent.reason` includes the score (`"Global (router 0.12)"`). The manual mode in the request still wins.
- No embedder (stub backend) or embedder error: keyword hints only.
- The question vector is returned so the global path does not embed twice.

### 4. Global answer (`brain/answer.py`)
`Brain.for_query(question, vector, filters, doc_meta, topic_id) -> GlobalContext | None`, where `GlobalContext` is `(chunks: list[RetrievedChunk], outline: str, scope_label: str | None, n_topics: int)`.
1. Allowed docs: the same predicate as local retrieval (`allowed()` over `project_filters` output). Keep topics with at least one allowed doc, with `doc_ids` intersected.
2. Candidates: the top-level topics; with `topic_id`, that topic's children (a leaf: its own docs as cards). No tree: each allowed doc's card is a pseudo-topic.
3. Rank candidates by cosine(question vector, centroid); keep up to 8.
4. Per topic: 1 representative chunk + the 2 best dense hits, scoped with `VectorStore.dense(..., doc_ids=topic docs)`, deduped. Dense hits use the vector of `retrieval_query(question, history)`: the router's question vector when there is no history, one extra embed when there is (so follow-ups keep their context, as in local mode). Topic ranking (step 3) uses the question vector.
5. Interleave round-robin across topics (rank order), then the existing `pack()` with the new `RetrievalSettings.global_token_budget` (default 6144), so trimming drops evenly across topics.
6. `outline`: one line per kept topic: label, doc count, subtopic labels.

Returns `None` when there are no cards or no allowed docs; the route then runs the local path with reason "No topic map yet".

Prompt: `llm.GLOBAL_SYSTEM_PROMPT` ("You answer questions about a whole library… synthesise across the topics… cite every claim with [document_id:page] markers from the passage headers…") + the outline, via the existing `system_prompt` parameter (`system_prompt_for` gains a `base` argument so language and extra instructions still apply). Passages go through the unchanged `_user_prompt`. `llm.py`'s streaming and `AnswerEngine` are unchanged.

Citations: `extract_citations(text, chunks)` with exactly the packed chunks; citations to anything else are dropped, as today.

### 5. API
- `GET /brain/topics` → `TopicTree { status: "none" | "building" | "ready", built_at: datetime | None, changed_since: int, topics: list[Topic] }`; `Topic { id, parent_id, level, label, keywords, doc_ids, rep_chunk_ids }`, flat.
- `POST /brain/recluster` → `Job` (kind `recluster`, already in `JobKind`). 409 when the index is blocked.
- `QueryRequest.topic_id: str | None`. When set, mode is global and `ModeEvent.scope` is the topic label. Unknown id: 404 before the stream starts.
- `IndexStats.topics`: real count.
- SSE frame order unchanged: `start → mode → sources → token* → citations → done`. No new event types.
- `StorePort` gains `brain`. The stub store's brain is a null object: empty tree, `for_query` returns `None`, recluster is a no-op job. Tests use the real backend with `FakeEmbedClient`.
- `api-types.ts` regenerated; `ipc.ts` + `queries.ts` gain `getTopics` / `recluster`.

### 6. UI
- Rail item **Topics** between Library and Convert; route `/topics`; `features/topics/topics-view.tsx`.
- Left: tree (label, doc count, expand to subtopics). Right: selected topic's keyword chips, subtopics, documents (click opens `/reader/$docId`), **Ask about this topic**.
- Header: "Built 12 min ago · 34 docs changed", **Re-cluster** button, progress while the job runs (existing jobs stream).
- Empty states: under 6 documents ("Topics appear once your library has 6+ documents"), building (progress), no embedder (link to Models).
- Ask about this topic → `/chat?topic=<id>`: new chat with a dismissible chip in the composer ("Topic: keychain · tokens"); queries send `topic_id` until dismissed.
- Chat: existing `ModeBadge` tooltip shows the router reason; global answers that were topic-scoped also show the scope label.
- New strings in every shipped catalog (en/it/fr/de/es).

## Data flow
Ingest → `_write` builds cards (same transaction as documents) → end of `ingest_source_async` → `Brain.changed(ids)` assigns to nearest leaf, bumps `changed_since` → maybe schedules recluster (thread) → `topics` table replaced.

`/query` → router embeds question → global? → `Brain.for_query` → packed chunks + outline → `sources` frame → `AnswerEngine.stream` with the global system prompt → tokens → `extract_citations` against packed chunks → `done`.

## Error handling
- Recluster raises: logged, previous tree kept, status back to `ready` (or `none`), manual job reports `failed`.
- Embedder down during routing: keyword hints; during global dense hits: representative chunks only.
- Backfill fails for one doc: that doc stays without a card (logged); next startup retries.
- Index blocked (embedder changed): recluster 409; `for_query` returns `None`.
- Cards from a different embedder can't exist: `rebuild_index` clears them with the vectors.

## Testing
- `test_brain_cluster.py`: separable blobs recovered; silhouette picks the true k; same seed same tree; PCA with n < 50; c-TF-IDF picks each cluster's distinguishing terms; 5,000 random 768-d vectors cluster in under 30 s.
- `test_brain_topics.py` (real store + `FakeEmbedClient`): ingest writes cards; removing a doc deletes its card; backfill fills missing cards; a new doc joins its nearest leaf; 15% churn schedules a recluster; `rebuild_index` and `wipe` clear the brain.
- `test_brain_route.py`: margin thresholds, keyword fallback in the dead zone, no-embedder path.
- `test_query.py`: global frame order; sources span several topics; citations validated against them; `topic_id` restricts passages to the subtree; no tree → local with reason; unknown `topic_id` → 404.
- API: `GET /brain/topics`, `POST /brain/recluster` returns a job.
- Vitest: flat→nested tree helper; chat sends `topic_id` while the chip is set.
- Eval: new corpus `fixtures/brain/` (~40 short Markdown docs I write across 6 themes: gardening, astronomy, personal finance, cooking, cycling, RAG) and `eval/golden/brain.jsonl` (~15 global questions with expected docs, ~15 local controls, English and Italian). `ragcore eval` gains `global_coverage` (share of expected docs present in the packed global context; no LLM) and `router_accuracy` (routed mode vs row `type`). Committed `eval/golden/brain.baseline.json`; CI fails on a drop of more than 2 points, like the recall gate. Targets: coverage ≥ 0.70, router accuracy ≥ 0.90.
- Manual: `ragcore ask` on global questions with a real connection; spot-check answers and citations; the Topics screen in the installed app.

## Rollout
Branch `feat/global-brain`. Commits: (1) shared stopwords + `cluster.py` with tests, (2) cards + SQLite tables + backfill, (3) `Brain` topics/assignment/recluster, (4) router, (5) global answer path + query route, (6) `/brain` routes + gen:types, (7) Topics screen + chat chip + i18n, (8) eval corpus, golden set, metrics, CI gate. After changes run `scripts/install-local.sh` (project rule). No release.

## Risks
- k-means on a skewed corpus can split one big theme and merge small ones. Silhouette selection and the eval set catch the worst; HDBSCAN is the upgrade.
- Router examples are hand-written; multilingual coverage is thin. Tuned on the eval set; the manual override stays.
- 24 passages may overflow small context windows (the built-in 1.7B model runs at 8192). `global_token_budget` 6144 leaves room for the prompt; lower it if local answers fail.
- Coverage is a retrieval proxy, not answer quality. Manual spot checks until a judge model is worth wiring into CI.
