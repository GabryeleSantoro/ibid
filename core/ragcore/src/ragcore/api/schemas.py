"""The API contract.

These models are the single source of truth for the UI: FastAPI turns them into
OpenAPI and `bun run gen:types` turns that into `src/lib/api-types.ts`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

# --------------------------------------------------------------------------- health


ProcessState = Literal["stopped", "starting", "ready", "restarting", "error"]


class ProcessStatus(BaseModel):
    name: str
    role: Literal["ragcore", "embedding", "reranking"]
    state: ProcessState
    pid: int | None = None
    port: int | None = None
    uptime_s: float = 0
    restarts: int = 0
    detail: str | None = None


class IndexStats(BaseModel):
    documents: int = 0
    chunks: int = 0
    parents: int = 0
    topics: int = 0
    size_bytes: int = 0
    embed_model: str
    embed_dim: int
    reranker_model: str
    schema_version: int
    last_indexed_at: datetime | None = None


class Health(BaseModel):
    status: Literal["ok", "degraded", "starting", "error"]
    version: str
    dev_mode: bool
    uptime_s: float
    stub: bool = Field(description="True while the core is serving fixture data.")
    processes: list[ProcessStatus]
    index: IndexStats
    models_ready: bool


# -------------------------------------------------------------------------- sources


class SourceCreate(BaseModel):
    path: str
    include_globs: list[str] = Field(default_factory=lambda: ["**/*"])
    exclude_globs: list[str] = Field(default_factory=list)
    max_file_mb: int = 100
    watch: bool = True
    project_id: str | None = None


class Source(SourceCreate):
    id: str
    document_count: int = 0
    indexed_count: int = 0
    error_count: int = 0
    added_at: datetime
    last_scan_at: datetime | None = None


# ------------------------------------------------------------------------ documents


DocumentStatus = Literal[
    "queued", "parsing", "ocr", "chunking", "embedding", "indexed", "error", "skipped"
]


class Document(BaseModel):
    id: str
    source_id: str
    path: str
    title: str
    ext: str
    mime: str
    size_bytes: int
    n_pages: int | None = None
    n_chunks: int = 0
    lang: str | None = None
    status: DocumentStatus
    error: str | None = None
    keywords: list[str] = Field(default_factory=list)
    mtime: datetime
    indexed_at: datetime | None = None


class FolderInput(BaseModel):
    name: str


class Folder(FolderInput):
    """A user-made group of documents. Virtual: files stay where they are on disk."""

    id: str
    doc_ids: list[str] = Field(default_factory=list)
    created_at: datetime


class FolderAssignment(BaseModel):
    folder_id: str | None = Field(description="None takes the document out of every folder.")


class DocumentPage(BaseModel):
    page: int
    section_path: str | None = None
    text: str


class DocumentChunkRef(BaseModel):
    """Where a passage sits inside a page, so the reader can highlight it."""

    chunk_id: str
    page: int
    char_start: int
    char_end: int
    text: str


class DocumentContent(BaseModel):
    doc_id: str
    title: str
    path: str
    n_pages: int
    pages: list[DocumentPage]
    chunks: list[DocumentChunkRef]


class DocumentList(BaseModel):
    items: list[Document]
    total: int
    offset: int
    limit: int


# ----------------------------------------------------------------------------- jobs


JobKind = Literal["index", "reindex", "remove", "download", "recluster", "eval"]
JobState = Literal["queued", "running", "done", "error", "cancelled"]


class Job(BaseModel):
    id: str
    kind: JobKind
    state: JobState
    label: str
    detail: str | None = None
    progress: float = 0
    completed: int = 0
    total: int | None = None
    error: str | None = None
    source_id: str | None = None
    model_id: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


class ReindexRequest(BaseModel):
    source_id: str | None = None
    doc_ids: list[str] | None = None
    full: bool = False


# ---------------------------------------------------------------------------- query


QueryMode = Literal["auto", "local", "global"]
Grounding = Literal["ok", "low", "none"]


class QueryFilters(BaseModel):
    source_ids: list[str] | None = None
    doc_ids: list[str] | None = None
    exts: list[str] | None = None
    langs: list[str] | None = None
    after: datetime | None = None
    before: datetime | None = None


class QueryRequest(BaseModel):
    q: str
    session_id: str | None = None
    query_id: str | None = Field(
        default=None,
        description="Client-generated, so cancellation works before the first frame arrives.",
    )
    mode: QueryMode = "auto"
    connection_id: str | None = None
    filters: QueryFilters = Field(default_factory=QueryFilters)
    lang: Literal["en", "it", "fr", "de", "es"] | None = Field(
        default=None, description="Language the answer is written in. None: the model's own choice."
    )


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    doc_title: str
    page_start: int
    page_end: int
    section_path: str | None = None
    text: str
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rrf_score: float = 0
    rerank_score: float = 0


class Citation(BaseModel):
    marker: str = Field(description='The literal "[doc:page]" the model wrote.')
    doc_id: str
    doc_title: str
    page: int
    chunk_id: str
    section_path: str | None = None
    snippet: str
    char_start: int | None = None
    char_end: int | None = None
    score: float = 0


class StageLatency(BaseModel):
    embed_ms: float = 0
    dense_ms: float = 0
    bm25_ms: float = 0
    rerank_ms: float = 0
    pack_ms: float = 0
    llm_first_token_ms: float = 0
    total_ms: float = 0


# SSE frames. Declared so they land in the OpenAPI schema even though FastAPI
# streams them as text/event-stream rather than returning them.
class StartEvent(BaseModel):
    query_id: str
    session_id: str


class ModeEvent(BaseModel):
    mode: Literal["local", "global"]
    reason: str
    scope: str | None = None


class SourcesEvent(BaseModel):
    chunks: list[RetrievedChunk]
    candidates: int
    kept: int


class TokenEvent(BaseModel):
    text: str


class CitationsEvent(BaseModel):
    citations: list[Citation]
    dropped: int = 0
    grounding: Grounding = "ok"


class DoneEvent(BaseModel):
    message_id: str
    latency: StageLatency
    connection_id: str | None = None
    remote: bool = False
    tokens_per_s: float | None = None


class ErrorEvent(BaseModel):
    message: str
    retryable: bool = False
    code: str | None = None
    params: dict[str, object] = Field(default_factory=dict)


# --------------------------------------------------------------- slide to text


class SlideConversionRequest(BaseModel):
    """Input for the slide conversion stream."""

    slide_ids: list[str] = Field(default_factory=list)
    file_paths: list[str] = Field(default_factory=list)
    research_query: str | None = None  # topics the model should dig into
    output_title: str | None = None
    language: Literal["it", "en", "fr", "de", "es"] = "it"
    depth: Literal["standard", "deep"] = "deep"

    @model_validator(mode="after")
    def has_input(self) -> SlideConversionRequest:
        if not self.slide_ids and not self.file_paths:
            raise ValueError("Select at least one indexed slide or local slide file")
        return self


class ConversionStartEvent(BaseModel):
    presentation_index: int
    presentation_total: int
    slide_id: str | None
    title: str


class ConversionSavedEvent(BaseModel):
    presentation_index: int
    path: str
    title: str
    document_id: str


class PresentationErrorEvent(BaseModel):
    presentation_index: int
    presentation_total: int
    title: str
    message: str
    code: str | None = None
    params: dict[str, object] = Field(default_factory=dict)


class ConversionDoneEvent(BaseModel):
    saved: list[ConversionSavedEvent]
    failed: list[PresentationErrorEvent]


# --------------------------------------------------------------------- connections


ConnectionKind = Literal["openai-compatible", "anthropic", "local"]
ThinkingLevel = Literal["off", "low", "medium", "high"]
ProviderSort = Literal["price", "throughput", "latency"]


class ConnectionInput(BaseModel):
    name: str
    kind: ConnectionKind
    base_url: str | None = None
    model_id: str
    # None: no cap, the model writes until its own limit.
    max_output_tokens: int | None = None
    thinking: ThinkingLevel = "off"
    is_remote: bool = True
    api_key: str | None = Field(
        default=None,
        description="Write-only. The durable copy lives in the OS keychain, held by the "
        "Rust shell; ragcore keeps one in memory for this process' lifetime because "
        "ragcore is what calls the provider. Never returned by any endpoint.",
    )
    provider_sort: ProviderSort | None = Field(
        default=None,
        description="OpenRouter provider routing: rank candidate providers by price, "
        "throughput or latency. Ignored outside OpenRouter.",
    )
    provider_order: list[str] | None = Field(
        default=None,
        description="OpenRouter provider routing: try these providers first, in order. "
        "Ignored outside OpenRouter.",
    )


class Connection(BaseModel):
    id: str
    name: str
    kind: ConnectionKind
    base_url: str | None = None
    model_id: str
    max_output_tokens: int | None = None
    thinking: ThinkingLevel
    is_remote: bool
    has_api_key: bool
    active: bool
    created_at: datetime
    provider_sort: ProviderSort | None = None
    provider_order: list[str] | None = None


class ConnectionSecret(BaseModel):
    api_key: str


class ConnectionTestRequest(BaseModel):
    connection_id: str | None = None
    kind: ConnectionKind | None = None
    base_url: str | None = None
    model_id: str | None = None
    api_key: str | None = None


class ConnectionTestResult(BaseModel):
    ok: bool
    reachable: bool
    model_found: bool
    streaming: bool
    tokens_per_second: float | None = None
    latency_ms: float | None = None
    error: str | None = None


# -------------------------------------------------------------------------- models


ModelRole = Literal["embedding", "reranking"]
FitVerdict = Literal["vram", "ram", "too-large", "unknown"]


class InstalledModel(BaseModel):
    id: str
    role: ModelRole
    name: str
    repo_id: str
    filename: str
    quant: str | None = None
    size_bytes: int
    sha256: str | None = None
    active: bool
    shipped: bool
    downloaded_at: datetime | None = None
    context_length: int | None = None


class HubModel(BaseModel):
    id: str
    author: str | None = None
    name: str
    downloads: int = 0
    likes: int = 0
    trending_score: float = 0
    pipeline_tag: str | None = None
    license: str | None = None
    tags: list[str] = Field(default_factory=list)
    updated_at: datetime | None = None
    gguf: bool = False
    recommended: bool = False
    recommended_reason: str | None = None


class HubSearchResult(BaseModel):
    items: list[HubModel]
    role: ModelRole | None = None
    query: str | None = None


class HubFile(BaseModel):
    path: str
    size_bytes: int
    quant: str | None = None
    fit: FitVerdict = "unknown"
    fit_note: str | None = None


class HubModelDetail(BaseModel):
    model: HubModel
    files: list[HubFile]
    readme_excerpt: str | None = None


class ModelInventory(BaseModel):
    installed: list[InstalledModel]
    active_embedding: str | None = None
    active_reranking: str | None = None


class DownloadRequest(BaseModel):
    repo_id: str
    filename: str
    role: ModelRole
    activate: bool = False
    """Set when swapping the embedder, which invalidates the whole index."""
    accept_reindex: bool = False


class HardwareInfo(BaseModel):
    os: str
    arch: str
    cpu_count: int
    ram_mb: int
    vram_mb: int
    gpu_backend: Literal["metal", "cuda", "vulkan", "cpu"]
    gpu_name: str | None = None
    profile: Literal["cpu", "balanced", "gpu"]


# ------------------------------------------------------------------------ settings


class RetrievalSettings(BaseModel):
    dense_top_k: int = 40
    bm25_top_k: int = 40
    rrf_k: int = 60
    rerank_candidates: int = 40
    top_k: int = 6
    min_score: float = 0.3
    context_token_budget: int = 4096
    history_token_budget: int = 2048


class PerformanceSettings(BaseModel):
    profile: Literal["cpu", "balanced", "gpu"] = "balanced"
    embed_batch: int = 32
    gpu_layers: int = 999
    max_parallel_parsers: int = 4


class AppSettings(BaseModel):
    onboarded: bool = False
    storage_path: str
    active_connection_id: str | None = None
    telemetry: bool = False
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    performance: PerformanceSettings = Field(default_factory=PerformanceSettings)
    chat_extra_instructions: str = Field(default="", max_length=1000)


class AppSettingsPatch(BaseModel):
    onboarded: bool | None = None
    storage_path: str | None = None
    active_connection_id: str | None = None
    telemetry: bool | None = None
    retrieval: RetrievalSettings | None = None
    performance: PerformanceSettings | None = None
    chat_extra_instructions: str | None = Field(default=None, max_length=1000)


class WipeRequest(BaseModel):
    confirm: str = Field(description='Must equal "DELETE" for the wipe to run.')
    keep_connections: bool = True


# ---------------------------------------------------------------------------- eval


class EvalRunRequest(BaseModel):
    set_name: str = "base"
    connection_id: str | None = None
    compare_baseline: bool = True


class EvalQuestionResult(BaseModel):
    q: str
    type: Literal["local", "global"]
    hit: bool
    rank: int | None = None
    ndcg: float = 0
    latency_ms: float = 0


class EvalMetrics(BaseModel):
    set_name: str
    n_questions: int
    recall_at_1: float
    recall_at_6: float
    mrr: float
    ndcg_at_6: float
    latency_p50: StageLatency
    latency_p95: StageLatency
    baseline_recall_at_6: float | None = None


class EvalResult(BaseModel):
    metrics: EvalMetrics
    questions: list[EvalQuestionResult]


class EvalSet(BaseModel):
    name: str
    n_questions: int
    description: str | None = None


# --------------------------------------------------------------------- chat history


class ChatProject(BaseModel):
    id: str
    name: str
    pinned: bool = False
    use_global_sources: bool = True
    created_at: datetime
    updated_at: datetime


class ChatProjectCreate(BaseModel):
    name: str


class ChatProjectPatch(BaseModel):
    name: str | None = None
    pinned: bool | None = None
    use_global_sources: bool | None = None


class ChatMessage(BaseModel):
    id: str
    session_id: str
    role: Literal["user", "assistant"]
    text: str
    mode: Literal["local", "global"] | None = None
    citations: list[Citation] = Field(default_factory=list)
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    grounding: Grounding | None = None
    created_at: datetime


class ChatSession(BaseModel):
    id: str
    title: str
    message_count: int
    scope_doc_id: str | None = None
    project_id: str | None = None
    pinned: bool = False
    created_at: datetime
    updated_at: datetime


class ChatSessionCreate(BaseModel):
    title: str | None = None
    scope_doc_id: str | None = None
    project_id: str | None = None


class ChatSessionPatch(BaseModel):
    title: str | None = None
    project_id: str | None = None
    pinned: bool | None = None


class Ok(BaseModel):
    ok: bool = True


class EvalProgressEvent(BaseModel):
    completed: int
    total: int
    question: str


class StreamEnvelope(BaseModel):
    """Every frame the SSE routes can emit, in one model.

    Streaming responses never pass through `response_model`, so without this the
    frame shapes would not reach OpenAPI and the UI would be typing them by
    hand. `GET /schema/events` returns an empty one; its purpose is the schema.
    """

    start: StartEvent | None = None
    mode: ModeEvent | None = None
    sources: SourcesEvent | None = None
    token: TokenEvent | None = None
    citations: CitationsEvent | None = None
    done: DoneEvent | None = None
    error: ErrorEvent | None = None
    job: Job | None = None
    eval_progress: EvalProgressEvent | None = None
    eval_result: EvalResult | None = None
    conversion_start: ConversionStartEvent | None = None
    conversion_saved: ConversionSavedEvent | None = None
    presentation_error: PresentationErrorEvent | None = None
    conversion_done: ConversionDoneEvent | None = None
