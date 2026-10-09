import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "@tanstack/react-router";
import { useTranslation } from "react-i18next";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { CloudIcon, FileTextIcon, PanelRightIcon } from "lucide-react";

import { Page, PageBody, PageHeader } from "@/components/shell/page";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { IconTooltip } from "@/components/ui/tooltip";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Composer } from "@/features/chat/composer";
import { AssistantMessage, UserMessage } from "@/features/chat/message";
import { SlowSpeedBanner } from "@/features/chat/slow-speed-banner";
import { SourcePanel } from "@/features/chat/source-panel";
import { useChat } from "@/features/chat/use-chat";
import type { CitationTarget } from "@/features/chat/answer-text";
import { api, type ChatMessage, type QueryMode } from "@/lib/ipc";
import { connectionsQuery, keys, projectsQuery, sourcesQuery } from "@/lib/queries";

export function ChatView({ sessionId }: { sessionId: string | null }) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const [mode, setMode] = useState<QueryMode>("auto");
  const [selectedDocumentIds, setSelectedDocumentIds] = useState<string[] | null>(null);
  const [panelOpen, setPanelOpen] = useState(true);
  const [selectedChunk, setSelectedChunk] = useState<string | null>(null);
  const bottom = useRef<HTMLDivElement>(null);

  const { pending, isStreaming, send, cancel, activeSession } = useChat(sessionId);
  const sources = useQuery(sourcesQuery);
  const projects = useQuery(projectsQuery);
  const documents = useQuery({
    queryKey: keys.documents({ limit: 500 }),
    queryFn: () => api.listDocuments({ limit: 500 }),
  });
  const connections = useQuery(connectionsQuery);

  const session = useQuery({
    queryKey: ["session", sessionId],
    queryFn: () => api.getSession(sessionId as string),
    enabled: Boolean(sessionId),
  });

  const history = useQuery({
    queryKey: keys.messages(sessionId ?? "none"),
    queryFn: () => api.getMessages(sessionId as string),
    enabled: Boolean(sessionId),
  });

  const active = connections.data?.find((connection) => connection.active);

  // A new chat gets its id from the first `start` frame. Move to its URL once
  // the stream is over, so the history query takes over from the live state.
  useEffect(() => {
    if (!sessionId && activeSession && pending && pending.status !== "retrieving" && pending.status !== "streaming") {
      void navigate({ to: "/chat/$sessionId", params: { sessionId: activeSession } });
    }
  }, [activeSession, navigate, pending, sessionId]);

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: pending ? "smooth" : "auto", block: "end" });
  }, [pending?.text, history.data?.length, pending]);

  const lastChunks = useMemo(() => {
    if (pending?.chunks.length) return pending.chunks;
    const lastAssistant = [...(history.data ?? [])].reverse().find((m) => m.role === "assistant");
    return lastAssistant?.chunks ?? [];
  }, [history.data, pending]);

  const lastCitations = useMemo(() => {
    if (pending?.citations.length) return pending.citations;
    const lastAssistant = [...(history.data ?? [])].reverse().find((m) => m.role === "assistant");
    return lastAssistant?.citations ?? [];
  }, [history.data, pending]);

  const onSelectCitation = (target: CitationTarget) => {
    if (!target.citation) return;
    setPanelOpen(true);
    setSelectedChunk(target.citation.chunk_id);
    requestAnimationFrame(() => {
      document
        .getElementById(`source-${target.citation!.chunk_id}`)
        ?.scrollIntoView({ behavior: "smooth", block: "center" });
    });
  };

  const scopeDocId = session.data?.scope_doc_id ?? null;
  const activeProject = projects.data?.find((project) => project.id === session.data?.project_id);
  const globalSourceIds = new Set(
    (sources.data ?? [])
      .filter((source) => source.project_id === null)
      .map((source) => source.id),
  );
  const availableSourceIds = new Set(
    (sources.data ?? [])
      .filter(
        (source) =>
          source.project_id === null || source.project_id === activeProject?.id,
      )
      .map((source) => source.id),
  );
  const availableDocumentIds = (documents.data?.items ?? [])
    .filter(
      (document) =>
        availableSourceIds.has(document.source_id) &&
        (!activeProject ||
          activeProject.use_global_sources ||
          !globalSourceIds.has(document.source_id)),
    )
    .map((document) => document.id);

  useEffect(() => {
    setSelectedDocumentIds(null);
  }, [activeProject?.id]);

  const availableDocumentIdSet = new Set(availableDocumentIds);
  const includedDocumentIds =
    selectedDocumentIds?.filter((documentId) => availableDocumentIdSet.has(documentId)) ??
    availableDocumentIds;

  const selectedDocumentCount = includedDocumentIds.length;

  const suggestionDocIds = [...includedDocumentIds].sort();
  const newChat = !sessionId && suggestionDocIds.length > 0;
  const model = useQuery({
    queryKey: ["suggestion-model"],
    queryFn: api.suggestionModel,
    enabled: newChat,
    refetchInterval: (query) => (query.state.data?.state === "downloading" ? 1500 : false),
  });
  const modelState = model.data?.state;
  const suggestions = useQuery({
    queryKey: ["suggestions", suggestionDocIds, modelState === "ready"],
    queryFn: () => api.suggestQuestions(suggestionDocIds),
    enabled: newChat && modelState !== undefined && modelState !== "downloading",
    staleTime: Infinity,
    retry: false,
  });
  const questions =
    suggestions.data?.source === "model"
      ? suggestions.data.questions
      : (suggestions.data?.topics ?? []).map((topic, index) =>
          t(`chat.suggestion.${index}`, { topic, lng: suggestions.data?.language ?? undefined }),
        );
  const installModel = useMutation({
    mutationFn: api.installSuggestionModel,
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: ["suggestion-model"] }),
  });

  const onSend = (text: string) => {
    send(text, {
      mode,
      filters: scopeDocId
        ? { doc_ids: [scopeDocId] }
        : { doc_ids: includedDocumentIds },
    });
  };

  return (
    <>
      <Page>
        <PageHeader
          title={sessionId ? undefined : t("chat.newChat")}
          actions={
            <IconTooltip label={panelOpen ? t("chat.hideResources") : t("chat.showResources")}>
              <Button
                variant="ghost"
                size="icon"
                className="size-8"
                aria-pressed={panelOpen}
                aria-label={panelOpen ? t("chat.hideResources") : t("chat.showResources")}
                onClick={() => setPanelOpen((open) => !open)}
              >
                <PanelRightIcon className="size-4" />
              </Button>
            </IconTooltip>
          }
        >
          <div className="min-w-0 flex-1">
            <h1 className="truncate text-sm font-semibold tracking-tight">
              {sessionId ? t("nav.chat") : t("chat.newChat")}
            </h1>
            <p className="truncate text-xs text-muted-foreground">
              {active ? `${active.name} · ${active.model_id}` : t("chat.noModel")}
            </p>
          </div>
        </PageHeader>

        {active?.is_remote ? (
          <div className="flex items-center gap-2 border-b border-status-warn/25 bg-status-warn/8 px-5 py-1.5 text-[0.6875rem] text-status-warn">
            <CloudIcon className="size-3.5 shrink-0" />
            {t("chat.remoteBanner", { name: active.name })}
          </div>
        ) : null}

        <PageBody>
          <div className="mx-auto w-full max-w-3xl space-y-6 px-5 py-6">
            {history.isLoading ? (
              <div className="space-y-3">
                <Skeleton className="h-8 w-2/3" />
                <Skeleton className="h-24 w-full" />
              </div>
            ) : null}

            {(history.data ?? []).map((message: ChatMessage) =>
              message.role === "user" ? (
                <UserMessage key={message.id} text={message.text} />
              ) : (
                <AssistantMessage
                  key={message.id}
                  text={message.text}
                  mode={message.mode ?? null}
                  citations={message.citations ?? []}
                  chunks={message.chunks ?? []}
                  grounding={message.grounding ?? null}
                  dropped={0}
                  latency={null}
                  onSelectCitation={onSelectCitation}
                />
              ),
            )}

            {pending ? (
              <>
                <UserMessage text={pending.question} />
                <AssistantMessage
                  text={pending.text}
                  mode={pending.mode}
                  modeReason={pending.modeReason}
                  citations={pending.citations}
                  chunks={pending.chunks}
                  grounding={pending.grounding}
                  dropped={pending.dropped}
                  latency={pending.latency}
                  streaming={pending.status === "streaming"}
                  retrieving={pending.status === "retrieving"}
                  error={pending.error}
                  onSelectCitation={onSelectCitation}
                />
                <SlowSpeedBanner done={pending.status === "done"} tokensPerS={pending.tokensPerS} />
              </>
            ) : null}

            {!pending && !sessionId ? (
              <div className="pt-10">
                <h2 className="text-lg font-semibold tracking-tight">{t("chat.emptyTitle")}</h2>
                <p className="mt-1 text-sm text-muted-foreground">
                  {t("chat.emptyLead")}
                </p>
                <div className="mt-5 grid gap-2">
                  {suggestions.isFetching
                    ? [0, 1, 2].map((i) => <Skeleton key={i} className="h-9 w-full rounded-lg" />)
                    : null}
                  {questions.map((suggestion) => (
                    <button
                      key={suggestion}
                      type="button"
                      onClick={() => onSend(suggestion)}
                      className="rounded-lg border border-border bg-card px-3 py-2 text-left text-[0.8125rem] transition-colors hover:border-muted-foreground/40"
                    >
                      {suggestion}
                    </button>
                  ))}
                </div>
                {newChat && modelState === "downloading" ? (
                  <p className="mt-3 text-xs text-muted-foreground">
                    {t("chat.suggestionModel.downloading", {
                      percent: Math.round((model.data?.progress ?? 0) * 100),
                    })}
                  </p>
                ) : null}
                {newChat && modelState === "missing" ? (
                  <button
                    type="button"
                    onClick={() => installModel.mutate()}
                    className="mt-3 text-xs text-muted-foreground underline underline-offset-2 hover:text-foreground"
                  >
                    {t("chat.suggestionModel.offer")}
                  </button>
                ) : null}
                {newChat && model.data?.error ? (
                  <p className="mt-1 text-xs text-destructive">
                    {t("chat.suggestionModel.failed", { error: model.data.error })}
                  </p>
                ) : null}
              </div>
            ) : null}

            <div ref={bottom} />
          </div>
        </PageBody>

        <Composer onSend={onSend} onStop={cancel} streaming={isStreaming}>
          {scopeDocId ? (
            <Badge variant="secondary" className="h-7 gap-1.5 font-normal">
              <FileTextIcon className="size-3" />
              {t("chat.scopedToDocument")}
            </Badge>
          ) : null}

          <Select value={mode} onValueChange={(value) => setMode(value as QueryMode)}>
            <SelectTrigger size="sm" className="h-7 gap-1 text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="auto">{t("chat.modeAuto")}</SelectItem>
              <SelectItem value="local">{t("chat.modeLocalOption")}</SelectItem>
              <SelectItem value="global">{t("chat.modeGlobalOption")}</SelectItem>
            </SelectContent>
          </Select>

          <Badge variant="secondary" className="h-7 gap-1.5 font-normal">
            <FileTextIcon className="size-3" />
            {t("chat.filesSelected", { count: scopeDocId ? 1 : selectedDocumentCount })}
          </Badge>
        </Composer>
      </Page>

      {panelOpen ? (
        <SourcePanel
          project={activeProject ?? null}
          selectedDocumentIds={selectedDocumentIds}
          onSelectionChange={setSelectedDocumentIds}
          chunks={lastChunks}
          citations={lastCitations}
          selectedChunkId={selectedChunk}
          onSelect={setSelectedChunk}
          onClose={() => setPanelOpen(false)}
        />
      ) : null}
    </>
  );
}
