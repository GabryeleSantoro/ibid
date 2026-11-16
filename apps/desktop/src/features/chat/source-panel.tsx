import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import { errorText } from "@/lib/errors";
import {
  BookOpenIcon,
  ChevronDownIcon,
  ChevronRightIcon,
  FileTextIcon,
  FolderIcon,
  FolderPlusIcon,
  SearchIcon,
  XIcon,
} from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { IconTooltip } from "@/components/ui/tooltip";
import { AddSourceDialog } from "@/features/library/add-source-dialog";
import { relativeTime, shortPath } from "@/lib/format";
import {
  api,
  type ChatProject,
  type Citation,
  type Document,
  type RetrievedChunk,
  type Source,
} from "@/lib/ipc";
import { foldersQuery, keys, sourcesQuery } from "@/lib/queries";
import { cn } from "@/lib/utils";

/**
 * The right-hand workspace for project resources and answer provenance.
 * Resource controls live here so they stay next to the chat they affect.
 */
export function SourcePanel({
  project,
  selectedDocumentIds,
  onSelectionChange,
  chunks,
  citations,
  selectedChunkId,
  onSelect,
  onClose,
}: {
  project: ChatProject | null;
  selectedDocumentIds: string[] | null;
  onSelectionChange: (documentIds: string[]) => void;
  chunks: RetrievedChunk[];
  citations: Citation[];
  selectedChunkId: string | null;
  onSelect: (chunkId: string) => void;
  onClose: () => void;
}) {
  const { t } = useTranslation();
  const sources = useQuery(sourcesQuery);
  const documents = useQuery({
    queryKey: keys.documents({ limit: 500 }),
    queryFn: () => api.listDocuments({ limit: 500 }),
  });
  const folders = useQuery(foldersQuery);
  const queryClient = useQueryClient();
  const [globalOpen, setGlobalOpen] = useState(true);
  const [projectOpen, setProjectOpen] = useState(true);
  const [knowledgeSearch, setKnowledgeSearch] = useState("");
  const citedChunks = new Map(citations.map((citation) => [citation.chunk_id, citation]));

  const updateProject = useMutation({
    mutationFn: (useGlobalSources: boolean) => {
      if (!project) throw new Error(t("chat.noProjectSelected"));
      return api.updateProject(project.id, { use_global_sources: useGlobalSources });
    },
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: keys.projects }),
    onError: (error: Error) =>
      toast.error(t("chat.updateResourcesFailed"), { description: errorText(error) }),
  });

  const allDocuments = documents.data?.items ?? [];
  const globalSources = (sources.data ?? []).filter((source) => source.project_id === null);
  const globalSourceIds = useMemo(
    () => new Set(globalSources.map((source) => source.id)),
    [globalSources],
  );
  const selectableSourceIds = useMemo(
    () =>
      new Set(
        (sources.data ?? [])
          .filter((source) => !project || source.project_id === null || source.project_id === project.id)
          .map((source) => source.id),
      ),
    [project, sources.data],
  );
  const selectableDocuments = allDocuments.filter((document) =>
    selectableSourceIds.has(document.source_id),
  );
  const selectedSet = useMemo(
    () => (selectedDocumentIds === null ? null : new Set(selectedDocumentIds)),
    [selectedDocumentIds],
  );
  const isSelected = (document: Document) =>
    (!globalSourceIds.has(document.source_id) || !project || project.use_global_sources) &&
    (selectedSet === null || selectedSet.has(document.id));
  const toggleDocuments = (items: Document[], checked: boolean) => {
    const next = new Set(selectedSet ?? selectableDocuments.map((document) => document.id));
    for (const document of items) {
      if (checked) next.add(document.id);
      else next.delete(document.id);
    }
    onSelectionChange([...next]);
  };
  // Plain chats can also pick folders that belong to projects.
  const projectSources = (sources.data ?? []).filter((source) =>
    project ? source.project_id === project.id : source.project_id !== null,
  );
  const normalizedSearch = knowledgeSearch.trim().toLowerCase();
  const documentsForSource = (source: Source) =>
    allDocuments.filter((document) => document.source_id === source.id);
  const visibleDocumentsForSource = (source: Source) => {
    const folderMatches = source.path.toLowerCase().includes(normalizedSearch);
    return documentsForSource(source).filter(
      (document) =>
        !normalizedSearch ||
        folderMatches ||
        document.title.toLowerCase().includes(normalizedSearch) ||
        document.path.toLowerCase().includes(normalizedSearch),
    );
  };
  const libraryFolders = (folders.data ?? [])
    .map((folder) => {
      const ids = new Set(folder.doc_ids);
      return { folder, files: selectableDocuments.filter((document) => ids.has(document.id)) };
    })
    .filter(({ files }) => files.length > 0);
  const globalDocuments = globalSources.flatMap(documentsForSource);
  const globalEnabled = !project || project.use_global_sources;
  const globalSelectedCount = globalDocuments.filter(isSelected).length;
  const globalChecked =
    globalEnabled &&
    globalDocuments.length > 0 &&
    globalSelectedCount === globalDocuments.length;
  const globalIndeterminate = globalEnabled && globalSelectedCount > 0 && !globalChecked;

  return (
    <aside className="flex w-90 shrink-0 flex-col border-l border-border bg-sidebar">
      <header
        data-tauri-drag-region="deep"
        className="drag-region flex items-center justify-between gap-2 px-3 pt-3 pb-2"
      >
        <div>
          <h2 className="text-[0.8125rem] font-semibold tracking-tight">{t("chat.resources")}</h2>
          <p className="text-[0.6875rem] text-muted-foreground">
            {project ? project.name : t("chat.globalLibrary")}
          </p>
        </div>
        <IconTooltip label={t("chat.closeResources")}>
          <Button
            data-tauri-drag-region="false"
            variant="ghost"
            size="icon"
            className="no-drag size-7"
            aria-label={t("chat.closeResources")}
            onClick={onClose}
          >
            <XIcon className="size-4" />
          </Button>
        </IconTooltip>
      </header>

      <ScrollArea className="min-h-0 flex-1">
        <div className="space-y-4 px-3 pb-4">
          <section className="space-y-1">
            <div className="flex items-center justify-between gap-2">
              <div>
                <h3 className="text-xs font-semibold tracking-tight">{t("chat.knowledge")}</h3>
                <p className="text-[0.6875rem] text-muted-foreground">
                  {project ? t("chat.knowledgeProject") : t("chat.knowledgeShared")}
                </p>
              </div>
              {!project ? (
                <AddSourceDialog
                  trigger={
                    <Button variant="ghost" size="sm" className="h-7 px-2 text-xs">
                      <FolderPlusIcon className="size-3.5" />
                      {t("chat.addFolder")}
                    </Button>
                  }
                />
              ) : null}
            </div>

            <div className="relative">
              <SearchIcon className="pointer-events-none absolute top-1/2 left-2.5 size-3.5 -translate-y-1/2 text-muted-foreground" />
              <Input
                value={knowledgeSearch}
                onChange={(event) => setKnowledgeSearch(event.target.value)}
                placeholder={t("chat.searchKnowledge")}
                aria-label={t("chat.searchKnowledgeLabel")}
                className="h-8 pl-8 text-xs"
              />
            </div>

            <Collapsible open={globalOpen} onOpenChange={setGlobalOpen}>
              <div className="flex items-center justify-between rounded-md border border-border bg-muted/30 px-2 py-1.5">
                <CollapsibleTrigger asChild>
                  <button
                    type="button"
                    className="flex min-w-0 items-center gap-2 text-left"
                    aria-label={globalOpen ? t("chat.collapseGlobal") : t("chat.expandGlobal")}
                  >
                    {globalOpen ? (
                      <ChevronDownIcon className="size-3.5 shrink-0 text-muted-foreground" />
                    ) : (
                      <ChevronRightIcon className="size-3.5 shrink-0 text-muted-foreground" />
                    )}
                    <FolderIcon className="size-4 shrink-0 text-primary" />
                    <span className="truncate text-xs font-medium">{t("chat.globalKnowledge")}</span>
                    <span className="text-[0.6875rem] text-muted-foreground">
                      {globalSources.length}
                    </span>
                  </button>
                </CollapsibleTrigger>
                <Checkbox
                  checked={globalIndeterminate ? "indeterminate" : globalChecked}
                  onCheckedChange={(checked) => {
                    const include = checked === true;
                    toggleDocuments(globalDocuments, include);
                    if (project && include !== project.use_global_sources) {
                      updateProject.mutate(include);
                    }
                  }}
                  disabled={updateProject.isPending}
                  aria-label={t("chat.includeAllGlobal")}
                />
              </div>
              <CollapsibleContent>
                <div className="mt-1 space-y-0.5 pl-7">
                  {globalSources.length ? (
                    globalSources.some((source) => visibleDocumentsForSource(source).length) ? (
                      globalSources.map((source) => (
                        <FolderGroup
                          key={source.id}
                          source={source}
                          files={documentsForSource(source)}
                          visibleFiles={visibleDocumentsForSource(source)}
                          selected={isSelected}
                          onToggle={toggleDocuments}
                          disabled={Boolean(project && !project.use_global_sources)}
                        />
                      ))
                    ) : (
                      <p className="py-2 text-xs text-muted-foreground">{t("chat.noMatchingFiles")}</p>
                    )
                  ) : (
                    <p className="py-2 text-xs text-muted-foreground">{t("chat.noGlobalFiles")}</p>
                  )}
                </div>
              </CollapsibleContent>
            </Collapsible>

            {libraryFolders.length ? (
              <div className="mt-2 space-y-0.5">
                <p className="px-1 text-[0.6875rem] font-medium text-muted-foreground">
                  {t("chat.libraryFolders")}
                </p>
                {libraryFolders.map(({ folder, files }) => (
                  <LibraryFolderGroup
                    key={folder.id}
                    name={folder.name}
                    files={files}
                    selected={isSelected}
                    onToggle={toggleDocuments}
                  />
                ))}
              </div>
            ) : null}

            {project || projectSources.length ? (
              <Collapsible open={projectOpen} onOpenChange={setProjectOpen}>
                <div className="mt-2 flex items-center justify-between rounded-md border border-border bg-muted/30 px-2 py-1.5">
                  <CollapsibleTrigger asChild>
                    <button
                      type="button"
                      className="flex min-w-0 items-center gap-2 text-left"
                      aria-label={projectOpen ? t("chat.collapseProject") : t("chat.expandProject")}
                    >
                      {projectOpen ? (
                        <ChevronDownIcon className="size-3.5 shrink-0 text-muted-foreground" />
                      ) : (
                        <ChevronRightIcon className="size-3.5 shrink-0 text-muted-foreground" />
                      )}
                      <FolderIcon className="size-4 shrink-0 text-primary" />
                      <span className="truncate text-xs font-medium">{t("chat.projectFolders")}</span>
                      <span className="text-[0.6875rem] text-muted-foreground">
                        {projectSources.length}
                      </span>
                    </button>
                  </CollapsibleTrigger>
                  {project ? (
                    <AddSourceDialog
                      projectId={project.id}
                      trigger={
                        <Button
                          variant="ghost"
                          size="icon"
                          className="size-6"
                          aria-label={t("chat.addProjectFolder")}
                        >
                          <FolderPlusIcon className="size-3.5" />
                        </Button>
                      }
                      triggerLabel={t("chat.addProjectFolder")}
                    />
                  ) : null}
                </div>
                <CollapsibleContent>
                  <div className="mt-1 space-y-0.5 pl-7">
                    {projectSources.length ? (
                      projectSources.some((source) => visibleDocumentsForSource(source).length) ? (
                        projectSources.map((source) => (
                          <FolderGroup
                            key={source.id}
                            source={source}
                            files={documentsForSource(source)}
                            visibleFiles={visibleDocumentsForSource(source)}
                            selected={isSelected}
                            onToggle={toggleDocuments}
                          />
                        ))
                      ) : (
                        <p className="py-2 text-xs text-muted-foreground">{t("chat.noMatchingFiles")}</p>
                      )
                    ) : (
                      <p className="py-2 text-xs text-muted-foreground">{t("chat.noProjectFiles")}</p>
                    )}
                  </div>
                </CollapsibleContent>
              </Collapsible>
            ) : null}
          </section>

          <section className="border-t border-border pt-3">
            <div className="mb-2 flex items-center justify-between">
              <div>
                <h3 className="text-xs font-semibold tracking-tight">{t("chat.answerSources")}</h3>
                <p className="text-[0.6875rem] text-muted-foreground">
                  {t("chat.passagesSent", { count: chunks.length })}
                </p>
              </div>
            </div>

            {chunks.length === 0 ? (
              <p className="py-5 text-center text-xs text-muted-foreground">
                {t("chat.askToSeePassages")}
              </p>
            ) : null}

            <div className="space-y-2">
              {chunks.map((chunk, index) => {
                const citation = citedChunks.get(chunk.chunk_id);
                const selected = selectedChunkId === chunk.chunk_id;
                return (
                  <article
                    key={chunk.chunk_id}
                    id={`source-${chunk.chunk_id}`}
                    onClick={() => onSelect(chunk.chunk_id)}
                    className={cn(
                      "cursor-pointer rounded-lg border bg-card p-3 transition-colors",
                      selected
                        ? "border-primary/60 ring-[3px] ring-primary/15"
                        : "border-border hover:border-muted-foreground/40",
                    )}
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div className="min-w-0">
                        <p className="truncate text-[0.8125rem] font-medium">{chunk.doc_title}</p>
                        <p className="truncate text-[0.6875rem] text-muted-foreground">
                          {chunk.section_path ?? "—"}
                        </p>
                      </div>
                      <span
                        className={cn(
                          "grid size-5 shrink-0 place-items-center rounded font-mono text-[0.625rem]",
                          citation ? "bg-primary/12 text-primary" : "bg-muted text-muted-foreground",
                        )}
                        title={citation ? t("chat.cited") : t("chat.notCited")}
                      >
                        {index + 1}
                      </span>
                    </div>

                    <p className="selectable mt-2 line-clamp-6 text-xs leading-[1.55] text-muted-foreground">
                      {chunk.text}
                    </p>

                    <div className="mt-2.5 flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-[0.625rem] text-muted-foreground tabular-nums">
                      <span>p.{chunk.page_start}</span>
                      <span title={t("chat.rerankerTitle")}>
                        rerank {chunk.rerank_score.toFixed(3)}
                      </span>
                      <span title={t("chat.denseTitle")}>dense {chunk.dense_rank ?? "—"}</span>
                      <span title={t("chat.keywordTitle")}>bm25 {chunk.bm25_rank ?? "—"}</span>
                    </div>

                    <Button
                      asChild
                      variant="ghost"
                      size="sm"
                      className="mt-2 h-7 w-full justify-start px-2"
                    >
                      <Link
                        to="/reader/$docId"
                        params={{ docId: chunk.doc_id }}
                        search={{ page: chunk.page_start, highlight: chunk.chunk_id }}
                      >
                        <BookOpenIcon className="size-3.5" />
                        {t("chat.openInReader")}
                      </Link>
                    </Button>
                  </article>
                );
              })}
            </div>
          </section>
        </div>
      </ScrollArea>
    </aside>
  );
}

function FolderGroup({
  source,
  files,
  visibleFiles,
  selected,
  onToggle,
  disabled = false,
}: {
  source: Source;
  files: Document[];
  visibleFiles: Document[];
  selected: (document: Document) => boolean;
  onToggle: (documents: Document[], checked: boolean) => void;
  disabled?: boolean;
}) {
  const { t } = useTranslation();
  const [open, setOpen] = useState(true);
  const selectedCount = files.filter(selected).length;
  const checked = !disabled && files.length > 0 && selectedCount === files.length;
  const indeterminate = !disabled && selectedCount > 0 && !checked;

  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <div className={cn("flex items-center gap-1 rounded-md px-1 py-1", disabled && "opacity-50")}>
        <Checkbox
          checked={indeterminate ? "indeterminate" : checked}
          onCheckedChange={(value) => onToggle(files, value === true)}
          disabled={disabled}
          aria-label={t("chat.includeAllIn", { path: shortPath(source.path) })}
        />
        <CollapsibleTrigger asChild>
          <button type="button" className="flex min-w-0 flex-1 items-center gap-2 text-left">
            {open ? (
              <ChevronDownIcon className="size-3.5 shrink-0 text-muted-foreground" />
            ) : (
              <ChevronRightIcon className="size-3.5 shrink-0 text-muted-foreground" />
            )}
            <FolderIcon className="size-3.5 shrink-0 text-muted-foreground" />
            <div className="min-w-0 flex-1">
              <p className="truncate text-xs font-medium">{shortPath(source.path)}</p>
              <p className="truncate text-[0.6875rem] text-muted-foreground">
                {t("chat.folderStats", {
                  selected: selectedCount,
                  files: files.length,
                  indexed: source.indexed_count,
                  total: source.document_count,
                })}
              </p>
            </div>
          </button>
        </CollapsibleTrigger>
      </div>
      <CollapsibleContent>
        <div className="ml-6 space-y-0.5 border-l border-border pl-2">
          {visibleFiles.length ? (
            visibleFiles.map((document) => (
              <FileRow
                key={document.id}
                document={document}
                selected={selected(document)}
                disabled={disabled}
                onToggle={(checked) => onToggle([document], checked)}
              />
            ))
          ) : (
            <p className="py-1 text-[0.6875rem] text-muted-foreground">{t("chat.noMatchingFiles")}</p>
          )}
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
}

function FileRow({
  document,
  selected,
  disabled,
  onToggle,
}: {
  document: Document;
  selected: boolean;
  disabled: boolean;
  onToggle: (checked: boolean) => void;
}) {
  const { t } = useTranslation();
  return (
    <div
      className={cn(
        "flex items-center gap-2 rounded-md px-2 py-1.5 transition-colors hover:bg-sidebar-accent/60",
        disabled && "cursor-not-allowed opacity-45",
      )}
    >
      <Checkbox
        checked={selected}
        onCheckedChange={(value) => onToggle(value === true)}
        disabled={disabled}
        aria-label={t("chat.includeFile", { title: document.title })}
      />
      <FileTextIcon className="size-3.5 shrink-0 text-muted-foreground" />
      <div className="min-w-0 flex-1">
        <p className="truncate text-xs font-medium">{document.title}</p>
        <p className="truncate text-[0.6875rem] text-muted-foreground">
          {t(`documentStatus.${document.status}`)} · {relativeTime(document.mtime)}
        </p>
      </div>
    </div>
  );
}

function LibraryFolderGroup({
  name,
  files,
  selected,
  onToggle,
}: {
  name: string;
  files: Document[];
  selected: (document: Document) => boolean;
  onToggle: (documents: Document[], checked: boolean) => void;
}) {
  const [open, setOpen] = useState(false);
  const count = files.filter(selected).length;
  const checked = count === files.length;
  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <div className="flex items-center gap-1 rounded-md border border-border bg-muted/30 px-2 py-1.5">
        <Checkbox
          checked={count > 0 && !checked ? "indeterminate" : checked}
          onCheckedChange={(value) => onToggle(files, value === true)}
          aria-label={name}
        />
        <CollapsibleTrigger asChild>
          <button type="button" className="flex min-w-0 flex-1 items-center gap-2 text-left">
            {open ? (
              <ChevronDownIcon className="size-3.5 shrink-0 text-muted-foreground" />
            ) : (
              <ChevronRightIcon className="size-3.5 shrink-0 text-muted-foreground" />
            )}
            <FolderIcon className="size-4 shrink-0 text-primary" />
            <span className="truncate text-xs font-medium">{name}</span>
            <span className="text-[0.6875rem] text-muted-foreground">
              {count}/{files.length}
            </span>
          </button>
        </CollapsibleTrigger>
      </div>
      <CollapsibleContent>
        <div className="ml-6 space-y-0.5 border-l border-border pl-2">
          {files.map((document) => (
            <FileRow
              key={document.id}
              document={document}
              selected={selected(document)}
              disabled={false}
              onToggle={(c) => onToggle([document], c)}
            />
          ))}
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
}
