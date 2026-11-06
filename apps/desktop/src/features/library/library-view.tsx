import { useState } from "react";
import { useTranslation } from "react-i18next";
import { Link } from "@tanstack/react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  ArrowDownIcon,
  ArrowUpIcon,
  BookOpenIcon,
  ChevronsUpDownIcon,
  FolderIcon,
  FolderInputIcon,
  FolderMinusIcon,
  ListFilterIcon,
  PencilIcon,
  RefreshCwIcon,
  SearchIcon,
  Trash2Icon,
} from "lucide-react";
import { toast } from "sonner";

import { Page, PageBody, PageHeader } from "@/components/shell/page";
import { StatusChip, documentTone } from "@/components/status";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Skeleton } from "@/components/ui/skeleton";
import { IconTooltip } from "@/components/ui/tooltip";
import { NameInput } from "@/features/library/folder-list";
import { nextSort, visibleRows, type Sort, type SortKey } from "@/features/library/rows";
import { SourceSidebar } from "@/features/library/source-sidebar";
import { errorText } from "@/lib/errors";
import { bytes, relativeTime } from "@/lib/format";
import { api, type DocumentStatus } from "@/lib/ipc";
import { useJobs } from "@/lib/jobs-context";
import { foldersQuery, keys } from "@/lib/queries";
import { cn } from "@/lib/utils";

function toggled<T>(set: Set<T>, value: T): Set<T> {
  const next = new Set(set);
  if (!next.delete(value)) next.add(value);
  return next;
}

/** `.pdf` → `PDF` */
const extLabel = (ext: string) => ext.replace(/^\./, "").toUpperCase() || "—";

function SortHead({
  column,
  sort,
  onSort,
  right,
  className,
  children,
}: {
  column: SortKey;
  sort: Sort;
  onSort: (column: SortKey) => void;
  right?: boolean;
  className?: string;
  children: React.ReactNode;
}) {
  const active = sort.key === column;
  const Icon = !active ? ChevronsUpDownIcon : sort.desc ? ArrowDownIcon : ArrowUpIcon;
  return (
    <TableHead
      className={cn(right && "text-right", className)}
      aria-sort={active ? (sort.desc ? "descending" : "ascending") : "none"}
    >
      <button
        type="button"
        onClick={() => onSort(column)}
        className={cn(
          "group/sort inline-flex items-center gap-1 rounded-sm hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring/50 focus-visible:outline-none",
          right && "flex-row-reverse",
          !active && "text-muted-foreground",
        )}
      >
        {children}
        <Icon
          className={cn(
            "size-3 shrink-0",
            !active && "opacity-0 group-hover/sort:opacity-60 group-focus-visible/sort:opacity-60",
          )}
        />
      </button>
    </TableHead>
  );
}

export function LibraryView() {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const [sourceId, setSourceId] = useState<string | null>(null);
  const [folderId, setFolderId] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [sort, setSort] = useState<Sort>({ key: "title", desc: false });
  const [statuses, setStatuses] = useState<Set<DocumentStatus>>(new Set());
  const [exts, setExts] = useState<Set<string>>(new Set());
  const [renaming, setRenaming] = useState<string | null>(null);
  const { active } = useJobs();
  const folders = useQuery(foldersQuery);

  const params = {
    source_id: sourceId ?? undefined,
    folder_id: folderId ?? undefined,
    q: search || undefined,
    limit: 500,
  };
  const documents = useQuery({
    queryKey: keys.documents(params),
    queryFn: () => api.listDocuments(params),
    refetchInterval: 5000,
  });

  const invalidate = () => {
    void queryClient.invalidateQueries({ queryKey: ["documents"] });
    void queryClient.invalidateQueries({ queryKey: keys.sources });
  };

  const reindex = useMutation({
    mutationFn: (docIds: string[]) => api.rebuild({ doc_ids: docIds }),
    onSuccess: () => toast.success(t("library.reindexStarted")),
    onError: (error: Error) => toast.error(t("library.reindexFailed"), { description: errorText(error) }),
  });

  const remove = useMutation({
    mutationFn: (id: string) => api.removeDocument(id),
    onSuccess: invalidate,
    onError: (error: Error) => toast.error(t("library.removeFailed"), { description: errorText(error) }),
  });

  const rename = useMutation({
    mutationFn: ({ id, title }: { id: string; title: string }) => api.renameDocument(id, title),
    onSuccess: (_, { id }) => {
      void queryClient.invalidateQueries({ queryKey: ["documents"] });
      void queryClient.invalidateQueries({ queryKey: keys.document(id) });
      void queryClient.invalidateQueries({ queryKey: keys.content(id) });
    },
    onError: (error: Error) => toast.error(t("library.renameFailed"), { description: errorText(error) }),
  });

  const move = useMutation({
    mutationFn: ({ docId, folderId }: { docId: string; folderId: string | null }) =>
      api.moveToFolder(docId, folderId),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: keys.folders });
      void queryClient.invalidateQueries({ queryKey: ["documents"] });
    },
    onError: (error: Error) => toast.error(t("library.moveFailed"), { description: errorText(error) }),
  });

  const folderOf = new Map<string, string>();
  for (const folder of folders.data ?? []) {
    for (const docId of folder.doc_ids) folderOf.set(docId, folder.id);
  }

  const items = documents.data?.items ?? [];
  const errored = items.filter((document) => document.status === "error");
  const statusLabel = (status: DocumentStatus) => t(`documentStatus.${status}`);
  const rows = visibleRows(items, { statuses, exts }, sort, statusLabel);
  const filterCount = statuses.size + exts.size;
  // Offer what this view holds, plus anything still ticked so it can be unticked.
  const statusOptions = [...new Set([...items.map((document) => document.status), ...statuses])].sort(
    (left, right) => statusLabel(left).localeCompare(statusLabel(right)),
  );
  const extOptions = [...new Set([...items.map((document) => document.ext), ...exts])].sort();
  const onSort = (column: SortKey) => setSort((current) => nextSort(current, column));

  return (
    <>
      <SourceSidebar
        selected={sourceId}
        onSelect={(id) => {
          setSourceId(id);
          setFolderId(null);
        }}
        selectedFolder={folderId}
        onSelectFolder={(id) => {
          setFolderId(id);
          setSourceId(null);
        }}
      />

      <Page>
        <PageHeader
          title={folders.data?.find((folder) => folder.id === folderId)?.name ?? t("nav.library")}
          description={
            documents.data
              ? t("library.documentCount", { count: documents.data.total }) +
                (active.length ? ` · ${t("library.jobsRunning", { count: active.length })}` : "")
              : t("common.loading")
          }
          actions={
            <>
              <div className="relative">
                <SearchIcon className="pointer-events-none absolute top-1/2 left-2.5 size-3.5 -translate-y-1/2 text-muted-foreground" />
                <Input
                  value={search}
                  onChange={(event) => setSearch(event.target.value)}
                  placeholder={t("library.filter")}
                  className="h-8 w-56 pl-8 text-xs"
                />
              </div>
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button
                    variant="secondary"
                    size="sm"
                    className="h-8"
                    disabled={items.length === 0 && filterCount === 0}
                  >
                    <ListFilterIcon className="size-3.5" />
                    {t("library.filters")}
                    {filterCount ? (
                      <span className="rounded-full bg-primary px-1.5 text-[0.625rem] leading-4 font-semibold text-primary-foreground tabular-nums">
                        {filterCount}
                      </span>
                    ) : null}
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end" className="w-48">
                  <DropdownMenuLabel className="text-xs text-muted-foreground">
                    {t("library.colStatus")}
                  </DropdownMenuLabel>
                  {statusOptions.map((status) => (
                    <DropdownMenuCheckboxItem
                      key={status}
                      checked={statuses.has(status)}
                      onCheckedChange={() => setStatuses((current) => toggled(current, status))}
                      onSelect={(event) => event.preventDefault()}
                    >
                      {statusLabel(status)}
                    </DropdownMenuCheckboxItem>
                  ))}
                  <DropdownMenuSeparator />
                  <DropdownMenuLabel className="text-xs text-muted-foreground">
                    {t("library.fileType")}
                  </DropdownMenuLabel>
                  {extOptions.map((ext) => (
                    <DropdownMenuCheckboxItem
                      key={ext}
                      checked={exts.has(ext)}
                      onCheckedChange={() => setExts((current) => toggled(current, ext))}
                      onSelect={(event) => event.preventDefault()}
                    >
                      {extLabel(ext)}
                    </DropdownMenuCheckboxItem>
                  ))}
                  {filterCount ? (
                    <>
                      <DropdownMenuSeparator />
                      <DropdownMenuItem
                        onSelect={() => {
                          setStatuses(new Set());
                          setExts(new Set());
                        }}
                      >
                        {t("library.clearFilters")}
                      </DropdownMenuItem>
                    </>
                  ) : null}
                </DropdownMenuContent>
              </DropdownMenu>
              <Button
                variant="secondary"
                size="sm"
                className="h-8"
                disabled={rows.length === 0 || reindex.isPending}
                onClick={() => reindex.mutate(rows.map((document) => document.id))}
              >
                <RefreshCwIcon className="size-3.5" />
                {t("library.reindexAll")}
              </Button>
            </>
          }
        />

        <PageBody>
          {errored.length > 0 ? (
            <div className="border-b border-status-error/25 bg-status-error/8 px-5 py-2 text-xs text-status-error">
              {t("library.failedToIndex", { count: errored.length })}
              {errored[0].error ? ` ${t("library.firstError", { error: errored[0].error })}` : ""}
            </div>
          ) : null}

          {documents.isLoading ? (
            <div className="space-y-2 p-5">
              <Skeleton className="h-9 w-full" />
              <Skeleton className="h-9 w-full" />
              <Skeleton className="h-9 w-full" />
            </div>
          ) : rows.length === 0 && (filterCount > 0 || search) ? (
            <div className="grid h-full place-items-center p-10 text-center">
              <div className="max-w-sm">
                <p className="text-sm font-medium">{t("library.noMatches")}</p>
                <p className="mt-1 text-xs text-muted-foreground">{t("library.noMatchesHint")}</p>
              </div>
            </div>
          ) : items.length === 0 && folderId ? (
            <div className="grid h-full place-items-center p-10 text-center">
              <div className="max-w-sm">
                <p className="text-sm font-medium">{t("library.folderEmpty")}</p>
                <p className="mt-1 text-xs text-muted-foreground">
                  {t("library.folderEmptyHint")}
                </p>
              </div>
            </div>
          ) : items.length === 0 ? (
            <div className="grid h-full place-items-center p-10 text-center">
              <div className="max-w-sm">
                <p className="text-sm font-medium">{t("library.emptyTitle")}</p>
                <p className="mt-1 text-xs text-muted-foreground">
                  {t("library.emptyHint")}
                </p>
              </div>
            </div>
          ) : (
            <Table>
              <TableHeader className="sticky top-0 z-10 bg-background">
                <TableRow>
                  <SortHead column="title" sort={sort} onSort={onSort} className="w-[42%]">
                    {t("library.colDocument")}
                  </SortHead>
                  <SortHead column="status" sort={sort} onSort={onSort}>
                    {t("library.colStatus")}
                  </SortHead>
                  <SortHead column="n_pages" sort={sort} onSort={onSort} right>
                    {t("library.colPages")}
                  </SortHead>
                  <SortHead column="n_chunks" sort={sort} onSort={onSort} right>
                    {t("library.colPassages")}
                  </SortHead>
                  <SortHead column="size_bytes" sort={sort} onSort={onSort} right>
                    {t("library.colSize")}
                  </SortHead>
                  <SortHead column="mtime" sort={sort} onSort={onSort} right>
                    {t("library.colModified")}
                  </SortHead>
                  <TableHead className="w-24" />
                </TableRow>
              </TableHeader>
              <TableBody>
                {rows.map((document) => (
                  <TableRow key={document.id} className="group">
                    <TableCell className="max-w-0">
                      {renaming === document.id ? (
                        <NameInput
                          initial={document.title}
                          onSubmit={(title) => {
                            setRenaming(null);
                            if (title !== document.title) rename.mutate({ id: document.id, title });
                          }}
                          onCancel={() => setRenaming(null)}
                        />
                      ) : (
                        <p
                          className="truncate text-[0.8125rem] font-medium"
                          onDoubleClick={() => setRenaming(document.id)}
                        >
                          {document.title}
                        </p>
                      )}
                      <p className="truncate font-mono text-[0.6875rem] text-muted-foreground">
                        {document.path}
                      </p>
                    </TableCell>
                    <TableCell>
                      <StatusChip
                        tone={documentTone(document.status)}
                        label={t(`documentStatus.${document.status}`)}
                        pulse={!["indexed", "error", "skipped"].includes(document.status)}
                      />
                    </TableCell>
                    <TableCell className="text-right font-mono text-xs tabular-nums">
                      {document.n_pages ?? "—"}
                    </TableCell>
                    <TableCell className="text-right font-mono text-xs tabular-nums">
                      {document.n_chunks}
                    </TableCell>
                    <TableCell className="text-right font-mono text-xs tabular-nums">
                      {bytes(document.size_bytes)}
                    </TableCell>
                    <TableCell className="text-right text-xs whitespace-nowrap text-muted-foreground">
                      {relativeTime(document.mtime)}
                    </TableCell>
                    <TableCell>
                      <div className="flex justify-end gap-0.5 opacity-0 transition-opacity group-hover:opacity-100 focus-within:opacity-100 has-[[data-state=open]]:opacity-100">
                        <IconTooltip label={t("library.renameNamed", { name: document.title })}>
                          <Button
                            variant="ghost"
                            size="icon"
                            className="size-7"
                            aria-label={t("library.renameNamed", { name: document.title })}
                            onClick={() => setRenaming(document.id)}
                          >
                            <PencilIcon className="size-3.5" />
                          </Button>
                        </IconTooltip>
                        <DropdownMenu>
                          <IconTooltip label={t("library.moveToFolder")}>
                            <DropdownMenuTrigger asChild>
                              <Button
                                variant="ghost"
                                size="icon"
                                className="size-7"
                                aria-label={t("library.moveToFolder")}
                              >
                                <FolderInputIcon className="size-3.5" />
                              </Button>
                            </DropdownMenuTrigger>
                          </IconTooltip>
                          <DropdownMenuContent align="end" className="w-48">
                            {(folders.data ?? []).length === 0 ? (
                              <DropdownMenuLabel className="text-xs font-normal text-muted-foreground">
                                {t("library.createFolderFirst")}
                              </DropdownMenuLabel>
                            ) : null}
                            {(folders.data ?? []).map((folder) => (
                              <DropdownMenuItem
                                key={folder.id}
                                disabled={folderOf.get(document.id) === folder.id}
                                onSelect={() => move.mutate({ docId: document.id, folderId: folder.id })}
                              >
                                <FolderIcon className="size-3.5" />
                                <span className="truncate">{folder.name}</span>
                              </DropdownMenuItem>
                            ))}
                            {folderOf.has(document.id) ? (
                              <>
                                <DropdownMenuSeparator />
                                <DropdownMenuItem
                                  onSelect={() => move.mutate({ docId: document.id, folderId: null })}
                                >
                                  <FolderMinusIcon className="size-3.5" />
                                  {t("library.removeFromFolder")}
                                </DropdownMenuItem>
                              </>
                            ) : null}
                          </DropdownMenuContent>
                        </DropdownMenu>
                        <IconTooltip label={t("library.openNamed", { title: document.title })}>
                          <Button asChild variant="ghost" size="icon" className="size-7">
                            <Link
                              to="/reader/$docId"
                              params={{ docId: document.id }}
                              aria-label={t("library.openNamed", { title: document.title })}
                            >
                              <BookOpenIcon className="size-3.5" />
                            </Link>
                          </Button>
                        </IconTooltip>
                        <IconTooltip label={t("library.reindex")}>
                          <Button
                            variant="ghost"
                            size="icon"
                            className="size-7"
                            aria-label={t("library.reindex")}
                            onClick={() => reindex.mutate([document.id])}
                          >
                            <RefreshCwIcon className="size-3.5" />
                          </Button>
                        </IconTooltip>
                        <IconTooltip label={t("common.remove")}>
                          <Button
                            variant="ghost"
                            size="icon"
                            className="size-7"
                            aria-label={t("common.remove")}
                            onClick={() => remove.mutate(document.id)}
                          >
                            <Trash2Icon className="size-3.5" />
                          </Button>
                        </IconTooltip>
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </PageBody>
      </Page>
    </>
  );
}
