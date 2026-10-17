import { useState } from "react";
import { useTranslation } from "react-i18next";
import { Link } from "@tanstack/react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  BookOpenIcon,
  FolderIcon,
  FolderInputIcon,
  FolderMinusIcon,
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
import { SourceSidebar } from "@/features/library/source-sidebar";
import { errorText } from "@/lib/errors";
import { bytes, relativeTime } from "@/lib/format";
import { api } from "@/lib/ipc";
import { useJobs } from "@/lib/jobs-context";
import { foldersQuery, keys } from "@/lib/queries";

export function LibraryView() {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const [sourceId, setSourceId] = useState<string | null>(null);
  const [folderId, setFolderId] = useState<string | null>(null);
  const [search, setSearch] = useState("");
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
              <Button
                variant="secondary"
                size="sm"
                className="h-8"
                disabled={items.length === 0 || reindex.isPending}
                onClick={() => reindex.mutate(items.map((document) => document.id))}
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
                  <TableHead className="w-[42%]">{t("library.colDocument")}</TableHead>
                  <TableHead>{t("library.colStatus")}</TableHead>
                  <TableHead className="text-right">{t("library.colPages")}</TableHead>
                  <TableHead className="text-right">{t("library.colPassages")}</TableHead>
                  <TableHead className="text-right">{t("library.colSize")}</TableHead>
                  <TableHead className="text-right">{t("library.colModified")}</TableHead>
                  <TableHead className="w-24" />
                </TableRow>
              </TableHeader>
              <TableBody>
                {items.map((document) => (
                  <TableRow key={document.id} className="group">
                    <TableCell className="max-w-0">
                      <p className="truncate text-[0.8125rem] font-medium">{document.title}</p>
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
