import { useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { useMutation, useQuery } from "@tanstack/react-query";
import { openPath, revealItemInDir } from "@tauri-apps/plugin-opener";
import {
  ArrowDownIcon,
  ArrowUpIcon,
  BookOpenTextIcon,
  BookTextIcon,
  CheckCircle2Icon,
  FileTextIcon,
  FolderOpenIcon,
  GripVerticalIcon,
  LoaderCircleIcon,
  SearchIcon,
  TriangleAlertIcon,
  XIcon,
} from "lucide-react";

import { Page, PageBody, PageHeader } from "@/components/shell/page";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Skeleton } from "@/components/ui/skeleton";
import { errorText } from "@/lib/errors";
import { api, type Document } from "@/lib/ipc";
import { keys } from "@/lib/queries";
import { cn } from "@/lib/utils";

const STORAGE_KEY = "ibid.book";
const LIST_PARAMS = { ext: ".md", limit: 500 };

type Outline = { title: string; ids: string[] };

/** The last book, so a reconverted chapter is one click from a fresh PDF. */
function loadOutline(): Outline {
  try {
    const saved: unknown = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "null");
    if (saved && typeof saved === "object") {
      const { title, ids } = saved as Partial<Outline>;
      return {
        title: typeof title === "string" ? title : "",
        ids: Array.isArray(ids) ? ids.filter((id): id is string => typeof id === "string") : [],
      };
    }
  } catch {
    // Unreadable storage: start empty.
  }
  return { title: "", ids: [] };
}

function move<T>(items: T[], from: number, to: number): T[] {
  const next = [...items];
  const [item] = next.splice(from, 1);
  next.splice(to, 0, item);
  return next;
}

// Converted chapters live in the data dir's global-files folder.
const isConverted = (document: Document) => /[\\/]global-files[\\/]/.test(document.path);

export function BookView() {
  const { t } = useTranslation();
  const [outline, setOutline] = useState<Outline>(loadOutline);
  const [search, setSearch] = useState("");
  const [dragging, setDragging] = useState<number | null>(null);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(outline));
    } catch {
      // Not persisted; the outline still works for this session.
    }
  }, [outline]);

  const documents = useQuery({
    queryKey: keys.documents(LIST_PARAMS),
    queryFn: () => api.listDocuments(LIST_PARAMS),
  });
  const items = documents.data?.items ?? [];
  const byId = useMemo(() => new Map(items.map((document) => [document.id, document])), [items]);
  // Chapters whose document was removed from the library drop out silently.
  const chapters = outline.ids.flatMap((id) => byId.get(id) ?? []);
  const needle = search.trim().toLowerCase();
  const visible = needle
    ? items.filter((document) => document.title.toLowerCase().includes(needle))
    : items;

  const book = useMutation({
    mutationFn: () =>
      api.createBook({ title: outline.title.trim(), document_ids: chapters.map((d) => d.id) }),
  });
  const busy = book.isPending;
  const result = book.data;

  const setIds = (ids: string[]) => setOutline((current) => ({ ...current, ids }));
  const toggle = (id: string, checked: boolean) =>
    setIds(checked ? [...outline.ids, id] : outline.ids.filter((other) => other !== id));
  const shift = (index: number, to: number) =>
    setIds(move(chapters.map((d) => d.id), index, to));

  return (
    <Page>
      <PageHeader
        title={t("nav.book")}
        description={t("book.chapters", { count: chapters.length })}
        actions={
          busy ? (
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <LoaderCircleIcon className="size-3.5 animate-spin text-primary" />
              <span>{t("book.generating")}</span>
            </div>
          ) : null
        }
      />

      <PageBody>
        <main className="mx-auto w-full max-w-6xl space-y-5 px-6 py-6">
          <p className="max-w-2xl text-sm leading-6 text-muted-foreground">{t("book.lead")}</p>

          <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_minmax(22rem,1fr)]">
            <section className="overflow-hidden rounded-xl border border-border bg-card">
              <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border px-4 py-3">
                <h2 className="text-sm font-semibold">{t("book.available")}</h2>
                <div className="relative w-full max-w-56">
                  <SearchIcon className="pointer-events-none absolute top-1/2 left-2.5 size-3.5 -translate-y-1/2 text-muted-foreground" />
                  <Input
                    value={search}
                    onChange={(event) => setSearch(event.target.value)}
                    placeholder={t("book.search")}
                    aria-label={t("book.search")}
                    className="h-8 pl-8 text-sm"
                  />
                </div>
              </div>

              {documents.isLoading ? (
                <div className="space-y-2 p-4">
                  <Skeleton className="h-12 w-full" />
                  <Skeleton className="h-12 w-full" />
                  <Skeleton className="h-12 w-full" />
                </div>
              ) : items.length === 0 ? (
                <div className="grid min-h-48 place-items-center p-8 text-center">
                  <div>
                    <BookOpenTextIcon className="mx-auto size-7 text-muted-foreground/50" />
                    <p className="mt-3 text-sm font-medium">{t("book.emptyTitle")}</p>
                    <p className="mt-1 max-w-xs text-xs leading-5 text-muted-foreground">
                      {t("book.emptyHint")}
                    </p>
                  </div>
                </div>
              ) : visible.length === 0 ? (
                <p className="p-6 text-center text-xs text-muted-foreground">{t("book.noMatch")}</p>
              ) : (
                <div className="max-h-[32rem] overflow-auto">
                  {visible.map((document) => (
                    <label
                      key={document.id}
                      className={cn(
                        "flex cursor-pointer items-start gap-3 border-b border-border px-4 py-3 last:border-0",
                        "transition-colors hover:bg-muted/45",
                        busy && "cursor-not-allowed opacity-55",
                      )}
                    >
                      <Checkbox
                        checked={outline.ids.includes(document.id)}
                        disabled={busy}
                        onCheckedChange={(value) => toggle(document.id, value === true)}
                        className="mt-0.5"
                      />
                      <FileTextIcon className="mt-0.5 size-4 shrink-0 text-primary/75" />
                      <span className="min-w-0 flex-1">
                        <span className="block truncate text-sm font-medium">{document.title}</span>
                        <span className="mt-0.5 block truncate font-mono text-[0.65rem] text-muted-foreground">
                          {document.path}
                        </span>
                      </span>
                      {isConverted(document) ? (
                        <Badge variant="secondary" className="shrink-0 text-[0.6rem]">
                          {t("book.converted")}
                        </Badge>
                      ) : null}
                    </label>
                  ))}
                </div>
              )}
            </section>

            <section className="rounded-xl border border-border bg-card p-5">
              <h2 className="text-sm font-semibold">{t("book.outline")}</h2>
              <div className="mt-4 space-y-4">
                <div className="space-y-1.5">
                  <Label htmlFor="book-title">{t("book.title")}</Label>
                  <Input
                    id="book-title"
                    value={outline.title}
                    onChange={(event) =>
                      setOutline((current) => ({ ...current, title: event.target.value }))
                    }
                    placeholder={chapters[0]?.title ?? t("book.titlePlaceholder")}
                    disabled={busy}
                  />
                </div>

                {chapters.length === 0 ? (
                  <p className="rounded-lg border border-dashed border-border px-4 py-6 text-center text-xs leading-5 text-muted-foreground">
                    {t("book.outlineEmpty")}
                  </p>
                ) : (
                  <div className="space-y-1.5">
                    <ol className="space-y-1.5">
                      {chapters.map((document, index) => (
                        <li
                          key={document.id}
                          draggable={!busy}
                          onDragStart={() => setDragging(index)}
                          onDragEnd={() => setDragging(null)}
                          onDragOver={(event) => {
                            event.preventDefault();
                            if (dragging === null || dragging === index) return;
                            shift(dragging, index);
                            setDragging(index);
                          }}
                          className={cn(
                            "flex items-center gap-2 rounded-lg border border-border bg-background px-2 py-1.5",
                            dragging === index && "border-primary/40 bg-primary/5",
                          )}
                        >
                          <GripVerticalIcon className="size-4 shrink-0 cursor-grab text-muted-foreground/60" />
                          <span className="w-5 shrink-0 text-right font-mono text-xs text-muted-foreground tabular-nums">
                            {index + 1}
                          </span>
                          <span className="min-w-0 flex-1 truncate text-sm">{document.title}</span>
                          <Button
                            type="button"
                            variant="ghost"
                            size="icon"
                            className="size-7"
                            disabled={busy || index === 0}
                            aria-label={t("book.moveUp", { title: document.title })}
                            onClick={() => shift(index, index - 1)}
                          >
                            <ArrowUpIcon className="size-3.5" />
                          </Button>
                          <Button
                            type="button"
                            variant="ghost"
                            size="icon"
                            className="size-7"
                            disabled={busy || index === chapters.length - 1}
                            aria-label={t("book.moveDown", { title: document.title })}
                            onClick={() => shift(index, index + 1)}
                          >
                            <ArrowDownIcon className="size-3.5" />
                          </Button>
                          <Button
                            type="button"
                            variant="ghost"
                            size="icon"
                            className="size-7"
                            disabled={busy}
                            aria-label={t("book.remove", { title: document.title })}
                            onClick={() => toggle(document.id, false)}
                          >
                            <XIcon className="size-3.5" />
                          </Button>
                        </li>
                      ))}
                    </ol>
                    <p className="text-[0.68rem] leading-4 text-muted-foreground">{t("book.dragHint")}</p>
                  </div>
                )}

                <Button
                  className="w-full"
                  disabled={busy || chapters.length === 0}
                  onClick={() => book.mutate()}
                >
                  {busy ? (
                    <LoaderCircleIcon className="size-4 animate-spin" />
                  ) : (
                    <BookTextIcon className="size-4" />
                  )}
                  {busy ? t("book.generating") : t("book.generate")}
                </Button>
                <p className="text-center text-[0.68rem] leading-4 text-muted-foreground">
                  {t("book.firstRunNote")}
                </p>
              </div>
            </section>
          </div>

          {book.isError ? (
            <Alert variant="destructive">
              <TriangleAlertIcon />
              <AlertTitle>{t("book.failed")}</AlertTitle>
              <AlertDescription>{errorText(book.error)}</AlertDescription>
            </Alert>
          ) : null}

          {result ? (
            <div className="rounded-xl border border-status-ok/25 bg-status-ok/6 p-4">
              <div className="flex items-center gap-2 text-status-ok">
                <CheckCircle2Icon className="size-4" />
                <p className="text-sm font-semibold">{t("book.ready")}</p>
              </div>
              <p className="mt-2 break-all font-mono text-[0.65rem] leading-5 text-muted-foreground">
                {result.pdf_path}
              </p>
              {result.had_errors ? (
                <p className="mt-2 text-xs leading-5 text-status-warn">{t("book.hadErrors")}</p>
              ) : null}
              <div className="mt-3 flex flex-wrap gap-2">
                <Button size="sm" onClick={() => void openPath(result.pdf_path)}>
                  <BookOpenTextIcon className="size-3.5" />
                  {t("book.openPdf")}
                </Button>
                <Button
                  size="sm"
                  variant="secondary"
                  onClick={() => void revealItemInDir(result.tex_path)}
                >
                  <FolderOpenIcon className="size-3.5" />
                  {t("book.reveal")}
                </Button>
              </div>
            </div>
          ) : null}
        </main>
      </PageBody>
    </Page>
  );
}
