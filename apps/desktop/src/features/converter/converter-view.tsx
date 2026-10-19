import { useState, useSyncExternalStore } from "react";
import { useTranslation } from "react-i18next";
import { Link } from "@tanstack/react-router";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { open } from "@tauri-apps/plugin-dialog";
import {
  BookOpenTextIcon,
  CheckCircle2Icon,
  ChevronLeftIcon,
  ChevronRightIcon,
  FileUpIcon,
  LoaderCircleIcon,
  LockKeyholeIcon,
  SearchIcon,
  SparklesIcon,
  XIcon,
  WandSparklesIcon,
} from "lucide-react";

import { Page, PageBody, PageHeader } from "@/components/shell/page";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Textarea } from "@/components/ui/textarea";
import {
  api,
  streamSlideConversion,
  type ConversionEvent,
  type Document,
  type StreamHandle,
} from "@/lib/ipc";
import { LANGUAGES, LANGUAGE_NAMES, currentLanguage, type Language } from "@/lib/i18n";
import { connectionsQuery, keys } from "@/lib/queries";
import { cn } from "@/lib/utils";

const PAGE_SIZE = 8;

type PresentationStatus = "generating" | "saved" | "error";
type PresentationRow = {
  index: number;
  title: string;
  status: PresentationStatus;
  output: string;
  savedPath: string | null;
  errorMessage: string | null;
};
type LocalSlide = { path: string; title: string; ext: string };

type RunState = { running: boolean; rows: PresentationRow[]; error: string | null };

// Module-level, not component state: a conversion keeps streaming while the user
// is on another page, and its progress must still be here when they come back.
let run: RunState = { running: false, rows: [], error: null };
let runStream: StreamHandle | null = null;
const runListeners = new Set<() => void>();

function setRun(patch: Partial<RunState> | ((current: RunState) => Partial<RunState>)) {
  run = { ...run, ...(typeof patch === "function" ? patch(run) : patch) };
  for (const listener of runListeners) listener();
}

function subscribeRun(listener: () => void) {
  runListeners.add(listener);
  return () => runListeners.delete(listener);
}

const setRows = (next: PresentationRow[] | ((rows: PresentationRow[]) => PresentationRow[])) =>
  setRun((current) => ({ rows: typeof next === "function" ? next(current.rows) : next }));
const setRunning = (running: boolean) => setRun({ running });
const setRequestError = (error: string | null) => setRun({ error });

function readableExtension(document: Document) {
  return document.ext.replace(".", "").toUpperCase() || "FILE";
}

function localSlideFromPath(path: string): LocalSlide {
  const filename = path.split(/[\\/]/).pop() ?? path;
  const dot = filename.lastIndexOf(".");
  return {
    path,
    title: dot > 0 ? filename.slice(0, dot) : filename,
    ext: dot > 0 ? filename.slice(dot + 1).toUpperCase() : "FILE",
  };
}

function ConverterStatus({ running, hasError }: { running: boolean; hasError: boolean }) {
  const { t } = useTranslation();
  return (
    <div className="flex items-center gap-2 text-xs text-muted-foreground">
      {running ? (
        <LoaderCircleIcon className="size-3.5 animate-spin text-primary" />
      ) : hasError ? (
        <span className="size-2 rounded-full bg-status-error" />
      ) : (
        <span className="size-2 rounded-full bg-status-idle" />
      )}
      <span>
        {running ? t("converter.running") : hasError ? t("converter.failed") : t("converter.ready")}
      </span>
    </div>
  );
}

function SlideRow({
  document,
  checked,
  disabled,
  onCheckedChange,
}: {
  document: Document;
  checked: boolean;
  disabled: boolean;
  onCheckedChange: (checked: boolean) => void;
}) {
  const { t } = useTranslation();
  return (
    <label
      className={cn(
        "group flex cursor-pointer items-start gap-3 border-b border-border px-4 py-3 last:border-0",
        "transition-colors hover:bg-muted/45",
        disabled && "cursor-not-allowed opacity-55",
      )}
    >
      <Checkbox
        checked={checked}
        disabled={disabled}
        onCheckedChange={(value) => onCheckedChange(value === true)}
        className="mt-0.5"
      />
      <BookOpenTextIcon className="mt-0.5 size-4 shrink-0 text-primary/75" />
      <span className="min-w-0 flex-1">
        <span className="block truncate text-sm font-medium">{document.title}</span>
        <span className="mt-0.5 block truncate font-mono text-[0.65rem] text-muted-foreground">
          {readableExtension(document)} · {t("converter.pages", { count: document.n_pages ?? 0 })} · {document.path}
        </span>
      </span>
      <ChevronRightIcon className="mt-0.5 size-3.5 text-muted-foreground/45 transition-transform group-hover:translate-x-0.5" />
    </label>
  );
}

export function ConverterView() {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [uploadedSlides, setUploadedSlides] = useState<LocalSlide[]>([]);
  const [researchQuery, setResearchQuery] = useState("");
  const [outputTitle, setOutputTitle] = useState("");
  const [language, setLanguage] = useState<Language>(currentLanguage);
  const [depth, setDepth] = useState<"standard" | "deep">("deep");
  const { running, rows, error: requestError } = useSyncExternalStore(subscribeRun, () => run);

  const connections = useQuery(connectionsQuery);
  const documents = useQuery({
    queryKey: keys.documents({ limit: 500 }),
    queryFn: () => api.listDocuments({ limit: 500 }),
  });
  const active = connections.data?.find((connection) => connection.active);
  const items = documents.data?.items ?? [];
  const [page, setPage] = useState(0);
  const pageCount = Math.max(1, Math.ceil(items.length / PAGE_SIZE));
  const pageItems = items.slice(Math.min(page, pageCount - 1) * PAGE_SIZE, (Math.min(page, pageCount - 1) + 1) * PAGE_SIZE);
  const selected = items.filter((document) => selectedIds.includes(document.id));
  const selectedCount = selected.length + uploadedSlides.length;
  const canConvert = Boolean(active && selectedCount && !running);

  const toggle = (id: string, checked: boolean) => {
    setSelectedIds((current) =>
      checked ? [...current, id] : current.filter((selectedId) => selectedId !== id),
    );
  };

  const updateRow = (index: number, patch: Partial<PresentationRow>) => {
    setRows((current) =>
      current.map((row) => (row.index === index ? { ...row, ...patch } : row)),
    );
  };

  const pickSlides = async () => {
    try {
      const picked = await open({
        directory: false,
        multiple: true,
        title: t("converter.pickTitle"),
        filters: [{ name: t("converter.slides"), extensions: ["pdf", "pptx"] }],
      });
      const paths = Array.isArray(picked) ? picked : picked ? [picked] : [];
      if (!paths.length) return;
      setUploadedSlides((current) => {
        const known = new Set(current.map((file) => file.path));
        return [...current, ...paths.filter((path) => !known.has(path)).map(localSlideFromPath)];
      });
      setRequestError(null);
    } catch (cause) {
      setRequestError(String(cause));
    }
  };

  const handleEvent = (event: ConversionEvent) => {
    if (event.event === "conversion_start") {
      setRows((current) => [
        ...current,
        {
          index: event.data.presentation_index,
          title: event.data.title,
          status: "generating",
          output: "",
          savedPath: null,
          errorMessage: null,
        },
      ]);
    } else if (event.event === "token") {
      setRows((current) => {
        const last = current[current.length - 1];
        if (!last) return current;
        return current.map((row) =>
          row.index === last.index
            ? { ...row, status: "generating", output: row.output + event.data.text }
            : row,
        );
      });
    } else if (event.event === "conversion_saved") {
      updateRow(event.data.presentation_index, { status: "saved", savedPath: event.data.path });
    } else if (event.event === "presentation_error") {
      const existing = run.rows.some((row) => row.index === event.data.presentation_index);
      if (!existing) {
        setRows((current) => [
          ...current,
          {
            index: event.data.presentation_index,
            title: event.data.title,
            status: "error",
            output: "",
            savedPath: null,
            errorMessage: event.data.message,
          },
        ]);
      } else {
        updateRow(event.data.presentation_index, { status: "error", errorMessage: event.data.message });
      }
    } else if (event.event === "conversion_done") {
      setRunning(false);
      void queryClient.invalidateQueries({ queryKey: ["documents"] });
      void queryClient.invalidateQueries({ queryKey: keys.sources });
    } else if (event.event === "error") {
      setRunning(false);
      setRequestError(event.data.message);
    }
  };

  const convert = () => {
    if (!canConvert) return;
    setRunning(true);
    setRows([]);
    setRequestError(null);
    runStream = streamSlideConversion(
      {
        slide_ids: selectedIds,
        file_paths: uploadedSlides.map((file) => file.path),
        research_query: researchQuery.trim() || null,
        output_title: outputTitle.trim() || null,
        language,
        depth,
      },
      {
        onEvent: handleEvent,
        // A stream that ends without conversion_done (sidecar error) must not spin forever.
        onClosed: () => setRunning(false),
        onFailed: (message) => {
          setRunning(false);
          setRequestError(message);
        },
      },
    );
  };

  const stop = () => {
    void runStream?.cancel();
    setRunning(false);
  };

  return (
    <Page>
      <PageHeader
        title={t("nav.convert")}
        description={active ? `${active.name} · ${active.model_id}` : t("chat.noModel")}
        actions={
          <>
            <ConverterStatus running={running} hasError={Boolean(requestError)} />
            <Button asChild variant="ghost" size="sm" className="h-8">
              <Link to="/chat">{t("converter.openChat")}</Link>
            </Button>
          </>
        }
      />

      <PageBody>
        <main className="mx-auto w-full max-w-6xl space-y-5 px-6 py-6">
          <p className="max-w-2xl text-sm leading-6 text-muted-foreground">{t("converter.lead")}</p>

          {!active ? (
            <Alert variant="destructive">
              <LockKeyholeIcon />
              <AlertTitle>{t("converter.noModelTitle")}</AlertTitle>
              <AlertDescription>
                {t("converter.noModelBody")}
                <Link
                  to="/settings/$section"
                  params={{ section: "connections" }}
                  className="mt-2 inline-flex font-medium underline underline-offset-4"
                >
                  {t("converter.goToConnections")}
                </Link>
              </AlertDescription>
            </Alert>
          ) : null}

          <div className="grid gap-5 xl:grid-cols-[minmax(0,1.25fr)_minmax(20rem,0.75fr)]">
            <section className="overflow-hidden rounded-xl border border-border bg-card">
              <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border px-4 py-3">
                <h2 className="text-sm font-semibold">{t("converter.sources")}</h2>
                <div className="flex items-center gap-2">
                  <Badge variant="secondary" className="shrink-0 font-mono text-[0.65rem]">
                    {t("converter.selected", { count: selectedCount })}
                  </Badge>
                  <Button
                    type="button"
                    variant="secondary"
                    size="sm"
                    className="h-7"
                    disabled={!active || running}
                    onClick={pickSlides}
                  >
                    <FileUpIcon className="size-3.5" />
                    {t("converter.upload")}
                  </Button>
                </div>
              </div>

              {documents.isLoading ? (
                <div className="space-y-2 p-4">
                  <Skeleton className="h-12 w-full" />
                  <Skeleton className="h-12 w-full" />
                  <Skeleton className="h-12 w-full" />
                </div>
              ) : items.length === 0 && uploadedSlides.length === 0 ? (
                <div className="grid min-h-48 place-items-center p-8 text-center">
                  <div>
                    <BookOpenTextIcon className="mx-auto size-7 text-muted-foreground/50" />
                    <p className="mt-3 text-sm font-medium">{t("converter.emptyTitle")}</p>
                    <p className="mt-1 max-w-xs text-xs leading-5 text-muted-foreground">
                      {t("converter.emptyHint")}
                    </p>
                  </div>
                </div>
              ) : (
                <div>
                  {uploadedSlides.map((file) => (
                    <div
                      key={file.path}
                      className="flex items-start gap-3 border-b border-primary/15 bg-primary/5 px-4 py-3"
                    >
                      <FileUpIcon className="mt-0.5 size-4 shrink-0 text-primary" />
                      <span className="min-w-0 flex-1">
                        <span className="block truncate text-sm font-medium">{file.title}</span>
                        <span className="mt-0.5 block truncate font-mono text-[0.65rem] text-muted-foreground">
                          {file.ext} · {t("converter.conversionOnly")} · {file.path}
                        </span>
                      </span>
                      <button
                        type="button"
                        className="rounded-md p-1 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
                        aria-label={t("converter.removeNamed", { title: file.title })}
                        onClick={() =>
                          setUploadedSlides((current) =>
                            current.filter((item) => item.path !== file.path),
                          )
                        }
                      >
                        <XIcon className="size-3.5" />
                      </button>
                    </div>
                  ))}
                  {pageItems.map((document) => (
                    <SlideRow
                      key={document.id}
                      document={document}
                      checked={selectedIds.includes(document.id)}
                      disabled={!active || running}
                      onCheckedChange={(checked) => toggle(document.id, checked)}
                    />
                  ))}
                </div>
              )}

              {pageCount > 1 ? (
                <div className="flex items-center justify-between border-t border-border px-4 py-2">
                  <span className="font-mono text-[0.65rem] text-muted-foreground tabular-nums">
                    {t("converter.pageOf", { page: Math.min(page, pageCount - 1) + 1, total: pageCount })}
                  </span>
                  <div className="flex gap-1">
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon"
                      className="size-7"
                      aria-label={t("converter.prevPage")}
                      disabled={page <= 0}
                      onClick={() => setPage((current) => Math.max(0, current - 1))}
                    >
                      <ChevronLeftIcon className="size-4" />
                    </Button>
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon"
                      className="size-7"
                      aria-label={t("converter.nextPage")}
                      disabled={page >= pageCount - 1}
                      onClick={() => setPage((current) => Math.min(pageCount - 1, current + 1))}
                    >
                      <ChevronRightIcon className="size-4" />
                    </Button>
                  </div>
                </div>
              ) : null}
            </section>

            <section className="rounded-xl border border-border bg-card p-5">
              <h2 className="text-sm font-semibold">{t("converter.brief")}</h2>
              <div className="mt-4 space-y-4">
                <div className="space-y-1.5">
                  <Label htmlFor="research-query">{t("converter.focus")}</Label>
                  <Textarea
                    id="research-query"
                    value={researchQuery}
                    onChange={(event) => setResearchQuery(event.target.value)}
                    placeholder={t("converter.focusPlaceholder")}
                    className="min-h-24 resize-none text-sm"
                    disabled={!active || running}
                  />
                  <p className="text-[0.68rem] leading-4 text-muted-foreground">
                    {t("converter.focusHint")}
                  </p>
                </div>
                <div className="space-y-1.5">
                  <Label htmlFor="output-title">{t("converter.outputTitle")}</Label>
                  <Input
                    id="output-title"
                    value={outputTitle}
                    onChange={(event) => setOutputTitle(event.target.value)}
                    placeholder={
                      selected[0]?.title ?? uploadedSlides[0]?.title ?? t("converter.outputTitlePlaceholder")
                    }
                    disabled={!active || running}
                  />
                </div>
                <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-1 2xl:grid-cols-2">
                  <div className="space-y-1.5">
                    <Label>{t("converter.language")}</Label>
                    <Select value={language} onValueChange={(value) => setLanguage(value as Language)}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {LANGUAGES.map((code) => (
                          <SelectItem key={code} value={code}>
                            {LANGUAGE_NAMES[code]}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-1.5">
                    <Label>{t("converter.depth")}</Label>
                    <Select value={depth} onValueChange={(value) => setDepth(value as "standard" | "deep")}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="deep">{t("converter.depthDeep")}</SelectItem>
                        <SelectItem value="standard">{t("converter.depthStandard")}</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
                <Button className="mt-1 w-full" disabled={!canConvert} onClick={convert}>
                  <WandSparklesIcon className="size-4" />
                  {t("converter.generate")}
                </Button>
                {running ? (
                  <Button variant="ghost" size="sm" className="w-full" onClick={stop}>
                    {t("converter.stop")}
                  </Button>
                ) : (
                  <p className="text-center text-[0.68rem] leading-4 text-muted-foreground">
                    {t("converter.savedNote")}
                  </p>
                )}
              </div>
            </section>
          </div>

          {requestError ? (
            <Alert variant="destructive">
              <SearchIcon />
              <AlertTitle>{t("converter.stopped")}</AlertTitle>
              <AlertDescription>{requestError}</AlertDescription>
            </Alert>
          ) : null}

          {rows.length ? (
            <div className="space-y-5">
              {rows.map((row) => (
                <section key={row.index} className="space-y-3">
                  <article className="overflow-hidden rounded-xl border border-border bg-card">
                    <div className="flex items-center justify-between border-b border-border px-4 py-3">
                      <div className="flex items-center gap-2">
                        <SparklesIcon className="size-4 text-primary" />
                        <h2 className="text-sm font-semibold">{row.title}</h2>
                      </div>
                      <Badge
                        variant={row.status === "error" ? "destructive" : "outline"}
                        className="font-mono text-[0.6rem]"
                      >
                        {row.status === "error" ? t("documentStatus.error") : ".md"}
                      </Badge>
                    </div>
                    <div className="max-h-[28rem] overflow-auto p-5">
                      {row.status === "error" ? (
                        <p className="text-sm text-status-error">{row.errorMessage}</p>
                      ) : row.output ? (
                        <pre className="selectable whitespace-pre-wrap font-sans text-sm leading-7 text-foreground">
                          {row.output}
                        </pre>
                      ) : (
                        <div className="space-y-3">
                          <Skeleton className="h-5 w-3/4" />
                          <Skeleton className="h-4 w-full" />
                          <Skeleton className="h-4 w-11/12" />
                          <Skeleton className="h-4 w-2/3" />
                        </div>
                      )}
                    </div>
                  </article>
                  {row.savedPath ? (
                    <div className="rounded-xl border border-status-ok/25 bg-status-ok/6 p-4">
                      <div className="flex items-center gap-2 text-status-ok">
                        <CheckCircle2Icon className="size-4" />
                        <p className="text-sm font-semibold">{t("converter.fileReady")}</p>
                      </div>
                      <p className="mt-2 break-all font-mono text-[0.65rem] leading-5 text-muted-foreground">
                        {row.savedPath}
                      </p>
                      <Link
                        to="/library"
                        className="mt-3 inline-flex text-xs font-medium text-primary underline underline-offset-4"
                      >
                        {t("converter.openInLibrary")}
                      </Link>
                    </div>
                  ) : null}
                </section>
              ))}
            </div>
          ) : null}
        </main>
      </PageBody>
    </Page>
  );
}
