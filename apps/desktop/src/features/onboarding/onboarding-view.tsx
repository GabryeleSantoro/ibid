import { useMemo, useState } from "react";
import { useNavigate } from "@tanstack/react-router";
import { useTranslation } from "react-i18next";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { errorText } from "@/lib/errors";
import {
  ArrowRightIcon,
  BookOpenIcon,
  CheckIcon,
  CloudIcon,
  CpuIcon,
  DownloadIcon,
  FolderPlusIcon,
  Loader2Icon,
  LockIcon,
  QuoteIcon,
  ServerIcon,
} from "lucide-react";
import { toast } from "sonner";

import { StatusChip } from "@/components/status";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { AddSourceDialog } from "@/features/library/add-source-dialog";
import { ConnectionDialog } from "@/features/settings/connection-dialog";
import { LocalModelControl } from "@/features/settings/local-model";
import { bytes } from "@/lib/format";
import { useJobs } from "@/lib/jobs-context";
import { api, type HardwareInfo, type InstalledModel, type ModelRole } from "@/lib/ipc";
import {
  connectionsQuery,
  hardwareQuery,
  healthQuery,
  keys,
  modelsQuery,
  runtimeQuery,
  sourcesQuery,
} from "@/lib/queries";
import { cn } from "@/lib/utils";

const STEPS = [
  { id: "welcome" },
  { id: "hardware" },
  { id: "models" },
  { id: "connection" },
  { id: "library" },
  { id: "done" },
] as const;

type StepId = (typeof STEPS)[number]["id"];

/**
 * The two models the pipeline cannot run without. Retrieval is entirely local,
 * so these download once and then never phone home.
 */
const CORE_MODELS: {
  role: ModelRole;
  name: string;
  repo_id: string;
  filename: string;
  whyKey: string;
}[] = [
  {
    role: "embedding",
    name: "EmbeddingGemma 300M",
    repo_id: "ggml-org/embeddinggemma-300M-qat-q4_0-GGUF",
    filename: "embeddinggemma-300M-qat-Q4_0.gguf",
    whyKey: "onboarding.models.embeddingWhy",
  },
  {
    role: "reranking",
    name: "Qwen3 Reranker 0.6B",
    repo_id: "ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF",
    filename: "qwen3-reranker-0.6b-q8_0.gguf",
    whyKey: "onboarding.models.rerankingWhy",
  },
];

const PROFILE_NOTE: Record<HardwareInfo["profile"], string> = {
  gpu: "onboarding.hardware.profileGpu",
  balanced: "onboarding.hardware.profileBalanced",
  cpu: "onboarding.hardware.profileCpu",
};

function StepFrame({
  title,
  lead,
  children,
}: {
  title: string;
  lead?: string;
  children?: React.ReactNode;
}) {
  return (
    <div className="space-y-5">
      <div className="space-y-1.5">
        <h1 className="text-lg font-semibold tracking-tight">{title}</h1>
        {lead ? (
          <p className="max-w-prose text-[0.8125rem] leading-[1.55] text-muted-foreground">
            {lead}
          </p>
        ) : null}
      </div>
      {children}
    </div>
  );
}

function Row({
  icon: Icon,
  title,
  body,
  aside,
  tone,
}: {
  icon: React.ComponentType<{ className?: string }>;
  title: React.ReactNode;
  body?: React.ReactNode;
  aside?: React.ReactNode;
  tone?: "ok";
}) {
  return (
    <div
      className={cn(
        "flex items-start gap-3 rounded-lg border bg-card p-3",
        tone === "ok" ? "border-status-ok/30" : "border-border",
      )}
    >
      <Icon className={cn("mt-0.5 size-4 shrink-0", tone === "ok" ? "text-status-ok" : "text-muted-foreground")} />
      <div className="min-w-0 flex-1">
        <p className="text-[0.8125rem] font-medium">{title}</p>
        {body ? (
          <div className="mt-0.5 text-[0.6875rem] leading-[1.5] text-muted-foreground">{body}</div>
        ) : null}
      </div>
      {aside ? <div className="shrink-0">{aside}</div> : null}
    </div>
  );
}

function WelcomeStep() {
  const { t } = useTranslation();
  return (
    <StepFrame
      title={t("onboarding.welcome.title")}
      lead={t("onboarding.welcome.lead")}
    >
      <div className="space-y-2">
        <Row
          icon={LockIcon}
          title={t("onboarding.welcome.privateTitle")}
          body={t("onboarding.welcome.privateBody")}
        />
        <Row
          icon={QuoteIcon}
          title={t("onboarding.welcome.citedTitle")}
          body={t("onboarding.welcome.citedBody")}
        />
        <Row
          icon={BookOpenIcon}
          title={t("onboarding.welcome.stepsTitle")}
          body={t("onboarding.welcome.stepsBody")}
        />
      </div>
    </StepFrame>
  );
}

function HardwareStep({ hardware }: { hardware: HardwareInfo | undefined }) {
  const { t } = useTranslation();
  if (!hardware) {
    return (
      <StepFrame
        title={t("onboarding.hardware.loadingTitle")}
        lead={t("onboarding.hardware.loadingLead")}
      >
        <Skeleton className="h-24 w-full" />
      </StepFrame>
    );
  }

  const gpu =
    hardware.gpu_backend === "cpu"
      ? t("onboarding.hardware.noGpu")
      : t("onboarding.hardware.gpu", {
          backend: hardware.gpu_backend,
          name: hardware.gpu_name ? ` · ${hardware.gpu_name}` : "",
          vram: (hardware.vram_mb / 1024).toFixed(1),
        });

  return (
    <StepFrame
      title={t("onboarding.hardware.title")}
      lead={t("onboarding.hardware.lead")}
    >
      <div className="grid gap-2 sm:grid-cols-2">
        <Row
          icon={CpuIcon}
          title={`${hardware.os} · ${hardware.arch}`}
          body={t("onboarding.hardware.cpuBody", {
            threads: hardware.cpu_count,
            ram: (hardware.ram_mb / 1024).toFixed(1),
          })}
        />
        <Row icon={ServerIcon} title={t("onboarding.hardware.graphics")} body={gpu} />
      </div>

      <div className="rounded-lg border border-primary/35 bg-primary/5 p-3">
        <p className="text-[0.8125rem] font-medium">
          {t("onboarding.hardware.profile")}{" "}
          <span className="capitalize">{t(`onboarding.hardware.profileName.${hardware.profile}`)}</span>
        </p>
        <p className="mt-0.5 text-[0.6875rem] leading-[1.5] text-muted-foreground">
          {t(PROFILE_NOTE[hardware.profile])}
        </p>
      </div>
    </StepFrame>
  );
}

function ModelRow({
  spec,
  installed,
}: {
  spec: (typeof CORE_MODELS)[number];
  installed: InstalledModel | undefined;
}) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const { byModel } = useJobs();
  const [startedModelId, setStartedModelId] = useState<string | null>(null);

  const job = startedModelId ? byModel.get(startedModelId) : undefined;

  const download = useMutation({
    mutationFn: () =>
      api.download({
        repo_id: spec.repo_id,
        filename: spec.filename,
        role: spec.role,
        activate: true,
        accept_reindex: true,
      }),
    onSuccess: (started) => {
      setStartedModelId(started.model_id ?? null);
      void queryClient.invalidateQueries({ queryKey: keys.jobs });
    },
    onError: (error: Error) =>
      toast.error(t("onboarding.models.downloadFailed", { role: t(`models.roles.${spec.role}`) }), {
        description: errorText(error),
      }),
  });

  if (installed) {
    return (
      <Row
        icon={CheckIcon}
        tone="ok"
        title={installed.name}
        body={
          <>
            {t(spec.whyKey)}
            <span className="mt-1 block font-mono text-[0.625rem]">
              {bytes(installed.size_bytes)}
              {installed.quant ? ` · ${installed.quant}` : ""}
              {installed.sha256 ? ` · sha256 ${installed.sha256.slice(0, 12)}…` : ""}
            </span>
          </>
        }
        aside={<StatusChip tone="ok" label={t("onboarding.models.verified")} />}
      />
    );
  }

  return (
    <Row
      icon={DownloadIcon}
      title={spec.name}
      body={
        <>
          {t(spec.whyKey)}
          {job ? (
            <span className="mt-2 block space-y-1">
              <Progress value={Math.round(job.progress * 100)} className="h-1" />
              <span className="block font-mono text-[0.625rem]">
                {job.detail ?? job.label} · {Math.round(job.progress * 100)}%
              </span>
            </span>
          ) : null}
        </>
      }
      aside={
        <Button
          size="sm"
          className="h-7"
          disabled={download.isPending || (job != null && job.state === "running")}
          onClick={() => download.mutate()}
        >
          {download.isPending || job?.state === "running" ? (
            <Loader2Icon className="size-3.5 animate-spin" />
          ) : (
            <DownloadIcon className="size-3.5" />
          )}
          {t("common.download")}
        </Button>
      }
    />
  );
}

function ModelsStep({ installed }: { installed: InstalledModel[] }) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const runtime = useQuery(runtimeQuery);
  const install = useMutation({
    mutationFn: api.installRuntime,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: keys.runtime }),
  });
  const data = runtime.data;
  const real = data !== undefined && data.state !== "unavailable";
  return (
    <StepFrame
      title={t("onboarding.models.title")}
      lead={t("onboarding.models.lead")}
    >
      <div className="space-y-2">
        {CORE_MODELS.map((spec) =>
          real ? (
            <Row
              key={spec.role}
              icon={CpuIcon}
              tone={data.state === "ready" ? "ok" : undefined}
              title={spec.name}
              body={t(spec.whyKey)}
            />
          ) : (
            <ModelRow
              key={spec.role}
              spec={spec}
              installed={installed.find((model) => model.role === spec.role && model.active)}
            />
          ),
        )}
      </div>
      {data?.state === "downloading" ? (
        <Progress value={Math.round(data.progress * 100)} />
      ) : null}
      {data?.error ? (
        <p className="text-xs text-status-error">
          {t("onboarding.models.runtimeFailed", { error: data.error })}
        </p>
      ) : null}
      {data?.state === "missing" ? (
        <Button size="sm" onClick={() => install.mutate()}>
          <DownloadIcon /> {t("onboarding.models.runtimeInstall")}
        </Button>
      ) : null}
    </StepFrame>
  );
}

function ConnectionStep() {
  const { t } = useTranslation();
  const connections = useQuery(connectionsQuery);
  const list = connections.data ?? [];
  const own = list.filter((connection) => connection.kind !== "local");

  return (
    <StepFrame
      title={t("onboarding.connection.title")}
      lead={t("onboarding.connection.lead")}
    >
      {connections.isLoading ? <Skeleton className="h-20 w-full" /> : null}

      <div className="space-y-2">
        {list.map((connection) => (
          <Row
            key={connection.id}
            icon={connection.is_remote ? CloudIcon : ServerIcon}
            tone={connection.active ? "ok" : undefined}
            title={
              <span className="flex items-center gap-2">
                {connection.kind === "local" ? t("localModel.name") : connection.name}
                {connection.active ? (
                  <span className="rounded-full bg-primary/12 px-1.5 py-0.5 text-[0.625rem] font-medium text-primary">
                    {t("common.active")}
                  </span>
                ) : null}
              </span>
            }
            body={
              connection.kind === "local" ? (
                t("localModel.blurb")
              ) : (
                <span className="font-mono text-[0.625rem]">
                  {connection.model_id}
                  {connection.base_url ? ` · ${connection.base_url}` : ""}
                </span>
              )
            }
            aside={
              connection.kind === "local" ? (
                <LocalModelControl connection={connection} />
              ) : (
                <ConnectionDialog
                  connection={connection}
                  trigger={
                    <Button variant="ghost" size="sm" className="h-7">
                      {t("common.edit")}
                    </Button>
                  }
                />
              )
            }
          />
        ))}
      </div>

      <ConnectionDialog
        trigger={
          <Button variant={own.length === 0 ? "default" : "secondary"} size="sm" className="h-7">
            {own.length === 0 ? t("onboarding.connection.add") : t("onboarding.connection.addAnother")}
          </Button>
        }
      />

      {list.some((connection) => connection.is_remote && connection.active) ? (
        <p className="rounded-md border border-status-warn/30 bg-status-warn/8 px-2.5 py-2 text-[0.6875rem] text-status-warn">
          {t("onboarding.connection.remoteWarning")}
        </p>
      ) : null}
    </StepFrame>
  );
}

function LibraryStep() {
  const { t } = useTranslation();
  const sources = useQuery(sourcesQuery);
  const { bySource } = useJobs();
  const list = sources.data ?? [];

  return (
    <StepFrame
      title={t("onboarding.library.title")}
      lead={t("onboarding.library.lead")}
    >
      {sources.isLoading ? <Skeleton className="h-20 w-full" /> : null}

      <div className="space-y-2">
        {list.map((source) => {
          const job = bySource.get(source.id);
          return (
            <Row
              key={source.id}
              icon={job ? Loader2Icon : CheckIcon}
              tone={job ? undefined : "ok"}
              title={source.path}
              body={
                job ? (
                  <>
                    <span className="mt-1 block">
                      <Progress value={Math.round(job.progress * 100)} className="h-1" />
                    </span>
                    <span className="mt-1 block">{job.detail ?? job.label}</span>
                  </>
                ) : (
                  t("onboarding.library.indexed", {
                    indexed: source.indexed_count,
                    total: source.document_count,
                  }) +
                  (source.error_count
                    ? ` · ${t("onboarding.library.failed", { count: source.error_count })}`
                    : "")
                )
              }
            />
          );
        })}
      </div>

      <AddSourceDialog
        trigger={
          <Button variant={list.length === 0 ? "default" : "secondary"} size="sm" className="h-7">
            <FolderPlusIcon className="size-3.5" />
            {list.length === 0 ? t("onboarding.library.choose") : t("onboarding.library.addAnother")}
          </Button>
        }
      />
    </StepFrame>
  );
}

function DoneStep({ documents }: { documents: number }) {
  const { t } = useTranslation();
  return (
    <StepFrame
      title={t("onboarding.done.title")}
      lead={t("onboarding.done.lead")}
    >
      <div className="space-y-2">
        <Row
          icon={CheckIcon}
          tone="ok"
          title={t("onboarding.done.documents", { count: documents })}
          body={t("onboarding.done.documentsBody")}
        />
        <Row
          icon={QuoteIcon}
          title={t("onboarding.done.citationTitle")}
          body={t("onboarding.done.citationBody")}
        />
        <Row
          icon={CpuIcon}
          title={t("onboarding.done.diagnosticsTitle")}
          body={t("onboarding.done.diagnosticsBody")}
        />
      </div>
    </StepFrame>
  );
}

export function OnboardingView() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [index, setIndex] = useState(0);

  const hardware = useQuery(hardwareQuery);
  const health = useQuery(healthQuery);
  const models = useQuery(modelsQuery);
  const connections = useQuery(connectionsQuery);
  const sources = useQuery(sourcesQuery);

  const installed = useMemo(() => models.data?.installed ?? [], [models.data]);
  const step: StepId = STEPS[index].id;

  const finish = useMutation({
    mutationFn: () => api.patchSettings({ onboarded: true }),
    onSuccess: (settings) => {
      queryClient.setQueryData(keys.settings, settings);
      void navigate({ to: "/chat", replace: true });
    },
    onError: (error: Error) =>
      toast.error(t("onboarding.saveFailed"), { description: errorText(error) }),
  });

  const runtime = useQuery(runtimeQuery);
  const coreReady =
    runtime.data?.state === "ready" ||
    (runtime.data?.state === "unavailable" &&
      CORE_MODELS.every((spec) =>
        installed.some((model) => model.role === spec.role && model.active),
      ));

  const blocked =
    (step === "hardware" && !hardware.data) || (step === "models" && !coreReady);

  const optional =
    (step === "connection" && !(connections.data ?? []).some((connection) => connection.active)) ||
    (step === "library" && (sources.data ?? []).length === 0);

  function next() {
    if (step === "done") {
      finish.mutate();
      return;
    }
    setIndex((current) => Math.min(current + 1, STEPS.length - 1));
  }

  return (
    <div className="flex h-full overflow-hidden">
      <aside className="flex w-56 shrink-0 flex-col border-r border-sidebar-border bg-sidebar">
        <div data-tauri-drag-region="deep" className="drag-region h-10 shrink-0" />
        <nav className="space-y-0.5 px-2">
          {STEPS.map((entry, position) => {
            const state = position < index ? "past" : position === index ? "current" : "future";
            return (
              <button
                key={entry.id}
                type="button"
                disabled={state === "future"}
                onClick={() => setIndex(position)}
                className={cn(
                  "flex w-full items-center gap-2.5 rounded-md px-2 py-1.5 text-left text-[0.8125rem]",
                  state === "current" && "bg-sidebar-accent font-medium text-sidebar-foreground",
                  state === "past" && "text-muted-foreground hover:bg-sidebar-accent/60",
                  state === "future" && "text-muted-foreground/50",
                )}
              >
                <span
                  className={cn(
                    "grid size-4 shrink-0 place-items-center rounded-full border text-[0.5625rem] tabular-nums",
                    state === "past"
                      ? "border-status-ok/50 text-status-ok"
                      : state === "current"
                        ? "border-primary text-primary"
                        : "border-border",
                  )}
                >
                  {state === "past" ? <CheckIcon className="size-2.5" /> : position + 1}
                </span>
                {t(`onboarding.steps.${entry.id}`)}
              </button>
            );
          })}
        </nav>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <div data-tauri-drag-region="deep" className="drag-region h-10 shrink-0" />

        <div className="min-h-0 flex-1 overflow-auto">
          <div className="mx-auto max-w-2xl px-8 pb-10">
            {step === "welcome" ? <WelcomeStep /> : null}
            {step === "hardware" ? <HardwareStep hardware={hardware.data} /> : null}
            {step === "models" ? <ModelsStep installed={installed} /> : null}
            {step === "connection" ? <ConnectionStep /> : null}
            {step === "library" ? <LibraryStep /> : null}
            {step === "done" ? <DoneStep documents={health.data?.index.documents ?? 0} /> : null}
          </div>
        </div>

        <footer className="flex shrink-0 items-center justify-between gap-3 border-t border-border px-8 py-3">
          <Button
            variant="ghost"
            size="sm"
            className="h-8"
            disabled={index === 0}
            onClick={() => setIndex((current) => Math.max(current - 1, 0))}
          >
            {t("common.back")}
          </Button>

          <div className="flex items-center gap-2">
            {optional ? (
              <span className="text-[0.6875rem] text-muted-foreground">
                {t("onboarding.later")}
              </span>
            ) : null}
            <Button size="sm" className="h-8" disabled={blocked || finish.isPending} onClick={next}>
              {finish.isPending ? <Loader2Icon className="size-3.5 animate-spin" /> : null}
              {step === "done"
                ? t("onboarding.openChat")
                : optional
                  ? t("onboarding.skip")
                  : t("onboarding.continue")}
              {step === "done" ? null : <ArrowRightIcon className="size-3.5" />}
            </Button>
          </div>
        </footer>
      </div>
    </div>
  );
}
