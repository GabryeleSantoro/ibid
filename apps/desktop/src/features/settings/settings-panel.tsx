import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { useParams } from "@tanstack/react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { CloudIcon, HardDriveIcon, PlusIcon, Trash2Icon } from "lucide-react";
import { disable, enable, isEnabled } from "@tauri-apps/plugin-autostart";
import { toast } from "sonner";

import { Page, PageBody, PageHeader } from "@/components/shell/page";
import { errorText } from "@/lib/errors";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { Switch } from "@/components/ui/switch";
import { Skeleton } from "@/components/ui/skeleton";
import { IconTooltip } from "@/components/ui/tooltip";
import { ConnectionDialog } from "@/features/settings/connection-dialog";
import { useTheme, type Theme } from "@/components/shell/theme-provider";
import { LocalModelControl } from "@/features/settings/local-model";
import { SETTINGS_SECTIONS } from "@/features/settings/settings-sidebar";
import { UpdatesSection } from "@/features/settings/updates";
import {
  api,
  shell,
  type AppSettings,
  type PerformanceSettings,
  type RetrievalSettings,
} from "@/lib/ipc";
import { connectionsQuery, hardwareQuery, keys, settingsQuery } from "@/lib/queries";
import {
  LANGUAGES,
  LANGUAGE_NAMES,
  currentLanguage,
  getLanguagePref,
  setLanguagePref,
  type LanguagePref,
} from "@/lib/i18n";
import { getFontSize, setFontSize, type FontSize } from "@/lib/font-size";
import { isMac } from "@/lib/platform";
import { SHORTCUTS } from "@/lib/shortcuts";
import { cn } from "@/lib/utils";

function Field({
  label,
  hint,
  wide,
  children,
}: {
  label: string;
  hint?: string;
  wide?: boolean;
  children: React.ReactNode;
}) {
  return (
    <div className="grid gap-1.5 sm:grid-cols-[14rem_1fr] sm:items-baseline sm:gap-6">
      <div>
        <Label className="text-[0.8125rem]">{label}</Label>
        {hint ? <p className="mt-0.5 text-[0.6875rem] text-muted-foreground">{hint}</p> : null}
      </div>
      <div className={wide ? "max-w-xl" : "max-w-xs"}>{children}</div>
    </div>
  );
}

function useSettingsDraft() {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const settings = useQuery(settingsQuery);
  const [draft, setDraft] = useState<AppSettings | null>(null);

  useEffect(() => {
    if (settings.data) setDraft(settings.data);
  }, [settings.data]);

  const save = useMutation({
    mutationFn: (patch: Parameters<typeof api.patchSettings>[0]) => api.patchSettings(patch),
    onSuccess: (updated) => {
      queryClient.setQueryData(keys.settings, updated);
      toast.success(t("settings.saved"));
    },
    onError: (error: Error) => toast.error(t("settings.saveFailed"), { description: errorText(error) }),
  });

  return { settings, draft, setDraft, save };
}

function ConnectionsSection() {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const connections = useQuery(connectionsQuery);

  const activate = useMutation({
    mutationFn: (id: string) => api.activateConnection(id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: keys.connections }),
  });

  const remove = useMutation({
    mutationFn: async (id: string) => {
      await api.deleteConnection(id);
      await shell.keychainDelete(id).catch(() => undefined);
    },
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: keys.connections });
      toast.success(t("settings.connections.removed"));
    },
  });

  return (
    <div className="space-y-3">
      {connections.isLoading ? <Skeleton className="h-20 w-full" /> : null}

      {(connections.data ?? []).map((connection) => (
        <div
          key={connection.id}
          className={cn(
            "flex items-start justify-between gap-4 rounded-lg border bg-card p-3",
            connection.active ? "border-primary/40" : "border-border",
          )}
        >
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <p className="truncate text-[0.8125rem] font-medium">
                {connection.kind === "local" ? t("localModel.name") : connection.name}
              </p>
              {connection.active ? (
                <span className="rounded-full bg-primary/12 px-1.5 py-0.5 text-[0.625rem] font-medium text-primary">
                  {t("common.active")}
                </span>
              ) : null}
              {connection.is_remote ? (
                <span className="inline-flex items-center gap-1 rounded-full bg-status-warn/12 px-1.5 py-0.5 text-[0.625rem] font-medium text-status-warn">
                  <CloudIcon className="size-2.5" />
                  {t("settings.connections.remote")}
                </span>
              ) : null}
            </div>
            {connection.kind === "local" ? (
              <p className="text-[0.6875rem] text-muted-foreground">{t("localModel.blurb")}</p>
            ) : (
              <>
                <p className="truncate font-mono text-[0.625rem] text-muted-foreground">
                  {connection.model_id}
                  {connection.base_url ? ` · ${connection.base_url}` : ""}
                </p>
                <p className="mt-1 font-mono text-[0.625rem] text-muted-foreground tabular-nums">
                  {connection.max_output_tokens
                    ? t("settings.connections.outputLimit", {
                        value: connection.max_output_tokens.toLocaleString(currentLanguage()),
                      })
                    : t("settings.connections.noOutputLimit")}
                  {connection.has_api_key ? ` · ${t("settings.connections.keyInKeychain")}` : ""}
                </p>
              </>
            )}
          </div>

          {connection.kind === "local" ? (
            <LocalModelControl connection={connection} />
          ) : (
          <div className="flex shrink-0 items-center gap-1">
            {!connection.active ? (
              <Button
                variant="secondary"
                size="sm"
                className="h-7"
                onClick={() => activate.mutate(connection.id)}
              >
                {t("models.use")}
              </Button>
            ) : null}
            <ConnectionDialog
              connection={connection}
              trigger={
                <Button variant="ghost" size="sm" className="h-7">
                  {t("common.edit")}
                </Button>
              }
            />
            <IconTooltip label={t("common.remove")}>
              <Button
                variant="ghost"
                size="icon"
                className="size-7"
                aria-label={t("common.remove")}
                onClick={() => remove.mutate(connection.id)}
              >
                <Trash2Icon className="size-3.5" />
              </Button>
            </IconTooltip>
          </div>
          )}
        </div>
      ))}

      <ConnectionDialog
        trigger={
          <Button variant="secondary" size="sm">
            <PlusIcon className="size-4" />
            {t("settings.connections.add")}
          </Button>
        }
      />
    </div>
  );
}

function RetrievalSection() {
  const { t } = useTranslation();
  const { draft, setDraft, save } = useSettingsDraft();
  if (!draft) return <Skeleton className="h-40 w-full" />;

  const retrieval = draft.retrieval as RetrievalSettings;
  const set = (key: keyof RetrievalSettings, value: number) =>
    setDraft({ ...draft, retrieval: { ...retrieval, [key]: value } });

  return (
    <div className="space-y-6">
      <div className="space-y-4">
        <Field
          label={t("settings.retrieval.topK")}
          hint={t("settings.retrieval.topKHint")}
        >
          <Input
            type="number"
            value={retrieval.top_k}
            onChange={(event) => set("top_k", Number(event.target.value) || 1)}
          />
        </Field>

        <Field
          label={t("settings.retrieval.minScore")}
          hint={t("settings.retrieval.minScoreHint")}
        >
          <Input
            type="number"
            step="0.05"
            min="0"
            max="1"
            value={retrieval.min_score}
            onChange={(event) => set("min_score", Number(event.target.value))}
          />
        </Field>

        <Field
          label={t("settings.retrieval.rerank")}
          hint={t("settings.retrieval.rerankHint")}
        >
          <Input
            type="number"
            value={retrieval.rerank_candidates}
            onChange={(event) => set("rerank_candidates", Number(event.target.value) || 1)}
          />
        </Field>
      </div>

      <Collapsible>
        <CollapsibleTrigger className="text-[0.8125rem] font-medium text-muted-foreground hover:text-foreground">
          {t("settings.retrieval.advanced")}
        </CollapsibleTrigger>
        <CollapsibleContent className="mt-4 space-y-4">
          <Field label={t("settings.retrieval.dense")} hint={t("settings.retrieval.denseHint")}>
            <Input
              type="number"
              value={retrieval.dense_top_k}
              onChange={(event) => set("dense_top_k", Number(event.target.value) || 1)}
            />
          </Field>
          <Field label={t("settings.retrieval.keyword")} hint={t("settings.retrieval.keywordHint")}>
            <Input
              type="number"
              value={retrieval.bm25_top_k}
              onChange={(event) => set("bm25_top_k", Number(event.target.value) || 1)}
            />
          </Field>
          <Field
            label={t("settings.retrieval.rrf")}
            hint={t("settings.retrieval.rrfHint")}
          >
            <Input
              type="number"
              value={retrieval.rrf_k}
              onChange={(event) => set("rrf_k", Number(event.target.value) || 1)}
            />
          </Field>
          <Field label={t("settings.retrieval.context")} hint={t("settings.retrieval.contextHint")}>
            <Input
              type="number"
              value={retrieval.context_token_budget}
              onChange={(event) => set("context_token_budget", Number(event.target.value) || 1)}
            />
          </Field>
          <Field label={t("settings.retrieval.history")} hint={t("settings.retrieval.historyHint")}>
            <Input
              type="number"
              value={retrieval.history_token_budget}
              onChange={(event) => set("history_token_budget", Number(event.target.value) || 1)}
            />
          </Field>
        </CollapsibleContent>
      </Collapsible>

      <Button
        disabled={save.isPending}
        onClick={() => save.mutate({ retrieval: draft.retrieval })}
      >
        {t("common.save")}
      </Button>
    </div>
  );
}

function PerformanceSection() {
  const { t } = useTranslation();
  const { draft, setDraft, save } = useSettingsDraft();
  const hardware = useQuery(hardwareQuery);
  if (!draft) return <Skeleton className="h-40 w-full" />;

  const performance = draft.performance as PerformanceSettings;
  const set = <K extends keyof PerformanceSettings>(key: K, value: PerformanceSettings[K]) =>
    setDraft({ ...draft, performance: { ...performance, [key]: value } });

  return (
    <div className="space-y-6">
      {hardware.data ? (
        <div className="rounded-lg border border-border bg-card p-3 text-[0.6875rem]">
          <p className="font-medium">
            {t("settings.performance.detected", {
              gpu: hardware.data.gpu_name ?? hardware.data.gpu_backend.toUpperCase(),
              ram: Math.round(hardware.data.ram_mb / 1024),
              cores: hardware.data.cpu_count,
            })}
          </p>
          <p className="mt-0.5 text-muted-foreground">
            {t("settings.performance.suggested", {
              profile: t(`onboarding.hardware.profileName.${hardware.data.profile}`),
            })}
          </p>
        </div>
      ) : null}

      <Field label={t("settings.performance.profile")} hint={t("settings.performance.profileHint")}>
        <Select
          value={performance.profile}
          onValueChange={(value) => set("profile", value as PerformanceSettings["profile"])}
        >
          <SelectTrigger>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="cpu">{t("settings.performance.cpuOnly")}</SelectItem>
            <SelectItem value="balanced">{t("settings.performance.balanced")}</SelectItem>
            <SelectItem value="gpu">{t("settings.performance.gpu")}</SelectItem>
          </SelectContent>
        </Select>
      </Field>

      <Field label={t("settings.performance.batch")} hint={t("settings.performance.batchHint")}>
        <Input
          type="number"
          value={performance.embed_batch}
          onChange={(event) => set("embed_batch", Number(event.target.value) || 1)}
        />
      </Field>

      <Field label={t("settings.performance.layers")} hint={t("settings.performance.layersHint")}>
        <Input
          type="number"
          value={performance.gpu_layers}
          onChange={(event) => set("gpu_layers", Number(event.target.value) || 0)}
        />
      </Field>

      <Field label={t("settings.performance.parsers")} hint={t("settings.performance.parsersHint")}>
        <Input
          type="number"
          value={performance.max_parallel_parsers}
          onChange={(event) => set("max_parallel_parsers", Number(event.target.value) || 1)}
        />
      </Field>

      <Button
        disabled={save.isPending}
        onClick={() => save.mutate({ performance: draft.performance })}
      >
        {t("common.save")}
      </Button>
    </div>
  );
}

function LaunchAtLogin() {
  const { t } = useTranslation();
  const [enabled, setEnabled] = useState<boolean | null>(null);

  useEffect(() => {
    isEnabled()
      .then(setEnabled)
      .catch(() => setEnabled(null));
  }, []);

  const toggle = async (next: boolean) => {
    try {
      await (next ? enable() : disable());
      setEnabled(next);
    } catch (error) {
      toast.error(t("settings.startup.failed"), { description: errorText(error) });
    }
  };

  // Browser or dev without the shell: the plugin call rejects and the row stays hidden.
  if (enabled === null) return null;
  return (
    <Field label={t("settings.startup.label")} hint={t("settings.startup.hint")}>
      <Switch checked={enabled} onCheckedChange={toggle} />
    </Field>
  );
}

function GeneralSection() {
  const { t } = useTranslation();
  const [pref, setPref] = useState<LanguagePref>(getLanguagePref());

  return (
    <div className="space-y-6">
      <Field label={t("settings.language.label")} hint={t("settings.language.hint")}>
        <Select
          value={pref}
          onValueChange={(value) => {
            setPref(value as LanguagePref);
            setLanguagePref(value as LanguagePref);
          }}
        >
          <SelectTrigger>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="system">{t("settings.language.system")}</SelectItem>
            {LANGUAGES.map((language) => (
              <SelectItem key={language} value={language}>
                {LANGUAGE_NAMES[language]}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </Field>
      <LaunchAtLogin />
    </div>
  );
}

function AppearanceSection() {
  const { t } = useTranslation();
  const { theme, setTheme } = useTheme();
  const [size, setSize] = useState<FontSize>(getFontSize());

  return (
    <div className="space-y-6">
      <Field label={t("settings.appearance.theme")}>
        <Select value={theme} onValueChange={(value) => setTheme(value as Theme)}>
          <SelectTrigger>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="system">{t("nav.themeSystem")}</SelectItem>
            <SelectItem value="light">{t("nav.themeLight")}</SelectItem>
            <SelectItem value="dark">{t("nav.themeDark")}</SelectItem>
          </SelectContent>
        </Select>
      </Field>
      <Field label={t("settings.appearance.fontSize")} hint={t("settings.appearance.fontSizeHint")}>
        <Select
          value={size}
          onValueChange={(value) => {
            setSize(value as FontSize);
            setFontSize(value as FontSize);
          }}
        >
          <SelectTrigger>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="s">{t("settings.appearance.small")}</SelectItem>
            <SelectItem value="m">{t("settings.appearance.medium")}</SelectItem>
            <SelectItem value="l">{t("settings.appearance.large")}</SelectItem>
          </SelectContent>
        </Select>
      </Field>
    </div>
  );
}

function ChatSection() {
  const { t } = useTranslation();
  const { draft, setDraft, save } = useSettingsDraft();
  if (!draft) return <Skeleton className="h-40 w-full" />;

  return (
    <div className="space-y-6">
      <Field label={t("settings.chat.extra")} hint={t("settings.chat.extraHint")} wide>
        <Textarea
          rows={5}
          maxLength={1000}
          placeholder={t("settings.chat.extraPlaceholder")}
          value={draft.chat_extra_instructions}
          onChange={(event) => setDraft({ ...draft, chat_extra_instructions: event.target.value })}
        />
      </Field>
      <Button
        disabled={save.isPending}
        onClick={() => save.mutate({ chat_extra_instructions: draft.chat_extra_instructions })}
      >
        {t("common.save")}
      </Button>
    </div>
  );
}

function ShortcutsSection() {
  const { t } = useTranslation();
  const mod = isMac ? "⌘" : "Ctrl+";
  return (
    <div className="max-w-md divide-y rounded-lg border">
      {SHORTCUTS.map((shortcut) => (
        <div key={shortcut.id} className="flex items-center justify-between px-3 py-2 text-[0.8125rem]">
          <span>{t(`settings.shortcuts.${shortcut.id}`)}</span>
          <kbd className="rounded bg-muted px-1.5 py-0.5 font-mono text-[0.6875rem]">
            {mod}
            {shortcut.key.toUpperCase()}
          </kbd>
        </div>
      ))}
    </div>
  );
}

function StorageSection() {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const { draft, setDraft, save } = useSettingsDraft();

  const wipe = useMutation({
    mutationFn: (keepConnections: boolean) => api.wipe(keepConnections),
    onSuccess: () => {
      void queryClient.invalidateQueries();
      toast.success(t("settings.storage.wiped"));
    },
    onError: (error: Error) => toast.error(t("settings.storage.wipeFailed"), { description: errorText(error) }),
  });

  if (!draft) return <Skeleton className="h-40 w-full" />;

  return (
    <div className="space-y-6">
      <Field label={t("settings.storage.location")} hint={t("settings.storage.locationHint")}>
        <Input
          value={draft.storage_path}
          onChange={(event) => setDraft({ ...draft, storage_path: event.target.value })}
        />
      </Field>

      <Field label={t("settings.storage.crash")} hint={t("settings.storage.crashHint")}>
        <div className="flex items-center gap-2">
          <Switch
            checked={draft.telemetry}
            onCheckedChange={(checked) => setDraft({ ...draft, telemetry: checked })}
          />
          <span className="text-[0.8125rem]">{draft.telemetry ? t("common.enabled") : t("common.disabled")}</span>
        </div>
      </Field>

      <Button
        disabled={save.isPending}
        onClick={() =>
          save.mutate({ storage_path: draft.storage_path, telemetry: draft.telemetry })
        }
      >
        {t("common.save")}
      </Button>

      <div className="rounded-lg border border-destructive/30 bg-destructive/5 p-4">
        <h3 className="flex items-center gap-1.5 text-[0.8125rem] font-semibold text-destructive">
          <HardDriveIcon className="size-3.5" />
          {t("settings.storage.wipeAll")}
        </h3>
        <p className="mt-1 max-w-prose text-[0.6875rem] leading-[1.5] text-muted-foreground">
          {t("settings.storage.wipeHint")}
        </p>

        <AlertDialog>
          <AlertDialogTrigger asChild>
            <Button variant="destructive" size="sm" className="mt-3">
              {t("settings.storage.wipeIndex")}
            </Button>
          </AlertDialogTrigger>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>{t("settings.storage.wipeTitle")}</AlertDialogTitle>
              <AlertDialogDescription>
                {t("settings.storage.wipeBody")}
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel>{t("common.cancel")}</AlertDialogCancel>
              <AlertDialogAction
                onClick={() => wipe.mutate(true)}
                className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
              >
                {t("settings.storage.wipeConfirm")}
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      </div>
    </div>
  );
}

export function SettingsPanel() {
  const { t } = useTranslation();
  const { section } = useParams({ from: "/_shell/settings/$section" });
  const meta = SETTINGS_SECTIONS.find((entry) => entry.slug === section);

  return (
    <Page>
      <PageHeader title={t(`settings.sections.${meta?.slug ?? "general"}.label`)}
        description={meta ? t(`settings.sections.${meta.slug}.blurb`) : undefined} />
      <PageBody>
        <div className="max-w-3xl p-5">
          {section === "general" ? <GeneralSection /> : null}
          {section === "connections" ? <ConnectionsSection /> : null}
          {section === "appearance" ? <AppearanceSection /> : null}
          {section === "chat" ? <ChatSection /> : null}
          {section === "retrieval" ? <RetrievalSection /> : null}
          {section === "performance" ? <PerformanceSection /> : null}
          {section === "shortcuts" ? <ShortcutsSection /> : null}
          {section === "storage" ? <StorageSection /> : null}
          {section === "updates" ? <UpdatesSection /> : null}
        </div>
      </PageBody>
    </Page>
  );
}
