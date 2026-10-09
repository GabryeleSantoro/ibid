import { useState } from "react";
import { useTranslation } from "react-i18next";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { CheckCircle2Icon, Loader2Icon, XCircleIcon } from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { errorText } from "@/lib/errors";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ms } from "@/lib/format";
import {
  api,
  shell,
  type Connection,
  type ConnectionKind,
  type ConnectionTestResult,
} from "@/lib/ipc";
import { keys } from "@/lib/queries";

type ProviderId = "openrouter" | "openai" | "anthropic" | "custom";

/** A preset prefills kind + base_url; "custom" is the only one with an editable URL. */
const PROVIDERS: {
  id: ProviderId;
  label: string;
  kind: ConnectionKind;
  baseUrl: string | null;
  hintKey: string;
}[] = [
  {
    id: "openrouter",
    label: "OpenRouter",
    kind: "openai-compatible",
    baseUrl: "https://openrouter.ai/api/v1",
    hintKey: "connection.hintOpenrouter",
  },
  {
    id: "openai",
    label: "OpenAI",
    kind: "openai-compatible",
    baseUrl: "https://api.openai.com/v1",
    hintKey: "connection.hintOpenai",
  },
  { id: "anthropic", label: "Anthropic", kind: "anthropic", baseUrl: null, hintKey: "connection.hintAnthropic" },
  {
    id: "custom",
    label: "Custom / localhost", // shown via connection.providerCustom
    kind: "openai-compatible",
    baseUrl: null,
    hintKey: "connection.hintCustom",
  },
];

function isRemoteUrl(url: string) {
  return !/localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1\]/.test(url);
}

function matchProvider(kind: ConnectionKind, base_url: string | null | undefined) {
  const normalized = (base_url ?? "").replace(/\/+$/, "");
  return (
    PROVIDERS.find((p) => p.baseUrl && p.kind === kind && p.baseUrl.replace(/\/+$/, "") === normalized) ??
    PROVIDERS.find((p) => p.baseUrl === null && p.kind === kind) ??
    PROVIDERS[PROVIDERS.length - 1]
  );
}

const BLANK = {
  name: "",
  kind: "openai-compatible" as ConnectionKind,
  base_url: "http://localhost:1234/v1",
  model_id: "",
  max_output_tokens: null as number | null,
  thinking: "off" as const,
  is_remote: false,
  provider_sort: null as "price" | "throughput" | "latency" | null,
  provider_order: null as string[] | null,
};

const SORT_MODES = ["auto", "price", "throughput", "latency"] as const;

function TestResult({ result }: { result: ConnectionTestResult }) {
  const { t } = useTranslation();
  const Icon = result.ok ? CheckCircle2Icon : XCircleIcon;
  return (
    <div
      className={`flex items-start gap-2 rounded-md border px-2.5 py-2 text-[0.6875rem] ${
        result.ok
          ? "border-status-ok/30 bg-status-ok/8 text-status-ok"
          : "border-status-warn/30 bg-status-warn/8 text-status-warn"
      }`}
    >
      <Icon className="mt-px size-3.5 shrink-0" />
      <div className="space-y-0.5">
        <p>
          {result.reachable ? t("connection.reachable") : t("connection.notReachable")}
          {result.latency_ms != null ? ` ${t("connection.inMs", { ms: ms(result.latency_ms) })}` : ""}
          {result.model_found ? ` · ${t("connection.modelFound")}` : ""}
          {result.streaming ? ` · ${t("connection.streaming")}` : ""}
        </p>
        {result.error ? <p className="opacity-80">{result.error}</p> : null}
      </div>
    </div>
  );
}

export function ConnectionDialog({
  connection,
  trigger,
  preset,
}: {
  connection?: Connection;
  trigger: React.ReactNode;
  /** Opens as a new OpenRouter connection for this model. */
  preset?: { model_id: string };
}) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const [open, setOpen] = useState(false);
  const [form, setForm] = useState(() => ({
    ...BLANK,
    ...(connection ?? {}),
    ...(preset
      ? {
          name: "OpenRouter",
          kind: "openai-compatible" as ConnectionKind,
          base_url: "https://openrouter.ai/api/v1",
          is_remote: true,
          model_id: preset.model_id,
        }
      : {}),
  }));
  const [provider, setProvider] = useState(() => matchProvider(form.kind, form.base_url));
  const [providerOrderText, setProviderOrderText] = useState(() =>
    (form.provider_order ?? []).join(", "),
  );
  const [apiKey, setApiKey] = useState("");
  const [result, setResult] = useState<ConnectionTestResult | null>(null);

  const set = <K extends keyof typeof form>(key: K, value: (typeof form)[K]) =>
    setForm((current) => ({ ...current, [key]: value }));

  const test = useMutation({
    mutationFn: () =>
      api.testConnection({
        kind: form.kind,
        base_url: form.base_url,
        model_id: form.model_id,
        api_key: apiKey || null,
      }),
    onSuccess: setResult,
    onError: (error: Error) => toast.error(t("connection.testFailed"), { description: errorText(error) }),
  });

  const save = useMutation({
    mutationFn: async () => {
      const payload = {
        name: form.name || form.model_id,
        kind: form.kind,
        base_url: form.kind === "anthropic" ? null : form.base_url,
        model_id: form.model_id,
        max_output_tokens: form.max_output_tokens,
        thinking: form.thinking,
        is_remote: form.is_remote,
        api_key: apiKey || null,
        provider_sort: provider.id === "openrouter" ? form.provider_sort : null,
        provider_order:
          provider.id === "openrouter" && providerOrderText.trim()
            ? providerOrderText
                .split(",")
                .map((entry) => entry.trim())
                .filter(Boolean)
            : null,
      };
      const saved = connection
        ? await api.updateConnection(connection.id, payload)
        : await api.createConnection(payload);
      // The keychain is the durable copy; the sidecar got a memory-only one in
      // the payload above, because the sidecar is what calls the provider.
      if (apiKey) await shell.keychainSet(saved.id, apiKey);
      // The slow-speed banner's shortcut exists to switch models, so the new one answers next.
      if (preset) await api.activateConnection(saved.id);
      return saved;
    },
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: keys.connections });
      setOpen(false);
      setApiKey("");
      toast.success(connection ? t("connection.updated") : t("connection.added"));
    },
    onError: (error: Error) => toast.error(t("settings.saveFailed"), { description: errorText(error) }),
  });

  const needsUrl = provider.id === "custom";

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>{trigger}</DialogTrigger>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>{connection ? t("connection.editTitle") : t("connection.addTitle")}</DialogTitle>
          <DialogDescription>
            {t("connection.lead")}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="space-y-1.5">
              <Label htmlFor="conn-kind">{t("connection.provider")}</Label>
              <Select
                value={provider.id}
                onValueChange={(value) => {
                  const next = PROVIDERS.find((entry) => entry.id === value)!;
                  setProvider(next);
                  setForm((current) => ({
                    ...current,
                    kind: next.kind,
                    base_url: next.baseUrl ?? (next.id === "custom" ? current.base_url : null),
                    is_remote:
                      next.id === "custom" ? isRemoteUrl(current.base_url ?? "") : true,
                  }));
                }}
              >
                <SelectTrigger id="conn-kind">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {PROVIDERS.map((entry) => (
                    <SelectItem key={entry.id} value={entry.id}>
                      {entry.id === "custom" ? t("connection.providerCustom") : entry.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <p className="text-[0.6875rem] text-muted-foreground">
                {t(provider.hintKey)}
                {provider.baseUrl ? ` · ${provider.baseUrl}` : ""}
              </p>
            </div>

            <div className="space-y-1.5">
              <Label htmlFor="conn-name">{t("connection.name")}</Label>
              <Input
                id="conn-name"
                value={form.name}
                placeholder="LM Studio"
                onChange={(event) => set("name", event.target.value)}
              />
            </div>
          </div>

          {provider.id === "openrouter" ? (
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-1.5">
                <Label htmlFor="conn-provider-sort">{t("connection.routing")}</Label>
                <Select
                  value={form.provider_sort ?? "auto"}
                  onValueChange={(value) =>
                    set(
                      "provider_sort",
                      value === "auto" ? null : (value as "price" | "throughput" | "latency"),
                    )
                  }
                >
                  <SelectTrigger id="conn-provider-sort">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {SORT_MODES.map((mode) => (
                      <SelectItem key={mode} value={mode}>
                        {t(`connection.sort.${mode}`)}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="conn-provider-order">{t("connection.preferred")}</Label>
                <Input
                  id="conn-provider-order"
                  value={providerOrderText}
                  placeholder={t("connection.preferredPlaceholder")}
                  onChange={(event) => setProviderOrderText(event.target.value)}
                />
                <p className="text-[0.6875rem] text-muted-foreground">
                  {t("connection.preferredHint")}
                </p>
              </div>
            </div>
          ) : null}

          {needsUrl ? (
            <div className="space-y-1.5">
              <Label htmlFor="conn-url">{t("connection.baseUrl")}</Label>
              <Input
                id="conn-url"
                value={form.base_url ?? ""}
                placeholder="http://localhost:1234/v1"
                onChange={(event) => {
                  set("base_url", event.target.value);
                  set(
                    "is_remote",
                    !/localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1\]/.test(event.target.value),
                  );
                }}
              />
              <p className="text-[0.6875rem] text-muted-foreground">
                {form.is_remote ? t("connection.remoteNote") : t("connection.localNote")}
              </p>
            </div>
          ) : null}

          <div className="grid gap-4 sm:grid-cols-3">
            <div className="space-y-1.5 sm:col-span-2">
              <Label htmlFor="conn-model">{t("connection.modelId")}</Label>
              <Input
                id="conn-model"
                value={form.model_id}
                placeholder="qwen3-8b-instruct"
                onChange={(event) => set("model_id", event.target.value)}
              />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="conn-thinking">{t("connection.thinking")}</Label>
              <Select
                value={form.thinking}
                onValueChange={(value) => set("thinking", value as typeof form.thinking)}
              >
                <SelectTrigger id="conn-thinking">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {["off", "low", "medium", "high"].map((level) => (
                    <SelectItem key={level} value={level}>
                      {t(`connection.level.${level}`)}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          <div className="grid gap-4 sm:grid-cols-2">
            <div className="space-y-1.5">
              <Label htmlFor="conn-output">{t("connection.maxOutput")}</Label>
              <Input
                id="conn-output"
                type="number"
                placeholder={t("connection.noLimit")}
                value={form.max_output_tokens ?? ""}
                onChange={(event) => set("max_output_tokens", Number(event.target.value) || null)}
              />
            </div>
          </div>

          <div className="space-y-1.5">
            <Label htmlFor="conn-key">
              {t("connection.apiKey")}{" "}
              {connection?.has_api_key ? t("connection.apiKeyStored") : ""}
            </Label>
            <Input
              id="conn-key"
              type="password"
              value={apiKey}
              placeholder={connection?.has_api_key ? "••••••••" : t("connection.apiKeyPlaceholder")}
              onChange={(event) => setApiKey(event.target.value)}
            />
            <p className="text-[0.6875rem] text-muted-foreground">
              {t("connection.apiKeyHint")}
            </p>
          </div>

          {result ? <TestResult result={result} /> : null}
        </div>

        <DialogFooter className="sm:justify-between">
          <Button variant="secondary" onClick={() => test.mutate()} disabled={test.isPending}>
            {test.isPending ? <Loader2Icon className="size-4 animate-spin" /> : null}
            {t("connection.test")}
          </Button>
          <div className="flex gap-2">
            <Button variant="ghost" onClick={() => setOpen(false)}>
              {t("common.cancel")}
            </Button>
            <Button
              disabled={!form.model_id.trim() || save.isPending}
              onClick={() => save.mutate()}
            >
              {t("common.save")}
            </Button>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
