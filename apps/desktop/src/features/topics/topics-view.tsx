import { useMemo, useState } from "react";
import { Link } from "@tanstack/react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { ChevronRightIcon, Loader2Icon, MessageSquarePlusIcon, RefreshCwIcon } from "lucide-react";
import { toast } from "sonner";

import { Page, PageBody, PageHeader } from "@/components/shell/page";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { nestTopics, type TopicNode } from "@/features/topics/tree";
import { errorText } from "@/lib/errors";
import { relativeTime } from "@/lib/format";
import { api } from "@/lib/ipc";
import { useJobs } from "@/lib/jobs-context";
import { healthQuery, keys, runtimeQuery, topicsQuery } from "@/lib/queries";
import { cn } from "@/lib/utils";

const MIN_DOCUMENTS = 6;

export function TopicsView() {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const topics = useQuery(topicsQuery);
  const health = useQuery(healthQuery);
  const runtime = useQuery(runtimeQuery);
  const documents = useQuery({
    queryKey: keys.documents({ limit: 500 }),
    queryFn: () => api.listDocuments({ limit: 500 }),
  });
  const { active } = useJobs();
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const { roots, byId } = useMemo(() => nestTopics(topics.data?.topics ?? []), [topics.data]);
  const titles = new Map((documents.data?.items ?? []).map((d) => [d.id, d.title]));
  const selected = (selectedId ? byId.get(selectedId) : undefined) ?? roots[0] ?? null;
  const building = topics.data?.status === "building" || active.some((j) => j.kind === "recluster");

  const recluster = useMutation({
    mutationFn: api.recluster,
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: keys.topics }),
    onError: (error: Error) =>
      toast.error(t("topics.reclusterFailed"), { description: errorText(error) }),
  });

  let empty: React.ReactNode = null;
  if (topics.data && roots.length === 0) {
    if (building) {
      empty = (
        <span className="flex items-center gap-2">
          <Loader2Icon className="size-3.5 animate-spin" />
          {t("topics.building")}
        </span>
      );
    } else if (runtime.data && runtime.data.state !== "ready") {
      empty = (
        <span>
          {t("topics.noEmbedder")}{" "}
          <Link to="/models" className="underline underline-offset-2 hover:text-foreground">
            {t("topics.openModels")}
          </Link>
        </span>
      );
    } else if ((health.data?.index.documents ?? 0) < MIN_DOCUMENTS) {
      empty = t("topics.under6");
    } else {
      empty = t("topics.empty");
    }
  }

  return (
    <Page>
      <PageHeader
        title={t("topics.title")}
        description={
          topics.data?.built_at
            ? `${t("topics.built", { when: relativeTime(topics.data.built_at) })} · ${t(
                "topics.changed",
                { count: topics.data.changed_since },
              )}`
            : undefined
        }
        actions={
          <Button
            size="sm"
            variant="secondary"
            disabled={building || recluster.isPending}
            onClick={() => recluster.mutate()}
          >
            {building ? (
              <Loader2Icon className="size-3.5 animate-spin" />
            ) : (
              <RefreshCwIcon className="size-3.5" />
            )}
            {building ? t("topics.reclustering") : t("topics.recluster")}
          </Button>
        }
      />
      <PageBody>
        {empty ? (
          <p className="px-5 py-10 text-sm text-muted-foreground">{empty}</p>
        ) : (
          <div className="grid min-h-0 grid-cols-[minmax(14rem,18rem)_1fr]">
            <ul className="space-y-0.5 border-r border-border p-2">
              {roots.map((node) => (
                <TopicRow
                  key={node.id}
                  node={node}
                  depth={0}
                  selectedId={selected?.id ?? null}
                  onSelect={setSelectedId}
                />
              ))}
            </ul>
            {selected ? (
              <TopicDetail node={selected} titles={titles} onSelect={setSelectedId} />
            ) : null}
          </div>
        )}
      </PageBody>
    </Page>
  );
}

function TopicRow({
  node,
  depth,
  selectedId,
  onSelect,
}: {
  node: TopicNode;
  depth: number;
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);
  return (
    <li>
      <div
        className={cn(
          "flex items-center gap-1 rounded-md pr-2 text-[0.8125rem]",
          node.id === selectedId ? "bg-primary/12 text-primary" : "hover:bg-sidebar-accent",
        )}
        style={{ paddingLeft: 4 + depth * 14 }}
      >
        {node.children.length ? (
          <button
            type="button"
            aria-label={open ? t("topics.collapse") : t("topics.expand")}
            aria-expanded={open}
            onClick={() => setOpen(!open)}
            className="grid size-5 place-items-center text-muted-foreground"
          >
            <ChevronRightIcon className={cn("size-3.5 transition-transform", open && "rotate-90")} />
          </button>
        ) : (
          <span className="size-5" />
        )}
        <button
          type="button"
          onClick={() => onSelect(node.id)}
          className="min-w-0 flex-1 truncate py-1.5 text-left"
        >
          {node.label}
        </button>
        <span className="text-[0.6875rem] tabular-nums text-muted-foreground">
          {node.doc_ids.length}
        </span>
      </div>
      {open && node.children.length ? (
        <ul>
          {node.children.map((child) => (
            <TopicRow
              key={child.id}
              node={child}
              depth={depth + 1}
              selectedId={selectedId}
              onSelect={onSelect}
            />
          ))}
        </ul>
      ) : null}
    </li>
  );
}

function TopicDetail({
  node,
  titles,
  onSelect,
}: {
  node: TopicNode;
  titles: Map<string, string>;
  onSelect: (id: string) => void;
}) {
  const { t } = useTranslation();
  return (
    <div className="min-w-0 space-y-5 px-6 py-5">
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <h2 className="truncate text-lg font-semibold tracking-tight">{node.label}</h2>
          <p className="text-xs text-muted-foreground">
            {t("topics.documents", { count: node.doc_ids.length })}
          </p>
        </div>
        <Button asChild size="sm">
          <Link to="/chat" search={{ topic: node.id }}>
            <MessageSquarePlusIcon className="size-3.5" />
            {t("topics.ask")}
          </Link>
        </Button>
      </div>

      <section className="space-y-1.5">
        <h3 className="text-xs font-medium text-muted-foreground">{t("topics.keywords")}</h3>
        <div className="flex flex-wrap gap-1.5">
          {node.keywords.map((keyword) => (
            <Badge key={keyword} variant="secondary" className="font-normal">
              {keyword}
            </Badge>
          ))}
        </div>
      </section>

      {node.children.length ? (
        <section className="space-y-1.5">
          <h3 className="text-xs font-medium text-muted-foreground">{t("topics.subtopics")}</h3>
          <div className="flex flex-wrap gap-1.5">
            {node.children.map((child) => (
              <button
                key={child.id}
                type="button"
                onClick={() => onSelect(child.id)}
                className="rounded-md border border-border px-2 py-1 text-xs hover:border-muted-foreground/40"
              >
                {child.label} · {child.doc_ids.length}
              </button>
            ))}
          </div>
        </section>
      ) : null}

      <section className="space-y-1.5">
        <h3 className="text-xs font-medium text-muted-foreground">{t("topics.docs")}</h3>
        <ul className="space-y-0.5">
          {node.doc_ids.map((docId) => (
            <li key={docId}>
              <Link
                to="/reader/$docId"
                params={{ docId }}
                className="block truncate rounded-md px-2 py-1 text-[0.8125rem] hover:bg-sidebar-accent"
              >
                {titles.get(docId) ?? docId}
              </Link>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}
