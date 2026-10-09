import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { errorText } from "@/lib/errors";
import { api, type Connection } from "@/lib/ipc";
import { keys } from "@/lib/queries";

/** Install, download progress, then Use: the built-in model's whole lifecycle in one slot. */
export function LocalModelControl({ connection }: { connection: Connection }) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const status = useQuery({
    queryKey: keys.localModel,
    queryFn: api.suggestionModel,
    refetchInterval: (query) => (query.state.data?.state === "downloading" ? 1000 : false),
  });
  const install = useMutation({
    mutationFn: api.installSuggestionModel,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: keys.localModel }),
  });
  const activate = useMutation({
    mutationFn: () => api.activateConnection("local"),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: keys.connections }),
    onError: (error: Error) => toast.error(errorText(error)),
  });

  const data = status.data;
  if (!data) return <Skeleton className="h-7 w-24" />;
  if (data.state === "downloading") {
    return (
      <span className="text-xs text-muted-foreground">
        {t("localModel.downloading", { percent: Math.round(data.progress * 100) })}
      </span>
    );
  }
  if (data.state === "missing") {
    return (
      <div className="flex items-center gap-2">
        {data.error ? (
          <span className="text-xs text-status-error">
            {t("localModel.failed", { error: data.error })}
          </span>
        ) : null}
        <Button size="sm" className="h-7" onClick={() => install.mutate()}>
          {t("localModel.install")}
        </Button>
      </div>
    );
  }
  return connection.active ? null : (
    <Button variant="secondary" size="sm" className="h-7" onClick={() => activate.mutate()}>
      {t("models.use")}
    </Button>
  );
}
