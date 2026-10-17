import { useState } from "react";
import { useTranslation } from "react-i18next";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { open } from "@tauri-apps/plugin-dialog";
import { FolderOpenIcon } from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
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
import { Switch } from "@/components/ui/switch";
import { IconTooltip } from "@/components/ui/tooltip";
import { errorText } from "@/lib/errors";
import { api } from "@/lib/ipc";
import { keys } from "@/lib/queries";

const DEFAULT_INCLUDE =
  "**/*.md, **/*.txt, **/*.pdf, **/*.pptx, **/*.docx, **/*.html, **/*.htm, **/*.csv";
const DEFAULT_EXCLUDE = "**/node_modules/**, **/.git/**";

function globs(value: string): string[] {
  return value
    .split(",")
    .map((glob) => glob.trim())
    .filter(Boolean);
}

export function AddSourceDialog({
  trigger,
  triggerLabel,
  projectId,
}: {
  trigger: React.ReactElement;
  triggerLabel?: string;
  projectId?: string;
}) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const [openDialog, setOpenDialog] = useState(false);
  const [path, setPath] = useState("");
  const [include, setInclude] = useState(DEFAULT_INCLUDE);
  const [exclude, setExclude] = useState(DEFAULT_EXCLUDE);
  const [maxFileMb, setMaxFileMb] = useState(100);
  const [watch, setWatch] = useState(true);

  const add = useMutation({
    mutationFn: () =>
      api.addSource({
        path,
        include_globs: globs(include),
        exclude_globs: globs(exclude),
        max_file_mb: maxFileMb,
        watch,
        project_id: projectId,
      }),
    onSuccess: (source) => {
      void queryClient.invalidateQueries({ queryKey: keys.sources });
      void queryClient.invalidateQueries({ queryKey: ["documents"] });
      setOpenDialog(false);
      setPath("");
      toast.success(t("library.sourceAdded"), { description: source.path });
    },
    onError: (error: Error) => toast.error(t("library.addSourceFailed"), { description: errorText(error) }),
  });

  const pick = async () => {
    const selected = await open({ directory: true, multiple: false, title: t("library.chooseFolder") });
    if (typeof selected === "string") setPath(selected);
  };

  return (
    <Dialog open={openDialog} onOpenChange={setOpenDialog}>
      {triggerLabel ? (
        <IconTooltip label={triggerLabel}>
          <DialogTrigger asChild>{trigger}</DialogTrigger>
        </IconTooltip>
      ) : (
        <DialogTrigger asChild>{trigger}</DialogTrigger>
      )}
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>{t("library.addSourceTitle")}</DialogTitle>
          <DialogDescription>
            {t("library.addSourceLead")}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          <div className="space-y-1.5">
            <Label htmlFor="source-path">{t("library.folder")}</Label>
            <div className="flex gap-2">
              <Input
                id="source-path"
                value={path}
                placeholder="/Users/you/Documents/research"
                onChange={(event) => setPath(event.target.value)}
              />
              <Button type="button" variant="secondary" onClick={pick}>
                <FolderOpenIcon className="size-4" />
                {t("library.browse")}
              </Button>
            </div>
          </div>

          <div className="grid gap-4 sm:grid-cols-2">
            <div className="space-y-1.5">
              <Label htmlFor="include">{t("library.include")}</Label>
              <Input id="include" value={include} onChange={(e) => setInclude(e.target.value)} />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="exclude">{t("library.exclude")}</Label>
              <Input id="exclude" value={exclude} onChange={(e) => setExclude(e.target.value)} />
            </div>
          </div>

          <div className="flex items-end gap-6">
            <div className="w-32 space-y-1.5">
              <Label htmlFor="max-mb">{t("library.maxFileSize")}</Label>
              <div className="flex items-center gap-2">
                <Input
                  id="max-mb"
                  type="number"
                  min={1}
                  value={maxFileMb}
                  onChange={(event) => setMaxFileMb(Number(event.target.value) || 1)}
                />
                <span className="text-xs text-muted-foreground">MB</span>
              </div>
            </div>
            <div className="flex items-center gap-2 pb-2">
              <Switch id="watch" checked={watch} onCheckedChange={setWatch} />
              <Label htmlFor="watch" className="font-normal">
                {t("library.watch")}
              </Label>
            </div>
          </div>
        </div>

        <DialogFooter>
          <Button variant="ghost" onClick={() => setOpenDialog(false)}>
            {t("common.cancel")}
          </Button>
          <Button disabled={!path.trim() || add.isPending} onClick={() => add.mutate()}>
            {add.isPending ? t("library.adding") : t("library.addSource")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
