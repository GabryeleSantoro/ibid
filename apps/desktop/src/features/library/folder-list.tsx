import { useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { FolderIcon, FolderPlusIcon, PencilIcon, Trash2Icon } from "lucide-react";
import { toast } from "sonner";

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
import { Input } from "@/components/ui/input";
import { errorText } from "@/lib/errors";
import { api, type Folder } from "@/lib/ipc";
import { foldersQuery, keys } from "@/lib/queries";
import { cn } from "@/lib/utils";

/** User-made folders. Virtual: filing a document never moves the file on disk. */
export function FolderList({
  selected,
  onSelect,
}: {
  selected: string | null;
  onSelect: (folderId: string | null) => void;
}) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const folders = useQuery(foldersQuery);
  const [creating, setCreating] = useState(false);

  const invalidate = () => {
    void queryClient.invalidateQueries({ queryKey: keys.folders });
    void queryClient.invalidateQueries({ queryKey: ["documents"] });
  };
  const onError = (error: Error) => toast.error(t("library.folderNotSaved"), { description: errorText(error) });

  const create = useMutation({
    mutationFn: api.createFolder,
    onSuccess: (folder) => {
      setCreating(false);
      invalidate();
      onSelect(folder.id);
    },
    onError,
  });
  const rename = useMutation({
    mutationFn: ({ id, name }: { id: string; name: string }) => api.renameFolder(id, name),
    onSuccess: invalidate,
    onError,
  });
  const remove = useMutation({
    mutationFn: api.deleteFolder,
    onSuccess: (_result, id) => {
      if (selected === id) onSelect(null);
      invalidate();
    },
    onError,
  });

  return (
    <div className="mt-3">
      <div className="flex items-center justify-between px-2 py-1">
        <span className="text-[0.6875rem] font-medium tracking-wide text-muted-foreground uppercase">
          {t("library.folders")}
        </span>
        <Button
          variant="ghost"
          size="icon"
          className="size-6"
          aria-label={t("library.newFolder")}
          onClick={() => setCreating(true)}
        >
          <FolderPlusIcon className="size-3.5" />
        </Button>
      </div>

      {creating ? (
        <NameInput
          placeholder={t("library.folderName")}
          onSubmit={(name) => create.mutate(name)}
          onCancel={() => setCreating(false)}
        />
      ) : null}

      {folders.data?.length === 0 && !creating ? (
        <p className="px-2 py-1 text-[0.6875rem] text-muted-foreground">
          {t("library.foldersHint")}
        </p>
      ) : null}

      <div className="flex flex-col gap-0.5">
        {(folders.data ?? []).map((folder) => (
          <FolderRow
            key={folder.id}
            folder={folder}
            active={selected === folder.id}
            onSelect={() => onSelect(folder.id)}
            onRename={(name) => rename.mutate({ id: folder.id, name })}
            onDelete={() => remove.mutate(folder.id)}
          />
        ))}
      </div>
    </div>
  );
}

function FolderRow({
  folder,
  active,
  onSelect,
  onRename,
  onDelete,
}: {
  folder: Folder;
  active: boolean;
  onSelect: () => void;
  onRename: (name: string) => void;
  onDelete: () => void;
}) {
  const { t } = useTranslation();
  const [editing, setEditing] = useState(false);

  if (editing) {
    return (
      <NameInput
        initial={folder.name}
        onSubmit={(name) => {
          setEditing(false);
          if (name !== folder.name) onRename(name);
        }}
        onCancel={() => setEditing(false)}
      />
    );
  }

  return (
    <div
      className={cn(
        "group flex items-center rounded-md pr-1 transition-colors",
        active ? "bg-sidebar-accent" : "hover:bg-sidebar-accent/60",
      )}
    >
      <button
        type="button"
        onClick={onSelect}
        onDoubleClick={() => setEditing(true)}
        className="flex min-w-0 flex-1 items-center gap-1.5 px-2 py-1.5 text-left"
      >
        <FolderIcon className="size-3.5 shrink-0 text-primary" />
        <span className={cn("truncate text-[0.8125rem]", active && "font-medium")}>{folder.name}</span>
        <span className="ml-auto pl-1 text-[0.6875rem] text-muted-foreground">{folder.doc_ids.length}</span>
      </button>

      <div className="flex opacity-0 transition-opacity group-hover:opacity-100 focus-within:opacity-100">
        <Button
          variant="ghost"
          size="icon"
          className="size-6"
          aria-label={t("library.renameNamed", { name: folder.name })}
          onClick={() => setEditing(true)}
        >
          <PencilIcon className="size-3" />
        </Button>
        <AlertDialog>
          <AlertDialogTrigger asChild>
            <Button variant="ghost" size="icon" className="size-6" aria-label={t("chat.deleteNamed", { name: folder.name })}>
              <Trash2Icon className="size-3" />
            </Button>
          </AlertDialogTrigger>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>{t("library.deleteFolderTitle", { name: folder.name })}</AlertDialogTitle>
              <AlertDialogDescription>
                {t("library.deleteFolderBody", { count: folder.doc_ids.length })}
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel>{t("common.cancel")}</AlertDialogCancel>
              <AlertDialogAction
                onClick={onDelete}
                className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
              >
                {t("library.deleteFolder")}
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      </div>
    </div>
  );
}

export function NameInput({
  initial = "",
  placeholder,
  onSubmit,
  onCancel,
}: {
  initial?: string;
  placeholder?: string;
  onSubmit: (name: string) => void;
  onCancel: () => void;
}) {
  const [value, setValue] = useState(initial);
  // Enter and Escape unmount the input, which fires a blur: settle only once.
  const settled = useRef(false);
  const settle = (save: boolean) => {
    if (settled.current) return;
    settled.current = true;
    if (save && value.trim()) onSubmit(value.trim());
    else onCancel();
  };

  return (
    <Input
      autoFocus
      value={value}
      placeholder={placeholder}
      onChange={(event) => setValue(event.target.value)}
      onBlur={() => settle(true)}
      onKeyDown={(event) => {
        if (event.key === "Enter") settle(true);
        if (event.key === "Escape") settle(false);
      }}
      className="my-0.5 h-7 text-[0.8125rem]"
    />
  );
}
