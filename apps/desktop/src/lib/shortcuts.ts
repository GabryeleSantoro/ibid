export type ShortcutId =
  | "newChat"
  | "focusComposer"
  | "openSettings"
  | "goChat"
  | "goLibrary"
  | "goConvert";

export const SHORTCUTS: { id: ShortcutId; key: string }[] = [
  { id: "newChat", key: "n" },
  { id: "focusComposer", key: "k" },
  { id: "openSettings", key: "," },
  { id: "goChat", key: "1" },
  { id: "goLibrary", key: "2" },
  { id: "goConvert", key: "3" },
];

export const COMPOSER_FOCUS_EVENT = "ibid:focus-composer";

type KeyLike = {
  key: string;
  metaKey: boolean;
  ctrlKey: boolean;
  altKey: boolean;
  shiftKey: boolean;
};

export function matchShortcut(event: KeyLike, mac: boolean): ShortcutId | null {
  const mod = mac ? event.metaKey && !event.ctrlKey : event.ctrlKey && !event.metaKey;
  if (!mod || event.altKey || event.shiftKey) return null;
  return SHORTCUTS.find((s) => s.key === event.key.toLowerCase())?.id ?? null;
}
