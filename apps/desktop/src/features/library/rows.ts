import type { Document, DocumentStatus } from "@/lib/ipc";

export type SortKey = "title" | "status" | "n_pages" | "n_chunks" | "size_bytes" | "mtime";
export type Sort = { key: SortKey; desc: boolean };
export type Filters = { statuses: Set<DocumentStatus>; exts: Set<string> };

const collator = new Intl.Collator(undefined, { numeric: true, sensitivity: "base" });

/** First click on a column: text ascending, numbers and dates largest/newest first. */
export function nextSort(current: Sort, key: SortKey): Sort {
  if (current.key === key) return { key, desc: !current.desc };
  return { key, desc: key !== "title" && key !== "status" };
}

/**
 * Filter and sort the loaded page of documents. `statusLabel` sorts the status
 * column by what the user reads, not by the raw enum.
 */
// ponytail: client-side over the loaded page (limit 500); move to the API if libraries outgrow it.
export function visibleRows(
  items: Document[],
  { statuses, exts }: Filters,
  { key, desc }: Sort,
  statusLabel: (status: DocumentStatus) => string = (status) => status,
): Document[] {
  const value = (document: Document) =>
    key === "title"
      ? document.title
      : key === "status"
        ? statusLabel(document.status)
        : key === "mtime"
          ? Date.parse(document.mtime)
          : (document[key] ?? -1);
  return items
    .filter(
      (document) =>
        (statuses.size === 0 || statuses.has(document.status)) &&
        (exts.size === 0 || exts.has(document.ext)),
    )
    .sort((left, right) => {
      const a = value(left);
      const b = value(right);
      const order =
        typeof a === "string" ? collator.compare(a, b as string) : a - (b as number);
      return desc ? -order : order;
    });
}
