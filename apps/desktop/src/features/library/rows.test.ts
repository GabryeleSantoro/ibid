import { describe, expect, it } from "vitest";

import type { Document } from "@/lib/ipc";

import { nextSort, visibleRows } from "./rows";

const doc = (title: string, extra: Partial<Document> = {}): Document => ({
  id: title,
  source_id: "s",
  path: `/docs/${title}`,
  title,
  ext: ".pdf",
  mime: "application/pdf",
  size_bytes: 100,
  n_pages: 1,
  n_chunks: 1,
  lang: null,
  status: "indexed",
  error: null,
  keywords: [],
  mtime: "2026-01-01T00:00:00Z",
  indexed_at: null,
  ...extra,
});

const none = { statuses: new Set<never>(), exts: new Set<string>() };
const titles = (rows: Document[]) => rows.map((row) => row.title);

describe("visibleRows", () => {
  const items = [
    doc("Chapter 10", { n_pages: 30, ext: ".md", mtime: "2026-03-01T00:00:00Z" }),
    doc("chapter 2", { n_pages: null, status: "error" }),
    doc("Appendix", { n_pages: 5, mtime: "2026-02-01T00:00:00Z" }),
  ];

  it("sorts titles naturally and case-insensitively", () => {
    const rows = visibleRows(items, none, { key: "title", desc: false });
    expect(titles(rows)).toEqual(["Appendix", "chapter 2", "Chapter 10"]);
  });

  it("sorts numbers descending with missing pages last", () => {
    const rows = visibleRows(items, none, { key: "n_pages", desc: true });
    expect(titles(rows)).toEqual(["Chapter 10", "Appendix", "chapter 2"]);
  });

  it("sorts dates by time, not by string", () => {
    const rows = visibleRows(items, none, { key: "mtime", desc: true });
    expect(titles(rows)).toEqual(["Chapter 10", "Appendix", "chapter 2"]);
  });

  it("filters by status and extension together", () => {
    const sort = { key: "title", desc: false } as const;
    expect(titles(visibleRows(items, { ...none, statuses: new Set(["error"]) }, sort))).toEqual([
      "chapter 2",
    ]);
    expect(titles(visibleRows(items, { ...none, exts: new Set([".pdf"]) }, sort))).toEqual([
      "Appendix",
      "chapter 2",
    ]);
  });
});

describe("nextSort", () => {
  it("flips direction on the same column, picks a default on a new one", () => {
    expect(nextSort({ key: "title", desc: false }, "title")).toEqual({ key: "title", desc: true });
    expect(nextSort({ key: "title", desc: false }, "size_bytes")).toEqual({
      key: "size_bytes",
      desc: true,
    });
    expect(nextSort({ key: "mtime", desc: true }, "status")).toEqual({ key: "status", desc: false });
  });
});
