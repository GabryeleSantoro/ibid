import { describe, expect, it } from "vitest";

import type { Topic } from "@/lib/ipc";

import { nestTopics } from "./tree";

const topic = (id: string, parent_id: string | null): Topic => ({
  id,
  parent_id,
  level: parent_id ? 1 : 0,
  label: id,
  keywords: [],
  doc_ids: [],
  rep_chunk_ids: [],
});

describe("nestTopics", () => {
  it("hangs children under their parent in server order", () => {
    const { roots } = nestTopics([
      topic("t1", null),
      topic("t1.1", "t1"),
      topic("t2", null),
      topic("t1.2", "t1"),
    ]);

    expect(roots.map((node) => node.id)).toEqual(["t1", "t2"]);
    expect(roots[0].children.map((node) => node.id)).toEqual(["t1.1", "t1.2"]);
    expect(roots[1].children).toEqual([]);
  });

  it("indexes every node by id", () => {
    const { byId } = nestTopics([topic("t1", null), topic("t1.1", "t1")]);

    expect(byId.get("t1.1")?.parent_id).toBe("t1");
    expect(byId.get("t1")?.children).toHaveLength(1);
  });

  it("keeps a node whose parent is missing as a root", () => {
    expect(nestTopics([topic("t3.1", "t3")]).roots.map((node) => node.id)).toEqual(["t3.1"]);
  });
});
