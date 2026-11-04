import type { Topic } from "@/lib/ipc";

export type TopicNode = Topic & { children: TopicNode[] };

/** The server sends the tree flat, parents first; the screen wants it nested. */
export function nestTopics(topics: Topic[]): { roots: TopicNode[]; byId: Map<string, TopicNode> } {
  const byId = new Map<string, TopicNode>(
    topics.map((topic) => [topic.id, { ...topic, children: [] }]),
  );
  const roots: TopicNode[] = [];
  for (const node of byId.values()) {
    const parent = node.parent_id ? byId.get(node.parent_id) : undefined;
    (parent ? parent.children : roots).push(node);
  }
  return { roots, byId };
}
