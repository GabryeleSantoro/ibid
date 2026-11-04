import { createFileRoute } from "@tanstack/react-router";

import { TopicsView } from "@/features/topics/topics-view";

export const Route = createFileRoute("/_shell/topics")({
  component: TopicsView,
});
