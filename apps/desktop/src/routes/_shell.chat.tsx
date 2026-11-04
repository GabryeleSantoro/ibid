import { Outlet, createFileRoute } from "@tanstack/react-router";

import { ChatSessionSidebar } from "@/features/chat/session-sidebar";

type ChatSearch = { topic?: string };

export const Route = createFileRoute("/_shell/chat")({
  validateSearch: (search: Record<string, unknown>): ChatSearch => ({
    topic: typeof search.topic === "string" ? search.topic : undefined,
  }),
  component: ChatSection,
});

function ChatSection() {
  return (
    <>
      <ChatSessionSidebar />
      <Outlet />
    </>
  );
}
