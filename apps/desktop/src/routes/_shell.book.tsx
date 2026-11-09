import { createFileRoute } from "@tanstack/react-router";

import { BookView } from "@/features/book/book-view";

export const Route = createFileRoute("/_shell/book")({
  component: BookView,
});
