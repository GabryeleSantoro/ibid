import React from "react";
import ReactDOM from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { RouterProvider, createRouter } from "@tanstack/react-router";

import { ThemeProvider } from "@/components/shell/theme-provider";
import { TooltipProvider } from "@/components/ui/tooltip";

import { applyFontSize, getFontSize } from "@/lib/font-size";
import { detectSystemLocale } from "@/lib/i18n";
import { routeTree } from "./routeTree.gen";
import "./app.css";

void detectSystemLocale();
applyFontSize(getFontSize());

// The shell's native file-drop handler is off (on macOS it swallowed every in-page
// drag); without it a file dropped on the window would replace the app with that file.
for (const type of ["dragover", "drop"] as const) {
  window.addEventListener(type, (event) => {
    if (event.dataTransfer?.types.includes("Files")) event.preventDefault();
  });
}

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // The sidecar is on loopback: refetching on focus buys nothing and
      // costs a request storm every time the window is raised.
      refetchOnWindowFocus: false,
      retry: 1,
      staleTime: 5_000,
    },
  },
});

const router = createRouter({
  routeTree,
  defaultPreload: "intent",
  context: { queryClient },
});

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <TooltipProvider delayDuration={400}>
          <RouterProvider router={router} />
        </TooltipProvider>
      </ThemeProvider>
    </QueryClientProvider>
  </React.StrictMode>,
);
