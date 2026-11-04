// @vitest-environment jsdom
import { act, renderHook } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { createElement, type ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const streamQuery = vi.fn((..._args: unknown[]) => ({
  streamId: "s",
  done: Promise.resolve(),
  cancel: vi.fn(),
}));
// queries.ts reads api.* at import time, so the mock carries empty api/shell objects.
vi.mock("@/lib/ipc", () => ({
  streamQuery: (...args: unknown[]) => streamQuery(...args),
  api: {},
  shell: {},
}));

const { useChat } = await import("./use-chat");

function wrapper({ children }: { children: ReactNode }) {
  return createElement(QueryClientProvider, { client: new QueryClient() }, children);
}

type Handlers = { onEvent: (event: unknown) => void };

beforeEach(() => streamQuery.mockClear());

describe("useChat", () => {
  it("sends the topic id while one is set", () => {
    const { result } = renderHook(() => useChat(null), { wrapper });

    act(() => result.current.send("What is here?", { topicId: "t2" }));

    expect(streamQuery.mock.calls[0][0]).toMatchObject({ q: "What is here?", topic_id: "t2" });
  });

  it("sends no topic id without one", () => {
    const { result } = renderHook(() => useChat(null), { wrapper });

    act(() => result.current.send("Why?"));

    expect(streamQuery.mock.calls[0][0]).toMatchObject({ topic_id: null });
  });

  it("keeps the topic scope the mode frame reports", () => {
    const { result } = renderHook(() => useChat(null), { wrapper });
    act(() => result.current.send("What is here?", { topicId: "t2" }));
    const handlers = streamQuery.mock.calls[0][1] as Handlers;

    act(() =>
      handlers.onEvent({
        event: "mode",
        data: { mode: "global", reason: "Asked about one topic", scope: "keychain · tokens" },
      }),
    );

    expect(result.current.pending?.modeScope).toBe("keychain · tokens");
  });
});
