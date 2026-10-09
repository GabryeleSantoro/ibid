import { useCallback, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";

import {
  type Citation,
  type Grounding,
  type QueryFilters,
  type RetrievedChunk,
  type StageLatency,
  type StreamHandle,
  streamQuery,
} from "@/lib/ipc";
import { keys } from "@/lib/queries";

export type PendingAnswer = {
  question: string;
  status: "retrieving" | "streaming" | "done" | "cancelled" | "error";
  mode: "local" | "global" | null;
  modeReason: string | null;
  chunks: RetrievedChunk[];
  candidates: number;
  text: string;
  citations: Citation[];
  dropped: number;
  grounding: Grounding | null;
  latency: StageLatency | null;
  remote: boolean;
  tokensPerS: number | null;
  error: string | null;
};

const EMPTY: Omit<PendingAnswer, "question"> = {
  status: "retrieving",
  mode: null,
  modeReason: null,
  chunks: [],
  candidates: 0,
  text: "",
  citations: [],
  dropped: 0,
  grounding: null,
  latency: null,
  remote: false,
  tokensPerS: null,
  error: null,
};

/**
 * Drives one question through the SSE pipeline.
 *
 * The frames arrive in a fixed order, so the reducer below is a straight
 * translation of the protocol rather than a general event bus.
 */
export function useChat(sessionId: string | null) {
  const queryClient = useQueryClient();
  const [pending, setPending] = useState<PendingAnswer | null>(null);
  const [activeSession, setActiveSession] = useState<string | null>(sessionId);
  const handle = useRef<StreamHandle | null>(null);

  const isStreaming = pending?.status === "retrieving" || pending?.status === "streaming";

  const send = useCallback(
    (question: string, options: { mode?: "auto" | "local" | "global"; filters?: QueryFilters } = {}) => {
      if (isStreaming) return;
      setPending({ question, ...EMPTY });

      handle.current = streamQuery(
        {
          q: question,
          session_id: sessionId,
          mode: options.mode ?? "auto",
          filters: options.filters ?? {},
        },
        {
          onEvent: (frame) => {
            setPending((current) => {
              if (!current) return current;
              switch (frame.event) {
                case "start":
                  setActiveSession(frame.data.session_id);
                  return current;
                case "mode":
                  return { ...current, mode: frame.data.mode, modeReason: frame.data.reason };
                case "sources":
                  return {
                    ...current,
                    chunks: frame.data.chunks,
                    candidates: frame.data.candidates,
                    status: "streaming",
                  };
                case "token":
                  return { ...current, status: "streaming", text: current.text + frame.data.text };
                case "citations":
                  return {
                    ...current,
                    citations: frame.data.citations,
                    dropped: frame.data.dropped,
                    grounding: frame.data.grounding,
                  };
                case "done":
                  return {
                    ...current,
                    status: "done",
                    latency: frame.data.latency,
                    remote: frame.data.remote,
                    tokensPerS: frame.data.tokens_per_s,
                  };
                case "error":
                  return { ...current, status: "error", error: frame.data.message };
                default:
                  return current;
              }
            });
          },
          onClosed: (reason) => {
            setPending((current) => {
              if (!current) return current;
              if (current.status === "done" || current.status === "error") return current;
              return { ...current, status: reason === "cancelled" ? "cancelled" : "done" };
            });
            // The sidecar persisted the turn; refresh the history it owns.
            void queryClient.invalidateQueries({ queryKey: keys.sessions });
          },
          onFailed: (message) => {
            setPending((current) =>
              current ? { ...current, status: "error", error: message } : current,
            );
          },
        },
      );
    },
    [isStreaming, queryClient, sessionId],
  );

  const cancel = useCallback(() => {
    void handle.current?.cancel();
  }, []);

  const reset = useCallback(() => setPending(null), []);

  return { pending, isStreaming, send, cancel, reset, activeSession };
}
