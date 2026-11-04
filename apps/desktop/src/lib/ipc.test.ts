import { beforeEach, describe, expect, it, vi } from "vitest";

const invoke = vi.fn();
const isTauri = vi.fn(() => true);

class FakeChannel<T> {
  onmessage: ((message: T) => void) | null = null;
}

vi.mock("@tauri-apps/api/core", () => ({
  invoke: (...args: unknown[]) => invoke(...args),
  isTauri: () => isTauri(),
  Channel: FakeChannel,
}));

const { api, IpcError, openStream, shell, streamEval, streamJobs, streamQuery } = await import(
  "./ipc"
);

/** The Channel the shell was handed, so a test can push frames through it. */
function channelFrom(call: number = 0): FakeChannel<unknown> {
  return invoke.mock.calls[call][1].channel as FakeChannel<unknown>;
}

beforeEach(() => {
  invoke.mockReset();
  invoke.mockResolvedValue(undefined);
  isTauri.mockReturnValue(true);
});

describe("request building", () => {
  it("reads the topic map and asks for a recluster", async () => {
    await api.getTopics();
    await api.recluster();

    expect(invoke).toHaveBeenNthCalledWith(1, "api_request", {
      req: { method: "GET", path: "/brain/topics", body: null },
    });
    expect(invoke).toHaveBeenNthCalledWith(2, "api_request", {
      req: { method: "POST", path: "/brain/recluster", body: null },
    });
  });

  it("sends method, path and body as one api_request", async () => {
    invoke.mockResolvedValue({ ok: true });

    await api.addSource({
      path: "/corpus",
      include_globs: ["**/*.md"],
      max_file_mb: 100,
      watch: true,
    });

    expect(invoke).toHaveBeenCalledWith("api_request", {
      req: {
        method: "POST",
        path: "/sources",
        body: {
          path: "/corpus",
          include_globs: ["**/*.md"],
          max_file_mb: 100,
          watch: true,
        },
      },
    });
  });

  it("sends an explicit null body when there is none", async () => {
    await api.health();

    expect(invoke.mock.calls[0][1].req.body).toBeNull();
  });

  it("interpolates ids into the path", async () => {
    await api.getContent("doc_1");

    expect(invoke.mock.calls[0][1].req.path).toBe("/documents/doc_1/content");
  });
});

describe("query strings", () => {
  it("omits undefined, null and empty filters", async () => {
    await api.listDocuments({ source_id: undefined, q: "", ext: ".md", limit: 50 });

    expect(invoke.mock.calls[0][1].req.path).toBe("/documents?ext=.md&limit=50");
  });

  it("produces no question mark when nothing is filtered", async () => {
    await api.listDocuments();

    expect(invoke.mock.calls[0][1].req.path).toBe("/documents");
  });

  it("keeps a zero, which is a real offset and not an empty value", async () => {
    await api.listDocuments({ offset: 0 });

    expect(invoke.mock.calls[0][1].req.path).toBe("/documents?offset=0");
  });

  it("escapes a value that would otherwise break the path", async () => {
    await api.listDocuments({ q: "rag & retrieval?" });

    expect(invoke.mock.calls[0][1].req.path).toBe("/documents?q=rag+%26+retrieval%3F");
  });

  it("sends the reindex acknowledgement when activating an embedder", async () => {
    await api.activateModel("embed-gemma-300m", true);

    expect(invoke.mock.calls[0][1].req.path).toBe(
      "/models/embed-gemma-300m/activate?accept_reindex=true",
    );
  });

  it("defaults the reindex acknowledgement to false", async () => {
    await api.activateModel("embed-gemma-300m");

    expect(invoke.mock.calls[0][1].req.path).toBe(
      "/models/embed-gemma-300m/activate?accept_reindex=false",
    );
  });
});

describe("errors", () => {
  it("explains the plain-browser case instead of failing obscurely", async () => {
    isTauri.mockReturnValue(false);

    await expect(api.health()).rejects.toBeInstanceOf(IpcError);
    await expect(api.health()).rejects.toThrow(/bun tauri dev/);
    expect(invoke).not.toHaveBeenCalled();
  });

  it("wraps a rejected command in an IpcError carrying its message", async () => {
    invoke.mockRejectedValue("404: document not found");

    await expect(api.getDocument("doc_nope")).rejects.toThrow("404: document not found");
  });

  it("stringifies a non-string rejection", async () => {
    invoke.mockRejectedValue(new Error("boom"));

    await expect(api.health()).rejects.toThrow(/boom/);
  });
});

describe("wipe", () => {
  it("always sends the literal confirmation the API demands", async () => {
    await api.wipe(true);

    expect(invoke.mock.calls[0][1].req.body).toEqual({
      confirm: "DELETE",
      keep_connections: true,
    });
  });
});

describe("shell commands", () => {
  it("passes the keychain arguments without a body wrapper", async () => {
    await shell.keychainSet("conn_1", "sk-secret");

    expect(invoke).toHaveBeenCalledWith("keychain_set", {
      connectionId: "conn_1",
      secret: "sk-secret",
    });
  });

  it("takes no arguments where the command needs none", async () => {
    await shell.hardware();

    expect(invoke).toHaveBeenCalledWith("hardware_info", undefined);
  });
});

describe("openStream", () => {
  it("mints a stream id up front so cancel works before the first frame", () => {
    const handle = openStream({ path: "/query" }, () => {});

    expect(handle.streamId).toMatch(/^s_/);
    expect(invoke.mock.calls[0][1].streamId).toBe(handle.streamId);
  });

  it("defaults to POST and passes a channel", () => {
    openStream({ path: "/query" }, () => {});

    const args = invoke.mock.calls[0][1];
    expect(args.method).toBe("POST");
    expect(args.channel).toBeInstanceOf(FakeChannel);
  });

  it("honours an explicit method", () => {
    openStream({ method: "GET", path: "/jobs/stream" }, () => {});

    expect(invoke.mock.calls[0][1].method).toBe("GET");
  });

  it("forwards each frame to the handler", () => {
    const frames: unknown[] = [];
    openStream({ path: "/query" }, (frame) => frames.push(frame));

    channelFrom().onmessage?.({ kind: "event", event: "token", data: { text: "hi" } });

    expect(frames).toEqual([{ kind: "event", event: "token", data: { text: "hi" } }]);
  });

  it("reports a command that never opened as a failed frame", async () => {
    invoke.mockRejectedValue("ragcore unreachable");
    const frames: unknown[] = [];

    const handle = openStream({ path: "/query" }, (frame) => frames.push(frame));
    await handle.done;

    expect(frames).toEqual([{ kind: "failed", message: "ragcore unreachable" }]);
  });

  it("keeps the error code on a failed frame so the UI can translate it", async () => {
    invoke.mockRejectedValue('{"code":"ragcore_unreachable","message":"down","params":{"reason":"x"}}');
    const frames: { kind: string; message?: string }[] = [];

    const handle = openStream({ path: "/query" }, (frame) => frames.push(frame));
    await handle.done;

    expect(JSON.parse(frames[0].message as string)).toEqual({
      code: "ragcore_unreachable",
      message: "down",
      params: { reason: "x" },
    });
  });

  it("cancels with the path the caller derived from the stream id", async () => {
    const handle = openStream(
      { path: "/query", cancelPath: (id) => `/query/${id}/cancel` },
      () => {},
    );

    await handle.cancel();

    expect(invoke).toHaveBeenLastCalledWith("api_cancel", {
      streamId: handle.streamId,
      cancelPath: `/query/${handle.streamId}/cancel`,
    });
  });

  it("cancels with no path when the caller gave none", async () => {
    const handle = openStream({ path: "/jobs/stream" }, () => {});

    await handle.cancel();

    expect(invoke.mock.calls[1][1].cancelPath).toBeNull();
  });
});

describe("streamQuery", () => {
  it("sends its own query id and the matching cancel path", () => {
    streamQuery({ q: "why rerank?", mode: "auto" }, { onEvent: () => {} });

    const args = invoke.mock.calls[0][1];
    expect(args.path).toBe("/query");
    expect(args.body.q).toBe("why rerank?");
    expect(args.body.query_id).toMatch(/^q_/);
  });

  it("asks for the answer in the UI language", () => {
    vi.stubGlobal("localStorage", { getItem: () => "de", setItem: () => {}, removeItem: () => {} });
    try {
      streamQuery({ q: "warum?", mode: "auto" }, { onEvent: () => {} });
    } finally {
      vi.unstubAllGlobals();
    }

    expect(invoke.mock.calls[0][1].body.lang).toBe("de");
  });

  it("routes event frames to onEvent and closed frames to onClosed", () => {
    const events: unknown[] = [];
    const closed: string[] = [];
    streamQuery(
      { q: "why rerank?", mode: "auto" },
      { onEvent: (event) => events.push(event), onClosed: (reason) => closed.push(reason) },
    );

    const channel = channelFrom();
    channel.onmessage?.({ kind: "event", event: "token", data: { text: "hi" } });
    channel.onmessage?.({ kind: "closed", reason: "complete" });

    expect(events).toEqual([{ event: "token", data: { text: "hi" } }]);
    expect(closed).toEqual(["complete"]);
  });

  it("routes a transport failure to onFailed", () => {
    const failures: string[] = [];
    streamQuery({ q: "why?", mode: "auto" }, { onEvent: () => {}, onFailed: (m) => failures.push(m) });

    channelFrom().onmessage?.({ kind: "failed", message: "ragcore unreachable" });

    expect(failures).toEqual(["ragcore unreachable"]);
  });

  it("keeps a sidecar-reported error as an ordinary event, not a failure", () => {
    const events: unknown[] = [];
    const failures: string[] = [];
    streamQuery(
      { q: "why?", mode: "auto" },
      { onEvent: (e) => events.push(e), onFailed: (m) => failures.push(m) },
    );

    channelFrom().onmessage?.({
      kind: "event",
      event: "error",
      data: { message: "boom", retryable: true },
    });

    expect(failures).toEqual([]);
    expect(events).toEqual([{ event: "error", data: { message: "boom", retryable: true } }]);
  });
});

describe("streamJobs", () => {
  it("opens the job stream with GET and no body", () => {
    streamJobs(() => {});

    const args = invoke.mock.calls[0][1];
    expect(args.method).toBe("GET");
    expect(args.path).toBe("/jobs/stream");
    expect(args.body).toBeNull();
  });

  it("delivers only job events", () => {
    const jobs: unknown[] = [];
    streamJobs((job) => jobs.push(job));

    const channel = channelFrom();
    channel.onmessage?.({ kind: "event", event: "job", data: { id: "job_index_0001" } });
    channel.onmessage?.({ kind: "event", event: "other", data: { id: "ignored" } });
    channel.onmessage?.({ kind: "closed", reason: "complete" });

    expect(jobs).toEqual([{ id: "job_index_0001" }]);
  });
});

describe("streamEval", () => {
  it("routes progress and result frames to onEvent", () => {
    const events: unknown[] = [];
    streamEval({ set_name: "base" }, { onEvent: (event) => events.push(event) });

    const channel = channelFrom();
    channel.onmessage?.({ kind: "event", event: "progress", data: { completed: 1 } });
    channel.onmessage?.({ kind: "event", event: "result", data: { metrics: {} } });

    expect(events.map((e) => (e as { event: string }).event)).toEqual(["progress", "result"]);
  });
});
