/**
 * Query keys and shared fetchers.
 *
 * Kept in one place so an invalidation in one feature cannot miss a cache the
 * other features read from.
 */

import { queryOptions } from "@tanstack/react-query";

import { api, shell } from "./ipc";

export const keys = {
  health: ["health"] as const,
  hardware: ["hardware"] as const,
  sidecars: ["sidecars"] as const,
  logs: ["sidecar-logs"] as const,
  settings: ["settings"] as const,
  localModel: ["local-model"] as const,
  runtime: ["runtime"] as const,
  sources: ["sources"] as const,
  folders: ["folders"] as const,
  documents: (params: Record<string, unknown>) => ["documents", params] as const,
  document: (id: string) => ["document", id] as const,
  content: (id: string) => ["content", id] as const,
  jobs: ["jobs"] as const,
  sessions: ["sessions"] as const,
  projects: ["chat-projects"] as const,
  messages: (id: string) => ["messages", id] as const,
  connections: ["connections"] as const,
  models: ["models"] as const,
  hubSearch: (params: Record<string, unknown>) => ["hub-search", params] as const,
  hubRecommended: (role: string) => ["hub-recommended", role] as const,
  hubDetail: (repoId: string) => ["hub-detail", repoId] as const,
  evalSets: ["eval-sets"] as const,
};

export const healthQuery = queryOptions({
  queryKey: keys.health,
  queryFn: api.health,
  refetchInterval: 5000,
});

export const hardwareQuery = queryOptions({
  queryKey: keys.hardware,
  queryFn: shell.hardware,
  staleTime: Infinity,
});

export const sidecarsQuery = queryOptions({
  queryKey: keys.sidecars,
  queryFn: shell.sidecars,
  refetchInterval: 2000,
});

export const logsQuery = queryOptions({
  queryKey: keys.logs,
  queryFn: shell.logs,
  refetchInterval: 1500,
});

export const settingsQuery = queryOptions({
  queryKey: keys.settings,
  queryFn: api.settings,
});

export const sourcesQuery = queryOptions({
  queryKey: keys.sources,
  queryFn: api.listSources,
});

export const foldersQuery = queryOptions({
  queryKey: keys.folders,
  queryFn: api.listFolders,
});

export const sessionsQuery = queryOptions({
  queryKey: keys.sessions,
  queryFn: api.listSessions,
});

export const projectsQuery = queryOptions({
  queryKey: keys.projects,
  queryFn: api.listProjects,
});

export const connectionsQuery = queryOptions({
  queryKey: keys.connections,
  queryFn: api.listConnections,
});

export const evalSetsQuery = queryOptions({
  queryKey: keys.evalSets,
  queryFn: api.evalSets,
  staleTime: Infinity,
});

export const runtimeQuery = queryOptions({
  queryKey: keys.runtime,
  queryFn: api.runtime,
  refetchInterval: (query) => (query.state.data?.state === "downloading" ? 1000 : false),
});

export const modelsQuery = queryOptions({
  queryKey: keys.models,
  queryFn: api.models,
});
