import { afterEach, describe, expect, it, vi } from "vitest";

import {
  loadSlowState,
  recordAnswer,
  saveSlowState,
  shouldShowBanner,
  shouldSuggest,
} from "./slow-speed";

const fresh = { streak: 0, dismissed: false };

describe("recordAnswer", () => {
  it("two slow answers in a row trigger the suggestion", () => {
    const one = recordAnswer(fresh, 3);
    expect(shouldSuggest(one)).toBe(false);
    expect(shouldSuggest(recordAnswer(one, 4))).toBe(true);
  });
  it("a fast answer resets the streak", () => {
    const slow = recordAnswer(recordAnswer(fresh, 2), 2);
    expect(recordAnswer(slow, 20).streak).toBe(0);
  });
  it("answers without a speed change nothing", () => {
    const one = recordAnswer(fresh, 3);
    expect(recordAnswer(one, null)).toEqual(one);
  });
  it("a dismissed suggestion never comes back", () => {
    const state = { streak: 5, dismissed: true };
    expect(shouldSuggest(state)).toBe(false);
    expect(shouldSuggest(recordAnswer(state, 1))).toBe(false);
  });
});

describe("shouldShowBanner", () => {
  const slow = { streak: 2, dismissed: false };

  it("shows under a finished slow local answer", () => {
    expect(shouldShowBanner(slow, 3, true)).toBe(true);
  });
  it("never shows under an answer with no local speed (cloud or stub)", () => {
    expect(shouldShowBanner(slow, null, true)).toBe(false);
  });
  it("waits for the answer to finish", () => {
    expect(shouldShowBanner(slow, 3, false)).toBe(false);
  });
});

describe("storage", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("round-trips", () => {
    const store = new Map<string, string>();
    vi.stubGlobal("localStorage", {
      getItem: (k: string) => store.get(k) ?? null,
      setItem: (k: string, v: string) => void store.set(k, v),
    });
    saveSlowState({ streak: 1, dismissed: true });
    expect(loadSlowState()).toEqual({ streak: 1, dismissed: true });
  });
  it("survives unavailable or corrupt storage", () => {
    vi.stubGlobal("localStorage", {
      getItem: () => {
        throw new Error("denied");
      },
      setItem: () => {
        throw new Error("denied");
      },
    });
    expect(loadSlowState()).toEqual(fresh);
    expect(() => saveSlowState(fresh)).not.toThrow();
    vi.stubGlobal("localStorage", { getItem: () => "{nope", setItem: () => {} });
    expect(loadSlowState()).toEqual(fresh);
  });
});
