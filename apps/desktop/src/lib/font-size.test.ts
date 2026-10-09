import { afterEach, describe, expect, it, vi } from "vitest";
import { getFontSize, parseFontSize } from "./font-size";

describe("font size", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("accepts s, m, l and falls back to m for anything else", () => {
    expect(parseFontSize("s")).toBe("s");
    expect(parseFontSize("l")).toBe("l");
    expect(parseFontSize("huge")).toBe("m");
    expect(parseFontSize(null)).toBe("m");
  });

  it("falls back to m when storage throws", () => {
    vi.stubGlobal("localStorage", {
      getItem: () => {
        throw new Error("blocked");
      },
    });
    expect(getFontSize()).toBe("m");
  });
});
