import { describe, expect, it } from "vitest";
import { matchShortcut } from "./shortcuts";

const key = (k: string, extra: Partial<Parameters<typeof matchShortcut>[0]> = {}) => ({
  key: k,
  metaKey: false,
  ctrlKey: false,
  altKey: false,
  shiftKey: false,
  ...extra,
});

describe("matchShortcut", () => {
  it("uses cmd on mac", () => {
    expect(matchShortcut(key("n", { metaKey: true }), true)).toBe("newChat");
    expect(matchShortcut(key(",", { metaKey: true }), true)).toBe("openSettings");
    expect(matchShortcut(key("2", { metaKey: true }), true)).toBe("goLibrary");
  });
  it("uses ctrl elsewhere and ignores cmd", () => {
    expect(matchShortcut(key("k", { ctrlKey: true }), false)).toBe("focusComposer");
    expect(matchShortcut(key("k", { metaKey: true }), false)).toBeNull();
  });
  it("never fires on a bare key (typing in inputs)", () => {
    expect(matchShortcut(key("n"), true)).toBeNull();
    expect(matchShortcut(key("1"), false)).toBeNull();
  });
  it("ignores extra modifiers and unknown keys", () => {
    expect(matchShortcut(key("n", { metaKey: true, shiftKey: true }), true)).toBeNull();
    expect(matchShortcut(key("n", { metaKey: true, altKey: true }), true)).toBeNull();
    expect(matchShortcut(key("z", { metaKey: true }), true)).toBeNull();
  });
});
