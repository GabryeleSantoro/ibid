// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { Citation } from "@/lib/ipc";
import "@/lib/i18n";

import { AnswerText } from "./answer-text";

afterEach(cleanup);

const cite = (marker: string) => ({ marker }) as Citation;

describe("AnswerText citations", () => {
  it("numbers known markers by citation order", () => {
    render(<AnswerText text="A [d1:2] and B [d2:5]." citations={[cite("[d1:2]"), cite("[d2:5]")]} />);
    expect(screen.getAllByRole("button").map((b) => b.textContent)).toEqual(["1", "2"]);
  });

  it("flags a marker with no matching citation as unknown", () => {
    render(<AnswerText text="Claim [ghost:9]." citations={[]} />);
    const chip = screen.getByRole("button");
    expect(chip.textContent).toBe("?");
    expect(chip.className).toContain("line-through");
  });

  it("reports the clicked target", () => {
    const onSelect = vi.fn();
    const c = cite("[d1:2]");
    render(<AnswerText text="See [d1:2]" citations={[c]} onSelect={onSelect} />);
    fireEvent.click(screen.getByRole("button"));
    expect(onSelect).toHaveBeenCalledWith({ docId: "d1", page: 2, citation: c });
  });

  it("renders bullets and skips blank lines", () => {
    const { container } = render(<AnswerText text={"- one\n\n- two"} citations={[]} />);
    expect(container.querySelectorAll("p")).toHaveLength(2);
  });
});
