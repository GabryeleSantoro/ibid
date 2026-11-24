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

  it("renders italic, bold, inline and display math, leaves prices as text", () => {
    const { container } = render(
      <AnswerText
        text={"Un *MDP* è **una tupla** $P(s' \\mid s,a)$ [d1:2]\n$$\\gamma \\in [0,1)$$\ncosta $5 e $10"}
        citations={[cite("[d1:2]")]}
      />,
    );
    expect(container.querySelector("em")?.textContent).toBe("MDP");
    expect(container.querySelector("strong")?.textContent).toBe("una tupla");
    expect(container.querySelectorAll(".katex")).toHaveLength(2);
    expect(container.querySelectorAll(".katex-display")).toHaveLength(1);
    expect(screen.getByRole("button").textContent).toBe("1");
    expect(container.textContent).toContain("costa $5 e $10");
  });

  it("renders headings and numbered lists", () => {
    const { container } = render(
      <AnswerText text={"## Definizione *MDP*\n1. Stati\n2) Azioni"} citations={[]} />,
    );
    const heading = screen.getByRole("heading", { level: 2 });
    expect(heading.textContent).toBe("Definizione MDP");
    expect(heading.querySelector("em")).not.toBeNull();
    const items = [...container.querySelectorAll("p")].map((p) => p.parentElement?.textContent);
    expect(items).toEqual(["1.Stati", "2.Azioni"]);
  });

  it("indents nested list items by depth, resets after a paragraph", () => {
    const { container } = render(
      <AnswerText
        text={"1. MDP\n    - stati\n        - finiti\n\n    - azioni\n2. POMDP\nFine.\n  - nuovo"}
        citations={[]}
      />,
    );
    const margins = [...container.querySelectorAll<HTMLElement>("[style]")].map(
      (row) => row.style.marginLeft,
    );
    expect(margins).toEqual(["0rem", "1.25rem", "2.5rem", "1.25rem", "0rem", "0rem"]);
  });

  it("renders bullets and skips blank lines", () => {
    const { container } = render(<AnswerText text={"- one\n\n- two"} citations={[]} />);
    expect(container.querySelectorAll("p")).toHaveLength(2);
  });
});
