import katex from "katex";
import "katex/dist/katex.min.css";
import { useTranslation } from "react-i18next";

import type { Citation } from "@/lib/ipc";
import { cn } from "@/lib/utils";

/**
 * One pass over a line, in priority order: citation marker [doc_id:page], inline math
 * ($…$ or \(…\)), **bold**, *italic*, `code`. A $ only opens math when a non-space
 * follows it and only closes when a non-space precedes it and no digit follows, so
 * "$5 and $10" stays text.
 */
const INLINE =
  /\[([A-Za-z0-9_.\-]+):(\d+)\]|\$(?=\S)([^$\n]*?\S)\$(?!\d)|\\\((.+?)\\\)|\*\*(.+?)\*\*|\*(?=\S)([^*\n]*?\S)\*|`([^`\n]+)`/g;

/** Display math: $$…$$ or \[…\], possibly across lines. */
const DISPLAY = /(\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\])/;

/** KaTeX's default (trust: false) refuses \href, \includegraphics and raw HTML. */
function TeX({ tex, display }: { tex: string; display?: boolean }) {
  const html = katex.renderToString(tex, { displayMode: display, throwOnError: false });
  return <span dangerouslySetInnerHTML={{ __html: html }} />;
}

export type CitationTarget = { docId: string; page: number; citation?: Citation };

function Chip({
  docId,
  page,
  index,
  known,
  onSelect,
}: {
  docId: string;
  page: number;
  index: number | null;
  known: boolean;
  onSelect: () => void;
}) {
  const { t } = useTranslation();
  const label = index ?? "?";
  return (
    <button
      type="button"
      onClick={onSelect}
      title={known ? t("chat.citationPage", { docId, page }) : t("chat.citationUnknown", { docId })}
      className={cn(
        "mx-0.5 inline-flex h-4.5 min-w-4.5 translate-y-[-1px] items-center justify-center",
        "rounded-[0.25rem] px-1 align-middle font-mono text-[0.625rem] font-medium tabular-nums",
        "transition-colors",
        known
          ? "bg-primary/12 text-primary hover:bg-primary/22"
          : "bg-destructive/12 text-destructive line-through",
      )}
    >
      {label}
    </button>
  );
}

/**
 * Renders an answer: light markdown (paragraphs, bullets, bold, italic, code), LaTeX
 * math via KaTeX, and citation chips.
 *
 * A full markdown renderer is deliberately not pulled in — the pieces that matter
 * here are the chips and the distinction between a citation that resolves and one
 * that does not.
 */
export function AnswerText({
  text,
  citations,
  onSelect,
  className,
}: {
  text: string;
  citations: Citation[];
  onSelect?: (target: CitationTarget) => void;
  className?: string;
}) {
  const order = new Map<string, number>();
  citations.forEach((citation, index) => order.set(citation.marker, index + 1));

  const renderInline = (line: string, keyPrefix: string): React.ReactNode[] => {
    const nodes: React.ReactNode[] = [];
    let cursor = 0;
    // Own regex per call: bold/italic recurse, and a shared lastIndex would clash.
    const pattern = new RegExp(INLINE);

    for (let match = pattern.exec(line); match; match = pattern.exec(line)) {
      if (match.index > cursor) nodes.push(line.slice(cursor, match.index));
      const [whole, docId, pageRaw, dollarMath, parenMath, bold, italic, code] = match;
      const key = `${keyPrefix}-${match.index}`;
      if (docId) {
        const page = Number(pageRaw);
        const citation = citations.find((c) => c.marker === whole);
        nodes.push(
          <Chip
            key={key}
            docId={docId}
            page={page}
            index={order.get(whole) ?? null}
            known={Boolean(citation)}
            onSelect={() => onSelect?.({ docId, page, citation })}
          />,
        );
      } else if (dollarMath ?? parenMath) {
        nodes.push(<TeX key={key} tex={dollarMath ?? parenMath} />);
      } else if (bold) {
        nodes.push(<strong key={key}>{renderInline(bold, key)}</strong>);
      } else if (italic) {
        nodes.push(<em key={key}>{renderInline(italic, key)}</em>);
      } else {
        nodes.push(
          <code key={key} className="rounded bg-muted px-1 font-mono text-[0.8125rem]">
            {code}
          </code>,
        );
      }
      cursor = match.index + whole.length;
    }
    if (cursor < line.length) nodes.push(line.slice(cursor));
    return nodes;
  };

  const renderLines = (segment: string, s: number) => {
    // Indent widths of the open list levels; a list item's depth is its index here.
    // Blank lines keep the list open (loose lists), any other line closes it.
    const levels: number[] = [];
    const depthOf = (line: string) => {
      const width = /^[ \t]*/.exec(line)![0].replace(/\t/g, "    ").length;
      while (levels.length && levels[levels.length - 1] > width) levels.pop();
      if (!levels.length || levels[levels.length - 1] < width) levels.push(width);
      return levels.length - 1;
    };
    const indent = (depth: number) => ({ marginLeft: `${depth * 1.25}rem` });

    return segment.split("\n").map((line, index) => {
      const trimmed = line.trim();
      const key = `${s}-${index}`;
      if (!trimmed) return null;
      const numbered = /^(\d+)[.)]\s+(.*)$/.exec(trimmed);
      const bullet = /^[-*+] /.test(trimmed);
      if (!numbered && !bullet) levels.length = 0;

      const heading = /^(#{1,6})\s+(.+)$/.exec(trimmed);
      if (heading) {
        const level = heading[1].length;
        return (
          <h4
            key={key}
            aria-level={level}
            className={cn(
              "mt-3 font-semibold first:mt-0",
              level === 1 ? "text-base" : level === 2 ? "text-[0.9375rem]" : "text-sm",
            )}
          >
            {renderInline(heading[2], `h${key}`)}
          </h4>
        );
      }
      if (numbered) {
        return (
          <div key={key} className="flex gap-2 pl-1" style={indent(depthOf(line))}>
            <span className="min-w-4 shrink-0 tabular-nums text-muted-foreground">
              {numbered[1]}.
            </span>
            <p className="flex-1">{renderInline(numbered[2], `n${key}`)}</p>
          </div>
        );
      }
      if (bullet) {
        const depth = depthOf(line);
        return (
          <div key={key} className="flex gap-2 pl-1" style={indent(depth)}>
            <span
              className={cn(
                "mt-[0.55em] size-1 shrink-0 rounded-full",
                depth % 2 ? "ring-1 ring-muted-foreground" : "bg-muted-foreground",
              )}
            />
            <p className="flex-1">{renderInline(trimmed.slice(2), `b${key}`)}</p>
          </div>
        );
      }
      return <p key={key}>{renderInline(trimmed, `p${key}`)}</p>;
    });
  };

  // Odd segments are display math, even ones are ordinary lines.
  const segments = text.split(DISPLAY);

  return (
    <div className={cn("selectable space-y-2 text-[0.875rem] leading-[1.6]", className)}>
      {segments.flatMap((segment, s) =>
        s % 2
          ? [
              <div key={`m${s}`} className="overflow-x-auto">
                <TeX tex={segment.slice(2, -2)} display />
              </div>,
            ]
          : renderLines(segment, s),
      )}
    </div>
  );
}
