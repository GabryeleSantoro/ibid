"""Markdown chapters → one LaTeX book (classic `book` class, XeTeX/Tectonic).

Text is always escaped; only math passes through raw, because the converted
chapters already write formulas in LaTeX. Math that could read or write files,
or redefine commands, is printed as code instead: the Markdown may come from a
model that read untrusted slides.
"""

from __future__ import annotations

import re

from markdown_it import MarkdownIt
from markdown_it.token import Token
from mdit_py_plugins.dollarmath import dollarmath_plugin
from mdit_py_plugins.texmath import texmath_plugin

_MD = (
    MarkdownIt("commonmark")
    .enable(["table", "strikethrough"])
    .use(dollarmath_plugin, allow_space=False, allow_digits=False, double_inline=True)
    .use(texmath_plugin, delimiters="brackets")
)

_BABEL = {"it": "italian", "en": "english", "fr": "french", "de": "ngerman", "es": "spanish"}

_ESCAPES = {
    "\\": r"\textbackslash{}",
    "{": r"\{",
    "}": r"\}",
    "$": r"\$",
    "&": r"\&",
    "#": r"\#",
    "^": r"\^{}",
    "_": r"\_",
    "%": r"\%",
    "~": r"\textasciitilde{}",
}
# Latin Modern Roman lacks Greek, math symbols, primes and Unicode sub/superscripts;
# the math font (unicode-math) has them all.
_SYMBOLS = re.compile(
    "[\u0370-\u03ff\u1d2c-\u1d6a\u2032-\u2037\u2070-\u209f\u2190-\u21ff"
    "\u2102\u2113\u2115\u2119-\u211d\u2124\u2200-\u22ff\u2308-\u230b\u27c0-\u27ff"
    "\u2a00-\u2aff\u2c7c\U0001d400-\U0001d7ff]+"
)

# Tectonic does not confine reads (../../ escapes its root), so math that could
# read, write or build a command name (to dodge this list) is never passed on.
_UNSAFE_MATH = re.compile(
    r"\^\^|(?i:file)|\\end\s*\{document\}"
    r"|\\(?:def|edef|gdef|xdef|let|futurelet|global|chardef|mathchardef)(?![a-zA-Z])"
    r"|\\[a-zA-Z@:_]*(?i:input|include|read|write|open|csname|string|detokenize|scantokens"
    r"|catcode|command|environment|explsyntax|makeat|package|class|lua|xetex|pdf|every"
    r"|after|roman|shipout|immediate|jobname|special|uppercase|lowercase|font)"
)
_DISPLAY_ENV = re.compile(r"\\begin\{(?:equation|align|gather|multline|flalign|eqnarray)\*?\}")
# Raw HTML is never passed on; models only use it for sub/superscripts.
_HTML = {"<sub>": r"\textsubscript{", "<sup>": r"\textsuperscript{"}
_HEADINGS = ("section", "subsection", "subsubsection", "paragraph", "subparagraph")

_BLOCKS = {
    "paragraph_open": "",
    "bullet_list_open": "\\begin{itemize}\n",
    "bullet_list_close": "\\end{itemize}\n\n",
    "ordered_list_close": "\\end{enumerate}\n\n",
    # `{}` stops LaTeX reading a leading "[" as the item's optional label.
    "list_item_open": "\\item{} ",
    "list_item_close": "\n",
    "blockquote_open": "\\begin{quote}\n",
    "blockquote_close": "\\end{quote}\n\n",
    "hr": "\\begin{center}\\rule{0.3\\linewidth}{0.4pt}\\end{center}\n\n",
}

_PREAMBLE = r"""\documentclass[11pt,a4paper,oneside,openany]{book}
\usepackage{amsmath}
\usepackage{unicode-math}
\usepackage[shorthands=off,<<babel>>]{babel}
\usepackage[margin=2.8cm]{geometry}
\usepackage{microtype}
\usepackage{array,longtable,booktabs}
\usepackage{enumitem}
\usepackage[hidelinks]{hyperref}
\pagestyle{plain}
\DeclareMathOperator*{\argmax}{arg\,max}
\DeclareMathOperator*{\argmin}{arg\,min}

\title{<<title>>}
\author{}
\date{\today}

\begin{document}
\frontmatter
\maketitle
\tableofcontents
\mainmatter

"""


def escape(text: str) -> str:
    escaped = "".join(_ESCAPES.get(char, char) for char in text)
    return _SYMBOLS.sub(lambda m: f"${m.group()}$", escaped)


def _math(source: str, display: bool) -> str:
    body = re.sub(r"\n\s*\n", "\n", source.strip())  # a blank line ends math mode
    body = re.sub(r"(?<!\\)([%#])", r"\\\1", body)
    # Models escape * and _ for Markdown even inside math: v^\* means v^*.
    body = re.sub(r"(?<!\\)\\([*_])", r"\1", body)
    if not body:
        return ""
    if _UNSAFE_MATH.search(body):
        return rf"\texttt{{{escape(source.strip())}}}"
    if not display:
        return rf"\({body}\)"
    if _DISPLAY_ENV.match(body):
        return body + "\n"
    return f"\\[\n{body}\n\\]\n"


def _link(href: str) -> str:
    if not re.match(r"(?i)(?:https?|mailto):", href):
        return "{"  # keep the text, drop the target
    url = re.sub(r"[{}\\]", "", href).replace("%", r"\%").replace("#", r"\#")
    return rf"\href{{{url}}}{{"


def _inline(children: list[Token] | None, heading: bool = False) -> str:
    out: list[str] = []
    opened = 0  # <sub>/<sup> still open: an unclosed tag must not unbalance braces
    for token in children or []:
        kind = token.type
        if kind == "text":
            out.append(escape(token.content))
        elif kind == "html_inline":
            tag = token.content.lower().replace(" ", "")
            if tag in _HTML:
                out.append(_HTML[tag])
                opened += 1
            elif tag in ("</sub>", "</sup>") and opened:
                out.append("}")
                opened -= 1
            elif tag in ("<br>", "<br/>"):
                out.append("\\newline\n")
            else:
                out.append(escape(token.content))  # e.g. List<T> is text, not a tag
        elif kind == "softbreak":
            out.append("\n")
        elif kind == "hardbreak":
            out.append("\\newline\n")  # unlike \\, never reads a following "[" or "*"
        elif kind == "code_inline":
            out.append(rf"\texttt{{{escape(token.content)}}}")
        elif kind == "strong_open":
            out.append(r"\textbf{")
        elif kind == "em_open":
            out.append(r"\emph{")
        elif kind in ("strong_close", "em_close", "link_close"):
            out.append("}")
        elif kind == "link_open":
            out.append(_link(str(token.attrs.get("href", ""))))
        elif kind == "image":
            # ponytail: alt text only; include the file when chapters start carrying images.
            out.append(rf"\emph{{{escape(token.content)}}}")
        elif kind == "math_inline":
            math = _math(token.content, display=False)
            # PDF bookmarks can't hold unicode-math glyphs: give them the source text.
            if heading:
                math = rf"\texorpdfstring{{{math}}}{{{escape(token.content)}}}"
            out.append(math)
        elif kind == "math_inline_double":
            out.append(_math(token.content, display=True))
    return "".join(out) + "}" * opened


def _code(source: str) -> str:
    lines = source.rstrip("\n").replace("\t", "    ").split("\n")
    # \mbox{} keeps the indent (glue after a line break is dropped); "\ " can wrap.
    body = "\\newline\n".join(r"\mbox{}" + escape(line).replace(" ", "\\ ") for line in lines)
    return f"\\begin{{flushleft}}\\small\\ttfamily\n{body}\n\\end{{flushleft}}\n\n"


def _table(tokens: list[Token]) -> str:
    rows: list[list[str]] = []
    for token in tokens:
        if token.type == "tr_open":
            rows.append([])
        elif token.type == "inline":
            rows[-1].append(_inline(token.children))
    width = max(len(row) for row in rows)
    # ponytail: equal-width wrapping columns; size them by content if tables look cramped.
    column = rf">{{\raggedright\arraybackslash}}p{{\dimexpr\linewidth/{width}-2\tabcolsep\relax}}"

    def line(cells: list[str]) -> str:
        cells = cells + [""] * (width - len(cells))
        return " & ".join("{}" + cell for cell in cells) + " \\\\\n"

    head, *body = rows
    return (
        f"\\begin{{longtable}}{{@{{}}{column * width}@{{}}}}\n\\toprule\n"
        + line([rf"\textbf{{{cell}}}" if cell else "" for cell in head])
        + "\\midrule\n\\endhead\n"
        + "".join(line(row) for row in body)
        + "\\bottomrule\n\\end{longtable}\n\n"
    )


def _blocks(tokens: list[Token], offset: int) -> str:
    out: list[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        kind = token.type
        if kind == "heading_open":
            level = min(max(int(token.tag[1]) - offset, 1), len(_HEADINGS))
            title = _inline(tokens[i + 1].children, heading=True)
            out.append(f"\\{_HEADINGS[level - 1]}{{{title}}}\n\n")
            i += 3
            continue
        if kind == "table_open":
            end = next(j for j in range(i, len(tokens)) if tokens[j].type == "table_close")
            out.append(_table(tokens[i:end]))
            i = end + 1
            continue
        if kind == "inline":
            out.append(_inline(token.children))
        elif kind == "paragraph_close":
            out.append("\n" if token.hidden else "\n\n")
        elif kind == "ordered_list_open":
            start = int(token.attrs.get("start", 1))
            out.append("\\begin{enumerate}" + (f"[start={start}]" if start != 1 else "") + "\n")
        elif kind in ("fence", "code_block"):
            out.append(_code(token.content))
        elif kind == "html_block":
            out.append(escape(re.sub(r"<[^>]*>", "", token.content)).strip() + "\n\n")
        elif kind.startswith("math_block"):
            out.append(_math(token.content, display=True) + "\n")
        else:
            out.append(_BLOCKS.get(kind, ""))
        i += 1
    return "".join(out)


def _chapter(title: str, markdown: str) -> str:
    # \[ … \] right under a line of text would continue that paragraph, where
    # CommonMark reads "\[" as a plain "[": give it the blank lines a block needs.
    markdown = re.sub(r"(?m)^([ \t]*\\\[)", r"\n\1", markdown)
    markdown = re.sub(r"(?m)(\\\][ \t]*)$", r"\1\n", markdown)
    tokens = _MD.parse(markdown)
    # The chapter heading is the document title; a leading H1 would repeat it.
    if tokens and tokens[0].type == "heading_open" and tokens[0].tag == "h1":
        tokens = tokens[3:]
    more_h1 = any(t.type == "heading_open" and t.tag == "h1" for t in tokens)
    return f"\\chapter{{{escape(title)}}}\n\n" + _blocks(tokens, 0 if more_h1 else 1)


def to_latex(title: str, chapters: list[tuple[str, str]], language: str | None) -> str:
    """`chapters` are (title, markdown) pairs, in book order."""
    preamble = _PREAMBLE.replace("<<babel>>", _BABEL.get(language or "", "english")).replace(
        "<<title>>", escape(title)
    )
    return preamble + "".join(_chapter(name, text) for name, text in chapters) + "\\end{document}\n"
