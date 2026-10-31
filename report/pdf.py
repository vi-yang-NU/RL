"""Markdown to PDF for the report, without pandoc or LaTeX.

Covers the Markdown that report/build.py and report/instructions.md use: headings, paragraphs,
**bold**, *italic*, `code`, links, bullet and numbered lists, fenced code blocks, tables,
block quotes, images (alt text `full-page` lets a tall one take most of a page) and horizontal rules, and
the list under a "Contents" heading becomes a table of contents with page numbers. Text is set in DejaVu, the font matplotlib ships, so symbols such as γ, ∞ and →
print.
"""
from __future__ import annotations

import re
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Flowable, Frame, HRFlowable, Image, PageTemplate, Paragraph, Preformatted,
                                Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
SANS, MONO = "DejaVuSans", "DejaVuSansMono"
TEXT = colors.HexColor("#1f2328")
MUTED = colors.HexColor("#59636e")
ACCENT = "#2a78d6"  # the dashboard's primary color
RULE = colors.HexColor("#d1d9e0")
SHADE = colors.HexColor("#f6f8fa")
MARGIN = 0.85 * inch
WIDTH = LETTER[0] - 2 * MARGIN
TOP_MARGIN, BOTTOM_MARGIN = 0.8 * inch, 0.85 * inch
FRAME_HEIGHT = LETTER[1] - TOP_MARGIN - BOTTOM_MARGIN
CELL_PAD = 5
CODE_SIZE = 8.2
CONTENTS = "Contents"  # an h2 with this text, followed by a list, is set as a table of contents with page numbers

CODE_SPAN = re.compile(r"`([^`]+)`")
LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
AUTOLINK = re.compile(r"&lt;(https?://\S+?)&gt;")
BOLD = re.compile(r"\*\*(.+?)\*\*")
ITALIC = re.compile(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])")
HEADING = re.compile(r"(#{1,4})\s+(.*)")
IMAGE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
LIST_ITEM = re.compile(r"(\s*)([-*+]|\d+[.)])\s+(.*)")
TABLE_RULE = re.compile(r"\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?")
HARD_BREAK = "\x00"


def _register_fonts() -> None:
    if SANS in pdfmetrics.getRegisteredFontNames():
        return
    for family in (SANS, MONO):
        for suffix in ("", "-Bold", "-Oblique", "-BoldOblique"):
            pdfmetrics.registerFont(TTFont(family + suffix, str(FONT_DIR / f"{family}{suffix}.ttf")))
        pdfmetrics.registerFontFamily(family, normal=family, bold=f"{family}-Bold", italic=f"{family}-Oblique",
                                      boldItalic=f"{family}-BoldOblique")


def _styles() -> dict[str, ParagraphStyle]:
    body = ParagraphStyle("body", fontName=SANS, fontSize=10, leading=14.5, textColor=TEXT, spaceAfter=7)
    heading = ParagraphStyle("heading", body, fontName=f"{SANS}-Bold", keepWithNext=1)
    cell = ParagraphStyle("cell", body, fontSize=8.5, leading=11.5, spaceAfter=0)
    return {
        "body": body,
        "h1": ParagraphStyle("h1", heading, fontSize=19, leading=24, spaceAfter=4),
        "h2": ParagraphStyle("h2", heading, fontSize=14, leading=18, spaceBefore=16, spaceAfter=6),
        "h3": ParagraphStyle("h3", heading, fontSize=11.5, leading=15, spaceBefore=12, spaceAfter=4),
        "h4": ParagraphStyle("h4", heading, fontSize=10, leading=14, spaceBefore=8, spaceAfter=3),
        "quote": ParagraphStyle("quote", body, textColor=MUTED, spaceAfter=4),
        "cell": cell,
        "head": ParagraphStyle("head", cell, fontName=f"{SANS}-Bold"),
        "item": ParagraphStyle("item", body, spaceAfter=3, bulletFontName=SANS),
        "code": ParagraphStyle("code", fontName=MONO, fontSize=CODE_SIZE, leading=CODE_SIZE * 1.35, textColor=TEXT),
    }


# --- inline markup ---------------------------------------------------------------------------

def _inline(text: str) -> str:
    """Markdown inline formatting -> reportlab paragraph markup."""
    # Code spans are set aside first, so emphasis can enclose them and nothing inside them is formatted.
    codes: list[str] = []

    def set_aside(match: re.Match) -> str:
        codes.append(match.group(1))
        return f"\x01{len(codes) - 1}\x02"

    text = escape(CODE_SPAN.sub(set_aside, text).replace("\\|", "|"))
    text = LINK.sub(rf'<a href="\2" color="{ACCENT}">\1</a>', text)
    text = AUTOLINK.sub(rf'<a href="\1" color="{ACCENT}">\1</a>', text)
    text = ITALIC.sub(r"<i>\1</i>", BOLD.sub(r"<b>\1</b>", text))
    text = re.sub(r"\x01(\d+)\x02", lambda m: f'<font face="{MONO}" backColor="#eff1f3">{escape(codes[int(m.group(1))])}</font>', text)
    return text.replace(HARD_BREAK, "<br/>")


def _join(lines: list[str]) -> str:
    """Paragraph lines -> one string; a line ending in two spaces keeps its line break."""
    return "".join(line.strip() + (HARD_BREAK if line.endswith("  ") else " ") for line in lines).strip(" " + HARD_BREAK)


def _segments(text: str) -> list[tuple[str, str]]:
    """A table cell's text as (plain text, font) pieces: code spans in the monospace font."""
    return [(piece, MONO) if i % 2 else (re.sub(r"\*+|\\", "", piece), SANS) for i, piece in enumerate(CODE_SPAN.split(text))]


def _text_width(text: str, size: float) -> float:
    return sum(pdfmetrics.stringWidth(piece, font, size) for piece, font in _segments(text))


def _longest_word(text: str, size: float) -> float:
    """Paragraphs only break at spaces, so a column must fit its longest word."""
    return max((pdfmetrics.stringWidth(w, font, size) for piece, font in _segments(text) for w in piece.split()), default=0)


# --- blocks ----------------------------------------------------------------------------------

def _starts_block(lines: list[str], i: int) -> bool:
    s = lines[i].strip()
    return (s.startswith(("```", ">", "|")) or bool(HEADING.match(s)) or bool(LIST_ITEM.match(lines[i]))
            or bool(IMAGE.fullmatch(s)))


def _blocks(markdown: str):
    """(kind, content) for each block of the document."""
    lines = re.sub(r"<!--.*?-->", "", markdown, flags=re.S).splitlines()
    i = 0
    while i < len(lines):
        line, s = lines[i], lines[i].strip()
        if not s:
            i += 1
        elif s.startswith("```"):
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith("```"):
                j += 1
            yield "code", "\n".join(lines[i + 1:j])
            i = j + 1
        elif HEADING.match(s):
            m = HEADING.match(s)
            yield f"h{len(m.group(1))}", m.group(2)
            i += 1
        elif re.fullmatch(r"-{3,}|\*{3,}|_{3,}", s):
            yield "hr", None
            i += 1
        elif IMAGE.fullmatch(s):
            m = IMAGE.fullmatch(s)
            yield "image", (m.group(1), m.group(2))
            i += 1
        elif s.startswith("|") and i + 1 < len(lines) and TABLE_RULE.fullmatch(lines[i + 1].strip()):
            rows, j = [_cells(line)], i + 2
            while j < len(lines) and lines[j].strip().startswith("|"):
                rows.append(_cells(lines[j]))
                j += 1
            yield "table", rows
            i = j
        elif s.startswith(">"):
            quoted = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                quoted.append(re.sub(r"^\s*> ?", "", lines[i]))
                i += 1
            yield "quote", quoted
        elif LIST_ITEM.fullmatch(line):
            items = []
            while i < len(lines):
                m = LIST_ITEM.fullmatch(lines[i])
                if m:
                    items.append([len(m.group(1).expandtabs(4)) // 2, m.group(2), [m.group(3)]])
                elif lines[i].strip() and lines[i][0] in " \t":
                    items[-1][2].append(lines[i])
                else:
                    break
                i += 1
            yield "list", items
        else:
            paragraph = [line]
            i += 1
            while i < len(lines) and lines[i].strip() and not _starts_block(lines, i):
                paragraph.append(lines[i])
                i += 1
            yield "paragraph", paragraph


def _cells(line: str) -> list[str]:
    s = line.strip()
    s = s[1:] if s.startswith("|") else s
    s = s[:-1] if s.endswith("|") and not s.endswith("\\|") else s
    return [c.strip() for c in re.split(r"(?<!\\)\|", s)]


def _column_widths(rows: list[list[str]], size: float) -> list[float]:
    """Full-width columns, as in an HTML table: each fits its longest word, and the remaining width goes
    to the columns in proportion to how much more they need to fit their text on one line."""
    n, pad = len(rows[0]), 2 * CELL_PAD + 1
    natural = [max(_text_width(row[c], size) for row in rows) + pad for c in range(n)]
    minimum = [max(_longest_word(row[c], size) for row in rows) + pad for c in range(n)]
    if sum(natural) <= WIDTH:
        return [w * WIDTH / sum(natural) for w in natural]
    if sum(minimum) >= WIDTH:
        return [w * WIDTH / sum(minimum) for w in minimum]
    slack = [nat - low for nat, low in zip(natural, minimum)]
    extra = WIDTH - sum(minimum)
    return [low + extra * s / sum(slack) for low, s in zip(minimum, slack)]


def _table(rows: list[list[str]], styles: dict[str, ParagraphStyle]) -> Table:
    n = len(rows[0])
    rows = [(row + [""] * n)[:n] for row in rows]
    data = [[Paragraph(_inline(c), styles["head" if r == 0 else "cell"]) for c in row] for r, row in enumerate(rows)]
    table = Table(data, colWidths=_column_widths(rows, styles["cell"].fontSize), repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, RULE),
        ("BACKGROUND", (0, 0), (-1, 0), SHADE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), CELL_PAD),
        ("RIGHTPADDING", (0, 0), (-1, -1), CELL_PAD),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def _boxed(flowables: list, background=None, bar=None) -> Table:
    """One-cell table around `flowables`: shaded for code, with a bar on the left for quotes."""
    box = Table([[flowables]], colWidths=[WIDTH], hAlign="LEFT")
    style = [("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
             ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]
    if background:
        style += [("BACKGROUND", (0, 0), (-1, -1), background), ("BOX", (0, 0), (-1, -1), 0.5, RULE)]
    if bar:
        style += [("LINEBEFORE", (0, 0), (0, -1), 2.5, bar), ("TOPPADDING", (0, 0), (-1, -1), 1),
                  ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]
    box.setStyle(TableStyle(style))
    return box


def _image(path: Path, styles: dict[str, ParagraphStyle]):
    """The image at up to 90 % of the text width and 42 % of the page height."""
    if not path.exists():
        return Paragraph(f"<i>Missing image: {escape(str(path))}</i>", styles["body"])
    w, h = ImageReader(str(path)).getSize()
    scale = min(0.9 * WIDTH / w, 0.42 * LETTER[1] / h)
    image = Image(str(path), width=w * scale, height=h * scale)
    image.hAlign = "CENTER"
    return image


class _TallFigure(Flowable):
    """A `full-page` image with its caption. It takes what is left of the current page when at least MIN_SHARE
    of a page is free, so it sits under the text that introduces it; otherwise it moves to the next page and
    gets up to MAX_SHARE of it. The caption is drawn with the image, so the two never part."""

    MAX_SHARE, MIN_SHARE = 0.8, 0.45  # of the page height

    def __init__(self, path: Path, caption: Paragraph | None):
        super().__init__()
        self.path, self.caption = str(path), caption
        self.image_w, self.image_h = ImageReader(self.path).getSize()
        self.hAlign = "CENTER"
        self._layout = (0.0, 0.0, 0.0)
        self._avail_width = WIDTH

    def _fit(self, avail_width: float, avail_height: float) -> tuple[float, float, float]:
        """(image width, image height, caption height with its gap) inside `avail_height`."""
        caption_h = self.caption.wrap(avail_width, avail_height)[1] + 6 if self.caption else 0.0
        scale = min(0.9 * WIDTH / self.image_w, (avail_height - caption_h) / self.image_h, self.MAX_SHARE * LETTER[1] / self.image_h)
        return scale * self.image_w, scale * self.image_h, caption_h

    def wrap(self, avail_width: float, avail_height: float) -> tuple[float, float]:
        w, h, caption_h = self._fit(avail_width, avail_height)
        if h < self.MIN_SHARE * LETTER[1] and self._fit(avail_width, FRAME_HEIGHT)[1] > 1.1 * h:
            return avail_width, avail_height + 1  # too little room left here; a fresh page gives it more
        self._layout, self._avail_width = (w, h, caption_h), avail_width
        self.width, self.height = w, h + caption_h
        return self.width, self.height

    def draw(self) -> None:
        w, h, caption_h = self._layout
        self.canv.drawImage(self.path, 0, caption_h, width=w, height=h, mask="auto")
        if self.caption:
            self.caption.wrapOn(self.canv, self._avail_width, caption_h)
            self.caption.drawOn(self.canv, (w - self._avail_width) / 2, 0)  # the caption spans the text width


def _toc(styles: dict[str, ParagraphStyle]) -> TableOfContents:
    """Entries for the h2 (bold) and h3 headings, with dot leaders and page numbers; multiBuild fills them in."""
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle("toc0", styles["body"], fontName=f"{SANS}-Bold", spaceBefore=7, spaceAfter=2),
        ParagraphStyle("toc1", styles["body"], leftIndent=18, spaceAfter=1),
    ]
    toc.dotsMinLevel = 0
    return toc


def _flowables(markdown: str, base_dir: Path) -> list:
    styles = _styles()
    story: list = []
    toc_next = False
    blocks = list(_blocks(markdown))
    i = 0
    while i < len(blocks):
        kind, content = blocks[i]
        i += 1
        if toc_next and kind == "list":  # the Markdown list under "## Contents" has no page numbers; this one does
            story += [_toc(styles), Spacer(1, 8)]
            toc_next = False
            continue
        toc_next = kind == "h2" and content.strip() == CONTENTS
        if kind[0] == "h":
            story.append(Paragraph(_inline(content), styles[kind]))
            if kind == "h1":
                story.append(HRFlowable(width="100%", thickness=0.8, color=RULE, spaceBefore=2, spaceAfter=8))
        elif kind == "paragraph":
            story.append(Paragraph(_inline(_join(content)), styles["body"]))
        elif kind == "code":
            longest = int((WIDTH - 18) / (0.602 * CODE_SIZE))
            code = Preformatted(content, styles["code"], maxLineLength=longest, newLineChars="  ")
            story += [_boxed([code], background=SHADE), Spacer(1, 8)]
        elif kind == "table":
            story += [_table(content, styles), Spacer(1, 9)]
        elif kind == "quote":
            paragraphs, current = [], []
            for line in content + [""]:
                if line.strip():
                    current.append(line)
                elif current:
                    paragraphs.append(Paragraph(_inline(_join(current)), styles["quote"]))
                    current = []
            if paragraphs:
                story += [_boxed(paragraphs, bar=RULE), Spacer(1, 8)]
        elif kind == "list":
            for number, (level, marker, lines) in enumerate(content, 1):
                indent = 15 + 14 * level
                bullet = marker if marker[0].isdigit() else ("•" if level == 0 else "◦")
                style = ParagraphStyle(f"item{level}", styles["item"], leftIndent=indent, bulletIndent=indent - 12)
                story.append(Paragraph(_inline(_join(lines)), style, bulletText=bullet))
            story.append(Spacer(1, 5))
        elif kind == "image":
            alt, src = content
            path = base_dir / src
            if alt.strip().lower() == "full-page" and path.exists():
                caption = None
                if i < len(blocks) and blocks[i][0] == "paragraph" and blocks[i][1][0].lstrip().startswith("*Figure"):
                    caption = Paragraph(_inline(_join(blocks[i][1])), styles["body"])  # drawn with the image
                    i += 1
                story += [_TallFigure(path, caption), Spacer(1, 4)]
            else:
                story += [_image(path, styles), Spacer(1, 4)]
        elif kind == "hr":
            story.append(HRFlowable(width="100%", thickness=0.6, color=RULE, spaceBefore=4, spaceAfter=10))
    return story


class _Document(BaseDocTemplate):
    """One frame per page with a footer. Every h1-h3 heading gets an outline entry (the viewer's sidebar) and,
    for h2 and h3, a table-of-contents entry; `multiBuild` renders again until the page numbers are stable."""

    def __init__(self, path: str, footer, **kwargs):
        super().__init__(path, **kwargs)
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="page")
        self.addPageTemplates([PageTemplate(id="page", frames=[frame], onPage=footer, pagesize=self.pagesize)])
        self._outline_level = -1

    def beforeDocument(self) -> None:
        self._outline_level = -1  # each pass starts the outline afresh

    def afterFlowable(self, flowable) -> None:
        name = getattr(getattr(flowable, "style", None), "name", "")
        if name not in ("h1", "h2", "h3") or not isinstance(flowable, Paragraph):
            return
        text = flowable.getPlainText()
        if text == CONTENTS:
            return
        level = min(int(name[1]) - 1, self._outline_level + 1)
        key = f"heading-{id(flowable)}"
        self.canv.bookmarkPage(key)
        self.canv.addOutlineEntry(text, key, level=level, closed=level > 0)
        self._outline_level = level
        if name != "h1":
            self.notify("TOCEntry", (int(name[1]) - 2, escape(text), self.page, key))


def markdown_to_pdf(markdown: str, path: Path, *, title: str, author: str = "", base_dir: Path | None = None) -> Path:
    """Render `markdown` to a US Letter PDF at `path`. Image paths are relative to `base_dir` (default: path's folder)."""
    _register_fonts()

    def footer(canvas, doc) -> None:
        canvas.saveState()
        canvas.setFont(SANS, 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(MARGIN, 0.5 * inch, title)
        canvas.drawRightString(LETTER[0] - MARGIN, 0.5 * inch, f"Page {doc.page}")
        canvas.restoreState()

    path.parent.mkdir(parents=True, exist_ok=True)
    doc = _Document(str(path), footer, pagesize=LETTER, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=TOP_MARGIN,
                    bottomMargin=BOTTOM_MARGIN, title=title, author=author)
    doc.multiBuild(_flowables(markdown, base_dir or path.parent))
    return path
