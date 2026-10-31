"""The report's picture of the Section 2 validation-test summary: the table in report/dp_tests_table.md (one row
per test: what it checks, how it validates, the pass criterion and the result) set in the report's typeface and
screenshotted in headless Chrome, so the report and the dashboard show it above the harness's own results.

    python scripts/tests_table.py                                       # results/dp/tests_table.png
    python scripts/tests_table.py --source FILE --out FILE --width 780 --font-size 13 --scale 2

The source is a Markdown pipe table whose first row is the header (a `\\|` is a literal bar inside a cell).
The picture is laid out for a portrait page: at --width 780 the report scales it to fit the page, so one CSS
pixel of the table is about 0.55 points there (13 px type reads as about 7 pt). Google Chrome is driven as in
explore_collage.py; set CHROME to the browser binary if it is not found.
"""
from __future__ import annotations

import argparse
import base64
import re
import sys
import tempfile
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib  # noqa: E402
from PIL import Image  # noqa: E402

from explore_collage import DevTools, headless_chrome  # noqa: E402
from rl.outputs import RESULTS  # noqa: E402

FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"  # DejaVu, the typeface of the report's PDF
SOURCE = ROOT / "report" / "dp_tests_table.md"
OUT = RESULTS / "dp" / "tests_table.png"
COLUMN_SHARES = (0.07, 0.15, 0.155, 0.18, 0.16, 0.285)  # #, Test, What it checks, How it validates, Pass criterion, Result
PAD = 10  # white margin around the table, CSS px
TABLE_RULE = re.compile(r"\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?")


def read_table(path: Path) -> list[list[str]]:
    """The rows of the first Markdown pipe table in `path` (header first), with `\\|` unescaped."""
    rows: list[list[str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s.startswith("|") or TABLE_RULE.fullmatch(s):
            if rows and not s.startswith("|"):
                break
            continue
        s = s[1:]
        s = s[:-1] if s.endswith("|") and not s.endswith("\\|") else s
        rows.append([c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", s)])
    if len(rows) < 2:
        raise SystemExit(f"{path}: no Markdown table with a header and at least one row")
    n = len(rows[0])
    return [(row + [""] * n)[:n] for row in rows]


def _font_face(file: str, weight: int) -> str:
    data = base64.b64encode((FONT_DIR / file).read_bytes()).decode()
    return (f'@font-face {{ font-family: "DejaVu Sans"; font-weight: {weight}; '
            f'src: url(data:font/ttf;base64,{data}) format("truetype"); }}')


def page(rows: list[list[str]], width: int, font_size: float) -> str:
    """A page holding just the table, styled like the report's tables, with the fonts embedded."""
    n = len(rows[0])
    shares = COLUMN_SHARES if n == len(COLUMN_SHARES) else (1 / n,) * n
    colgroup = "".join(f'<col style="width:{100 * share:.2f}%">' for share in shares)
    head = "".join(f"<th>{escape(c)}</th>" for c in rows[0])
    body = "".join("<tr>" + "".join(f"<td>{escape(c)}</td>" for c in row) + "</tr>" for row in rows[1:])
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Validation tests</title>
<style>
{_font_face("DejaVuSans.ttf", 400)}
{_font_face("DejaVuSans-Bold.ttf", 700)}
html, body {{ margin: 0; background: #fff; }}
body {{ font-family: "DejaVu Sans", sans-serif; font-size: {font_size}px; line-height: 1.4; color: #1f2328; }}
#wrap {{ display: inline-block; padding: {PAD}px; background: #fff; }}
table {{ border-collapse: collapse; table-layout: fixed; width: {width}px; }}
th, td {{ border: 1px solid #d1d9e0; padding: 4px 6px; vertical-align: top; text-align: left; overflow-wrap: anywhere; }}
th {{ background: #f6f8fa; font-weight: 700; }}
td:first-child {{ white-space: nowrap; color: #59636e; }}
td:nth-child(2) {{ font-weight: 700; }}
</style></head>
<body><div id="wrap"><table><colgroup>{colgroup}</colgroup><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div></body></html>
"""


def render(html: Path, width: int, scale: float) -> Image.Image:
    """Screenshot of the table (with its white margin) at `scale` device pixels per CSS pixel."""
    viewport = width + 2 * PAD + 40
    with headless_chrome(viewport, 1600) as port:
        dt = DevTools(port, html.resolve().as_uri())
        dt.call("Emulation.setDeviceMetricsOverride", width=viewport, height=1600, deviceScaleFactor=scale, mobile=False)
        dt.js("document.fonts.ready.then(() => true)")
        if not dt.js('document.fonts.check(\'12px "DejaVu Sans"\') && document.fonts.check(\'bold 12px "DejaVu Sans"\')'):
            raise RuntimeError("the embedded DejaVu fonts did not load")
        x, y, w, h = dt.js("(() => { const r = document.getElementById('wrap').getBoundingClientRect(); "
                           "return [r.x, r.y, r.width, r.height]; })()")
        return dt.screenshot(x, y, w, h)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", type=Path, default=SOURCE, help="Markdown file holding the table")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--width", type=int, default=780, help="CSS width of the table (the report scales it to its text width)")
    parser.add_argument("--font-size", type=float, default=13, help="CSS px")
    parser.add_argument("--scale", type=float, default=2.0, help="device pixels per CSS pixel")
    args = parser.parse_args()

    rows = read_table(args.source)
    with tempfile.TemporaryDirectory(prefix="tests-table-") as tmp:
        html = Path(tmp) / "table.html"
        html.write_text(page(rows, args.width, args.font_size), encoding="utf-8")
        image = render(html, args.width, args.scale)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.out, optimize=True)
    print(f"saved {args.out} ({image.width}×{image.height} px): {len(rows) - 1} rows, {len(rows[0])} columns, "
          f"from {args.source}")


if __name__ == "__main__":
    main()
