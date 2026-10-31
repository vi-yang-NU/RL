"""The report's picture of tab 2's "Explore the problem" panel: the two-station scene and its summary tiles at
the start of each of the first periods of the optimal policy (handout parameters, start state 0, seed 7, the
explorer's defaults), tiled into one labelled collage.

    python scripts/explore_collage.py                        # results/dp/explore_steps.png (+ .json with the rows)
    python scripts/explore_collage.py --out FILE --steps 10 --columns 2 --width 640 --scale 2

Google Chrome is driven headless over DevTools (tornado, which ships with Streamlit, speaks the websocket);
Pillow assembles the collage. Set CHROME to the browser binary if it is not found.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from tornado.ioloop import IOLoop  # noqa: E402
from tornado.websocket import websocket_connect  # noqa: E402

from rl.dp.explore import write_explorer  # noqa: E402
from rl.dp.shuttle import ShuttleParams  # noqa: E402
from rl.outputs import RESULTS  # noqa: E402

CHROME_CANDIDATES = [
    os.environ.get("CHROME", ""),
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "google-chrome", "google-chrome-stable", "chromium", "chromium-browser",
]
HIDE_CSS = "#scenePanel .controls, #scenePanel .hint { display: none; }"  # the buttons repeat in every shot
FONT_DIR = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "matplotlib" / "mpl-data" / "fonts" / "ttf"


def find_chrome() -> str:
    for candidate in CHROME_CANDIDATES:
        if candidate and (Path(candidate).is_file() or shutil.which(candidate)):
            return candidate
    raise SystemExit("Google Chrome / Chromium not found; set CHROME=/path/to/chrome")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


class DevTools:
    """A blocking DevTools session over tornado's websocket client."""

    def __init__(self, port: int, url: str):
        request = urllib.request.Request(f"http://localhost:{port}/json/new?{url}", method="PUT")
        target = json.loads(urllib.request.urlopen(request, timeout=10).read())
        self.loop = IOLoop.current()
        self.ws = self.loop.run_sync(lambda: websocket_connect(target["webSocketDebuggerUrl"], max_message_size=64 * 2**20))
        self.n = 0

    def call(self, method: str, **params):
        self.n += 1
        self.loop.run_sync(lambda: self.ws.write_message(json.dumps({"id": self.n, "method": method, "params": params})))
        while True:
            message = self.loop.run_sync(self.ws.read_message)
            if message is None:
                raise RuntimeError("DevTools connection closed")
            data = json.loads(message)
            if data.get("id") == self.n:
                if "error" in data:
                    raise RuntimeError(f"{method}: {data['error']}")
                return data.get("result", {})

    def js(self, expression: str):
        result = self.call("Runtime.evaluate", expression=expression, returnByValue=True, awaitPromise=True)
        if "exceptionDetails" in result:
            raise RuntimeError(result["exceptionDetails"].get("exception", {}).get("description") or result["exceptionDetails"].get("text"))
        return result["result"].get("value")

    def wait_for(self, expression: str, timeout: float = 15.0, every: float = 0.1):
        deadline = time.time() + timeout
        while time.time() < deadline:
            value = self.js(expression)
            if value:
                return value
            time.sleep(every)
        raise TimeoutError(f"timed out waiting for {expression}")

    def screenshot(self, x: float, y: float, w: float, h: float) -> Image.Image:
        result = self.call("Page.captureScreenshot", format="png", captureBeyondViewport=True,
                           clip={"x": x, "y": y, "width": w, "height": h, "scale": 1})
        return Image.open(io.BytesIO(base64.b64decode(result["data"]))).convert("RGB")


def wait_for_chrome(port: int, timeout: float = 20.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://localhost:{port}/json/version", timeout=2).read()
            return
        except OSError:
            time.sleep(0.2)
    raise TimeoutError("Chrome did not start")


@contextmanager
def headless_chrome(width: int, height: int = 1400) -> Iterator[int]:
    """Google Chrome, headless with a throw-away profile: yields its DevTools port and quits it afterwards."""
    chrome = find_chrome()
    port = free_port()
    profile = tempfile.mkdtemp(prefix="headless-chrome-")
    proc = subprocess.Popen(
        [chrome, "--headless=new", f"--remote-debugging-port={port}", f"--user-data-dir={profile}", "--no-first-run",
         "--hide-scrollbars", f"--window-size={width},{height}", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        wait_for_chrome(port)
        yield port
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(profile, ignore_errors=True)


def capture(page: Path, steps: int, width: int, scale: float) -> tuple[list[Image.Image], list[dict]]:
    """One screenshot of the scene panel at the start of each period 0 … steps-1, and the period's row of the
    explorer's "Period by period" table once the step has been taken."""
    shots, rows = [], []
    with headless_chrome(width) as port:
        dt = DevTools(port, page.resolve().as_uri())
        dt.call("Emulation.setDeviceMetricsOverride", width=width, height=1400, deviceScaleFactor=scale, mobile=False)
        dt.wait_for("document.querySelectorAll('#tiles .tile').length >= 6")
        dt.js(f"""(() => {{
            const style = document.createElement('style'); style.textContent = {json.dumps(HIDE_CSS)};
            document.head.appendChild(style);
            const speed = document.getElementById('selSpeed'); speed.value = '0';
            speed.dispatchEvent(new Event('change'));
            window.scrollTo(0, 0);
            return true;
        }})()""")
        time.sleep(0.3)
        for t in range(steps):
            dt.wait_for(f"document.getElementById('scPeriod').textContent.startsWith('period {t} ')")
            time.sleep(0.15)
            x, y, w, h = dt.js("(() => { const r = document.getElementById('scenePanel').getBoundingClientRect(); return [r.x, r.y, r.width, r.height]; })()")
            shots.append(dt.screenshot(x, y, w, h))
            if dt.js("document.getElementById('btnStep').disabled"):
                raise RuntimeError(f"the Step button is disabled at period {t}: the replay has fewer than {steps} periods")
            dt.js("document.getElementById('btnStep').click(); true")
            dt.wait_for(f"document.getElementById('scPeriod').textContent.startsWith('period {t + 1}')")
            cells = dt.js("Array.from(document.querySelectorAll('#stepsTable tbody tr:not(.pending)')).slice(-1)[0].innerText.split('\\t')")
            heads = dt.js("Array.from(document.querySelectorAll('#stepsTable thead th')).map(th => th.innerText)")
            rows.append(dict(zip(heads, cells)))
    return shots, rows


def font(name: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = FONT_DIR / name
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def label_for(row: dict) -> tuple[str, list[str]]:
    """"Step n · period t = …" and the period's state, value, Q-values, decision, cost and transition."""
    t = int(row["t"])
    q = ", ".join(f"{k[len('Q*(s_t, '):-1]} {row[k]}" for k in row if k.startswith("Q*("))
    return f"Step {t + 1} · period t = {t}", [
        f"s = {row['s_t']}", f"V*(s) = {row['V*(s_t)']}", f"Q*(s, ·): {q}", f"policy: {row['taken'].lower()}",
        f"cost c = {row['c_t']}", f"{row['A_t']} arrive → s = {row['s_{t+1}']}",
    ]


def wrap(parts: list[str], text_font, width: float, sep: str = " · ") -> list[str]:
    lines: list[str] = []
    for part in parts:
        if lines and text_font.getlength(lines[-1] + sep + part) <= width:
            lines[-1] += sep + part
        else:
            lines.append(part)
    return lines


def collage(shots: list[Image.Image], rows: list[dict], columns: int, scale: float) -> Image.Image:
    gap, margin = int(24 * scale), int(16 * scale)
    title_font, text_font = font("DejaVuSans-Bold.ttf", int(18 * scale)), font("DejaVuSans.ttf", int(13 * scale))
    cell_w = max(s.width for s in shots)
    labels = [(title, wrap(parts, text_font, cell_w)) for title, parts in map(label_for, rows)]
    line_h = int(17 * scale)
    label_h = int(26 * scale) + line_h * max(len(lines) for _, lines in labels) + int(6 * scale)
    cell_h = label_h + max(s.height for s in shots)
    n_rows = -(-len(shots) // columns)
    out = Image.new("RGB", (2 * margin + columns * cell_w + (columns - 1) * gap, 2 * margin + n_rows * cell_h + (n_rows - 1) * gap), "white")
    draw = ImageDraw.Draw(out)
    for i, (shot, (title, lines)) in enumerate(zip(shots, labels)):
        x = margin + (i % columns) * (cell_w + gap)
        y = margin + (i // columns) * (cell_h + gap)
        draw.text((x, y), title, fill="#0b0b0b", font=title_font)
        for j, line in enumerate(lines):
            draw.text((x, y + int(24 * scale) + j * line_h), line, fill="#52514e", font=text_font)
        out.paste(shot, (x, y + label_h))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=RESULTS / "dp" / "explore_steps.png")
    parser.add_argument("--steps", type=int, default=10, help="periods to show (the explorer replays at most 10)")
    parser.add_argument("--columns", type=int, default=2)
    parser.add_argument("--width", type=int, default=640, help="CSS width of the explorer page for each screenshot")
    parser.add_argument("--scale", type=float, default=2.0, help="device pixels per CSS pixel")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="explore-page-") as tmp:
        page = write_explorer(Path(tmp) / "explore.html", ShuttleParams(), theme="light")
        shots, rows = capture(page, args.steps, args.width, args.scale)
    image = collage(shots, rows, args.columns, args.scale)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.out, optimize=True)
    args.out.with_suffix(".json").write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"saved {args.out} ({image.width}×{image.height} px) and {args.out.with_suffix('.json')}")
    for row in rows:
        print("  " + "  ".join(f"{k}={v}" for k, v in row.items()))


if __name__ == "__main__":
    main()
