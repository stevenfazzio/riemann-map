"""Stage 11: screenshot the public page for its link card.

Reads   docs/index.html and its data files (stage 10 with --public)
Writes  docs/social-preview.png   1200 x 630, the image a shared link shows

    uv run python -I scripts/10_render_map.py --public
    uv run python -I scripts/11_social_preview.py

The card is the interactive map itself, not a separate static plot, so a shared link
previews what the reader will get. The map is WebGL, so this drives the installed Chrome
through Playwright; no bundled browser is downloaded. Run it again whenever stage 10's
public build changes what the opening view shows.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import io
import os
import threading
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

from riemann_map import load_stage

OG_SIZE = (1200, 630)
SCALE = 2  # capture at 2x and downsample, so labels and points stay crisp
SETTLE_MS = 6000

# Controls that do nothing in a still image.
HIDE_CONTROLS = """
for (const id of ['search-container', 'colormap-selector-container']) {
  const el = document.getElementById(id);
  if (el) (el.closest('.container-box') || el).style.display = 'none';
}
"""
# Link previews print og:title under the image, so the card spends its space on the map.
HIDE_TITLE = """
const t = document.getElementById('title-container');
if (t) (t.closest('.container-box') || t).style.display = 'none';
"""


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def main() -> None:
    render = load_stage("10_render_map")
    ap = argparse.ArgumentParser()
    # Wheel ticks at the centre of the frame. The card is wider than the page's opening
    # view, which leaves the cloud small in the middle. Two ticks were chosen by eye from
    # 0, 1 and 2 (2026-11-23): the whole cloud stays in frame, four region names show, and
    # none is caught half-faded by the label collision filter. Look again if the names change.
    ap.add_argument("--zoom", type=int, default=2, help="wheel ticks towards the map")
    ap.add_argument("--pan-x", type=int, default=0, help="drag the map right N px")
    ap.add_argument("--pan-y", type=int, default=0, help="drag the map down N px")
    ap.add_argument("--title", action="store_true", help="keep the title panel")
    ap.add_argument("--out", type=Path, default=render.SOCIAL_PREVIEW)
    args = ap.parse_args()
    if not render.PUBLIC_HTML.exists():
        raise SystemExit(f"{render.PUBLIC_HTML} does not exist; run stage 10 with --public first")

    handler = functools.partial(QuietHandler, directory=str(render.DOCS))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}/{render.PUBLIC_HTML.name}"
    w, h = OG_SIZE
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(channel="chrome")
            page = browser.new_page(viewport={"width": w, "height": h}, device_scale_factor=SCALE)
            page.goto(url, wait_until="load", timeout=120_000)
            page.wait_for_load_state("networkidle", timeout=120_000)
            page.wait_for_timeout(SETTLE_MS)
            page.mouse.move(w / 2, h / 2)
            for _ in range(args.zoom):
                # deck.gl zooms by a function of delta x speed, so scaling the delta keeps a
                # tick meaning what it does at deck.gl's own speed of 0.01.
                page.mouse.wheel(0, -50 * 0.01 / render.ZOOM_SPEED)
                page.wait_for_timeout(150)
            if args.pan_x or args.pan_y:
                page.mouse.down()
                page.mouse.move(w / 2 + args.pan_x, h / 2 + args.pan_y, steps=8)
                page.mouse.up()
            page.mouse.move(0, 0)  # park the cursor off the points: no hovercard in the shot
            page.evaluate(HIDE_CONTROLS)
            if not args.title:
                page.evaluate(HIDE_TITLE)
            page.wait_for_timeout(4000)
            shot = page.screenshot()
            browser.close()
    finally:
        server.shutdown()

    with Image.open(io.BytesIO(shot)) as im:
        card = im.convert("RGB").resize(OG_SIZE, Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    card.save(buf, format="PNG", optimize=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".tmp")
    tmp.write_bytes(buf.getvalue())
    os.replace(tmp, args.out)
    print(f"wrote {args.out} {card.size} ({args.out.stat().st_size / 1e3:.0f} KB)")


if __name__ == "__main__":
    main()
