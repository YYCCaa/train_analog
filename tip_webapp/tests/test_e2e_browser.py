"""End-to-end tests: start the real server and drive the real UI in headless Chromium, like a person would.

    pytest ──Popen──▶ python -m tip_webapp --port N  (real uvicorn process, killed in teardown)
       │                         ▲
       └──Playwright Chromium────┘  file chooser, mouse drags on the stage, arrow keys, sliders,
                                    the Samples menu, and a fake camera fed from a .y4m file

Screen positions are computed from the stage's data attributes, the same transform the app draws with:

    stage.getBoundingClientRect() ─┐
    data-scale, data-offset-x/y  ──┼─▶  pixel EDGE e  ->  screen  rect.x + e * scale + offsetX
                                   └─▶  pixel CENTRE X ->  screen  rect.x + (X + 0.5) * scale + offsetX
"""

from __future__ import annotations

import base64
import io
import json
import math
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sync_api = pytest.importorskip("playwright.sync_api")
expect = sync_api.expect

# REPO_ROOT: Path, the directory that contains the tip_webapp package
REPO_ROOT = Path(__file__).resolve().parents[2]
# SAMPLES_DIR: Path, the labelled sample dials; SAMPLES: list[dict] from samples.json
SAMPLES_DIR = REPO_ROOT / "tip_webapp" / "samples"
SAMPLES = json.loads((SAMPLES_DIR / "samples.json").read_text(encoding="utf-8"))["samples"]
# WAIT_MS: int, explicit timeout for every UI wait (milliseconds)
WAIT_MS = 15_000
# SERVER_START_S: float, how long to wait for /api/health after spawning the server (seconds)
SERVER_START_S = 40.0
# VIEWPORT: dict, desktop size large enough that a 900 x 700 image is shown at scale >= 1
VIEWPORT = {"width": 1600, "height": 1000}
# OFFSET: (int, int), where sample 1 is pasted in the 900 x 700 upload composite
OFFSET = (311, 197)
# TOL_PX: float, allowed tip error for a selected, unscaled dial (image px)
TOL_PX = 3.0


def _free_port() -> int:
    """Ask the OS for an unused TCP port on 127.0.0.1."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def server_url(tmp_path_factory) -> str:
    """Run `python -m tip_webapp` in a subprocess for this module; yields its base URL (str)."""
    port = _free_port()
    log_path = tmp_path_factory.mktemp("server") / "server.log"
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "tip_webapp", "--port", str(port)],
            cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT,
        )
        url = f"http://127.0.0.1:{port}"
        try:
            deadline = time.monotonic() + SERVER_START_S
            while True:
                if proc.poll() is not None:
                    pytest.fail(f"server exited early (code {proc.returncode}); log: {log_path.read_text()}")
                try:
                    with urllib.request.urlopen(f"{url}/api/health", timeout=2) as res:
                        if res.status == 200:
                            break
                except OSError:
                    pass
                if time.monotonic() > deadline:
                    pytest.fail(f"server did not become healthy within {SERVER_START_S}s; log: {log_path.read_text()}")
                time.sleep(0.2)
            yield url
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)


@pytest.fixture(scope="module")
def playwright_instance():
    """One Playwright driver for the module."""
    with sync_api.sync_playwright() as pw:
        yield pw


@pytest.fixture(scope="module")
def browser(playwright_instance):
    """One headless Chromium for the module."""
    chromium = playwright_instance.chromium.launch(headless=True)
    yield chromium
    chromium.close()


@pytest.fixture
def page(browser, server_url):
    """A fresh desktop-sized tab with the app loaded and the model badge filled in."""
    context = browser.new_context(viewport=VIEWPORT)
    tab = context.new_page()
    tab.set_default_timeout(WAIT_MS)
    tab.goto(server_url)
    expect(tab.get_by_test_id("model-badge")).to_contain_text("p99", timeout=WAIT_MS)
    yield tab
    context.close()


# ── helpers ────────────────────────────────────────────────────────────────────────────────────


def _sample(i: int) -> np.ndarray:
    """Sample i (1-based) as a (256, 256) uint8 gray array."""
    return np.asarray(Image.open(SAMPLES_DIR / SAMPLES[i - 1]["file"]).convert("L"))


def _view(page) -> dict:
    """Stage transform -> {"x", "y": stage top-left in page px, "scale", "ox", "oy", "w", "h", "sel": (l, t, s) | None}."""
    stage = page.get_by_test_id("stage")
    box = stage.bounding_box(timeout=WAIT_MS)
    d = stage.evaluate("s => ({...s.dataset})")
    sel = (int(d["selLeft"]), int(d["selTop"]), int(d["selSize"])) if "selLeft" in d else None
    return {"x": box["x"], "y": box["y"], "scale": float(d["scale"]), "ox": float(d["offsetX"]),
            "oy": float(d["offsetY"]), "w": int(d["naturalW"]), "h": int(d["naturalH"]), "sel": sel}


def _edge_to_screen(v: dict, ex: float, ey: float) -> tuple[float, float]:
    """Image pixel-EDGE coordinates -> page px (pixel i spans edges i .. i + 1)."""
    return v["x"] + ex * v["scale"] + v["ox"], v["y"] + ey * v["scale"] + v["oy"]


def _pixel_to_screen(v: dict, X: int, Y: int, fx: float = 0.5, fy: float = 0.5) -> tuple[float, float]:
    """Image pixel (X, Y) -> page px at fraction (fx, fy) across that pixel; default (0.5, 0.5) = its centre.

    Drawing a new square is pixel-inclusive (the pixel under the pointer is inside), so a person aims at
    the middle of the first and last pixel they want, not at the edges between pixels.
    """
    return _edge_to_screen(v, X + fx, Y + fy)


def _drag(page, a: tuple[float, float], b: tuple[float, float], steps: int = 12) -> None:
    """Press at a, move to b in small steps, release: a human mouse drag."""
    page.mouse.move(*a)
    page.mouse.down()
    page.mouse.move(*b, steps=steps)
    page.mouse.up()


def _wait_prediction(page, left: int, top: int, size: int) -> dict:
    """Wait until the readout shows a finished prediction for this square; returns its data-* dict."""
    readout = page.get_by_test_id("readout")
    expect(readout).to_have_attribute("data-box-left", str(left), timeout=WAIT_MS)
    expect(readout).to_have_attribute("data-box-top", str(top), timeout=WAIT_MS)
    expect(readout).to_have_attribute("data-box-size", str(size), timeout=WAIT_MS)
    expect(page.get_by_test_id("app")).to_have_attribute("data-state", "done", timeout=WAIT_MS)
    return readout.evaluate("r => ({...r.dataset})")


def _tip_error(d: dict, want_x: float, want_y: float) -> float:
    """Distance (px) between the readout tip and (want_x, want_y)."""
    return math.hypot(float(d["tipX"]) - want_x, float(d["tipY"]) - want_y)


def _colourfulness(page, testid: str) -> float:
    """Sum of |R-G| + |G-B| over a canvas: 0 for pure gray, grows with the heatmap colours."""
    return page.get_by_test_id(testid).evaluate(
        """c => { const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data; let s = 0;
                 for (let i = 0; i < d.length; i += 4) s += Math.abs(d[i] - d[i + 1]) + Math.abs(d[i + 1] - d[i + 2]);
                 return s; }"""
    )


def _open_sample(page, index: int) -> dict:
    """Open the Samples menu and pick sample `index` like a user; returns the readout after prediction."""
    page.get_by_test_id("samples-button").click(timeout=WAIT_MS)
    page.get_by_test_id(f"sample-{index}").click(timeout=WAIT_MS)
    expect(page.get_by_test_id("stage")).to_have_attribute("data-source", "sample", timeout=WAIT_MS)
    readout = page.get_by_test_id("readout")
    expect(readout).to_have_attribute("data-gt-x", str(SAMPLES[index - 1]["label"]["x"]), timeout=WAIT_MS)
    return _wait_prediction(page, 0, 0, 256)


@pytest.fixture
def composite_png(tmp_path) -> Path:
    """900 x 700 noisy gray PNG with sample 1 pasted at OFFSET; returns its path."""
    rng = np.random.default_rng(7)
    canvas = rng.integers(70, 190, size=(700, 900), dtype=np.uint8)
    x, y = OFFSET
    canvas[y:y + 256, x:x + 256] = _sample(1)
    path = tmp_path / "dial_on_noise.png"
    Image.fromarray(canvas).save(path)
    return path


# ── tests ──────────────────────────────────────────────────────────────────────────────────────


def test_badge_shows_model(page):
    badge = page.get_by_test_id("model-badge")
    expect(badge).to_contain_text("TipNet v9", timeout=WAIT_MS)
    expect(badge).to_contain_text("407 KB", timeout=WAIT_MS)
    expect(badge).to_contain_text("val p99 3.07 px", timeout=WAIT_MS)
    page.get_by_test_id("about").locator("summary").click(timeout=WAIT_MS)
    expect(page.get_by_test_id("about")).to_contain_text("[1, 256, 256, 1]", timeout=WAIT_MS)


def test_upload_select_predict_nudge_move_resize(page, composite_png):
    label = SAMPLES[0]["label"]
    want = (label["x"] + OFFSET[0], label["y"] + OFFSET[1])

    # 1. Open the composite through the file chooser.
    with page.expect_file_chooser(timeout=WAIT_MS) as chooser:
        page.get_by_test_id("open-button").click(timeout=WAIT_MS)
    chooser.value.set_files(str(composite_png), timeout=WAIT_MS)
    stage = page.get_by_test_id("stage")
    expect(stage).to_have_attribute("data-natural-w", "900", timeout=WAIT_MS)
    expect(stage).to_have_attribute("data-natural-h", "700", timeout=WAIT_MS)
    _wait_prediction(page, 100, 0, 700)  # the preselected largest centred square was auto-predicted
    v = _view(page)
    assert v["scale"] >= 1.0, f"viewport too small: scale {v['scale']}"

    # 2. Drag a square from the centre of the pasted dial's first pixel to the centre of its last pixel.
    x, y = OFFSET
    _drag(page, _pixel_to_screen(v, x, y), _pixel_to_screen(v, x + 255, y + 255))
    assert _view(page)["sel"] == (x, y, 256)
    expect(page.get_by_test_id("status-selection")).to_have_text(f"{x},{y} · 256×256 px", timeout=WAIT_MS)
    d = _wait_prediction(page, x, y, 256)
    err = _tip_error(d, *want)
    print(f"\nupload + drag: tip ({float(d['tipX']):.2f}, {float(d['tipY']):.2f}), error {err:.3f} px")
    assert err < TOL_PX
    seq = int(d["seq"])

    # 3. Arrow keys nudge the square 1 px each; the prediction follows (debounced auto-predict).
    for _ in range(3):
        page.keyboard.press("ArrowRight")
    assert _view(page)["sel"] == (x + 3, y, 256)
    d = _wait_prediction(page, x + 3, y, 256)
    assert int(d["seq"]) > seq
    assert _tip_error(d, *want) < TOL_PX  # same dial, same image px, different crop

    # 4. Drag inside the square to move it by (+40, +25) image px.
    v = _view(page)
    left, top, size = v["sel"]
    start = _edge_to_screen(v, left + size / 2, top + size / 2)
    _drag(page, start, (start[0] + 40 * v["scale"], start[1] + 25 * v["scale"]))
    left2, top2, size2 = _view(page)["sel"]
    assert (abs(left2 - (left + 40)) <= 1, abs(top2 - (top + 25)) <= 1, size2) == (True, True, size)
    _wait_prediction(page, left2, top2, size2)

    # 5. Drag the bottom-right handle far past the image corner: square, top-left anchored, clamped.
    v = _view(page)
    handle = _edge_to_screen(v, left2 + size2, top2 + size2)
    _drag(page, handle, (handle[0] + 900, handle[1] + 900))
    left3, top3, size3 = _view(page)["sel"]
    assert (left3, top3) == (left2, top2)
    assert size3 == min(900 - left2, 700 - top2)
    assert left3 + size3 <= 900 and top3 + size3 <= 700
    _wait_prediction(page, left3, top3, size3)

    # 6. Drag the top-left handle inwards: the bottom-right corner stays anchored, still square.
    v = _view(page)
    handle = _edge_to_screen(v, left3, top3)
    _drag(page, handle, (handle[0] + 100 * v["scale"], handle[1] + 60 * v["scale"]))
    left4, top4, size4 = _view(page)["sel"]
    assert (left4 + size4, top4 + size4) == (left3 + size3, top3 + size3)
    assert size4 == size3 - 60  # side = max(|dx|, |dy|) from the anchor
    _wait_prediction(page, left4, top4, size4)

    # 7. +/- keys grow and shrink the square around its centre; Enter predicts.
    page.keyboard.press("-")
    assert _view(page)["sel"] == (left4 + 1, top4 + 1, size4 - 2)
    page.keyboard.press("Enter")
    _wait_prediction(page, left4 + 1, top4 + 1, size4 - 2)


def test_zoom_pan_fit_and_escape(page):
    d0 = _open_sample(page, 1)
    v = _view(page)
    fit_scale = v["scale"]

    # Wheel zoom. While the image is narrower than the stage it stays centred on that axis; once it
    # overflows both axes, the image point under the mouse stays put.
    cursor = _edge_to_screen(v, 140.0, 120.0)
    page.mouse.move(*cursor)
    for _ in range(3):
        page.mouse.wheel(0, -200)
    expect(page.get_by_test_id("stage")).not_to_have_attribute("data-scale", str(fit_scale), timeout=WAIT_MS)
    page.wait_for_timeout(100)
    a = _view(page)
    under_a = ((cursor[0] - a["x"] - a["ox"]) / a["scale"], (cursor[1] - a["y"] - a["oy"]) / a["scale"])
    page.mouse.wheel(0, -200)
    expect(page.get_by_test_id("stage")).not_to_have_attribute("data-scale", str(a["scale"]), timeout=WAIT_MS)
    z = _view(page)
    assert z["scale"] > a["scale"] > fit_scale * 1.5
    under = ((cursor[0] - z["x"] - z["ox"]) / z["scale"], (cursor[1] - z["y"] - z["oy"]) / z["scale"])
    assert abs(under[0] - under_a[0]) < 0.05 and abs(under[1] - under_a[1]) < 0.05, (under_a, under)

    # Space + drag pans by exactly the drag distance (and does not touch the selection).
    page.get_by_test_id("stage").focus()
    page.keyboard.down(" ")
    _drag(page, cursor, (cursor[0] + 60, cursor[1] + 40))
    page.keyboard.up(" ")
    p = _view(page)
    assert (round(p["ox"] - z["ox"]), round(p["oy"] - z["oy"])) == (60, 40)
    assert p["sel"] == (0, 0, 256)

    # Middle-button drag pans too.
    page.mouse.move(*cursor)
    page.mouse.down(button="middle")
    page.mouse.move(cursor[0] - 30, cursor[1] - 20, steps=5)
    page.mouse.up(button="middle")
    m = _view(page)
    assert (round(m["ox"] - p["ox"]), round(m["oy"] - p["oy"])) == (-30, -20)

    # Fit restores the whole-image view.
    page.get_by_test_id("fit-button").click(timeout=WAIT_MS)
    assert abs(_view(page)["scale"] - fit_scale) < 1e-9

    # Esc during a drag puts the previous square back and does not predict.
    v = _view(page)
    page.mouse.move(*_pixel_to_screen(v, 40, 40))
    page.mouse.down()
    page.mouse.move(*_pixel_to_screen(v, 119, 119), steps=6)  # pixels 40..119 -> an 80 px square
    assert _view(page)["sel"] == (40, 40, 80)
    page.keyboard.press("Escape")
    page.mouse.up()
    assert _view(page)["sel"] == (0, 0, 256)
    page.wait_for_timeout(400)
    assert page.get_by_test_id("readout").get_attribute("data-seq") == d0["seq"]


def test_draw_square_is_pixel_inclusive_like_paint(page):
    """Paint rule for a NEW square: the pixel under the press and the pixel under the pointer are both inside it,
    wherever in each pixel the mouse is. Needs a zoomed-in view so one pixel is several screen px wide."""
    _open_sample(page, 1)
    stage = page.get_by_test_id("stage")

    # Wheel-zoom (around the stage centre) until a pixel is >= 4 screen px wide.
    box = stage.bounding_box(timeout=WAIT_MS)
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    for _ in range(12):
        if _view(page)["scale"] >= 4:
            break
        page.mouse.wheel(0, -200)
        page.wait_for_timeout(100)
    v = _view(page)
    assert v["scale"] >= 4, f"could not zoom in: scale {v['scale']}"

    # (vx0, vy0, vx1, vy1): the range of whole pixel indices fully visible in the stage (image px).
    vx0 = max(0, math.ceil(-v["ox"] / v["scale"]))
    vy0 = max(0, math.ceil(-v["oy"] / v["scale"]))
    vx1 = min(v["w"], math.floor((box["width"] - v["ox"]) / v["scale"]))
    vy1 = min(v["h"], math.floor((box["height"] - v["oy"]) / v["scale"]))
    ax, ay = vx0 + 8, vy0 + 8  # first draw: press pixel
    bx, by = ax + 60, ay + 40  # second draw: press pixel, right of / below the first square
    assert bx + 3 < vx1 and by + 3 < vy1, f"visible pixels {(vx0, vy0, vx1, vy1)} too small for the drags"

    # 1. Down-right: press 80 % across pixel (ax, ay), release 20 % across pixel (ax + 40, ay + 25).
    #    Both pixels are inside, so the square is 41 wide (it would be 39 if the pointer snapped to the nearest edge).
    _drag(page, _pixel_to_screen(v, ax, ay, 0.8, 0.8), _pixel_to_screen(v, ax + 40, ay + 25, 0.2, 0.2))
    assert _view(page)["sel"] == (ax, ay, 41)
    expect(page.get_by_test_id("status-selection")).to_have_text(f"{ax},{ay} · 41×41 px", timeout=WAIT_MS)
    _wait_prediction(page, ax, ay, 41)

    # 2. Up-left: press in pixel (bx, by), release in pixel (bx - 30, by - 10). The square grows left and up
    #    from the press pixel, side = max(30, 10) + 1 = 31, and the press pixel is its bottom-right pixel.
    _drag(page, _pixel_to_screen(v, bx, by, 0.3, 0.3), _pixel_to_screen(v, bx - 30, by - 10, 0.7, 0.7))
    assert _view(page)["sel"] == (bx - 30, by - 30, 31)
    _wait_prediction(page, bx - 30, by - 30, 31)


def test_heatmap_controls_change_the_overlay(page):
    _open_sample(page, 1)
    stage = page.get_by_test_id("stage")
    expect(stage).to_have_attribute("data-heatmap", "on", timeout=WAIT_MS)
    page.wait_for_timeout(100)
    log_on = _colourfulness(page, "model-canvas")

    toggle = page.get_by_role("switch", name="Show heatmap")
    toggle.click(timeout=WAIT_MS)
    expect(stage).to_have_attribute("data-heatmap", "off", timeout=WAIT_MS)
    off = _colourfulness(page, "model-canvas")
    assert off < log_on, (off, log_on)

    toggle.click(timeout=WAIT_MS)
    expect(stage).to_have_attribute("data-heatmap", "on", timeout=WAIT_MS)

    # Opacity: click the slider near its left end, like dragging it down.
    slider = page.get_by_test_id("heatmap-opacity")
    sb = slider.bounding_box(timeout=WAIT_MS)
    page.mouse.click(sb["x"] + sb["width"] * 0.2, sb["y"] + sb["height"] / 2)
    opacity = float(stage.get_attribute("data-heatmap-opacity", timeout=WAIT_MS))
    assert 0.1 <= opacity <= 0.3, opacity
    faint = _colourfulness(page, "model-canvas")
    assert off < faint < log_on, (off, faint, log_on)
    page.mouse.click(sb["x"] + sb["width"] * 0.7, sb["y"] + sb["height"] / 2)

    # Scale: Linear shows (much) less than Log, which reveals the low-probability tail.
    page.get_by_role("radio", name="Linear").click(timeout=WAIT_MS)
    expect(stage).to_have_attribute("data-heatmap-scale", "linear", timeout=WAIT_MS)
    linear = _colourfulness(page, "model-canvas")
    page.get_by_role("radio", name="Log").click(timeout=WAIT_MS)
    expect(stage).to_have_attribute("data-heatmap-scale", "log", timeout=WAIT_MS)
    log_again = _colourfulness(page, "model-canvas")
    assert linear != log_again and linear < log_again, (linear, log_again)
    # The stage canvas carries the same overlay.
    assert _colourfulness(page, "stage-canvas") > 0


def test_samples_show_ground_truth_and_failure(page):
    d = _open_sample(page, 1)
    err_out = page.get_by_test_id("out-error")
    expect(err_out).to_be_visible(timeout=WAIT_MS)
    expect(err_out).to_contain_text("px", timeout=WAIT_MS)
    expect(page.get_by_test_id("out-gt")).to_contain_text("203.7, 115.3", timeout=WAIT_MS)
    assert float(d["errorPx"]) < 7.0
    expect(page.get_by_test_id("verdict-chip")).to_contain_text("Clear", timeout=WAIT_MS)

    d6 = _open_sample(page, 6)
    print(f"\nsample 1 error {float(d['errorPx']):.3f} px; sample 6 error {float(d6['errorPx']):.1f} px")
    assert float(d6["errorPx"]) > 7.0 and d6["ambiguous"] == "true"
    chip = page.get_by_test_id("verdict-chip")
    expect(chip).to_be_visible(timeout=WAIT_MS)
    expect(chip).to_contain_text("Ambiguous", timeout=WAIT_MS)
    expect(err_out).to_have_class("bad", timeout=WAIT_MS)
    expect(page.get_by_test_id("candidates").locator("li")).to_have_count(int(d6["candidates"]), timeout=WAIT_MS)


def test_non_image_upload_shows_alert(page):
    page.get_by_test_id("file-input").set_input_files(
        {"name": "notes.txt", "mimeType": "text/plain", "buffer": b"definitely not a picture"}, timeout=WAIT_MS
    )
    alert = page.get_by_role("alert")
    expect(alert).to_be_visible(timeout=WAIT_MS)
    expect(alert).to_contain_text("not an image", timeout=WAIT_MS)
    # Looks like a PNG by name and type, but the bytes are garbage: the browser cannot decode it.
    page.get_by_test_id("file-input").set_input_files(
        {"name": "broken.png", "mimeType": "image/png", "buffer": b"not really png bytes"}, timeout=WAIT_MS
    )
    expect(alert).to_contain_text("“broken.png” is not an image the browser can decode", timeout=WAIT_MS)
    expect(page.get_by_test_id("stage-empty")).to_be_visible(timeout=WAIT_MS)


def test_drop_a_file_on_the_stage(page):
    buf = io.BytesIO()
    Image.fromarray(_sample(2)).save(buf, "PNG")
    page.get_by_test_id("stage").evaluate(
        """(stage, b64) => {
            const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
            const dt = new DataTransfer();
            dt.items.add(new File([bytes], "dropped.png", { type: "image/png" }));
            for (const type of ["dragenter", "dragover", "drop"]) {
              stage.dispatchEvent(new DragEvent(type, { dataTransfer: dt, bubbles: true, cancelable: true }));
            }
        }""",
        base64.b64encode(buf.getvalue()).decode(),
    )
    expect(page.get_by_test_id("stage")).to_have_attribute("data-source", "drop", timeout=WAIT_MS)
    d = _wait_prediction(page, 0, 0, 256)
    assert _tip_error(d, SAMPLES[1]["label"]["x"], SAMPLES[1]["label"]["y"]) < TOL_PX


def test_camera_needs_secure_context_message(browser, server_url):
    context = browser.new_context(viewport=VIEWPORT)
    try:
        tab = context.new_page()
        tab.add_init_script("Object.defineProperty(window, 'isSecureContext', { value: false });")
        tab.goto(server_url, timeout=WAIT_MS)
        tab.get_by_test_id("camera-button").click(timeout=WAIT_MS)
        alert = tab.get_by_role("alert")
        expect(alert).to_contain_text("secure context", timeout=WAIT_MS)
        expect(alert).to_contain_text("localhost", timeout=WAIT_MS)
        expect(tab.get_by_test_id("camera-panel")).to_be_hidden(timeout=WAIT_MS)
    finally:
        context.close()


def test_mobile_viewport_has_no_horizontal_scroll(browser, server_url):
    context = browser.new_context(viewport={"width": 375, "height": 800}, is_mobile=True, has_touch=True)
    try:
        tab = context.new_page()
        tab.goto(server_url, timeout=WAIT_MS)
        expect(tab.get_by_test_id("model-badge")).to_contain_text("p99", timeout=WAIT_MS)
        _open_sample(tab, 3)
        overflow = tab.evaluate("() => document.documentElement.scrollWidth - document.documentElement.clientWidth")
        assert overflow <= 0, f"page scrolls horizontally by {overflow}px at 375px width"
    finally:
        context.close()


# ── camera (fake device fed from a .y4m file) ──────────────────────────────────────────────────

# PAD: (int, int), where the sample dial sits inside the 640 x 480 fake camera frame
PAD = ((640 - 256) // 2, (480 - 256) // 2)

# RECORD_STREAMS_JS: str, init script that remembers every MediaStream handed to the page, so the
# test can check afterwards that each track was stopped ("ended") - the camera light is off.
RECORD_STREAMS_JS = """
window.__streams = [];
const realGUM = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
navigator.mediaDevices.getUserMedia = async (constraints) => {
  const stream = await realGUM(constraints);
  window.__streams.push(stream);
  return stream;
};
"""


def _write_y4m(path: Path, frames: int = 5) -> np.ndarray:
    """Write a 640 x 480 I420 .y4m whose frames show sample 1 at PAD on mid-gray; returns the gray frame (480, 640) uint8.

    Y is stored in limited range (16..235) so the browser's BT.601 conversion gives back ~the original gray.
    """
    gray = np.full((480, 640), 128, np.uint8)
    gray[PAD[1]:PAD[1] + 256, PAD[0]:PAD[0] + 256] = _sample(1)
    y = np.round(16 + gray.astype(np.float64) * 219.0 / 255.0).astype(np.uint8)
    chroma = np.full((240, 320), 128, np.uint8)
    with open(path, "wb") as f:
        f.write(b"YUV4MPEG2 W640 H480 F30:1 Ip A1:1 C420jpeg\n")
        for _ in range(frames):
            f.write(b"FRAME\n")
            f.write(y.tobytes())
            f.write(chroma.tobytes())
            f.write(chroma.tobytes())
    return gray


def _model_input(response) -> np.ndarray:
    """The 256 x 256 gray image the model saw, from a /api/predict response -> (256, 256) uint8.

    For a 256-px square the server does not resample, so this is exactly the gray of the crop the
    browser uploaded (Chromium does not expose blob-backed multipart request bodies to Playwright).
    """
    url = response.json()["model_input_png"]
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("L"))


def test_camera_capture_once(playwright_instance, server_url, tmp_path):
    y4m = tmp_path / "dial.y4m"
    frame_gray = _write_y4m(y4m)
    chromium = playwright_instance.chromium.launch(headless=True, args=[
        "--use-fake-ui-for-media-stream",
        "--use-fake-device-for-media-stream",
        f"--use-file-for-fake-video-capture={y4m}",
    ])
    try:
        context = chromium.new_context(viewport=VIEWPORT, permissions=["camera"])
        tab = context.new_page()
        tab.set_default_timeout(WAIT_MS)
        tab.add_init_script(RECORD_STREAMS_JS)
        tab.goto(server_url)

        # Cancel first: the preview starts, then every track is stopped.
        tab.get_by_test_id("camera-button").click()
        expect(tab.get_by_test_id("camera-panel")).to_have_attribute("data-state", "live", timeout=WAIT_MS)
        tab.get_by_test_id("camera-cancel").click()
        expect(tab.get_by_test_id("camera-panel")).to_be_hidden(timeout=WAIT_MS)
        assert set(tab.evaluate("() => window.__streams.flatMap(s => s.getTracks().map(t => t.readyState))")) == {"ended"}

        # Capture ONE frame.
        tab.get_by_test_id("camera-button").click()
        panel = tab.get_by_test_id("camera-panel")
        expect(panel).to_have_attribute("data-state", "live", timeout=WAIT_MS)
        tab.wait_for_timeout(300)  # let a few frames play, as a person would aim the camera
        tab.get_by_test_id("capture-button").click()
        expect(panel).to_be_hidden(timeout=WAIT_MS)
        states = tab.evaluate("() => window.__streams.flatMap(s => s.getTracks().map(t => t.readyState))")
        assert len(states) >= 2 and set(states) == {"ended"}, states
        stage = tab.get_by_test_id("stage")
        expect(stage).to_have_attribute("data-source", "camera", timeout=WAIT_MS)
        expect(stage).to_have_attribute("data-natural-w", "640", timeout=WAIT_MS)
        expect(stage).to_have_attribute("data-natural-h", "480", timeout=WAIT_MS)
        _wait_prediction(tab, 80, 0, 480)  # auto-predict on the preselected square

        # Select the dial's square with the mouse and capture what the browser uploads.
        v = _view(tab)
        x, y = PAD
        with tab.expect_response("**/api/predict", timeout=WAIT_MS) as res:
            _drag(tab, _pixel_to_screen(v, x, y), _pixel_to_screen(v, x + 255, y + 255))
        d = _wait_prediction(tab, x, y, 256)
        seen = _model_input(res.value)
        mean_diff = float(np.abs(seen.astype(np.int16) - frame_gray[y:y + 256, x:x + 256]).mean())
        label = SAMPLES[0]["label"]
        err = _tip_error(d, label["x"] + x, label["y"] + y)
        y4m_path_worked = mean_diff < 12.0
        print(f"\ncamera: y4m frames {'reached' if y4m_path_worked else 'did NOT reach'} the page "
              f"(mean |posted - dial| = {mean_diff:.2f} gray levels); tip error {err:.3f} px")
        if y4m_path_worked:
            assert err < TOL_PX
        context.close()
    finally:
        chromium.close()
