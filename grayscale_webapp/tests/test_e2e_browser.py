"""End-to-end tests: start the real server and drive the real UI in headless Chromium, like a person would.

    pytest ──Popen──▶ python -m grayscale_webapp --port N  (real uvicorn process)
       │                         ▲
       └──Playwright Chromium────┘  choose / drop / paste files, change dropdowns, click Download
"""

from __future__ import annotations

import base64
import io
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

# REPO_ROOT: Path, the directory that contains the grayscale_webapp package
REPO_ROOT = Path(__file__).resolve().parents[2]
# WAIT_MS: int, explicit timeout for every UI wait (milliseconds)
WAIT_MS = 15_000
# SERVER_START_S: float, how long to wait for /api/health after spawning the server (seconds)
SERVER_START_S = 30.0


def _free_port() -> int:
    """Ask the OS for an unused TCP port on 127.0.0.1."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="session")
def server_url(tmp_path_factory) -> str:
    """Run `python -m grayscale_webapp` in a subprocess for the whole session; yields its base URL (str)."""
    port = _free_port()
    log_path = tmp_path_factory.mktemp("server") / "server.log"
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "grayscale_webapp", "--port", str(port)],
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


@pytest.fixture(scope="session")
def browser():
    """One headless Chromium for the session."""
    with sync_api.sync_playwright() as pw:
        chromium = pw.chromium.launch(headless=True)
        yield chromium
        chromium.close()


@pytest.fixture
def page(browser, server_url):
    """A fresh desktop-sized tab with the app loaded and the method list fetched."""
    context = browser.new_context(viewport={"width": 1280, "height": 900}, accept_downloads=True)
    tab = context.new_page()
    tab.set_default_timeout(WAIT_MS)
    tab.goto(server_url)
    expect(tab.get_by_test_id("method-select").locator("option")).to_have_count(4, timeout=WAIT_MS)
    yield tab
    context.close()


@pytest.fixture
def colourful_png(tmp_path) -> tuple[Path, np.ndarray]:
    """A 64x48 colourful PNG on disk; returns (path, rgb array of shape (48, 64, 3) uint8)."""
    yy, xx = np.mgrid[0:48, 0:64]
    rgb = np.stack([xx * 4, yy * 5, (xx + yy) * 2], axis=-1).clip(0, 255).astype(np.uint8)
    path = tmp_path / "sunset.png"
    Image.fromarray(rgb).save(path)
    return path, rgb


def _download_result(page, tmp_path: Path) -> tuple[Image.Image, str]:
    """Click the Download button like a user; returns (decoded image, suggested filename)."""
    with page.expect_download(timeout=WAIT_MS) as info:
        page.get_by_test_id("download-button").click(timeout=WAIT_MS)
    download = info.value
    target = tmp_path / download.suggested_filename
    download.save_as(target)
    img = Image.open(target)
    img.load()
    return img, download.suggested_filename


def _wait_for_result(page, method: str) -> None:
    """Wait until the result image for the given method id is visible."""
    expect(page.locator(f'[data-testid="result-img"][data-method="{method}"]')).to_be_visible(timeout=WAIT_MS)


def test_choose_file_convert_and_download(page, colourful_png, tmp_path):
    path, rgb = colourful_png
    page.get_by_test_id("file-input").set_input_files(str(path), timeout=WAIT_MS)

    _wait_for_result(page, "luma601")
    expect(page.get_by_test_id("original-img")).to_be_visible(timeout=WAIT_MS)
    expect(page.get_by_test_id("result-meta")).to_contain_text("64 × 48", timeout=WAIT_MS)
    expect(page.get_by_test_id("original-meta")).to_contain_text("64 × 48", timeout=WAIT_MS)
    expect(page.get_by_test_id("error-box")).to_be_hidden(timeout=WAIT_MS)

    img, name = _download_result(page, tmp_path)
    assert name == "sunset_gray.png"
    assert img.mode in ("L", "LA") and img.size == (64, 48)
    assert np.array_equal(np.asarray(img.convert("L")), np.asarray(Image.fromarray(rgb).convert("L")))


def test_switching_method_reconverts(page, colourful_png, tmp_path):
    path, rgb = colourful_png
    page.get_by_test_id("file-input").set_input_files(str(path), timeout=WAIT_MS)
    _wait_for_result(page, "luma601")

    page.get_by_test_id("method-select").select_option("average", timeout=WAIT_MS)
    _wait_for_result(page, "average")
    expect(page.get_by_test_id("method-description")).to_contain_text("(R + G + B) / 3", timeout=WAIT_MS)

    img, _ = _download_result(page, tmp_path)
    expected = np.floor(rgb.astype(np.float64).sum(axis=2) / 3.0 + 0.5).astype(np.uint8)
    assert np.array_equal(np.asarray(img.convert("L")), expected)

    # Changing the output format re-converts the same file into JPEG.
    page.get_by_test_id("format-select").select_option("jpeg", timeout=WAIT_MS)
    expect(page.locator('[data-testid="result-img"][data-format="jpeg"]')).to_be_visible(timeout=WAIT_MS)
    jpeg, name = _download_result(page, tmp_path)
    assert name == "sunset_gray.jpg" and jpeg.format == "JPEG" and jpeg.mode == "L"


def test_drag_and_drop(page, colourful_png):
    path, _ = colourful_png
    b64 = base64.b64encode(path.read_bytes()).decode()
    page.get_by_test_id("drop-zone").evaluate(
        """(zone, b64) => {
            const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
            const dt = new DataTransfer();
            dt.items.add(new File([bytes], "dropped.png", { type: "image/png" }));
            zone.dispatchEvent(new DragEvent("dragenter", { dataTransfer: dt, bubbles: true, cancelable: true }));
            zone.dispatchEvent(new DragEvent("dragover", { dataTransfer: dt, bubbles: true, cancelable: true }));
            zone.dispatchEvent(new DragEvent("drop", { dataTransfer: dt, bubbles: true, cancelable: true }));
        }""",
        b64,
    )
    _wait_for_result(page, "luma601")
    expect(page.get_by_test_id("result-meta")).to_contain_text("64 × 48", timeout=WAIT_MS)
    assert page.get_by_test_id("download-button").get_attribute("download") == "dropped_gray.png"


def test_keyboard_opens_file_picker(page):
    page.get_by_test_id("drop-zone").focus()
    with page.expect_file_chooser(timeout=WAIT_MS):
        page.keyboard.press("Enter")


def test_text_file_shows_error(page):
    page.get_by_test_id("file-input").set_input_files(
        {"name": "notes.txt", "mimeType": "text/plain", "buffer": b"definitely not a picture"}, timeout=WAIT_MS
    )
    error = page.get_by_test_id("error-box")
    expect(error).to_be_visible(timeout=WAIT_MS)
    expect(error).to_contain_text("not an image", timeout=WAIT_MS)
    expect(page.get_by_test_id("result-img")).to_be_hidden(timeout=WAIT_MS)


def test_server_error_detail_is_shown(page):
    # Looks like a PNG to the browser, but the bytes are garbage: the server's 400 detail must reach the user.
    page.get_by_test_id("file-input").set_input_files(
        {"name": "broken.png", "mimeType": "image/png", "buffer": b"not really png bytes"}, timeout=WAIT_MS
    )
    error = page.get_by_test_id("error-box")
    expect(error).to_be_visible(timeout=WAIT_MS)
    expect(error).to_contain_text("not a recognised image", timeout=WAIT_MS)
    expect(page.get_by_test_id("result-img")).to_be_hidden(timeout=WAIT_MS)


def test_reset_returns_to_start(page, colourful_png):
    path, _ = colourful_png
    page.get_by_test_id("file-input").set_input_files(str(path), timeout=WAIT_MS)
    _wait_for_result(page, "luma601")
    page.get_by_test_id("reset-button").click(timeout=WAIT_MS)
    expect(page.get_by_test_id("results")).to_be_hidden(timeout=WAIT_MS)
    expect(page.get_by_test_id("drop-zone")).to_be_focused(timeout=WAIT_MS)


def test_mobile_viewport_has_no_horizontal_scroll(browser, server_url, colourful_png):
    path, _ = colourful_png
    context = browser.new_context(viewport={"width": 375, "height": 800}, is_mobile=True, has_touch=True)
    try:
        tab = context.new_page()
        tab.goto(server_url, timeout=WAIT_MS)
        tab.get_by_test_id("file-input").set_input_files(str(path), timeout=WAIT_MS)
        _wait_for_result(tab, "luma601")
        overflow = tab.evaluate(
            "() => document.documentElement.scrollWidth - document.documentElement.clientWidth"
        )
        assert overflow <= 0, f"page scrolls horizontally by {overflow}px at 375px width"
    finally:
        context.close()
