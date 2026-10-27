"""Browser tests: start the real `python -m val_viewer` on a fake dataset and drive the page like a person.

    pytest --Popen--> python -m val_viewer --data-dir <fake> --run <fake run> --device cpu --port N
       |                              ^
       +--Playwright Chromium---------+   type in n, click scale / markers, click entries, arrow keys,
                                          resize the window, scroll

Screen positions are derived from the DOM the same way the page draws them:
    panel x on screen = panel_left + (data_x) * panel_width / 256        (data_x = label + 0.5)
"""

from __future__ import annotations

import contextlib
import json
import re
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

from val_viewer.tests.conftest import build_fake_data, write_params_jsonl

sync_api = pytest.importorskip("playwright.sync_api")
expect = sync_api.expect

# REPO_ROOT: Path, the directory that contains the val_viewer package
REPO_ROOT = Path(__file__).resolve().parents[2]
# WAIT_MS: int, explicit timeout for every UI wait (milliseconds)
WAIT_MS = 15_000
# SERVER_START_S: float, how long to wait for /api/health (ranking all 200 images on cpu included)
SERVER_START_S = 150.0
# DESKTOP: dict, default window size
DESKTOP = {"width": 1400, "height": 1000}


def _free_port() -> int:
    """Ask the OS for an unused TCP port on 127.0.0.1."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextlib.contextmanager
def serve(fd, log_path: Path):
    """Run `python -m val_viewer` over fd (namespace with data_dir, run_dir, cache_dir: Path) until the block ends.

    Yields the base url (str) once /api/health answers; the server's output goes to log_path (Path)."""
    port = _free_port()
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "val_viewer", "--data-dir", str(fd.data_dir), "--run", str(fd.run_dir),
             "--cache-dir", str(fd.cache_dir), "--device", "cpu", "--port", str(port)],
            cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
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
                    pytest.fail(f"server not healthy within {SERVER_START_S}s; log: {log_path.read_text()}")
                time.sleep(0.3)
            yield url
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


@pytest.fixture(scope="module")
def served(tmp_path_factory) -> SimpleNamespace:
    """One server per module over the fake dataset: SimpleNamespace(url: str, fd: fake_data namespace, log: Path of the server output)."""
    fd = build_fake_data(tmp_path_factory.mktemp("fake"))
    write_params_jsonl(fd)  # saved parameters: pointer.gray of row r is 100 + r
    log_path = tmp_path_factory.mktemp("server") / "server.log"
    with serve(fd, log_path) as url:
        yield SimpleNamespace(url=url, fd=fd, log=log_path)


@pytest.fixture(scope="module")
def browser():
    """Headless Chromium shared by the module."""
    with sync_api.sync_playwright() as pw:
        b = pw.chromium.launch(headless=True)
        yield b
        b.close()


@pytest.fixture
def page(browser, served):
    """A fresh page (1400 x 1000) on the viewer, loaded and showing the default 24 + 24 entries."""
    ctx = browser.new_context(viewport=DESKTOP)
    ctx.set_default_timeout(WAIT_MS)
    pg = ctx.new_page()
    pg.goto(served.url + "/")
    expect(pg.locator("#col-best .entry")).to_have_count(24)
    expect(pg.locator("#col-worst .entry")).to_have_count(24)
    yield pg
    ctx.close()


def api(served, path: str):
    """GET served.url + path and decode the JSON (list | dict)."""
    with urllib.request.urlopen(served.url + path, timeout=30) as res:
        return json.loads(res.read())


def set_n(page, text: str) -> None:
    """Type `text` into the n field and press Enter, like a person."""
    box = page.locator("#n-input")
    box.fill(text)
    box.press("Enter")


def test_page_shows_both_columns(page, served):
    summary = api(served, "/api/summary")
    head = page.locator("#summary").inner_text()
    assert "30" in head and summary["run"] in head
    first = api(served, "/api/list?side=best&n=1")[0]
    cap = page.locator("#col-best .entry").first.locator(".caption").inner_text()
    wanted = ["#1", f"val {first['split_pos']}", first["name"], f"err {first['err']:.2f} px",
              f"conf {first['conf']:.3f}",
              f"true ({first['label'][0]:.1f}, {first['label'][1]:.1f})",
              f"pred ({first['tip'][0]:.1f}, {first['tip'][1]:.1f})"]
    for w in wanted:
        assert w in cap, f"{w!r} missing in caption {cap!r}"


def test_change_n(page):
    set_n(page, "5")
    expect(page.locator("#col-best .entry")).to_have_count(5)
    expect(page.locator("#col-worst .entry")).to_have_count(5)
    set_n(page, "0")
    expect(page.locator("#n-input")).to_have_value("1")
    expect(page.locator("#col-best .entry")).to_have_count(1)
    expect(page.locator("#col-worst .entry")).to_have_count(1)
    set_n(page, "5000")
    expect(page.locator("#n-input")).to_have_value("1000")
    expect(page.locator("#col-best .entry")).to_have_count(30)
    expect(page.locator("#col-worst .entry")).to_have_count(30)


def _heat_pixels(page) -> list:
    """RGBA bytes (list[int]) of the first heatmap image, read back through a canvas once it is loaded."""
    return page.evaluate("""async () => {
        const img = document.querySelector('#col-best .entry img.heatmap');
        if (!img.complete || !img.naturalWidth) await new Promise(r => img.addEventListener('load', r, {once: true}));
        const c = document.createElement('canvas'); c.width = img.naturalWidth; c.height = img.naturalHeight;
        const g = c.getContext('2d'); g.drawImage(img, 0, 0);
        return Array.from(g.getImageData(0, 0, c.width, c.height).data);
    }""")


def _wait_first_heatmap(page, scale: str) -> None:
    """Wait until the first heatmap shows the `scale` ('log' | 'linear') image fully loaded."""
    page.wait_for_function(
        """(s) => { const i = document.querySelector('#col-best .entry img.heatmap');
        return i.currentSrc.includes('scale=' + s) && i.complete && i.naturalWidth > 0; }""", arg=scale)


def test_scale_switch(page):
    expect(page.locator("#col-best .entry img.heatmap").first).to_have_attribute("src", re.compile("scale=log"))
    _wait_first_heatmap(page, "log")
    log_px = _heat_pixels(page)
    page.locator("input[name=scale][value=linear]").check()
    srcs = page.locator(".entry img.heatmap").evaluate_all("els => els.map(e => e.getAttribute('src'))")
    assert srcs and all("scale=linear" in s for s in srcs)
    _wait_first_heatmap(page, "linear")
    assert _heat_pixels(page) != log_px
    page.locator("input[name=scale][value=log]").check()
    srcs = page.locator(".entry img.heatmap").evaluate_all("els => els.map(e => e.getAttribute('src'))")
    assert srcs and all("scale=log" in s for s in srcs)
    _wait_first_heatmap(page, "log")
    assert _heat_pixels(page) == log_px


def test_markers_toggle(page):
    lists = []
    page.on("request", lambda r: lists.append(r.url) if "/api/list" in r.url else None)
    visible = page.locator("svg.markers:visible")
    expect(visible.first).to_be_visible()
    page.locator("#markers-toggle").uncheck()
    expect(visible).to_have_count(0)
    page.locator("#markers-toggle").check()
    expect(visible.first).to_be_visible()
    assert lists == []


def test_marker_positions(page, served):
    worst = api(served, "/api/list?side=worst&n=3")
    for e in worst:
        entry = page.locator(f"#col-worst .entry[data-rank='{e['rank']}']")
        entry.scroll_into_view_if_needed()
        for img_cls in ("dial", "heatmap"):
            panel = entry.locator(f"img.{img_cls}").bounding_box()
            svg = entry.locator(f".panel:has(img.{img_cls}) svg.markers")
            true_tip, pred_tip = svg.locator(".true-tip"), svg.locator(".pred-tip")
            assert float(true_tip.get_attribute("data-x")) == pytest.approx(e["label"][0] + 0.5)
            assert float(true_tip.get_attribute("data-y")) == pytest.approx(e["label"][1] + 0.5)
            assert float(pred_tip.get_attribute("data-x")) == pytest.approx(e["tip"][0] + 0.5)
            assert float(pred_tip.get_attribute("data-y")) == pytest.approx(e["tip"][1] + 0.5)
            box = true_tip.bounding_box()
            cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
            want_x = panel["x"] + (e["label"][0] + 0.5) * panel["width"] / 256
            want_y = panel["y"] + (e["label"][1] + 0.5) * panel["height"] / 256
            assert abs(cx - want_x) <= 1 and abs(cy - want_y) <= 1, (img_cls, cx, cy, want_x, want_y)


def test_detail_view(page):
    page.locator("#col-worst .entry[data-rank='3'] img.dial").click()
    detail = page.locator("#detail")
    expect(detail).to_be_visible()
    expect(detail).to_have_attribute("data-rank", "3")
    assert detail.get_attribute("data-side") == "worst"
    for cls in ("dial", "heatmap"):
        box = detail.locator(f"img.{cls}").bounding_box()
        assert abs(box["width"] - 512) <= 1 and abs(box["height"] - 512) <= 1
    page.keyboard.press("ArrowRight")
    expect(detail).to_have_attribute("data-rank", "4")
    page.keyboard.press("ArrowLeft")
    page.keyboard.press("ArrowLeft")
    expect(detail).to_have_attribute("data-rank", "2")
    page.keyboard.press("ArrowLeft")
    expect(detail).to_have_attribute("data-rank", "1")
    page.keyboard.press("ArrowLeft")
    expect(detail).to_have_attribute("data-rank", "1")
    page.locator("#detail-next").click()
    expect(detail).to_have_attribute("data-rank", "2")
    page.locator("#detail-prev").click()
    expect(detail).to_have_attribute("data-rank", "1")
    page.keyboard.press("Escape")
    expect(detail).to_be_hidden()
    page.locator("#col-worst .entry[data-rank='1'] img.dial").click()
    expect(detail).to_be_visible()
    page.locator("#detail-close").click()
    expect(detail).to_be_hidden()


def test_zoom_clamped_at_border(page, served):
    set_n(page, "1000")
    expect(page.locator("#col-best .entry")).to_have_count(30)
    row = int(served.fd.val_idx[2])  # dataset row of val position 2 (its tip is at (2, 3))
    page.locator(f"#col-best .entry[data-row='{row}'] img.dial").click()
    detail = page.locator("#detail")
    expect(detail).to_be_visible()
    expect(detail).to_have_attribute("data-row", str(row))
    zooms = detail.locator(".zoom")
    assert zooms.count() >= 1
    for i in range(zooms.count()):
        z = zooms.nth(i)
        assert z.get_attribute("data-x0") == "0" and z.get_attribute("data-y0") == "0"
        tip = z.locator(".true-tip")
        assert float(tip.get_attribute("data-x")) == pytest.approx(2.5)
        assert float(tip.get_attribute("data-y")) == pytest.approx(3.5)
        zb, tb = z.bounding_box(), tip.bounding_box()
        cx, cy = tb["x"] + tb["width"] / 2, tb["y"] + tb["height"] / 2
        assert abs(cx - (zb["x"] + 2.5 * zb["width"] / 32)) <= 1
        assert abs(cy - (zb["y"] + 3.5 * zb["height"] / 32)) <= 1


def test_layout_stacks_when_narrow(page):
    def boxes():
        return page.locator("#col-best").bounding_box(), page.locator("#col-worst").bounding_box()

    page.set_viewport_size({"width": 1400, "height": 1000})
    b, w = boxes()
    assert w["x"] >= b["x"] + b["width"] - 1 and abs(w["y"] - b["y"]) <= 1
    page.set_viewport_size({"width": 1100, "height": 1000})
    b, w = boxes()
    assert w["x"] >= b["x"] + b["width"] - 1 and abs(w["y"] - b["y"]) <= 1
    assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
    page.set_viewport_size({"width": 800, "height": 1000})
    b, w = boxes()
    assert w["y"] >= b["y"] + b["height"] - 1


def test_lazy_images(browser, served):
    ctx = browser.new_context(viewport={"width": 1400, "height": 700})
    ctx.set_default_timeout(WAIT_MS)
    pg = ctx.new_page()
    # With n = 30 every val image is in both columns, so the same 30 URLs serve 60 heatmap <img> elements and
    # the browser cache may answer the second one without a request; count loaded <img> elements instead.
    loaded_js = "document.querySelectorAll('.entry img.heatmap').length && " \
                "[...document.querySelectorAll('.entry img.heatmap')].filter(i => i.complete && i.naturalWidth > 0).length"
    try:
        pg.goto(served.url + "/")
        set_n(pg, "30")
        expect(pg.locator("#col-worst .entry")).to_have_count(30)
        expect(pg.locator(".entry img.heatmap")).to_have_count(60)
        pg.wait_for_load_state("networkidle")
        before = pg.evaluate(loaded_js)
        assert 0 < before < 60, before
        deadline = time.monotonic() + 30
        n_loaded = before
        while n_loaded < 60 and time.monotonic() < deadline:
            pg.mouse.wheel(0, 500)  # a person scrolls down step by step
            pg.wait_for_timeout(150)
            n_loaded = pg.evaluate(loaded_js)
        assert n_loaded == 60, n_loaded
    finally:
        ctx.close()


def test_failed_load_can_be_retried(page):
    aborted: list = []

    def flaky(route):
        """Abort the first /api/list request, let every later one through."""
        if not aborted:
            aborted.append(route.request.url)
            route.abort()
        else:
            route.continue_()

    page.route("**/api/list*", flaky)
    set_n(page, "7")
    expect(page.locator("#status")).not_to_have_text("")
    assert len(aborted) == 1
    set_n(page, "7")  # same value again, like a person retrying
    expect(page.locator("#col-best .entry")).to_have_count(7)
    expect(page.locator("#col-worst .entry")).to_have_count(7)
    expect(page.locator("#status")).to_have_text("")


def _focus_entry_by_tab(page, side: str, rank: int) -> None:
    """Press Tab from the n field until the entry (side, rank) has focus (max 60 presses)."""
    page.locator("#n-input").focus()
    sel = f"#col-{side} .entry[data-rank='{rank}']"
    for _ in range(60):
        page.keyboard.press("Tab")
        if page.evaluate("(s) => document.activeElement === document.querySelector(s)", sel):
            return
    pytest.fail(f"could not Tab to {sel}")


def test_keyboard_opens_detail_and_traps_focus(page):
    detail = page.locator("#detail")
    inside = "() => !!document.activeElement.closest('#detail')"
    _focus_entry_by_tab(page, "best", 1)
    page.keyboard.press("Enter")
    expect(detail).to_be_visible()
    expect(detail).to_have_attribute("data-side", "best")
    expect(detail).to_have_attribute("data-rank", "1")
    assert page.evaluate(inside)
    for _ in range(8):
        page.keyboard.press("Tab")
        assert page.evaluate(inside)
    for _ in range(4):
        page.keyboard.press("Shift+Tab")
        assert page.evaluate(inside)
    page.keyboard.press("Escape")
    expect(detail).to_be_hidden()
    assert page.evaluate("() => document.activeElement === document.querySelector(\"#col-best .entry[data-rank='1']\")")
    _focus_entry_by_tab(page, "worst", 2)
    page.keyboard.press("Space")
    expect(detail).to_be_visible()
    expect(detail).to_have_attribute("data-side", "worst")
    expect(detail).to_have_attribute("data-rank", "2")
    assert page.evaluate(inside)
    page.keyboard.press("Escape")
    expect(detail).to_be_hidden()
    assert page.evaluate("() => document.activeElement === document.querySelector(\"#col-worst .entry[data-rank='2']\")")


SRC_JS = "els => els.map(e => e.getAttribute('src'))"  # str, reads the src attributes of the matched <img>s


def test_png_urls_carry_the_cache_key(page, served):
    key = api(served, "/api/summary")["key"]
    assert key
    want = "v=" + key
    srcs = page.locator(".entry img").evaluate_all(SRC_JS)
    assert len(srcs) == 96 and all(want in s for s in srcs), srcs[:3]
    page.locator("input[name=scale][value=linear]").check()  # a scale change keeps the key on every url
    srcs = page.locator(".entry img").evaluate_all(SRC_JS)
    assert all(want in s for s in srcs) and all("scale=linear" in s for s in srcs if "heatmap" in s), srcs[:3]
    page.locator("#col-best .entry[data-rank='1'] img.dial").click()
    expect(page.locator("#detail")).to_be_visible()
    srcs = page.locator("#detail img").evaluate_all(SRC_JS)
    assert len(srcs) == 4 and all(want in s for s in srcs), srcs


def test_legend_symbols_are_coloured(page):
    def strokes(sel):
        return page.locator(sel).evaluate_all("els => els.map(e => getComputedStyle(e).stroke)")

    for legend, marker in (("#legend-true", "#col-best .entry svg.markers .true-tip"),
                           ("#legend-pred", "#col-best .entry svg.markers .pred-tip")):
        got = strokes(f"{legend} .fg")
        want = strokes(f"{marker} .fg")[: len(got)]
        assert got and all(g != "none" for g in got), got
        assert got == want, (got, want)
    t = strokes("#legend-true .fg")[0]
    r, g, b = [int(v) for v in re.findall(r"\d+", t)[:3]]
    assert g > r and g > b, t
    p = strokes("#legend-pred .fg")[0]
    r, g, b = [int(v) for v in re.findall(r"\d+", p)[:3]]
    assert r > b and r >= g, p


def test_server_log_has_no_access_lines(page, served):
    page.wait_for_load_state("networkidle")
    log = served.log.read_text(errors="replace")
    assert "[rank]" in log
    assert "GET /api/" not in log and "GET /static" not in log, log[-600:]


# ---------------- val / train / all, saved parameters ----------------

def pick_set(page, name: str) -> None:
    """Click the radio of set `name` ('val' | 'train' | 'all') like a person."""
    page.locator(f"input[name=set][value={name}]").check()


def entry_rows(page, side: str = "best") -> list:
    """data-row (list[str]) of the entries of column `side`, top to bottom."""
    return page.locator(f"#col-{side} .entry").evaluate_all("els => els.map(e => e.dataset.row)")


def entry_tags(page, side: str = "best") -> list:
    """Text of the train / val tag (list[str]) of each entry of column `side`."""
    return page.locator(f"#col-{side} .entry .tag").evaluate_all("els => els.map(e => e.textContent.trim())")


def test_set_switch(page, served):
    first_val = entry_rows(page)
    assert set(entry_tags(page)) == {"val"}
    pick_set(page, "train")
    expect(page.locator("#summary")).to_contain_text("train 170")
    expect(page.locator("#col-best .entry")).to_have_count(24)
    assert set(entry_tags(page)) == {"train"} and set(entry_tags(page, "worst")) == {"train"}
    want = [str(e["row"]) for e in api(served, "/api/list?set=train&side=best&n=24")]
    assert entry_rows(page) == want
    pick_set(page, "all")
    expect(page.locator("#summary")).to_contain_text("all 200 (train 170 · val 30)")
    expect(page.locator("#col-best .entry")).to_have_count(24)
    both = entry_tags(page) + entry_tags(page, "worst")
    assert set(both) == {"train", "val"}, both
    pick_set(page, "val")
    expect(page.locator("#summary")).to_contain_text(re.compile(r"· val 30 ·"))  # not the all-header's "val 30)"
    expect(page.locator("#col-best .entry")).to_have_count(24)
    assert entry_rows(page) == first_val and set(entry_tags(page)) == {"val"}


def param_gray(page):
    """Text of the pointer.gray value shown in the Parameters column (str)."""
    return page.locator("#params [data-section='pointer'] [data-key='gray'] .v").inner_text()


def test_detail_shows_params(page, served):
    worst = api(served, "/api/list?side=worst&n=2")
    page.locator("#col-worst .entry[data-rank='1'] img.dial").click()
    expect(page.locator("#detail")).to_be_visible()
    expect(page.locator("#params [data-section='pointer']")).to_be_visible()
    expect(page.locator("#params [data-key='gray'] .v").first).to_have_text(str(100 + worst[0]["row"]))
    assert param_gray(page) == str(100 + worst[0]["row"])
    page.keyboard.press("ArrowRight")
    expect(page.locator("#detail")).to_have_attribute("data-rank", "2")
    expect(page.locator("#params [data-section='pointer'] [data-key='gray'] .v")).to_have_text(
        str(100 + worst[1]["row"]))
    head = page.locator("#params").inner_text()
    assert f"seed {1000 + worst[1]['row']}" in head


def test_detail_no_params_message(page):
    page.route("**/api/params/*", lambda route: route.fulfill(
        status=404, content_type="application/json",
        body=json.dumps({"detail": "no parameters saved for this dataset"})))
    page.locator("#col-worst .entry[data-rank='1'] img.dial").click()
    expect(page.locator("#detail")).to_be_visible()
    expect(page.locator("#params")).to_contain_text("No parameters saved for this dataset")
    expect(page.locator("#params")).to_contain_text("studio")
    assert page.locator("#params [data-section]").count() == 0


def test_params_layout_narrow(page):
    page.set_viewport_size({"width": 800, "height": 1000})
    page.locator("#col-worst .entry[data-rank='1'] img.dial").click()
    expect(page.locator("#detail")).to_be_visible()
    expect(page.locator("#params [data-section='pointer']")).to_be_visible()
    zoom_bottom = max(page.locator(f"#detail {sel}").bounding_box()["y"]
                      + page.locator(f"#detail {sel}").bounding_box()["height"]
                      for sel in (".big-row", ".zoom-row"))
    assert page.locator("#params").bounding_box()["y"] >= zoom_bottom - 1
    assert page.evaluate("document.documentElement.scrollWidth") <= 800
    page.set_viewport_size({"width": 1400, "height": 1000})  # wide: the column sits to the right of the panels
    big = page.locator("#detail .big-row").bounding_box()
    assert page.locator("#params").bounding_box()["x"] >= big["x"] + big["width"] - 1


def test_captions_tag_split(page, served):
    pick_set(page, "all")
    set_n(page, "1000")
    expect(page.locator("#col-best .entry")).to_have_count(200)
    lst = api(served, "/api/list?set=all&side=best&n=1000")
    for split in ("train", "val"):
        e = next(e for e in lst if e["split"] == split)
        cap = page.locator(f"#col-best .entry[data-row='{e['row']}'] .caption").inner_text()
        assert f"{split} {e['split_pos']}" in cap, cap
        tag = page.locator(f"#col-best .entry[data-row='{e['row']}'] .tag").inner_text().strip()
        assert tag == split


def test_fast_set_switch_keeps_last(page, served):
    held = []  # list[Route]: set=train responses held back until the person has moved on

    def hold_train(route):
        held.append(route)

    page.route(re.compile(r"/api/(list|summary)\?.*set=train"), hold_train)
    pick_set(page, "train")
    for _ in range(100):
        if len(held) >= 3:
            break
        page.wait_for_timeout(100)
    assert len(held) >= 3, len(held)  # summary + best + worst of the train load are in flight
    pick_set(page, "all")
    expect(page.locator("#summary")).to_contain_text("all 200")
    expect(page.locator("#col-best .entry")).to_have_count(24)
    for route in held:  # now the late train answers arrive
        route.continue_()
    page.wait_for_timeout(1000)
    assert set(entry_tags(page)) | set(entry_tags(page, "worst")) == {"train", "val"}
    want = [str(e["row"]) for e in api(served, "/api/list?set=all&side=best&n=24")]
    assert entry_rows(page) == want
    assert "all 200" in page.locator("#summary").inner_text()
    assert page.locator("input[name=set][value=all]").is_checked()


def test_generated_dataset_shows_its_parameters(browser, tmp_path):
    """A dataset the generator wrote with save_params=True: the detail view lists each image's own pointer.gray."""
    import generate_synthetic_v7_mp as g
    import v7_config
    from val_viewer.tests.conftest import make_model_file
    cfg = v7_config.default_config()
    fd = SimpleNamespace(data_dir=tmp_path / "gen", run_dir=tmp_path / "run", cache_dir=tmp_path / "vv_cache")
    g.generate_dataset(cfg, str(fd.data_dir), 40, seed=5, workers=1, save_params=True)
    fd.run_dir.mkdir()
    make_model_file(fd.run_dir / "tip.pth", 1234)
    with serve(fd, tmp_path / "server.log") as url:
        n_val = api(SimpleNamespace(url=url), "/api/summary")["n"]  # int, val images of the 40
        ctx = browser.new_context(viewport=DESKTOP)
        ctx.set_default_timeout(WAIT_MS)
        pg = ctx.new_page()
        try:
            pg.goto(url + "/")
            expect(pg.locator("#col-worst .entry")).to_have_count(n_val)
            pg.locator("#col-worst .entry[data-rank='1'] img.dial").click()
            for rank in ("1", "2"):
                if rank == "2":
                    pg.keyboard.press("ArrowRight")
                expect(pg.locator("#detail")).to_have_attribute("data-rank", rank)
                row = int(pg.locator(f"#col-worst .entry[data-rank='{rank}']").get_attribute("data-row"))
                gray = g.render_image(cfg, g._seed_for_index(5, row)).params["pointer"]["gray"]  # int
                expect(pg.locator("#params [data-section='pointer'] [data-key='gray'] .v")).to_have_text(str(gray))
        finally:
            ctx.close()
