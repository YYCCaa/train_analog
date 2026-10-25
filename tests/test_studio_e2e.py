"""Studio browser E2E: drive the real page (Playwright Chromium, headless) the
way a person would -- clicks, typing, file pickers -- against a live
`studio/run_studio.py` server on port 8799."""
import csv
import json
import math
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest
import yaml

import v7_config

playwright_sync = pytest.importorskip("playwright.sync_api")

# repo root (Path); run_studio.py must be launched from here
REPO_ROOT = Path(__file__).resolve().parents[1]
# server port (int) and base URL (str) of the studio under test
PORT = 8799
BASE = f"http://127.0.0.1:{PORT}"
# per-wait timeout for UI updates (ms, int)
UI_TIMEOUT = 30_000

# JS predicate: 9 tile <img> are decoded at 256 px natural width
_TILES_LOADED_JS = """() => {
  const imgs = [...document.querySelectorAll('[data-testid=tile] img')];
  return imgs.length === 9 && imgs.every(i => i.complete && i.naturalWidth === 256);
}"""

# JS predicate: no debounce pending / request in flight, latest response applied
_SETTLED_JS = """() => {
  const g = document.querySelector('[data-testid=preview-grid]');
  return !!g && g.dataset.busy === 'false' && g.dataset.appliedSeq === g.dataset.sentSeq
      && Number(g.dataset.appliedSeq) > 0;
}"""


def _port_open(port):
    """True if something already listens on 127.0.0.1:port (int)."""
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _kill_tree(proc):
    """Kill proc (subprocess.Popen) and every child (uvicorn + pool workers)."""
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       capture_output=True, timeout=30)
    else:
        proc.kill()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    """Start run_studio.py --port 8799 --no-browser; yield BASE (str); kill tree."""
    assert not _port_open(PORT), f"port {PORT} already in use (stale studio server?)"
    log_path = tmp_path_factory.mktemp("studio_server") / "server.log"  # uvicorn log (Path)
    log = open(log_path, "wb")
    proc = subprocess.Popen(
        [sys.executable, str(REPO_ROOT / "studio" / "run_studio.py"),
         "--port", str(PORT), "--no-browser"],
        cwd=str(REPO_ROOT), stdout=log, stderr=subprocess.STDOUT)
    try:
        deadline = time.time() + 300  # build may run first (s)
        while True:
            if proc.poll() is not None:
                log.flush()
                pytest.fail(f"studio exited early ({proc.returncode}):\n"
                            + log_path.read_text(errors="replace")[-3000:])
            try:
                with urllib.request.urlopen(BASE + "/api/schema", timeout=2) as r:
                    if r.status == 200:
                        break
            except OSError:
                pass
            if time.time() > deadline:
                pytest.fail("studio not ready within 300 s:\n"
                            + log_path.read_text(errors="replace")[-3000:])
            time.sleep(0.5)
        yield BASE
    finally:
        _kill_tree(proc)
        log.close()


@pytest.fixture(scope="module")
def pw():
    """The module's single Playwright driver (sync API allows one per thread)."""
    with playwright_sync.sync_playwright() as p:
        yield p


@pytest.fixture(scope="module")
def browser(pw):
    """One headless Chromium for the module (playwright Browser)."""
    b = pw.chromium.launch(headless=True)
    try:
        yield b
    finally:
        b.close()


@pytest.fixture(scope="module")
def firefox_browser(pw):
    """Headless Firefox (playwright Browser); skipped if it is not installed."""
    try:
        b = pw.firefox.launch(headless=True)
    except Exception as e:  # browser binary missing
        pytest.skip(f"Playwright Firefox unavailable ({e}); "
                    "run `python -m playwright install firefox`")
    try:
        yield b
    finally:
        b.close()


def _open_page(browser, server):
    """New 1400x900 context + page on the studio with tiles loaded -> (ctx, page)."""
    ctx = browser.new_context(viewport={"width": 1400, "height": 900}, accept_downloads=True)
    pg = ctx.new_page()
    pg.set_default_timeout(UI_TIMEOUT)
    pg.goto(server)
    _wait_settled(pg)
    return ctx, pg


@pytest.fixture
def page(server, browser):
    """Fresh Chromium page on the studio, preview tiles loaded (playwright Page)."""
    ctx, pg = _open_page(browser, server)
    try:
        yield pg
    finally:
        ctx.close()


@pytest.fixture
def firefox_page(server, firefox_browser):
    """Fresh Firefox page on the studio, preview tiles loaded (playwright Page)."""
    ctx, pg = _open_page(firefox_browser, server)
    try:
        yield pg
    finally:
        ctx.close()


def _wait_settled(page):
    """Wait until the latest preview is applied and all 9 tiles are decoded."""
    page.wait_for_function(_SETTLED_JS, timeout=UI_TIMEOUT)
    page.wait_for_function(_TILES_LOADED_JS, timeout=UI_TIMEOUT)


def _applied_seq(page):
    """Grid's data-applied-seq (int)."""
    return int(page.get_by_test_id("preview-grid").get_attribute("data-applied-seq"))


def _commit_and_wait(page, locator, value):
    """Type value (str) into a number input, press Enter, wait for a new preview."""
    old = _applied_seq(page)
    locator.fill(value)
    locator.press("Enter")
    page.wait_for_function(
        f"() => Number(document.querySelector('[data-testid=preview-grid]')"
        f".dataset.appliedSeq) > {old}", timeout=UI_TIMEOUT)
    _wait_settled(page)


def _num(page, param, handle=None):
    """Number input of control data-param=param (str); handle 'lo'/'hi' for ranges."""
    sel = f'[data-param="{param}"] input[type=number]'
    if handle:
        sel += f'[data-handle="{handle}"]'
    return page.locator(sel)


def _tile_srcs(page):
    """list[str]: src of each tile image in grid order."""
    return page.eval_on_selector_all("[data-testid=tile] img", "els => els.map(e => e.src)")


def _marker_geometry(page):
    """list[dict]: per tile seed (int), marker centre and img box in CSS px."""
    return page.eval_on_selector_all("[data-testid=tile]", """figs => figs.map(f => {
        const img = f.querySelector('img').getBoundingClientRect();
        const m = f.querySelector('[data-testid=tip-marker]');
        const r = m ? m.getBoundingClientRect() : null;
        return {seed: Number(f.dataset.seed), ix: img.x, iy: img.y, iw: img.width,
                cx: r ? r.x + r.width / 2 : null, cy: r ? r.y + r.height / 2 : null};
    })""")


def _save_yaml(page, dest):
    """Click save-yaml, store the download at dest (Path); return its text."""
    with page.expect_download() as dl:
        page.get_by_test_id("save-yaml").click()
    assert dl.value.suggested_filename == "v7_config.yaml"
    dl.value.save_as(str(dest))
    return dest.read_text(encoding="utf-8")


def _post_json(base, path, body):
    """POST body (dict) as JSON to base+path; return parsed JSON response."""
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def _wait_input(page, selector, js_cond, timeout=UI_TIMEOUT):
    """Wait until js_cond (JS expr over number `v`) holds for the input at selector."""
    page.wait_for_function(
        f"() => {{ const e = document.querySelector('{selector}');"
        f" const v = Number(e && e.value); return !!e && ({js_cond}); }}", timeout=timeout)


def _drag(page, x0, x1, y):
    """Mouse-drag from (x0, y) to (x1, y) in CSS px (floats), in small steps."""
    page.mouse.move(x0, y)
    page.mouse.down()
    page.mouse.move(x1, y, steps=15)
    page.mouse.up()


def test_page_shows_preview(page):
    # naturalWidth of each tile image, list[int] of length 9
    widths = page.eval_on_selector_all("[data-testid=tile] img",
                                       "els => els.map(e => e.naturalWidth)")
    assert widths == [256] * 9


def test_every_param_has_control(page, server):
    with urllib.request.urlopen(server + "/api/schema", timeout=10) as r:
        schema = json.loads(r.read())  # section -> key -> leaf dict
    for sec, keys in schema.items():
        assert page.locator(f'section[data-section="{sec}"] h2').count() == 1, sec
        for key, leaf in keys.items():
            ctl = page.locator(f'[data-param="{sec}.{key}"]')  # control container (Locator)
            assert ctl.count() >= 1, f"{sec}.{key} has no control"
            assert leaf["help"] in ctl.first.inner_text(), f"{sec}.{key} help missing"
    assert _num(page, "canvas.img_size").is_disabled()
    assert page.locator('[data-param="canvas.img_size"] input[type=range]').is_disabled()


def test_tick_slider_changes_only_ticks(page):
    page.get_by_test_id("overlay-toggle").check()
    before_src = _tile_srcs(page)  # tile srcs before the change, list[str] (9)
    before_mk = _marker_geometry(page)  # marker geometry before, list[dict] (9)
    assert all(m["cx"] is not None for m in before_mk)
    _commit_and_wait(page, _num(page, "ticks.len_range", "hi"), "14")
    _commit_and_wait(page, _num(page, "ticks.len_range", "lo"), "12")
    after_src = _tile_srcs(page)  # tile srcs after, list[str] (9)
    after_mk = _marker_geometry(page)  # marker geometry after, list[dict] (9)
    assert any(a != b for a, b in zip(before_src, after_src)), "no tile changed"
    for b, a in zip(before_mk, after_mk):
        assert a["seed"] == b["seed"]
        assert abs(a["cx"] - b["cx"]) < 0.5 and abs(a["cy"] - b["cy"]) < 0.5, (b, a)


def test_overlay_marker_on_apex(page):
    with page.expect_response(lambda r: r.url.endswith("/api/preview")
                              and r.request.method == "POST") as resp:
        page.get_by_test_id("reshuffle").click()
    tips = {t["seed"]: t for t in resp.value.json()["tiles"]}  # seed (int) -> tile dict
    _wait_settled(page)
    page.get_by_test_id("overlay-toggle").check()
    markers = page.get_by_test_id("tip-marker")  # all tip markers (Locator)
    assert markers.count() == 9
    for i in range(9):
        assert markers.nth(i).is_visible()
    geo = _marker_geometry(page)  # per-tile marker/img geometry, list[dict] (9)
    assert sorted(g["seed"] for g in geo) == sorted(tips)
    for g in geo:
        scale = g["iw"] / 256.0  # displayed px per image px (float)
        t = tips[g["seed"]]  # tile dict of this figure's seed
        # pixel-centre labels: image px i's centre is drawn at (i + 0.5) * scale
        assert abs(g["cx"] - (g["ix"] + (t["tip_x"] + 0.5) * scale)) <= 1, (g, t)
        assert abs(g["cy"] - (g["iy"] + (t["tip_y"] + 0.5) * scale)) <= 1, (g, t)


def test_rapid_drag_shows_latest(page, server):
    """Hold every /api/preview response, release the newest first and the stale
    ones after it: the grid must keep showing the newest config."""
    held = []  # intercepted preview requests not yet answered, list[playwright Route]
    page.route("**/api/preview", lambda route: held.append(route))
    finished = []  # preview requests whose response reached the page, list[Request]
    page.on("requestfinished",
            lambda req: finished.append(req) if req.url.endswith("/api/preview") else None)
    grid = page.get_by_test_id("preview-grid")  # preview grid (Locator)
    sent0 = int(grid.get_attribute("data-sent-seq"))  # sent seq before typing (int)
    hi = _num(page, "ticks.len_range", "hi")  # hi number input (Locator)
    values = [str(v) for v in range(10, 20)]  # 10 successive hi values (str)
    for v in values:
        hi.fill(v)
        hi.press("Enter")
        page.wait_for_timeout(200)  # human pause just over the 150 ms debounce
    deadline = time.time() + 30  # s (float)
    while len(held) < len(values) and time.time() < deadline:
        page.wait_for_timeout(50)
    assert len(held) == len(values), f"expected one request per value, got {len(held)}"
    assert int(grid.get_attribute("data-sent-seq")) == sent0 + len(values)
    # newest first ...
    held[-1].fulfill(response=held[-1].fetch())
    _wait_settled(page)
    latest_src = _tile_srcs(page)  # tiles of the newest response, list[str] (9)
    # ... then the stale ones, oldest last
    for route in reversed(held[:-1]):
        route.fulfill(response=route.fetch())
    while len(finished) < len(values) and time.time() < deadline:
        page.wait_for_timeout(50)
    page.wait_for_timeout(300)  # let the page's stale handlers run (they must drop)
    assert grid.get_attribute("data-applied-seq") == str(sent0 + len(values))
    assert grid.get_attribute("data-sent-seq") == str(sent0 + len(values))
    assert hi.input_value() == values[-1]
    assert _tile_srcs(page) == latest_src
    # the tiles shown are exactly the render of the latest config
    cfg = v7_config.default_config()  # expected config, section -> key -> value
    cfg["ticks"]["len_range"] = [6, int(values[-1])]
    seeds = page.eval_on_selector_all(  # seeds shown in grid order, list[int] (9)
        "[data-testid=tile]", "fs => fs.map(f => Number(f.dataset.seed))")
    # reference render of the latest config, list[tile dict] (9)
    want = _post_json(server, "/api/preview", {"config": cfg, "seeds": seeds})["tiles"]
    assert latest_src == ["data:image/png;base64," + t["png_b64"] for t in want]


def test_range_handles_usable_at_max(page):
    """lo == hi == max: dragging left from the handle must move lo down."""
    lo = _num(page, "ticks.len_range", "lo")  # lo number input (Locator)
    hi = _num(page, "ticks.len_range", "hi")  # hi number input (Locator)
    _commit_and_wait(page, hi, "20")
    _commit_and_wait(page, lo, "20")
    # slider box in CSS px, dict x/y/width/height (floats)
    box = page.locator('[data-param="ticks.len_range"] input[type=range][data-handle="lo"]'
                       ).bounding_box()
    y = box["y"] + box["height"] / 2  # slider centre line (float px)
    _drag(page, box["x"] + box["width"] - 7, box["x"] + box["width"] * 0.4, y)
    _wait_input(page, '[data-param="ticks.len_range"] input[type=number][data-handle="lo"]',
                "v < 20")
    assert int(lo.input_value()) < 20
    assert hi.input_value() == "20"


def test_range_slider_drag_firefox(firefox_page):
    """Dual-range handles are draggable in Firefox too (default browser may be FF)."""
    lo = _num(firefox_page, "dataset.angle_deg_range", "lo")  # lo number input (Locator)
    assert lo.input_value() == "0"
    # slider box in CSS px, dict x/y/width/height (floats)
    box = firefox_page.locator(
        '[data-param="dataset.angle_deg_range"] input[type=range][data-handle="lo"]'
    ).bounding_box()
    y = box["y"] + box["height"] / 2  # slider centre line (float px)
    _drag(firefox_page, box["x"] + 7, box["x"] + box["width"] * 0.5, y)
    _wait_input(firefox_page,
                '[data-param="dataset.angle_deg_range"] input[type=number][data-handle="lo"]',
                "v > 0")
    assert int(lo.input_value()) > 0
    assert _num(firefox_page, "dataset.angle_deg_range", "hi").input_value() == "359"


def test_invalid_yaml_shows_error(page, tmp_path):
    bad = tmp_path / "bad.yaml"  # YAML with a syntax error (Path)
    bad.write_text("ticks: [\n", encoding="utf-8")
    page.get_by_test_id("load-yaml").set_input_files(str(bad))
    errs = page.get_by_test_id("config-errors")  # error box (Locator)
    errs.get_by_text("yaml").first.wait_for(timeout=UI_TIMEOUT)
    assert "yaml" in errs.inner_text()
    page.wait_for_function(_TILES_LOADED_JS, timeout=UI_TIMEOUT)
    assert _num(page, "ticks.len_range", "hi").input_value() == "9"


def test_load_old_yaml_shows_adjustments(page, tmp_path):
    old = tmp_path / "old.yaml"  # old-format YAML (Path)
    old.write_text("reflections:\n  perspective: 0.4\n  warp_grid: 4\n  warp_amplitude: 3\n"
                   "foo:\n  bar: 1\n", encoding="utf-8")
    seq = _applied_seq(page)  # applied seq before loading (int)
    page.get_by_test_id("load-yaml").set_input_files(str(old))
    note = page.get_by_test_id("config-warnings")  # adjustments notice (Locator)
    note.get_by_text("perspective").first.wait_for(timeout=UI_TIMEOUT)
    assert "Loaded with" in note.inner_text() and "adjustments" in note.inner_text()
    assert "foo" in note.inner_text()
    assert page.get_by_test_id("config-errors").inner_text().strip() == ""
    _wait_applied_after(page, seq)
    assert _num(page, "reflections.perspective_range", "lo").input_value() == "0"
    assert _num(page, "reflections.perspective_range", "hi").input_value() == "0.4"
    assert _num(page, "reflections.warp_grid_range", "lo").input_value() == "4"
    page.wait_for_function(_TILES_LOADED_JS, timeout=UI_TIMEOUT)
    note.get_by_role("button", name="Dismiss").click()
    assert not note.is_visible()


def test_save_load_roundtrip(page, tmp_path):
    prob = _num(page, "digits.show_prob")  # show_prob number input (Locator)
    assert prob.input_value() == "0.5"
    _commit_and_wait(page, prob, "1")
    saved = tmp_path / "saved.yaml"  # downloaded YAML (Path)
    _save_yaml(page, saved)
    page.reload()
    _wait_settled(page)
    assert _num(page, "digits.show_prob").input_value() == "0.5"
    old = _applied_seq(page)  # applied seq before loading (int)
    page.get_by_test_id("load-yaml").set_input_files(str(saved))
    page.wait_for_function(
        f"() => Number(document.querySelector('[data-testid=preview-grid]')"
        f".dataset.appliedSeq) > {old}", timeout=UI_TIMEOUT)
    assert _num(page, "digits.show_prob").input_value() == "1"


def _progress(page):
    """(value, max) of the gen-progress bar, floats."""
    return page.get_by_test_id("gen-progress").evaluate("e => [e.value, e.max]")


def test_generate_and_cancel(page, tmp_path):
    out = tmp_path / "out"  # first (complete) output folder, absolute (Path)
    page.get_by_test_id("gen-output-dir").fill(str(out))
    page.get_by_test_id("gen-num-images").fill("20")
    page.get_by_test_id("gen-seed").fill("3")
    page.get_by_test_id("gen-workers").fill("2")
    page.get_by_test_id("gen-start").click()
    status = page.get_by_test_id("gen-status")  # status line (Locator)
    status.get_by_text("done").wait_for(timeout=180_000)
    assert _progress(page) == [20, 20]
    assert len(list(out.glob("dial_*.png"))) == 20
    with open(out / "labels.csv", newline="") as f:
        assert len(list(csv.DictReader(f))) == 20
    # out/config.yaml as a dict (render sections + run block)
    written = yaml.safe_load((out / "config.yaml").read_text(encoding="utf-8"))
    # config of the current UI state, from its saved YAML (section -> key -> value)
    ui_cfg = v7_config.parse_config_text(_save_yaml(page, tmp_path / "ui.yaml"))
    assert v7_config.strip_run(written) == ui_cfg
    assert written["run"]["completed"] is True

    out2 = tmp_path / "out2"  # second (cancelled) output folder, absolute (Path)
    page.get_by_test_id("gen-output-dir").fill(str(out2))
    page.get_by_test_id("gen-num-images").fill("400")
    page.get_by_test_id("gen-workers").fill("1")
    page.get_by_test_id("gen-start").click()
    # max == 400 makes sure the bar belongs to this job, not the finished 20-image one
    page.wait_for_function(
        "() => { const p = document.querySelector('[data-testid=gen-progress]');"
        " return p.max === 400 && p.value > 0; }",
        timeout=120_000)
    page.get_by_test_id("gen-cancel").click()
    status.get_by_text("cancelled").wait_for(timeout=120_000)
    # out2/config.yaml as a dict
    written2 = yaml.safe_load((out2 / "config.yaml").read_text(encoding="utf-8"))
    assert written2["run"]["completed"] is False
    with open(out2 / "labels.csv", newline="") as f:
        rows = len(list(csv.DictReader(f)))  # labelled images kept after cancel (int)
    assert 0 < rows < 400
    assert rows == len(list(out2.glob("dial_*.png")))


def test_generate_refuses_non_integer_inputs(page, tmp_path):
    """Typing garbage into Images / Seed shows an error and starts no job;
    '2.5' is not truncated to 2, '' / 'abc' are not read as 0."""
    out = tmp_path / "bad"  # output folder that must never be created (Path)
    status = page.get_by_test_id("gen-status")  # status line (Locator)
    page.get_by_test_id("gen-output-dir").fill(str(out))
    # (input test id, typed text) pairs (tuple[str, str]) that must be refused
    for field, text in (("gen-num-images", "abc"), ("gen-num-images", "2.5"),
                        ("gen-num-images", ""), ("gen-seed", "abc"),
                        ("gen-seed", "1.5")):
        page.get_by_test_id("gen-num-images").fill("5")
        page.get_by_test_id("gen-seed").fill("0")
        page.get_by_test_id(field).fill(text)
        page.get_by_test_id("gen-start").click()
        label = "Images" if field == "gen-num-images" else "Seed"
        status.get_by_text(f"{label}: must be a whole number").wait_for()
        # no request reached the backend: no folder, job still in its old state
        assert not out.exists(), (field, text)
        with urllib.request.urlopen(BASE + "/api/jobs/current", timeout=10) as r:
            state = json.loads(r.read())["state"]  # backend job state (str)
        assert state != "running", (field, text)
        assert status.locator(".state").inner_text() != "running"




def _wait_tile_count(page, n):
    """Wait until the grid is settled and shows exactly n (int) decoded tiles."""
    page.wait_for_function(_SETTLED_JS, timeout=UI_TIMEOUT)
    page.wait_for_function(
        f"""() => {{
          const imgs = [...document.querySelectorAll('[data-testid=tile] img')];
          return imgs.length === {n} && imgs.every(i => i.complete && i.naturalWidth === 256);
        }}""", timeout=UI_TIMEOUT)


def _set_count(page, text, expect_tiles):
    """Type text (str) into the preview-count input, press Enter, wait for
    expect_tiles (int) tiles to be shown and settled."""
    box = page.get_by_test_id("preview-count")  # preview count input (Locator)
    box.fill(text)
    box.press("Enter")
    _wait_tile_count(page, expect_tiles)


def _tile_seeds(page):
    """list[int]: data-seed of each tile in grid order."""
    return page.eval_on_selector_all("[data-testid=tile]",
                                     "els => els.map(e => Number(e.dataset.seed))")


def _grid_columns(page):
    """int: number of columns in the computed grid-template-columns of the preview grid."""
    css = page.eval_on_selector("[data-testid=preview-grid]",
                                "e => getComputedStyle(e).gridTemplateColumns")
    return len(css.split())


def test_preview_count_changes_tile_count(page):
    box = page.get_by_test_id("preview-count")  # count input (Locator)
    assert box.input_value() == "9"
    base_seeds = _tile_seeds(page)  # seeds of the 9 initial tiles, list[int]
    base_srcs = _tile_srcs(page)  # their image srcs, list[str]
    assert base_seeds == list(range(9))

    _set_count(page, "4", 4)
    assert page.get_by_test_id("tile").count() == 4
    assert _grid_columns(page) == 2
    assert _tile_seeds(page) == base_seeds[:4]
    assert _tile_srcs(page) == base_srcs[:4]

    _set_count(page, "16", 16)
    assert page.get_by_test_id("tile").count() == 16
    assert _grid_columns(page) == 4
    assert _tile_seeds(page)[:4] == base_seeds[:4]
    assert _tile_srcs(page)[:4] == base_srcs[:4]
    assert _tile_seeds(page) == list(range(16))  # sequential scheme continues

    _set_count(page, "40", 16)
    assert box.input_value() == "16"
    _set_count(page, "0", 1)
    assert box.input_value() == "1"
    assert page.get_by_test_id("tile").count() == 1
    assert _grid_columns(page) == 1


def test_preview_count_survives_reshuffle(page):
    _set_count(page, "5", 5)
    before = set(_tile_seeds(page))  # seeds before reshuffle, set[int]
    old = _applied_seq(page)
    page.get_by_test_id("reshuffle").click()
    page.wait_for_function(
        f"() => Number(document.querySelector('[data-testid=preview-grid]')"
        f".dataset.appliedSeq) > {old}", timeout=UI_TIMEOUT)
    _wait_tile_count(page, 5)
    after = _tile_seeds(page)  # seeds after reshuffle, list[int]
    assert len(after) == 5
    assert not (set(after) & before), (before, after)
    assert page.get_by_test_id("preview-count").input_value() == "5"
    # growing after a reshuffle keeps the 5 and appends random (non-sequential) seeds
    _set_count(page, "7", 7)
    assert _tile_seeds(page)[:5] == after


def _reset_btn(page, param):
    """Reset button (Locator) of control param (str, 'section.key')."""
    return page.get_by_test_id(f"reset-{param}")


def _wait_applied_after(page, old):
    """Wait until grid applied seq (int) exceeds old (int) and the preview settled."""
    page.wait_for_function(
        f"() => Number(document.querySelector('[data-testid=preview-grid]')"
        f".dataset.appliedSeq) > {old}", timeout=UI_TIMEOUT)
    _wait_settled(page)


def test_reset_slider_to_default(page):
    base_srcs = _tile_srcs(page)  # default-config tile srcs, list[str] (9)
    rng = _reset_btn(page, "ticks.len_range")  # range reset button (Locator)
    sca = _reset_btn(page, "digits.show_prob")  # scalar reset button (Locator)
    assert rng.is_disabled() and sca.is_disabled()
    _commit_and_wait(page, _num(page, "ticks.len_range", "hi"), "14")
    _commit_and_wait(page, _num(page, "digits.show_prob"), "0.9")
    assert rng.is_enabled() and sca.is_enabled()
    assert _tile_srcs(page) != base_srcs
    old = _applied_seq(page)
    rng.click()
    _wait_applied_after(page, old)
    old = _applied_seq(page)
    sca.click()
    _wait_applied_after(page, old)
    assert _num(page, "ticks.len_range", "lo").input_value() == "6"
    assert _num(page, "ticks.len_range", "hi").input_value() == "9"
    assert _num(page, "digits.show_prob").input_value() == "0.5"
    assert rng.is_disabled() and sca.is_disabled()
    assert _tile_srcs(page) == base_srcs
    assert "Reset to default" in rng.get_attribute("title")
    assert _reset_btn(page, "canvas.img_size").count() == 0 \
        or _reset_btn(page, "canvas.img_size").is_disabled()


def test_reset_toggle_to_default(page):
    box = page.locator('[data-param="tri_artifacts.enabled"] input[type=checkbox]')
    btn = _reset_btn(page, "tri_artifacts.enabled")  # toggle reset button (Locator)
    assert box.is_checked() and btn.is_disabled()
    old = _applied_seq(page)
    box.uncheck()
    _wait_applied_after(page, old)
    assert btn.is_enabled()
    old = _applied_seq(page)
    btn.click()
    _wait_applied_after(page, old)
    assert box.is_checked() and btn.is_disabled()


def test_cut_tip_toggle_moves_marker_to_visible_tip(page):
    # a user flips cut_tip off and on and watches the tip marker on each tile
    page.get_by_test_id("overlay-toggle").check()
    box = page.locator('[data-param="cut_tip.enabled"] input[type=checkbox]')  # (Locator)
    assert box.is_checked()
    on1 = _marker_geometry(page)  # cut on (default), list[dict] (9)
    old = _applied_seq(page)
    box.uncheck()
    _wait_applied_after(page, old)
    off = _marker_geometry(page)  # cut off: marker on the visible apex end, list[dict] (9)
    old = _applied_seq(page)
    box.check()
    _wait_applied_after(page, old)
    on2 = _marker_geometry(page)  # cut on again, list[dict] (9)
    shifts = []  # list[float], marker shift per tile in image px
    for a, b, c in zip(on1, off, on2):
        assert a["seed"] == b["seed"] == c["seed"]
        assert abs(a["cx"] - c["cx"]) < 0.5 and abs(a["cy"] - c["cy"]) < 0.5, (a, c)
        scale = b["iw"] / 256.0  # displayed px per image px (float)
        mid_x, mid_y = b["ix"] + b["iw"] / 2.0, b["iy"] + b["iw"] / 2.0  # tile centre (CSS px)
        # marker shift in image px (float): radius_range default [7, 10], minus
        # what the edge ring / rim already hid with the cut off (the label is
        # the visible end), so anything from 0 (tip under the rim) to ~11
        shift = math.hypot(a["cx"] - b["cx"], a["cy"] - b["cy"]) / scale
        assert shift <= 11.0, (a, b, shift)
        shifts.append(shift)
        d_on = math.hypot(a["cx"] - mid_x, a["cy"] - mid_y)  # float CSS px
        d_off = math.hypot(b["cx"] - mid_x, b["cy"] - mid_y)  # float CSS px
        assert d_on <= d_off + 0.5, (a, b)  # the cut never pushes the tip outward
    assert max(shifts) >= 3.0, shifts  # the user does see the marker move


def _centre_dists(marks):
    """list[float]: marker distance to its tile's image centre, image px, per tile
    (marks = _marker_geometry output)."""
    out = []  # list[float]
    for m in marks:
        scale = m["iw"] / 256.0  # displayed px per image px (float)
        mid_x, mid_y = m["ix"] + m["iw"] / 2.0, m["iy"] + m["iw"] / 2.0  # tile centre (CSS px)
        out.append(math.hypot(m["cx"] - mid_x, m["cy"] - mid_y) / scale)
    return out


def test_min_pixels_moves_tip_marker_inward(page):
    # a user shows the tip markers, sets tip_label.min_pixels to 0 (label at the
    # first pointer pixel), then to 8 by typing into its number input: the
    # markers retreat toward the hub or stay, never move outward
    page.get_by_test_id("overlay-toggle").check()
    box = _num(page, "tip_label.min_pixels")  # number input (Locator)
    box.scroll_into_view_if_needed()
    assert box.input_value() == "2"
    _commit_and_wait(page, box, "0")
    low = _marker_geometry(page)  # min_pixels 0, list[dict] (9)
    _commit_and_wait(page, box, "8")
    high = _marker_geometry(page)  # min_pixels 8, list[dict] (9)
    assert [m["seed"] for m in low] == [m["seed"] for m in high]
    assert all(m["cx"] is not None for m in low + high)
    d_low, d_high = _centre_dists(low), _centre_dists(high)  # list[float] image px
    for a, b in zip(d_low, d_high):
        assert b <= a + 0.25, (d_low, d_high)  # never away from the centre
    assert any(b < a - 1.0 for a, b in zip(d_low, d_high)), (d_low, d_high)


def test_reset_after_yaml_load_uses_builtin_default(page, tmp_path):
    saved = tmp_path / "default.yaml"  # downloaded default YAML (Path)
    text = _save_yaml(page, saved)
    import re
    patched = re.sub(r"show_prob:\s*0\.5", "show_prob: 0.9", text)  # YAML with 0.9 (str)
    assert patched != text
    saved.write_text(patched, encoding="utf-8")
    old = _applied_seq(page)
    page.get_by_test_id("load-yaml").set_input_files(str(saved))
    _wait_applied_after(page, old)
    assert _num(page, "digits.show_prob").input_value() == "0.9"
    btn = _reset_btn(page, "digits.show_prob")  # scalar reset button (Locator)
    assert btn.is_enabled()
    old = _applied_seq(page)
    btn.click()
    _wait_applied_after(page, old)
    assert _num(page, "digits.show_prob").input_value() == "0.5"
    assert btn.is_disabled()


def test_min_bg_contrast_slider_changes_preview(page):
    # pointer gray range straddling the default bg range [154, 179]
    _commit_and_wait(page, _num(page, "pointer.gray_range", "lo"), "100")
    _commit_and_wait(page, _num(page, "pointer.gray_range", "hi"), "230")
    before = _tile_srcs(page)  # tile srcs with contrast off, list[str] (9)
    btn = _reset_btn(page, "pointer.min_bg_contrast")  # scalar reset button (Locator)
    assert btn.is_disabled()
    assert _num(page, "pointer.min_bg_contrast").input_value() == "0"
    # slider box in CSS px, dict x/y/width/height (floats)
    slider = page.locator('[data-param="pointer.min_bg_contrast"] input[type=range]')  # (Locator)
    slider.scroll_into_view_if_needed()  # a person scrolls the panel down to it first
    box = slider.bounding_box()
    y = box["y"] + box["height"] / 2  # slider centre line (float px)
    old = _applied_seq(page)
    _drag(page, box["x"] + 7, box["x"] + box["width"] * 0.3, y)  # human drag right
    _wait_input(page, '[data-param="pointer.min_bg_contrast"] input[type=number]', "v > 0")
    _wait_applied_after(page, old)
    assert int(_num(page, "pointer.min_bg_contrast").input_value()) > 0
    after = _tile_srcs(page)  # tile srcs with contrast on, list[str] (9)
    assert any(a != b for a, b in zip(before, after)), "no tile changed"
    assert btn.is_enabled()
    old = _applied_seq(page)
    btn.click()
    _wait_applied_after(page, old)
    assert _num(page, "pointer.min_bg_contrast").input_value() == "0"
    assert btn.is_disabled()
    assert _tile_srcs(page) == before


def test_tight_gray_range_shows_live_warning(page):
    notice = page.get_by_test_id("range-warnings")  # live range notice (Locator)
    assert notice.count() == 0  # the default config is fine
    # a person types a pointer range squeezed against white
    _commit_and_wait(page, _num(page, "pointer.gray_range", "lo"), "230")
    notice.wait_for(timeout=UI_TIMEOUT)
    assert "pointer.gray_range" in notice.inner_text()
    assert "no room" in notice.inner_text()
    assert page.get_by_test_id("config-warnings").count() == 0  # separate from load notice
    # ... and back to the full range: the notice goes away
    _commit_and_wait(page, _num(page, "pointer.gray_range", "lo"), "0")
    notice.wait_for(state="detached", timeout=UI_TIMEOUT)
    # raising min_bg_contrast to 120 leaves most images no allowed gray
    _commit_and_wait(page, _num(page, "pointer.min_bg_contrast"), "120")
    notice.wait_for(timeout=UI_TIMEOUT)
    assert "pointer.min_bg_contrast" in notice.inner_text()
    assert "point mass" in notice.inner_text()
    items = notice.locator("li").all_inner_texts()  # one warning per item, list[str]
    assert [w.split(":")[0] for w in items] == ["pointer.min_bg_contrast"]


def _tile_means(page):
    """list[float]: mean gray (0-255) of each decoded tile image, grid order.

    Reads the pixels the person sees by drawing each <img> onto a canvas.
    """
    return page.eval_on_selector_all("[data-testid=tile] img", """els => els.map(e => {
        const c = document.createElement('canvas');
        c.width = e.naturalWidth; c.height = e.naturalHeight;
        const g = c.getContext('2d');
        g.drawImage(e, 0, 0);
        const d = g.getImageData(0, 0, c.width, c.height).data;
        let s = 0;
        for (let i = 0; i < d.length; i += 4) s += d[i];
        return s / (d.length / 4);
    })""")


def test_shadow_strength_darkens_preview(page):
    _commit_and_wait(page, _num(page, "shadows.prob"), "1")  # shadow on every tile
    # strength [0, 0]: lo first, then hi, so lo <= hi holds at every step
    _commit_and_wait(page, _num(page, "shadows.strength_range", "lo"), "0")
    _commit_and_wait(page, _num(page, "shadows.strength_range", "hi"), "0")
    none_src, none_mean = _tile_srcs(page), _tile_means(page)  # list[str] / list[float] (9)
    # strength [1, 1]: hi first, then lo
    _commit_and_wait(page, _num(page, "shadows.strength_range", "hi"), "1")
    _commit_and_wait(page, _num(page, "shadows.strength_range", "lo"), "1")
    full_src, full_mean = _tile_srcs(page), _tile_means(page)  # list[str] / list[float] (9)
    assert all(a != b for a, b in zip(none_src, full_src)), "a tile did not change"
    # every tile is darker with full-strength shadows, by a visible margin
    for n, f in zip(none_mean, full_mean):
        assert f < n - 2.0, (n, f)


def test_shadow_distance_slider_casts_farther(page, tmp_path):
    # a person turns shadows on for every tile, then pushes the cast distance
    # (fraction of the dial radius) up by typing into its range's max box
    _commit_and_wait(page, _num(page, "shadows.prob"), "1")  # shadow on every tile
    for key in ("shadows.distance_frac_range", "shadows.blur_min", "shadows.blur_per_px"):
        assert page.locator(f'[data-param="{key}"]').count() == 1, key  # sliders rendered
    near = _tile_srcs(page)  # tile srcs at the default [0.05, 0.3], list[str] (9)
    _commit_and_wait(page, _num(page, "shadows.distance_frac_range", "hi"), "0.55")
    _commit_and_wait(page, _num(page, "shadows.distance_frac_range", "lo"), "0.45")
    far = _tile_srcs(page)  # tile srcs at [0.45, 0.55], list[str] (9)
    assert all(a != b for a, b in zip(near, far)), "a tile did not change"
    _, _, params, _ = _save_tile(page, 0, tmp_path / "far")  # dict, params.json
    sh = params["sampled"]["shadows"]  # dict, the tile's shadow draws
    radius = params["sampled"]["geometry"]["dial_radius"]  # int px
    assert sh["active"] is True and "offset" not in sh, sh
    assert 0.45 <= sh["distance_frac"] <= 0.55, sh
    # distance_frac is recorded to 4 decimals: d is good to radius * 5e-5
    assert abs(sh["distance"] - sh["distance_frac"] * radius) < 0.02, (sh, radius)
    assert params["config"]["shadows"]["distance_frac_range"] == [0.45, 0.55]


def test_glares_and_reflections_change_preview(page):
    # the shapes are font glyphs rendered in memory: no data files needed
    # a person makes the glares pure white first (value [255, 255]), then turns them on
    # (typing a value the box already shows sends no new preview, so only
    # the ends that differ are typed; hi first, so lo <= hi at every step)
    for handle in ("hi", "lo"):
        box = _num(page, "glares.value_range", handle)  # number input (Locator)
        if box.input_value() != "255":
            _commit_and_wait(page, box, "255")
    assert _num(page, "glares.value_range", "lo").input_value() == "255"
    plain = _tile_srcs(page)  # tile srcs with both effects off, list[str] (9)
    _commit_and_wait(page, _num(page, "glares.prob"), "1")
    glared = _tile_srcs(page)  # tile srcs with glares on, list[str] (9)
    assert all(a != b for a, b in zip(plain, glared)), "a tile shows no glare"
    _commit_and_wait(page, _num(page, "reflections.prob"), "1")
    both = _tile_srcs(page)  # tile srcs with glares + reflections, list[str] (9)
    assert all(a != b for a, b in zip(glared, both)), "a tile shows no reflection"


def test_preview_count_keeps_seeds_after_reshuffle_and_typing(page):
    old = _applied_seq(page)
    page.get_by_test_id("reshuffle").click()
    _wait_applied_after(page, old)
    recorded = _tile_seeds(page)  # the 9 reshuffled seeds, list[int]
    assert len(recorded) == 9
    box = page.get_by_test_id("preview-count")  # count input (Locator)
    box.click()
    box.press("Control+a")
    box.press_sequentially("16", delay=50)  # key by key: "1" is applied before "16"
    box.press("Enter")
    _wait_tile_count(page, 16)
    assert _tile_seeds(page)[:9] == recorded
    _set_count(page, "4", 4)
    _set_count(page, "9", 9)
    assert _tile_seeds(page) == recorded


# ── per-tile Save: zip of the tile PNG + params.json + config.yaml ──────────

def _slider_keys(page, param, handle, key, times):
    """Focus the range slider of param (str) / handle ('lo'|'hi') and press key (str)
    `times` (int) times, the way a keyboard user nudges a slider."""
    s = page.locator(f'[data-param="{param}"] input[type=range][data-handle="{handle}"]')
    s.focus()
    for _ in range(times):
        s.press(key)


def _save_tile(page, index, dest_dir):
    """Click the Save button of tile `index` (int, 0-based) and unpack the download.

    Returns (zip name str, png bytes, params dict, config.yaml text str)."""
    import zipfile
    btn = page.get_by_test_id("tile").nth(index).get_by_test_id("tile-save")  # Locator
    with page.expect_download() as dl:
        btn.click()
    name = dl.value.suggested_filename  # str, e.g. "v7_seed1.zip"
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / name  # Path of the saved zip
    dl.value.save_as(str(path))
    with zipfile.ZipFile(path) as zf:
        stem = name[:-len(".zip")]  # str, "v7_seed<seed>"
        return (name, zf.read(f"{stem}.png"), json.loads(zf.read("params.json")),
                zf.read("config.yaml").decode("utf-8"))


def _tile_png(page, index):
    """PNG bytes shown by tile `index` (int): its data-URL <img> src decoded."""
    import base64
    src = _tile_srcs(page)[index]  # str, "data:image/png;base64,..."
    return base64.b64decode(src.split(",", 1)[1])


def test_tile_save_bundles_image_and_params(page, tmp_path):
    # nudge the ticks.len_range hi slider 9 -> 14 by keyboard, wait for the preview
    old = _applied_seq(page)
    _slider_keys(page, "ticks.len_range", "hi", "ArrowRight", 5)
    _wait_applied_after(page, old)
    _wait_settled(page)
    assert _num(page, "ticks.len_range", "hi").input_value() == "14"
    tile = page.get_by_test_id("tile").nth(1)  # second tile (Locator)
    seed = int(tile.get_attribute("data-seed"))  # int, its image seed
    btn = tile.get_by_test_id("tile-save")  # its Save button (Locator)
    assert btn.get_attribute("aria-label") == f"Save image and parameters for seed {seed}"
    name, png, params, text = _save_tile(page, 1, tmp_path)
    assert name == f"v7_seed{seed}.zip"
    assert png == _tile_png(page, 1)
    assert params["seed"] == seed and params["sampled"]["angle"]["angle_deg"] is not None
    assert params["config"]["ticks"]["len_range"] == [6, 14]
    assert yaml.safe_load(text)["ticks"]["len_range"] == [6, 14]

    # stale-snapshot safety: hold the next preview, move the slider again, Save
    # before the new tiles arrive -> the bundle still matches the shown tile
    held = []  # intercepted /api/preview routes not yet answered, list[Route]
    page.route("**/api/preview", lambda route: held.append(route))
    shown = _tile_png(page, 1)  # bytes, the tile still on screen
    _slider_keys(page, "ticks.len_range", "hi", "ArrowRight", 1)
    deadline = time.time() + 30  # s (float)
    while not held and time.time() < deadline:
        page.wait_for_timeout(50)
    assert held, "the slider move sent no preview request"
    assert _num(page, "ticks.len_range", "hi").input_value() == "15"
    name2, png2, params2, text2 = _save_tile(page, 1, tmp_path / "stale")
    assert png2 == shown == _tile_png(page, 1)
    assert params2["config"]["ticks"]["len_range"] == [6, 14]
    assert yaml.safe_load(text2)["ticks"]["len_range"] == [6, 14]
    for route in held:
        route.fulfill(response=route.fetch())
    _wait_settled(page)
    page.unroute("**/api/preview")
    # once the new tiles are applied, Save follows the new config
    _, png3, params3, _ = _save_tile(page, 1, tmp_path / "fresh")
    assert png3 == _tile_png(page, 1)
    assert params3["config"]["ticks"]["len_range"] == [6, 15]


def test_min_bg_contrast_avoids_shadowed_face(page, tmp_path):
    # a person turns strong shadows on for every tile, then asks for a
    # pointer/face contrast: the pointer gray is redrawn at least c away from
    # the bare face AND from the darkest shadowed face (two +-c windows; grays
    # between the two faces are allowed)
    _commit_and_wait(page, _num(page, "shadows.prob"), "1")  # shadow on every tile
    # strength 0.8 (+-0.05 jitter): the darkest face bg * (0.15-0.25) sits far
    # enough below the bare face (154-179) that the two windows mostly split
    _commit_and_wait(page, _num(page, "shadows.strength_range", "hi"), "0.8")
    _commit_and_wait(page, _num(page, "shadows.strength_range", "lo"), "0.8")
    # and a flat pointer (no gradient, no noise): the gray draw then needs no
    # headroom from 0 / 255, so the whole gray range is open to the contrast rule
    _commit_and_wait(page, _num(page, "pointer.gradient_range", "lo"), "0")
    _commit_and_wait(page, _num(page, "pointer.gradient_range", "hi"), "0")
    _commit_and_wait(page, _num(page, "pointer.noise_amplitude"), "0")
    before_src = _tile_srcs(page)  # tile srcs with the contrast off, list[str] (9)
    _commit_and_wait(page, _num(page, "pointer.min_bg_contrast"), "60")
    after_src = _tile_srcs(page)  # tile srcs with the contrast on, list[str] (9)
    assert any(a != b for a, b in zip(before_src, after_src)), "no tile changed"
    # the saved parameters of each of three tiles show the gray outside every window
    for i in range(3):
        _, _, params, _ = _save_tile(page, i, tmp_path / f"save{i}")  # dict, params.json
        pt, sh = params["sampled"]["pointer"], params["sampled"]["shadows"]  # dicts
        bg = params["sampled"]["palette"]["bg_gray"]  # int, bare face gray
        assert sh["active"] is True and "d_max" not in sh, sh
        assert params["config"]["shadows"]["strength_range"] == [0.8, 0.8]
        # darkest shadowed face; s_p / s_r are recorded to 4 decimals, so the
        # recomputed level is good to bg * 5e-5 < 0.02 levels
        shd = bg * (1 - max(sh["s_p"], sh["s_r"]))  # float, darkest shadowed face gray
        # the windows a person expects: shd +- 60 and bg +- 60, one when they overlap
        want = ([[shd - 60, bg + 60]] if shd + 60 > bg - 60
                else [[shd - 60, shd + 60], [bg - 60, bg + 60]])  # list of [lo, hi] floats
        win = pt["forbidden"]  # list of [lo, hi] floats, recorded merged windows
        if abs((shd + 60) - (bg - 60)) >= 0.02:  # merge decision clear at 4-decimal s
            assert len(win) == len(want), (win, want)
            assert all(abs(a - b) < 0.02 for w, v in zip(win, want)
                       for a, b in zip(w, v)), (win, want)
        assert pt["headroom"] == 0 and pt["gray_fallback"] is False, pt
        g = pt["gray"]  # int, pointer body gray
        assert all(g <= lo or g >= hi for lo, hi in win), (g, win)
        assert abs(g - bg) >= 60 and abs(g - shd) >= 60 - 0.02, (g, bg, shd)


def test_edge_max_frac_slider_changes_tiles(page):
    # a person sets the edge clamp to 1 (no clamp), then drags its slider all
    # the way left to 0: the thin pointer ends lose their edge band, tiles change
    box = _num(page, "pointer.edge_max_frac")  # number input (Locator)
    box.scroll_into_view_if_needed()
    assert box.input_value() == "0.25"
    _commit_and_wait(page, box, "1")
    before = _tile_srcs(page)  # tile srcs at frac 1, list[str] (9)
    slider = page.locator('[data-param="pointer.edge_max_frac"] input[type=range]')  # (Locator)
    slider.scroll_into_view_if_needed()
    sb = slider.bounding_box()  # slider box in CSS px, dict x/y/width/height (floats)
    y = sb["y"] + sb["height"] / 2  # slider centre line (float px)
    old = _applied_seq(page)
    _drag(page, sb["x"] + sb["width"] - 7, sb["x"] - 20, y)  # human drag to the far left
    _wait_input(page, '[data-param="pointer.edge_max_frac"] input[type=number]', "v === 0")
    _wait_applied_after(page, old)
    assert float(box.input_value()) == 0.0
    after = _tile_srcs(page)  # tile srcs at frac 0, list[str] (9)
    assert any(a != b for a, b in zip(before, after)), "no tile changed"


# ── no clamping: capped exposure jitter, triangles always placed ────────────

def _dial_disc_255_frac(png, sampled):
    """Fraction (float) of dial-face pixels at 255 in a saved tile PNG (bytes).

    The dial face = the stretched dial ellipse shrunk by the rim band and 2 px of
    blur margin, about the dial centre (canvas centre + canvas offset), read
    from the tile's params (sampled: dict, params.json "sampled")."""
    import io
    import numpy as np
    from PIL import Image
    im = np.asarray(Image.open(io.BytesIO(png)), dtype=np.uint8)  # (256, 256) uint8
    d, g = sampled["derived"], sampled["geometry"]  # dicts
    c = g["canvas_size"] // 2  # int px, canvas centre (pixel-centre coords)
    cx, cy = c + d["canvas_offset_x"], c + d["canvas_offset_y"]  # int px, dial centre (image)
    ys, xs = np.mgrid[:im.shape[0], :im.shape[1]]  # (256, 256) int grids
    rx, ry = d["dial_rx"] - 5, d["dial_ry"] - 5  # float px, face radii (rim + blur margin)
    face = ((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2 <= 1  # (256, 256) bool
    return float((im[face] == 255).mean())


def test_saved_tile_has_capped_jitter_and_placed_triangles(page, tmp_path):
    # a person pushes the exposure up (brightness [1.5, 2.0]: hi first, so lo <= hi
    # at every step) and asks for a bright pointer (gray [200, 255])
    _commit_and_wait(page, _num(page, "jitter.brightness_range", "hi"), "2")
    _commit_and_wait(page, _num(page, "jitter.brightness_range", "lo"), "1.5")
    _commit_and_wait(page, _num(page, "pointer.gray_range", "lo"), "200")
    assert _num(page, "pointer.gray_range", "hi").input_value() == "255"
    for i in range(3):
        _, png, params, _ = _save_tile(page, i, tmp_path / f"bright{i}")  # bytes, dict
        jit = params["sampled"]["jitter"]  # dict, factors + caps
        assert params["config"]["jitter"]["brightness_range"] == [1.5, 2.0]
        assert jit["brightness_max"] is not None, jit
        assert jit["brightness"] <= jit["brightness_max"] + 1e-4, jit
        assert jit["capped"]["brightness"] is True, jit
        # the dial face shows (almost) no blown-out white: <= 0.5% of its pixels
        frac = _dial_disc_255_frac(png, params["sampled"])  # float
        assert frac <= 0.005, (i, frac)
    # then six triangles on every image: every one lands on the dial
    _commit_and_wait(page, _num(page, "tri_artifacts.count_range", "lo"), "6")
    assert _num(page, "tri_artifacts.count_range", "hi").input_value() == "6"
    _, _, params, _ = _save_tile(page, 0, tmp_path / "tris")  # dict, params.json
    tri = params["sampled"]["tri_artifacts"]  # dict
    assert tri["n"] == 6 and len(tri["items"]) == 6, tri
    assert all(it["placed"] is True and it["pixels"] >= 1 for it in tri["items"]), tri


# ── pointer raster: no "ball" at a thin, cut tip ────────────────────────────

# path of the user's seed-8 sample config (Path; from the studio Save zip
# inspect_tip_location/v7_seed8.zip: slender pointer, edge thickness [0, 0], cut on)
_SEED8_YAML = REPO_ROOT / "tests" / "data" / "v7_seed8_config.yaml"


def _clean_seed8_yaml(dest):
    """Write the user's seed-8 YAML with a clean look (flat face 200, black flat
    pointer, rim / outer / circle 220, light ticks, no noise / shadows /
    effects / jitter / digits / triangles) to dest (Path); the geometry, stretch,
    edge thickness [0, 0] and cut-tip settings stay the user's.  Returns dest."""
    cfg = yaml.safe_load(_SEED8_YAML.read_text(encoding="utf-8"))  # dict, user config
    pins = {"palette": {"bg_gray_range": [200, 200], "circle_gray_range": [220, 220],
                        "outer_gray_range": [220, 220], "rim_gray_range": [220, 220]},
            "pointer": {"gray_range": [0, 0], "gradient_range": [0, 0], "noise_amplitude": 0,
                        "edge_noise_amplitude": 0, "min_bg_contrast": 0},
            "background": {"noise_intensity": 0, "variation": 0, "dial_circle_blur": False},
            "hier_noise": {"block_variance": 0, "pixel_variance": 0},
            "jitter": {"brightness_range": [1, 1], "contrast_range": [1, 1],
                       "sharpness_range": [1, 1]},
            "shadows": {"prob": 0}, "reflections": {"prob": 0}, "glares": {"prob": 0},
            "digits": {"show_prob": 0}, "ticks": {"gray_range": [215, 215]},
            "tri_artifacts": {"enabled": False}}  # section -> key -> value overrides (dict)
    for sec, kv in pins.items():
        cfg[sec].update(kv)
    assert cfg["pointer"]["edge_thickness_range"] == [0, 0] and cfg["cut_tip"]["enabled"]
    dest.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return dest


def _tip_coverage(png, params, n_lines=12, face=200):
    """Effective pointer width (list[float], px) on the scan lines from the saved
    label toward the base, from a saved tile PNG (bytes) and its params (dict).

    Scan lines are rows when the label axis (axis_apex -> axis_base_mid) is
    closer to vertical, else columns; on each line the pointer coverage of a
    pixel is clip((face - gray) / face, 0, 1) (black pointer on a face-gray
    dial), summed over a window of 4 px + the triangle's width about the axis."""
    import io
    import numpy as np
    from PIL import Image
    im = np.asarray(Image.open(io.BytesIO(png)), dtype=np.float64)  # (256, 256) float64
    cov = np.clip((face - im) / face, 0.0, 1.0)  # (256, 256) float64, pointer coverage
    d = params["sampled"]["derived"]  # dict
    ax, ay = d["axis_apex_x"], d["axis_apex_y"]  # float px, A'
    mx, my = d["axis_base_mid_x"], d["axis_base_mid_y"]  # float px, M'
    g = params["sampled"]["geometry"]  # dict
    rows = abs(my - ay) >= abs(mx - ax)  # bool, scan rows
    s0, s1 = (ay, my) if rows else (ax, mx)  # float, apex / base scan coords
    u0, u1 = (ax, mx) if rows else (ay, my)  # float, along-line coords
    step = 1 if s1 > s0 else -1  # int, apex -> base
    lab = params["tip_y"] if rows else params["tip_x"]  # float, label scan coord
    s = int(math.floor(lab + 0.5))  # int, the label's line
    widths = []  # list[float]
    for _ in range(n_lines + 1):
        f = (s - s0) / (s1 - s0)  # float, fraction of the height
        uc = u0 + f * (u1 - u0)  # float, axis on this line
        reach = 4.0 + 2.0 * g["chord_half"] * f  # float px, half window
        lo, hi = int(math.ceil(uc - reach)), int(math.floor(uc + reach))  # int window
        widths.append(float(cov[s, lo:hi + 1].sum() if rows else cov[lo:hi + 1, s].sum()))
        s += step
    return widths


def test_saved_tile_tip_has_no_ball(page, tmp_path):
    # a person loads the seed-8 sample YAML (slender pointer, no edge band, cut
    # tip) with a clean look, saves three tiles and measures the pointer width
    # row by row from the saved label toward the hub: it never shrinks by more
    # than half a pixel (the old Pillow fill made a 3-px "ball" on a 2-px neck)
    old = _applied_seq(page)
    page.get_by_test_id("load-yaml").set_input_files(str(_clean_seed8_yaml(tmp_path / "s8.yaml")))
    _wait_applied_after(page, old)
    _wait_settled(page)
    assert _num(page, "pointer.edge_thickness_range", "hi").input_value() == "0"
    assert _num(page, "cut_tip.radius_range", "lo").input_value() == "7"
    for i in range(3):
        _, png, params, _ = _save_tile(page, i, tmp_path / f"tip{i}")  # bytes, dict
        assert params["config"]["pointer"]["edge_thickness_range"] == [0, 0]
        assert params["config"]["cut_tip"]["enabled"] is True
        assert params["sampled"]["derived"]["tip_fallback"] is False
        w = _tip_coverage(png, params)  # list[float] px
        print("tile %d widths from the label: %s" % (i, [round(v, 2) for v in w]))
        assert all(b >= a - 0.5 for a, b in zip(w, w[1:])), (i, w)
        assert w[-1] > w[0], (i, w)
