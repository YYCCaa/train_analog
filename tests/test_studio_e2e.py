"""Studio browser E2E: drive the real page (Playwright Chromium, headless) the
way a person would -- clicks, typing, file pickers -- against a live
`studio/run_studio.py` server on port 8799."""
import csv
import json
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
        assert abs(g["cx"] - (g["ix"] + t["tip_x"] * scale)) <= 3, (g, t)
        assert abs(g["cy"] - (g["iy"] + t["tip_y"] * scale)) <= 3, (g, t)


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
    bad = tmp_path / "bad.yaml"  # YAML with lo > hi (Path)
    bad.write_text("ticks:\n  len_range: [9, 6]\n", encoding="utf-8")
    page.get_by_test_id("load-yaml").set_input_files(str(bad))
    errs = page.get_by_test_id("config-errors")  # error box (Locator)
    errs.get_by_text("ticks.len_range").first.wait_for(timeout=UI_TIMEOUT)
    assert "ticks.len_range" in errs.inner_text()
    page.wait_for_function(_TILES_LOADED_JS, timeout=UI_TIMEOUT)
    assert _num(page, "ticks.len_range", "hi").input_value() == "9"


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
