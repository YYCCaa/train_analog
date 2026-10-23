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
def browser():
    """One headless Chromium for the module (playwright Browser)."""
    with playwright_sync.sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        try:
            yield b
        finally:
            b.close()


@pytest.fixture
def page(server, browser):
    """Fresh 1400x900 page on the studio, preview tiles loaded (playwright Page)."""
    ctx = browser.new_context(viewport={"width": 1400, "height": 900}, accept_downloads=True)
    pg = ctx.new_page()
    pg.set_default_timeout(UI_TIMEOUT)
    pg.goto(server)
    _wait_settled(pg)
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


def test_page_shows_preview(page):
    widths = page.eval_on_selector_all("[data-testid=tile] img",
                                       "els => els.map(e => e.naturalWidth)")
    assert widths == [256] * 9


def test_every_param_has_control(page, server):
    with urllib.request.urlopen(server + "/api/schema", timeout=10) as r:
        schema = json.loads(r.read())
    for sec, keys in schema.items():
        assert page.locator(f'section[data-section="{sec}"] h2').count() == 1, sec
        for key, leaf in keys.items():
            ctl = page.locator(f'[data-param="{sec}.{key}"]')
            assert ctl.count() >= 1, f"{sec}.{key} has no control"
            assert leaf["help"] in ctl.first.inner_text(), f"{sec}.{key} help missing"
    assert _num(page, "canvas.img_size").is_disabled()
    assert page.locator('[data-param="canvas.img_size"] input[type=range]').is_disabled()


def test_tick_slider_changes_only_ticks(page):
    page.get_by_test_id("overlay-toggle").check()
    before_src = _tile_srcs(page)
    before_mk = _marker_geometry(page)
    assert all(m["cx"] is not None for m in before_mk)
    _commit_and_wait(page, _num(page, "ticks.len_range", "hi"), "14")
    _commit_and_wait(page, _num(page, "ticks.len_range", "lo"), "12")
    after_src = _tile_srcs(page)
    after_mk = _marker_geometry(page)
    assert any(a != b for a, b in zip(before_src, after_src)), "no tile changed"
    for b, a in zip(before_mk, after_mk):
        assert a["seed"] == b["seed"]
        assert abs(a["cx"] - b["cx"]) < 0.5 and abs(a["cy"] - b["cy"]) < 0.5, (b, a)


def test_overlay_marker_on_apex(page):
    with page.expect_response(lambda r: r.url.endswith("/api/preview")
                              and r.request.method == "POST") as resp:
        page.get_by_test_id("reshuffle").click()
    tips = {t["seed"]: t for t in resp.value.json()["tiles"]}  # seed -> tile dict
    _wait_settled(page)
    page.get_by_test_id("overlay-toggle").check()
    markers = page.get_by_test_id("tip-marker")
    assert markers.count() == 9
    for i in range(9):
        assert markers.nth(i).is_visible()
    geo = _marker_geometry(page)
    assert sorted(g["seed"] for g in geo) == sorted(tips)
    for g in geo:
        scale = g["iw"] / 256.0  # displayed px per image px (float)
        t = tips[g["seed"]]
        assert abs(g["cx"] - (g["ix"] + t["tip_x"] * scale)) <= 3, (g, t)
        assert abs(g["cy"] - (g["iy"] + t["tip_y"] * scale)) <= 3, (g, t)


def test_rapid_drag_shows_latest(page, server):
    hi = _num(page, "ticks.len_range", "hi")
    values = [str(v) for v in range(10, 20)]  # 10 successive hi values (str)
    t0 = time.time()
    for v in values:
        hi.fill(v)
        hi.press("Enter")
    assert time.time() - t0 < 1.5, "typing took too long to exercise the race"
    _wait_settled(page)
    grid = page.get_by_test_id("preview-grid")
    assert grid.get_attribute("data-applied-seq") == grid.get_attribute("data-sent-seq")
    assert hi.input_value() == values[-1]
    # the tiles shown are exactly the render of the latest config
    cfg = v7_config.default_config()
    cfg["ticks"]["len_range"] = [6, int(values[-1])]
    seeds = page.eval_on_selector_all("[data-testid=tile]",
                                      "fs => fs.map(f => Number(f.dataset.seed))")
    want = _post_json(server, "/api/preview", {"config": cfg, "seeds": seeds})["tiles"]
    assert _tile_srcs(page) == ["data:image/png;base64," + t["png_b64"] for t in want]


def test_invalid_yaml_shows_error(page, tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("ticks:\n  len_range: [9, 6]\n", encoding="utf-8")
    page.get_by_test_id("load-yaml").set_input_files(str(bad))
    errs = page.get_by_test_id("config-errors")
    errs.get_by_text("ticks.len_range").first.wait_for(timeout=UI_TIMEOUT)
    assert "ticks.len_range" in errs.inner_text()
    page.wait_for_function(_TILES_LOADED_JS, timeout=UI_TIMEOUT)
    assert _num(page, "ticks.len_range", "hi").input_value() == "9"


def test_save_load_roundtrip(page, tmp_path):
    prob = _num(page, "digits.show_prob")
    assert prob.input_value() == "0.5"
    _commit_and_wait(page, prob, "1")
    saved = tmp_path / "saved.yaml"
    _save_yaml(page, saved)
    page.reload()
    _wait_settled(page)
    assert _num(page, "digits.show_prob").input_value() == "0.5"
    old = _applied_seq(page)
    page.get_by_test_id("load-yaml").set_input_files(str(saved))
    page.wait_for_function(
        f"() => Number(document.querySelector('[data-testid=preview-grid]')"
        f".dataset.appliedSeq) > {old}", timeout=UI_TIMEOUT)
    assert _num(page, "digits.show_prob").input_value() == "1"


def _progress(page):
    """(value, max) of the gen-progress bar, floats."""
    return page.get_by_test_id("gen-progress").evaluate("e => [e.value, e.max]")


def test_generate_and_cancel(page, tmp_path):
    out = tmp_path / "out"
    page.get_by_test_id("gen-output-dir").fill(str(out))
    page.get_by_test_id("gen-num-images").fill("20")
    page.get_by_test_id("gen-seed").fill("3")
    page.get_by_test_id("gen-workers").fill("2")
    page.get_by_test_id("gen-start").click()
    status = page.get_by_test_id("gen-status")
    status.get_by_text("done").wait_for(timeout=180_000)
    assert _progress(page) == [20, 20]
    assert len(list(out.glob("dial_*.png"))) == 20
    with open(out / "labels.csv", newline="") as f:
        assert len(list(csv.DictReader(f))) == 20
    written = yaml.safe_load((out / "config.yaml").read_text(encoding="utf-8"))
    ui_cfg = v7_config.parse_config_text(_save_yaml(page, tmp_path / "ui.yaml"))
    assert v7_config.strip_run(written) == ui_cfg
    assert written["run"]["completed"] is True

    out2 = tmp_path / "out2"
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
    written2 = yaml.safe_load((out2 / "config.yaml").read_text(encoding="utf-8"))
    assert written2["run"]["completed"] is False
    with open(out2 / "labels.csv", newline="") as f:
        rows = len(list(csv.DictReader(f)))
    assert 0 < rows < 400
    assert rows == len(list(out2.glob("dial_*.png")))
