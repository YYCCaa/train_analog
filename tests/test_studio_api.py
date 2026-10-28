"""Studio backend API tests (TestClient; lifespan runs via the context manager)."""
import base64
import copy
import csv
import io
import json
import math
import os
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import v7_config
import generate_synthetic_v7_mp as gen
from studio.backend.app import create_app

# TestClient base URL (str): a loopback Host the app's TrustedHostMiddleware accepts
BASE_URL = "http://127.0.0.1"


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(), base_url=BASE_URL) as c:
        yield c


def _wait_state(client, states, timeout=120):
    """Poll /api/jobs/current until state in states (set[str]); fail at timeout (s)."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = client.get("/api/jobs/current").json()
        if st["state"] in states:
            return st
        time.sleep(0.2)
    raise AssertionError(f"job did not reach {states} in {timeout}s: {st}")


def test_schema_and_default(client):
    assert "len_range" in client.get("/api/schema").json()["ticks"]
    assert "len_range" in client.get("/api/config/default").json()["config"]["ticks"]


def test_preview_tiles(client):
    cfg = v7_config.default_config()
    r = client.post("/api/preview", json={"config": cfg, "seeds": [1, 2, 3]})
    assert r.status_code == 200
    tiles = r.json()["tiles"]
    assert [t["seed"] for t in tiles] == [1, 2, 3]
    for t in tiles:
        img = Image.open(io.BytesIO(base64.b64decode(t["png_b64"])))
        assert img.mode == "L" and img.size == (256, 256)
        ref = gen.render_image(cfg, t["seed"])
        assert np.array_equal(np.array(img), ref.image)
        assert t["angle_deg"] == pytest.approx(ref.angle_deg)


def test_config_parse_returns_warnings_for_old_yaml(client):
    text = "reflections:\n  perspective: 0.4\n  warp_amplitude: 3\nfoo:\n  bar: 1\n"
    r = client.post("/api/config/parse", json={"text": text})
    assert r.status_code == 200
    body = r.json()
    assert body["config"]["reflections"]["perspective_range"] == [0, 0.4]
    assert any(w.startswith("reflections.perspective:") for w in body["warnings"])
    assert any(w.startswith("foo") for w in body["warnings"])


def test_config_parse_clean_yaml_has_empty_warnings(client):
    text = client.post("/api/config/dump", json={"config": v7_config.default_config()}).json()["text"]
    assert client.post("/api/config/parse", json={"text": text}).json()["warnings"] == []


def test_config_check_range_warnings(client):
    cfg = v7_config.default_config()  # default cfg (dict)
    r = client.post("/api/config/check", json={"config": cfg})
    assert r.status_code == 200 and r.json() == {"warnings": []}
    tight = copy.deepcopy(cfg)  # cfg with a too-tight pointer gray range
    tight["pointer"]["gray_range"] = [230, 255]
    tight["pointer"]["min_bg_contrast"] = 120
    warns = client.post("/api/config/check", json={"config": tight}).json()["warnings"]
    assert warns == gen.pointer_gray_warnings(tight)
    assert [w.split(":")[0] for w in warns] == ["pointer.gray_range", "pointer.min_bg_contrast"]
    bad = copy.deepcopy(cfg)
    bad["pointer"]["nope"] = 1
    assert client.post("/api/config/check", json={"config": bad}).status_code == 422


def test_config_parse_appends_range_warnings(client):
    # old key (migration warning) + a too-tight range: migration first, then range
    text = "reflections:\n  perspective: 0.4\npointer:\n  gray_range: [100, 120]\n"
    warns = client.post("/api/config/parse", json={"text": text}).json()["warnings"]
    assert warns[0].startswith("reflections.perspective:")
    assert warns[-1].startswith("pointer.gray_range:") and "21 levels" in warns[-1]


def test_preview_still_rejects_unknown_key(client):
    cfg = copy.deepcopy(v7_config.default_config())
    cfg["reflections"]["perspective"] = 0.4
    r = client.post("/api/preview", json={"config": cfg, "seeds": [1]})
    assert r.status_code == 422
    assert any("reflections.perspective" in e for e in r.json()["errors"])


def test_preview_invalid_config_422(client):
    cfg = copy.deepcopy(v7_config.default_config())
    cfg["ticks"]["len_range"] = [9, 6]
    r = client.post("/api/preview", json={"config": cfg, "seeds": [1]})
    assert r.status_code == 422
    assert any("ticks.len_range" in e for e in r.json()["errors"])


def test_parse_and_dump_roundtrip(client):
    cfg = v7_config.default_config()
    text = client.post("/api/config/dump", json={"config": cfg}).json()["text"]
    assert client.post("/api/config/parse", json={"text": text}).json()["config"] == cfg
    bad = client.post("/api/config/parse", json={"text": "ticks: [\n"})
    assert bad.status_code == 422 and bad.json()["errors"]


def test_generate_rejects_relative_path(client):
    r = client.post("/api/generate", json={
        "config": v7_config.default_config(), "output_dir": "rel/out",
        "num_images": 2, "seed": 0, "workers": 1, "overwrite": False})
    assert r.status_code == 400


def test_generate_job_runs(client, tmp_path):
    out = str(tmp_path / "out")
    body = {"config": v7_config.default_config(), "output_dir": out,
            "num_images": 8, "seed": 0, "workers": 2, "overwrite": False}
    r = client.post("/api/generate", json=body)
    assert r.status_code == 202 and r.json() == {"state": "running"}
    st = _wait_state(client, {"done", "error", "cancelled"})
    assert st["state"] == "done" and st["done"] == st["total"] == 8
    assert len([f for f in os.listdir(out) if f.startswith("dial_")]) == 8
    # old data without overwrite -> 409 mentioning overwrite
    r = client.post("/api/generate", json=body)
    assert r.status_code == 409 and "overwrite" in r.json()["errors"][0]


def test_generate_conflict_and_cancel(client, tmp_path):
    out = str(tmp_path / "big")
    body = {"config": v7_config.default_config(), "output_dir": out,
            "num_images": 300, "seed": 0, "workers": 2, "overwrite": False}
    assert client.post("/api/generate", json=body).status_code == 202
    assert client.post("/api/generate", json=body).status_code == 409
    client.post("/api/jobs/current/cancel", json={})
    st = _wait_state(client, {"done", "error", "cancelled"})
    assert st["state"] == "cancelled"
    pngs = {f for f in os.listdir(out) if f.startswith("dial_")}
    with open(os.path.join(out, "labels.csv"), newline="") as f:
        labels = {row["filename"] for row in csv.DictReader(f)}
    assert pngs == labels
    import yaml
    with open(os.path.join(out, "config.yaml"), encoding="utf-8") as f:
        assert yaml.safe_load(f)["run"]["completed"] is False


def _gen_body(out, n, **kw):
    """Generate request body (dict) with default cfg; kw overrides fields."""
    body = {"config": v7_config.default_config(), "output_dir": out,
            "num_images": n, "seed": 0, "workers": 2, "overwrite": False}
    body.update(kw)
    return body


def test_preview_recovers_from_broken_pool(client):
    from concurrent.futures.process import BrokenProcessPool

    class Dead:
        """Pool stand-in whose submit raises like a dead ProcessPoolExecutor."""

        def submit(self, *a, **k):
            raise BrokenProcessPool("dead")

        def shutdown(self, *a, **k):
            pass

    client.app.state.pool = Dead()
    r = client.post("/api/preview", json={"config": v7_config.default_config(), "seeds": [4]})
    assert r.status_code == 200 and len(r.json()["tiles"]) == 1


def test_malformed_bodies_use_errors_contract(client):
    r = client.post("/api/preview", json={"config": 5, "seeds": ["a"]})
    assert r.status_code == 422 and r.json()["errors"]
    r = client.post("/api/preview", content="{not json",
                    headers={"content-type": "application/json"})
    assert r.status_code == 422 and r.json()["errors"]
    r = client.post("/api/generate", json=_gen_body("/x", "many"))
    assert r.status_code == 422 and r.json()["errors"]


def test_overwrite_replaces_previous_run(client, tmp_path):
    out = str(tmp_path / "o")
    assert client.post("/api/generate", json=_gen_body(out, 6)).status_code == 202
    _wait_state(client, {"done", "error", "cancelled"})
    assert client.post("/api/generate", json=_gen_body(out, 3, overwrite=True)).status_code == 202
    st = _wait_state(client, {"done", "error", "cancelled"})
    assert st["state"] == "done"
    assert len([f for f in os.listdir(out) if f.startswith("dial_")]) == 3


def test_generate_bad_config_and_zero_images(client, tmp_path):
    cfg = v7_config.default_config()
    cfg["ticks"]["len_range"] = [9, 6]
    r = client.post("/api/generate", json=_gen_body(str(tmp_path / "b"), 2, config=cfg))
    assert r.status_code == 422 and any("ticks.len_range" in e for e in r.json()["errors"])
    r = client.post("/api/generate", json=_gen_body(str(tmp_path / "b"), 0))
    assert r.status_code == 422


def test_preview_works_during_generation(client, tmp_path):
    out = str(tmp_path / "p")
    assert client.post("/api/generate", json=_gen_body(out, 300)).status_code == 202
    r = client.post("/api/preview", json={"config": v7_config.default_config(), "seeds": [1]})
    assert r.status_code == 200
    client.post("/api/jobs/current/cancel", json={})
    _wait_state(client, {"done", "error", "cancelled"})


def test_cancel_when_idle():
    with TestClient(create_app(), base_url=BASE_URL) as c:
        r = c.post("/api/jobs/current/cancel", json={})
        assert r.status_code == 200 and r.json()["state"] == "idle"


# ── request guards (cross-site / DNS-rebinding) and body bounds ──────────

def test_post_without_json_content_type_415(client, monkeypatch):
    _no_job(monkeypatch, client)
    # a no-cors "simple" POST from another site carries text/plain or no type;
    # FastAPI would still parse this full, valid body without the guard
    body = json.dumps(_gen_body("Z:/never/used", 2)).encode()  # bytes
    r = client.post("/api/generate", content=body)
    assert r.status_code == 415 and r.json()["errors"]
    r = client.post("/api/jobs/current/cancel",
                    content=b"{}", headers={"content-type": "text/plain"})
    assert r.status_code == 415 and r.json()["errors"]


def _no_job(monkeypatch, client):
    """Make JobRunner.start fail the test, so a guard regression never writes files."""
    def start(*a, **k):
        """Stand-in for JobRunner.start that must never be reached."""
        raise AssertionError("request reached JobRunner.start")
    monkeypatch.setattr(client.app.state.jobs, "start", start)


def test_foreign_origin_403(client, tmp_path, monkeypatch):
    _no_job(monkeypatch, client)
    r = client.post("/api/generate", json=_gen_body(str(tmp_path / "o"), 2),
                    headers={"Origin": "http://evil.example"})
    assert r.status_code == 403 and r.json()["errors"]
    r = client.get("/api/schema", headers={"Origin": "http://evil.example"})
    assert r.status_code == 403
    # loopback origins (the page itself, the Vite dev server) are fine
    for origin in ("http://127.0.0.1:8765", "http://localhost:5173"):
        assert client.get("/api/schema", headers={"Origin": origin}).status_code == 200


def test_foreign_host_400(client):
    # DNS rebinding: evil.example resolving to 127.0.0.1 still sends Host: evil.example
    r = client.get("/api/schema", headers={"Host": "evil.example"})
    assert r.status_code == 400
    assert client.get("/api/schema", headers={"Host": "localhost:8765"}).status_code == 200


def test_generate_workers_and_num_images_bounds(client, tmp_path, monkeypatch):
    _no_job(monkeypatch, client)
    out = str(tmp_path / "bounds")
    for w in (0, (os.cpu_count() or 1) + 1, 10 ** 9):
        r = client.post("/api/generate", json=_gen_body(out, 2, workers=w))
        assert r.status_code == 422, w
        assert r.json()["errors"][0].startswith("workers:")
    r = client.post("/api/generate", json=_gen_body(out, 1_000_001))
    assert r.status_code == 422 and r.json()["errors"][0].startswith("num_images:")
    assert not os.path.exists(out)


def test_not_empty_message_points_at_checkbox(client, tmp_path):
    out = tmp_path / "ne"
    out.mkdir()
    (out / "labels.csv").write_text("old")
    r = client.post("/api/generate", json=_gen_body(str(out), 2))
    assert r.status_code == 409
    msg = r.json()["errors"][0]
    assert "overwrite box" in msg and "--overwrite" not in msg


def test_auto_workers_leave_one_core_free(tmp_path):
    with TestClient(create_app(), base_url=BASE_URL) as c:
        seen = {}  # args JobRunner.start received (dict[str, object])

        def fake_start(cfg, output_dir, num_images, seed, workers, overwrite, save_params=False):
            """Stand-in for JobRunner.start: record workers, start nothing."""
            seen["workers"] = workers

        c.app.state.jobs.start = fake_start
        r = c.post("/api/generate", json=_gen_body(str(tmp_path / "w"), 2, workers=None))
        assert r.status_code == 202
        assert seen["workers"] == max(1, (os.cpu_count() or 1) - 1)


def _fx_cfg():
    """Default cfg with reflections + glares always on (dict), so items get recorded."""
    cfg = copy.deepcopy(v7_config.default_config())
    cfg["reflections"]["prob"] = 1.0
    cfg["glares"]["prob"] = 1.0
    return cfg


def test_preview_tiles_carry_params(client):
    r = client.post("/api/preview", json={"config": _fx_cfg(), "seeds": [4]})
    t = r.json()["tiles"][0]  # dict, one tile
    assert t["params"]["angle"]["angle_deg"] == t["angle_deg"]
    assert len(t["params"]["reflections"]["items"]) == t["params"]["reflections"]["n"]


def test_bundle_zip_matches_preview(client):
    import zipfile
    cfg, seed = _fx_cfg(), 5
    r = client.post("/api/bundle", json={"config": cfg, "seed": seed})
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/zip"
    assert r.headers["content-disposition"] == f'attachment; filename="v7_seed{seed}.zip"'
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    assert sorted(zf.namelist()) == sorted([f"v7_seed{seed}.png", "params.json", "config.yaml"])
    tile = client.post("/api/preview", json={"config": cfg, "seeds": [seed]}).json()["tiles"][0]
    assert zf.read(f"v7_seed{seed}.png") == base64.b64decode(tile["png_b64"])
    p = json.loads(zf.read("params.json"))  # dict, the bundle's params.json
    assert p["generator"] == "generate_synthetic_v7_mp" and p["seed"] == seed
    assert p["tip_x"] == tile["tip_x"] and p["tip_y"] == tile["tip_y"]
    assert p["angle_deg"] == tile["angle_deg"] and p["img_size"] == 256
    assert p["sampled"] == tile["params"] and p["config"] == cfg
    assert isinstance(p["created"], str) and "T" in p["created"]
    text = zf.read("config.yaml").decode("utf-8")  # str, YAML dump of cfg
    assert client.post("/api/config/parse", json={"text": text}).json()["config"] == cfg


def test_bundle_invalid_config_422(client):
    cfg = copy.deepcopy(v7_config.default_config())
    cfg["ticks"]["len_range"] = [9, 6]
    r = client.post("/api/bundle", json={"config": cfg, "seed": 1})
    assert r.status_code == 422
    assert any("ticks.len_range" in e for e in r.json()["errors"])
    assert client.post("/api/bundle", json={"config": cfg}).status_code == 422


@pytest.mark.skipif(os.name != "nt", reason="drive-relative paths are a Windows thing")
def test_driveless_windows_path_400(client, monkeypatch):
    _no_job(monkeypatch, client)
    r = client.post("/api/generate", json=_gen_body("\\foo\\bar", 2))
    assert r.status_code == 400 and r.json()["errors"][0].startswith("output_dir:")


def _read_params_lines(out):
    """params.jsonl in folder out (str) -> list[dict], one parsed record per line."""
    with open(os.path.join(out, "params.jsonl"), encoding="utf-8") as f:
        return [json.loads(ln) for ln in f.read().splitlines() if ln]


def _read_run(out):
    """config.yaml 'run' block of folder out (str) -> dict."""
    import yaml
    with open(os.path.join(out, "config.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)["run"]


def test_generate_save_params(client, tmp_path):
    out = str(tmp_path / "sp")
    r = client.post("/api/generate", json=_gen_body(out, 8, save_params=True))
    assert r.status_code == 202
    assert _wait_state(client, {"done", "error", "cancelled"})["state"] == "done"
    recs = _read_params_lines(out)
    with open(os.path.join(out, "labels.csv"), newline="") as f:
        rows = [row["filename"] for row in csv.DictReader(f)]
    assert len(recs) == 8 and [r_["filename"] for r_ in recs] == rows
    assert _read_run(out)["save_params"] is True


def test_generate_default_writes_no_params(client, tmp_path):
    out = str(tmp_path / "nop")
    assert client.post("/api/generate", json=_gen_body(out, 4)).status_code == 202
    assert _wait_state(client, {"done", "error", "cancelled"})["state"] == "done"
    assert not os.path.exists(os.path.join(out, "params.jsonl"))
    assert _read_run(out)["save_params"] is False


@pytest.mark.parametrize("bad", ["yes", 1])
def test_generate_save_params_must_be_bool(client, tmp_path, bad):
    out = tmp_path / "bad"
    r = client.post("/api/generate", json=_gen_body(str(out), 2, save_params=bad))
    assert r.status_code == 422 and r.json()["errors"]
    assert not out.exists()


def test_not_empty_counts_params_file(client, tmp_path):
    out = tmp_path / "pe"
    out.mkdir()
    (out / "params.jsonl").write_text("{}\n")
    r = client.post("/api/generate", json=_gen_body(str(out), 2))
    assert r.status_code == 409
    msg = r.json()["errors"][0]
    assert "overwrite box" in msg and "params.jsonl" in msg


# ── tri_artifacts.clear_distance: triangles kept d px off pointer, hub and hole ──

# slack (float px) allowed when checking a triangle's polygon against the hole
# circle.  The generator measures d between PIXEL CENTRES: every pixel of the
# triangle raster vs every pixel of the drawn hole (Pillow's ellipse, whose
# inclusive bbox makes it up to ~1.5 px larger, plus the bicubic stretch spread,
# so the zone it measures from is never smaller than the analytic circle used
# here).  The test measures between the EXACT shapes instead: the polygon from
# verts and the analytic stretched hole ellipse.  Each exact shape can reach up
# to half a pixel diagonal (~0.71 px) past the pixel centres the generator saw,
# so the exact gap may read ~1.4 px under d; 2.5 px covers that and nothing more.
TRI_HOLE_SLACK = 2.5


def _hole_circle(sampled):
    """Mounting hole of one image as the test sees it, from its sampled params.

    sampled: dict, params "sampled" (geometry + derived sections).  The hub
    centre is the canvas centre (canvas_size // 2) + the pointer displacement;
    the hole centre is the hub centre + the hole displacement, radius
    hole_radius; both are then stretched about the canvas centre (stretch_h,
    stretch_v) and moved into the image by the canvas offset, like the
    generator does.  Returns (hx, hy, rx, ry): floats, hole centre and the
    stretched x / y radii in image px.
    """
    g, d = sampled["geometry"], sampled["derived"]  # dicts
    c = g["canvas_size"] // 2  # int px, canvas centre (x = y)
    hub_x = c + g["ptr_displacement_mag"] * math.cos(g["ptr_displacement_angle_rad"])  # float px
    hub_y = c + g["ptr_displacement_mag"] * math.sin(g["ptr_displacement_angle_rad"])
    hole_x = hub_x + g["hole_displacement_mag"] * math.cos(g["hole_displacement_angle_rad"])
    hole_y = hub_y + g["hole_displacement_mag"] * math.sin(g["hole_displacement_angle_rad"])
    hx = c + (hole_x - c) * g["stretch_h"] + d["canvas_offset_x"]  # float image px
    hy = c + (hole_y - c) * g["stretch_v"] + d["canvas_offset_y"]
    return hx, hy, g["hole_radius"] * g["stretch_h"], g["hole_radius"] * g["stretch_v"]


def _point_triangle_dist(px, py, verts):
    """float px: distance from point (px, py) to the filled triangle verts
    (3 [x, y] floats); 0 when the point is inside it."""
    (x1, y1), (x2, y2), (x3, y3) = verts
    # signed areas of the point against each edge (floats); mixed signs = outside
    s = [(px - x2) * (y1 - y2) - (x1 - x2) * (py - y2),
         (px - x3) * (y2 - y3) - (x2 - x3) * (py - y3),
         (px - x1) * (y3 - y1) - (x3 - x1) * (py - y1)]
    if not (any(v < 0 for v in s) and any(v > 0 for v in s)):
        return 0.0
    best = math.inf  # float px, nearest edge so far
    for (ax, ay), (bx, by) in zip(verts, verts[1:] + verts[:1]):
        vx, vy = bx - ax, by - ay  # edge vector (floats)
        t = max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / ((vx * vx + vy * vy) or 1.0)))
        best = min(best, math.hypot(px - (ax + t * vx), py - (ay + t * vy)))
    return best


def tri_hole_gaps(sampled):
    """list[float]: per triangle item of sampled["tri_artifacts"]["items"], the
    gap (px) between its polygon (verts) and the hole circle of _hole_circle.

    The gap is distance(polygon, hole centre) - max(rx, ry): the ellipse lies
    inside that circle, so this never overstates the true gap.  Also used by
    tests/test_studio_e2e.py on a saved tile's params.json.
    """
    hx, hy, rx, ry = _hole_circle(sampled)
    return [_point_triangle_dist(hx, hy, it["verts"]) - max(rx, ry)
            for it in sampled["tri_artifacts"]["items"]]


def test_clear_distance_in_schema_and_default(client):
    # the studio builds its slider from this leaf: a float 0-80 px, default 16
    leaf = client.get("/api/schema").json()["tri_artifacts"]["clear_distance"]  # dict
    assert leaf["type"] == "float" and leaf["min"] == 0 and leaf["max"] == 80, leaf
    assert leaf["help"], leaf
    cfg = client.get("/api/config/default").json()["config"]  # dict, default config
    assert cfg["tri_artifacts"]["clear_distance"] == 16.0
    assert isinstance(cfg["tri_artifacts"]["clear_distance"], float)


def test_parse_old_tri_yaml_fills_clear_distance(client):
    # an old YAML (written before clear_distance existed) loads with the default 16
    text = ("tri_artifacts:\n  enabled: true\n  count_range: [0, 6]\n  size_range: [15, 40]\n"
            "  gray_range: [65, 184]\n  pixel_var: 5\n  vertex_dist_range: [0.3, 1.0]\n")
    r = client.post("/api/config/parse", json={"text": text})
    assert r.status_code == 200, r.json()
    body = r.json()  # dict {config, warnings}
    assert body["config"]["tri_artifacts"]["clear_distance"] == 16.0
    assert not [w for w in body["warnings"] if w.startswith("tri_artifacts")], body["warnings"]


def test_generate_keeps_triangles_off_hole(client, tmp_path):
    # 6 triangles per image, a big hub (so a big hole) and a 40 px clear zone
    cfg = copy.deepcopy(v7_config.default_config())  # dict, request config
    cfg["tri_artifacts"]["count_range"] = [6, 6]
    cfg["geometry"]["hub_ratio_range"] = [0.5, 0.5]
    cfg["tri_artifacts"]["clear_distance"] = 40.0
    out = str(tmp_path / "clear")  # str, absolute output folder
    r = client.post("/api/generate", json=_gen_body(out, 4, config=cfg, save_params=True))
    assert r.status_code == 202, r.json()
    assert _wait_state(client, {"done", "error", "cancelled"})["state"] == "done"
    import yaml
    with open(os.path.join(out, "config.yaml"), encoding="utf-8") as f:
        written = yaml.safe_load(f)  # dict, the dataset's config.yaml
    assert written["tri_artifacts"]["clear_distance"] == 40.0
    recs = _read_params_lines(out)  # list[dict], one per image
    assert len(recs) == 4
    for rec in recs:
        tri = rec["sampled"]["tri_artifacts"]  # dict
        assert tri["n"] == 6 and len(tri["items"]) == 6, tri
        for it in tri["items"]:
            assert {"tries", "trimmed", "verts"} <= set(it), it
            assert isinstance(it["tries"], int) and isinstance(it["trimmed"], bool), it
            assert len(it["verts"]) == 3 and all(len(v) == 2 for v in it["verts"]), it
            assert it["placed"] is True and it["pixels"] >= 1, it
            # a trimmed triangle used every try and is painted only up to the zone,
            # so its full polygon may reach it; a placed-whole one may not
            assert 1 <= it["tries"] <= 64 and (it["tries"] == 64 or not it["trimmed"]), it
        gaps = tri_hole_gaps(rec["sampled"])  # list[float] px, one per triangle
        for it, gap in zip(tri["items"], gaps):
            if not it["trimmed"]:
                assert gap >= 40.0 - TRI_HOLE_SLACK, (rec["filename"], gap, it)
