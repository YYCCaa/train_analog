"""Studio backend API tests (TestClient; lifespan runs via the context manager)."""
import base64
import copy
import csv
import io
import os
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import v7_config
import generate_synthetic_v7_mp as gen
from studio.backend.app import create_app


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as c:
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
    bad = client.post("/api/config/parse", json={"text": "ticks:\n  len_range: [9, 6]\n"})
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
    client.post("/api/jobs/current/cancel")
    st = _wait_state(client, {"done", "error", "cancelled"})
    assert st["state"] == "cancelled"
    pngs = {f for f in os.listdir(out) if f.startswith("dial_")}
    with open(os.path.join(out, "labels.csv"), newline="") as f:
        labels = {row["filename"] for row in csv.DictReader(f)}
    assert pngs == labels
