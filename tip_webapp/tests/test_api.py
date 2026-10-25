"""HTTP API tests via FastAPI's TestClient (in-process, no real server).

    TestClient ──▶ tip_webapp.app.main.app ──▶ inference.predict ──▶ the shipped TFLite model

Covers /api/health, /api/model, /api/samples, the static files, /api/predict on every sample,
the exactness of the box mapping (pasted and 2x-upscaled composites), the default box on a
non-square image, EXIF-upright boxes, every error status, and the `--model` CLI guard.
"""

from __future__ import annotations

import base64
import io
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tip_webapp.app import main

# SAMPLES_DIR: Path, folder with the labelled sample dials
SAMPLES_DIR = Path(main.SAMPLES_DIR)
# SAMPLES: list[dict], samples.json entries ({"file", "label": {"x", "y"}, "note", ...})
SAMPLES = json.loads((SAMPLES_DIR / "samples.json").read_text(encoding="utf-8"))["samples"]
# REPO_ROOT: Path, the directory that contains the tip_webapp package (cwd for the CLI test)
REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def client() -> TestClient:
    """TestClient bound to the FastAPI app."""
    return TestClient(main.app)


def _png(arr: np.ndarray) -> bytes:
    """PNG bytes of a uint8 array (H, W) gray or (H, W, 3) RGB."""
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return buf.getvalue()


def _sample(i: int) -> np.ndarray:
    """Sample i (1-based) as a (256, 256) uint8 gray array."""
    return np.asarray(Image.open(SAMPLES_DIR / SAMPLES[i - 1]["file"]).convert("L"))


def _predict(client: TestClient, data: bytes, box: str | None = None, name: str = "dial.png"):
    """POST /api/predict like the browser does; returns the httpx Response."""
    form = {"box": box} if box is not None else None
    return client.post("/api/predict", files={"file": (name, data, "image/png")}, data=form)


def _heatmap(body: dict) -> np.ndarray:
    """Decode the response heatmap -> (128, 128) float32."""
    hm = body["heatmap"]
    return np.frombuffer(base64.b64decode(hm["data"]), dtype="<f4").reshape(hm["height"], hm["width"])


# ── simple endpoints ───────────────────────────────────────────────────────────────────────────


def test_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200 and res.json() == {"status": "ok"}


def test_index_static_and_sample_files(client):
    page = client.get("/")
    assert page.status_code == 200 and "text/html" in page.headers["content-type"]
    assert 'data-testid="stage"' in page.text
    for path in ("/static/app.js", "/static/styles.css", "/samples/v9_val_1.png"):
        assert client.get(path).status_code == 200, path


def test_model_info(client):
    res = client.get("/api/model")
    assert res.status_code == 200
    body = res.json()
    assert body["name"].startswith("TipNet v9")
    assert body["served_file"] == "tipnet_v9.tflite"
    assert body["file_bytes"] == (main.inference.DEFAULT_MODEL_PATH).stat().st_size == 416_408
    io_ = body["interpreter"]
    assert io_["input_shape"] == [1, 256, 256, 1] and io_["output_shape"] == [1, 128, 128, 1]
    assert io_["input_dtype"] == io_["output_dtype"] == "float32"
    assert body["validation"]["p99_px"] == pytest.approx(3.073)


def test_samples(client):
    body = client.get("/api/samples").json()
    assert len(body["samples"]) == 6
    for i, s in enumerate(body["samples"], start=1):
        assert s["id"] == i and s["url"] == f"/samples/{s['file']}"
        assert set(s["label"]) == {"x", "y"}
        assert client.get(s["url"]).status_code == 200


# ── predictions ────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("index", range(1, 7))
def test_predict_each_sample(client, index):
    data = (SAMPLES_DIR / SAMPLES[index - 1]["file"]).read_bytes()
    res = _predict(client, data)
    assert res.status_code == 200, res.text
    body = res.json()
    assert set(body) >= {"box", "tip", "tip_model", "confidence", "peak_prob", "candidates", "heatmap",
                         "model_input_png", "timing_ms"}
    assert body["box"] == {"left": 0, "top": 0, "size": 256}
    assert body["tip"] == body["tip_model"]  # 256-px box at the origin: image px == model px

    hm = _heatmap(body)
    assert hm.shape == (128, 128) and body["heatmap"]["dtype"] == "float32"
    assert body["heatmap"]["encoding"] == "base64-le"
    assert hm.sum() == pytest.approx(1.0, abs=1e-4)
    assert body["peak_prob"] == pytest.approx(float(hm.max()), rel=1e-6)
    assert 0.0 < body["confidence"] <= 1.0 and body["confidence"] >= body["peak_prob"]

    cands = body["candidates"]
    assert 1 <= len(cands) <= 5
    assert (cands[0]["x"], cands[0]["y"]) == (body["tip"]["x"], body["tip"]["y"])
    assert (cands[0]["x_model"], cands[0]["y_model"]) == (body["tip_model"]["x"], body["tip_model"]["y"])
    assert cands[0]["mass"] == pytest.approx(body["confidence"])
    assert all(c["mass"] >= 1e-3 for c in cands)
    assert [c["mass"] for c in cands[1:]] == sorted((c["mass"] for c in cands[1:]), reverse=True)

    prefix = "data:image/png;base64,"
    assert body["model_input_png"].startswith(prefix)
    seen = Image.open(io.BytesIO(base64.b64decode(body["model_input_png"][len(prefix):])))
    assert seen.size == (256, 256) and seen.mode == "L"
    assert np.array_equal(np.asarray(seen), _sample(index))
    assert set(body["timing_ms"]) == {"preprocess", "inference", "total"}
    assert all(v >= 0 for v in body["timing_ms"].values())

    label = SAMPLES[index - 1]["label"]
    err = math.hypot(body["tip"]["x"] - label["x"], body["tip"]["y"] - label["y"])
    if index <= 5:
        assert err < 7.0
    else:
        assert err > 7.0 and cands[1]["mass"] > 0.2 * cands[0]["mass"]  # the look-alike shows as runner-up


def test_composite_box_gives_identical_tip(client):
    raw = _predict(client, _png(_sample(1))).json()
    canvas = np.full((480, 640), 90, np.uint8)
    canvas[61:61 + 256, 137:137 + 256] = _sample(1)
    body = _predict(client, _png(canvas), "137,61,256").json()
    assert body["box"] == {"left": 137, "top": 61, "size": 256}
    assert abs(body["tip"]["x"] - (raw["tip"]["x"] + 137)) < 1e-3
    assert abs(body["tip"]["y"] - (raw["tip"]["y"] + 61)) < 1e-3
    assert body["tip_model"] == raw["tip_model"]
    for c, rc in zip(body["candidates"], raw["candidates"]):
        assert abs(c["x"] - (rc["x"] + 137)) < 1e-3 and abs(c["y"] - (rc["y"] + 61)) < 1e-3


def test_upscaled_composite_maps_back_with_centre_convention(client):
    # Sample 2 (raw error 0.07 px): any model error is doubled by the 2x mapping, so a sample with a
    # small raw error isolates the mapping itself.
    label = SAMPLES[1]["label"]
    raw = _predict(client, _png(_sample(2))).json()
    big = np.asarray(Image.fromarray(_sample(2)).resize((512, 512), Image.Resampling.BICUBIC))
    canvas = np.full((700, 900), 128, np.uint8)
    left, top = 211, 97
    canvas[top:top + 512, left:left + 512] = big
    body = _predict(client, _png(canvas), f"{left},{top},512").json()
    # centre convention: crop px X = (x + 0.5) * 2 - 0.5, image px = offset + X
    want_x = left + (label["x"] + 0.5) * 2 - 0.5
    want_y = top + (label["y"] + 0.5) * 2 - 0.5
    err = math.hypot(body["tip"]["x"] - want_x, body["tip"]["y"] - want_y)
    raw_x = left + (raw["tip"]["x"] + 0.5) * 2 - 0.5
    raw_y = top + (raw["tip"]["y"] + 0.5) * 2 - 0.5
    err_raw = math.hypot(body["tip"]["x"] - raw_x, body["tip"]["y"] - raw_y)
    print(f"\n2x upscaled composite: {err:.3f} px from the 2x-mapped label, {err_raw:.3f} px from the 2x-mapped raw tip")
    assert err < 1.5 and err_raw < 1.5
    # tip in image px == model px mapped through the 512 box
    assert body["tip"]["x"] == pytest.approx(left + (body["tip_model"]["x"] + 0.5) * 2 - 0.5)


def test_default_box_on_non_square_image(client):
    raw = _predict(client, _png(_sample(3))).json()
    canvas = np.full((256, 400), 30, np.uint8)
    canvas[:, 72:72 + 256] = _sample(3)  # (400 - 256) // 2 = 72: the default box is exactly the dial
    body = _predict(client, _png(canvas)).json()
    assert body["box"] == {"left": 72, "top": 0, "size": 256}
    assert body["image"] == {"width": 400, "height": 256}
    assert abs(body["tip"]["x"] - (raw["tip"]["x"] + 72)) < 1e-3 and abs(body["tip"]["y"] - raw["tip"]["y"]) < 1e-3


def test_box_uses_exif_upright_coordinates(client):
    # Stored 300 wide x 200 tall with EXIF orientation 6 -> upright 200 x 300. "0,100,200" fits only upright.
    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, size=(200, 300), dtype=np.uint8))
    exif = Image.Exif()
    exif[0x0112] = 6
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif.tobytes())
    res = client.post("/api/predict", files={"file": ("phone.jpg", buf.getvalue(), "image/jpeg")}, data={"box": "0,100,200"})
    assert res.status_code == 200, res.text
    assert res.json()["image"] == {"width": 200, "height": 300}


# ── errors ─────────────────────────────────────────────────────────────────────────────────────


def test_missing_file_is_400(client):
    res = client.post("/api/predict", data={"box": "0,0,16"})
    assert res.status_code == 400 and isinstance(res.json()["detail"], str)


def test_file_sent_as_text_field_is_400(client):
    res = client.post("/api/predict", data={"file": "not a file upload"})
    assert res.status_code == 400 and isinstance(res.json()["detail"], str)


@pytest.mark.parametrize(
    ("payload", "needle"),
    [(b"just some text", "not an image"), (b"", "empty"), (b"\x89PNG\r\n\x1a\n" + b"\x00" * 40, "")],
)
def test_not_an_image_is_400(client, payload, needle):
    res = client.post("/api/predict", files={"file": ("x.png", payload, "image/png")})
    assert res.status_code == 400
    assert needle in res.json()["detail"]


def test_too_big_is_413(client, monkeypatch):
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 64)
    res = _predict(client, _png(_sample(1)))
    assert res.status_code == 413 and "MB" in res.json()["detail"]


def test_too_many_pixels_is_413(client, monkeypatch):
    monkeypatch.setattr(main, "MAX_PIXELS", 256 * 256 - 1)
    res = _predict(client, _png(_sample(1)))
    assert res.status_code == 413 and "megapixels" in res.json()["detail"]


@pytest.mark.parametrize(
    ("box", "needle"),
    [("1,2", "Malformed"), ("a,b,c", "Malformed"), ("1,2,3,4", "Malformed"), ("0.5,0,64", "Malformed"),
     ("-1,0,64", "outside"), ("200,0,64", "outside"), ("0,0,257", "outside"), ("10,10,15", "at least 16")],
)
def test_bad_box_is_422(client, box, needle):
    res = _predict(client, _png(_sample(1)), box)
    assert res.status_code == 422
    assert needle in res.json()["detail"]


def test_tiny_image_is_422(client):
    res = _predict(client, _png(np.zeros((12, 40), np.uint8)))
    assert res.status_code == 422 and "at least 16" in res.json()["detail"]


def test_cli_rejects_missing_model():
    proc = subprocess.run(
        [sys.executable, "-m", "tip_webapp", "--model", "does_not_exist.tflite", "--port", "1"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 2 and "not found" in proc.stderr
