"""HTTP API tests via FastAPI's TestClient (in-process, no real server)."""

from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from grayscale_webapp.app import main


@pytest.fixture
def client() -> TestClient:
    """TestClient bound to the FastAPI app."""
    return TestClient(main.app)


@pytest.fixture
def png_bytes() -> bytes:
    """PNG bytes of a 6x4 colourful RGB image."""
    # rgb: (4, 6, 3) uint8 gradient
    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    rgb[..., 0] = np.arange(6) * 40
    rgb[..., 1] = (np.arange(4) * 60)[:, None]
    rgb[..., 2] = 120
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "PNG")
    return buf.getvalue()


def test_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200 and res.json() == {"status": "ok"}


def test_methods(client):
    res = client.get("/api/methods")
    assert res.status_code == 200
    body = res.json()
    assert [m["id"] for m in body] == ["luma601", "luma709", "average", "lightness"]
    assert all(set(m) == {"id", "label", "description"} for m in body)


def test_index_and_static(client):
    page = client.get("/")
    assert page.status_code == 200 and "text/html" in page.headers["content-type"]
    assert 'data-testid="drop-zone"' in page.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/styles.css").status_code == 200


def test_convert_success_headers(client, png_bytes):
    res = client.post(
        "/api/grayscale",
        files={"file": ("holiday photo.png", png_bytes, "image/png")},
        data={"method": "average"},
    )
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/png"
    assert 'filename="holiday_photo_gray.png"' in res.headers["content-disposition"]
    assert "filename*=UTF-8''holiday%20photo_gray.png" in res.headers["content-disposition"]
    assert res.headers["x-image-width"] == "6" and res.headers["x-image-height"] == "4"
    assert res.headers["x-gray-method"] == "average"
    assert Image.open(io.BytesIO(res.content)).mode == "L"


def test_convert_defaults_and_format(client, png_bytes):
    res = client.post("/api/grayscale", files={"file": ("a.png", png_bytes, "image/png")})
    assert res.status_code == 200 and res.headers["x-gray-method"] == "luma601"
    res = client.post("/api/grayscale", files={"file": ("a.png", png_bytes, "image/png")}, data={"format": "jpeg"})
    assert res.headers["content-type"] == "image/jpeg"
    assert 'filename="a_gray.jpg"' in res.headers["content-disposition"]


def test_text_file_is_400(client):
    res = client.post("/api/grayscale", files={"file": ("notes.txt", b"just some text", "text/plain")})
    assert res.status_code == 400
    assert "not a recognised image" in res.json()["detail"]


def test_missing_file_is_400(client):
    res = client.post("/api/grayscale", data={"method": "luma601"})
    assert res.status_code == 400
    assert isinstance(res.json()["detail"], str)


def test_too_big_is_413(client, png_bytes, monkeypatch):
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 32)
    res = client.post("/api/grayscale", files={"file": ("a.png", png_bytes, "image/png")})
    assert res.status_code == 413 and "MB" in res.json()["detail"]


def test_too_many_pixels_is_413(client, png_bytes, monkeypatch):
    monkeypatch.setattr(main, "MAX_PIXELS", 10)
    res = client.post("/api/grayscale", files={"file": ("a.png", png_bytes, "image/png")})
    assert res.status_code == 413 and "megapixels" in res.json()["detail"]


def test_bad_method_and_format_are_422(client, png_bytes):
    res = client.post("/api/grayscale", files={"file": ("a.png", png_bytes, "image/png")}, data={"method": "sepia"})
    assert res.status_code == 422 and "sepia" in res.json()["detail"]
    res = client.post("/api/grayscale", files={"file": ("a.png", png_bytes, "image/png")}, data={"format": "gif"})
    assert res.status_code == 422 and "gif" in res.json()["detail"]
