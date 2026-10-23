"""FastAPI application: serves the static frontend and the grayscale conversion API."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from . import converter

# STATIC_DIR: Path, folder holding index.html / styles.css / app.js
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
# MAX_UPLOAD_BYTES: int, byte limit per upload (module-level so tests can monkeypatch it small)
MAX_UPLOAD_BYTES = converter.MAX_BYTES
# MAX_PIXELS: int, pixel limit per image (module-level so tests can monkeypatch it small)
MAX_PIXELS = converter.MAX_PIXELS

# _STATUS_FOR: dict[type, int], converter exception class -> HTTP status code
_STATUS_FOR = {
    converter.InvalidImageError: 400,
    converter.ImageTooLargeError: 413,
    converter.UnknownMethodError: 422,
    converter.UnknownFormatError: 422,
}

# app: FastAPI, the ASGI application (run with `python -m grayscale_webapp`)
app = FastAPI(title="Grayscale", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _download_name(original: str | None, extension: str) -> tuple[str, str]:
    """Build the output filename from the uploaded one.

    original: str | None, client filename such as "My photo.JPG"; extension: str such as "png".
    Returns (ascii_fallback, utf8_name), e.g. ("My_photo_gray.png", "My photo_gray.png").
    """
    stem = Path(original or "").stem.strip() or "image"
    utf8_name = f"{stem}_gray.{extension}"
    ascii_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "image"
    return f"{ascii_stem}_gray.{extension}", utf8_name


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Serve the single-page frontend."""
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")


@app.get("/api/health")
def health() -> dict[str, str]:
    """Liveness probe -> {"status": "ok"}."""
    return {"status": "ok"}


@app.get("/api/methods")
def methods() -> list[dict[str, str]]:
    """List grayscale methods as [{"id", "label", "description"}, ...] in display order."""
    return [{"id": m.id, "label": m.label, "description": m.description} for m in converter.METHODS.values()]


@app.post("/api/grayscale")
async def grayscale(
    file: UploadFile | None = File(None),
    method: str = Form(converter.DEFAULT_METHOD),
    out_format: str = Form("same", alias="format"),
) -> Response:
    """Convert one uploaded image; returns the image bytes as an attachment, or {"detail": str} on error."""
    if file is None:
        raise HTTPException(status_code=400, detail="No file was uploaded. Attach an image in the 'file' field.")
    # Validate cheap parameters before reading the body.
    if method not in converter.METHODS:
        raise HTTPException(422, f"Unknown method '{method}'. Choose one of: {', '.join(converter.METHODS)}.")
    if out_format not in converter.FORMAT_CHOICES:
        raise HTTPException(422, f"Unknown format '{out_format}'. Choose one of: {', '.join(converter.FORMAT_CHOICES)}.")

    # data: bytes, at most MAX_UPLOAD_BYTES + 1 so an oversize file is detected without reading it all
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    try:
        result = await run_in_threadpool(
            converter.convert_to_grayscale, data, method, out_format,
            max_bytes=MAX_UPLOAD_BYTES, max_pixels=MAX_PIXELS,
        )
    except converter.ConversionError as exc:
        raise HTTPException(status_code=_STATUS_FOR.get(type(exc), 400), detail=str(exc)) from exc

    ascii_name, utf8_name = _download_name(file.filename, result.extension)
    headers = {
        "Content-Disposition": f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(utf8_name)}",
        "X-Image-Width": str(result.width),
        "X-Image-Height": str(result.height),
        "X-Gray-Method": result.method,
        "Cache-Control": "no-store",
    }
    return Response(content=result.data, media_type=result.media_type, headers=headers)
