"""FastAPI app: static page + JSON lists + dial / heatmap PNGs over one ValData.

    GET /                     static/index.html          GET /static/*  static files
    GET /api/health           {"status": "ok"}
    GET /api/summary          ValData.summary()
    GET /api/list?side&n      best|worst entries (422 bad side / n outside 1..1000)
    GET /api/image/{pos}.png  256x256 mode L dial        (404 pos outside 0..n_val-1)
    GET /api/heatmap/{pos}.png?scale=log|linear          128x128 RGB (422 bad scale)

Model forwards are serialised by the lock inside ValData; handlers are sync so FastAPI runs them
in its thread pool, and the PNG routes are safe under many concurrent requests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from . import render
from .data import ValData

# STATIC_DIR: Path, folder holding index.html and the page's css / js
STATIC_DIR = Path(__file__).resolve().parents[1] / "static"
# PNG_HEADERS: dict[str, str], response headers of the PNG routes
PNG_HEADERS = {"Cache-Control": "max-age=3600"}


def create_app(data: ValData) -> FastAPI:
    """Build the FastAPI app serving `data` (a loaded ValData)."""
    app = FastAPI(title="Val viewer", version="1.0.0")
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    def check_pos(pos: int) -> None:
        """404 unless 0 <= pos < n_val."""
        if not 0 <= pos < data.n_val:
            raise HTTPException(status_code=404, detail=f"no val image {pos}")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/api/summary")
    def summary() -> dict:
        return data.summary()

    @app.get("/api/list")
    def list_entries(side: Literal["best", "worst"], n: int = Query(24, ge=1, le=1000)) -> list[dict]:
        return data.entries(side, n)

    @app.get("/api/image/{pos}.png")
    def image(pos: int) -> Response:
        check_pos(pos)
        return Response(render.dial_png(data.image(pos)), media_type="image/png", headers=PNG_HEADERS)

    @app.get("/api/heatmap/{pos}.png")
    def heatmap(pos: int, scale: Literal["log", "linear"] = "log") -> Response:
        check_pos(pos)
        return Response(render.heatmap_png(data.logits(pos), scale), media_type="image/png",
                        headers=PNG_HEADERS)

    return app
