"""FastAPI app: static page + JSON lists + dial / heatmap PNGs over one ValData.

    GET /                     static/index.html          GET /static/*  static files
    GET /api/health           {"status": "ok"}
    GET /api/summary?set      ValData.summary(set); set = val (default) | train | all (422 otherwise)
    GET /api/list?set&side&n  best|worst entries of the set (422 bad set / side / n outside 1..1000)
    GET /api/image/{row}.png  256x256 mode L dial        (404 row outside 0..N-1)
    GET /api/heatmap/{row}.png?scale=log|linear          128x128 RGB (422 bad scale)
    GET /api/params/{row}     saved parameter record of the row (404 no file / no record, 409 wrong file name or tip /
                              file changed since start)

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
from .params import NoParamsFile, NoRecord, ParamsChanged, RecordMismatch

# STATIC_DIR: Path, folder holding index.html and the page's css / js
STATIC_DIR = Path(__file__).resolve().parents[1] / "static"
# PNG_HEADERS: dict[str, str], response headers of the PNG routes
PNG_HEADERS = {"Cache-Control": "max-age=3600"}
# SetName: the allowed values of the `set` query parameter
SetName = Literal["val", "train", "all"]


def create_app(data: ValData) -> FastAPI:
    """Build the FastAPI app serving `data` (a loaded ValData)."""
    app = FastAPI(title="Val viewer", version="1.0.0")
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    def check_row(row: int) -> None:
        """404 unless 0 <= row < N (dataset rows)."""
        if not 0 <= row < data.n_total:
            raise HTTPException(status_code=404, detail=f"no image {row}")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/api/summary")
    def summary(set: SetName = "val") -> dict:
        return data.summary(set)

    @app.get("/api/list")
    def list_entries(side: Literal["best", "worst"], n: int = Query(24, ge=1, le=1000),
                     set: SetName = "val") -> list[dict]:
        return data.entries(set, side, n)

    @app.get("/api/image/{row}.png")
    def image(row: int) -> Response:
        check_row(row)
        return Response(render.dial_png(data.image(row)), media_type="image/png", headers=PNG_HEADERS)

    @app.get("/api/heatmap/{row}.png")
    def heatmap(row: int, scale: Literal["log", "linear"] = "log") -> Response:
        check_row(row)
        return Response(render.heatmap_png(data.logits(row), scale), media_type="image/png",
                        headers=PNG_HEADERS)

    @app.get("/api/params/{row}")
    def params(row: int) -> dict:
        check_row(row)
        try:
            return {**data.params.get(row), "row": row}  # the real row wins over a hand-edited "row"
        except NoParamsFile:
            raise HTTPException(status_code=404, detail="no parameters saved for this dataset") from None
        except NoRecord:
            raise HTTPException(status_code=404, detail="no parameters saved for this image") from None
        except ParamsChanged:
            raise HTTPException(
                status_code=409,
                detail="params.jsonl changed on disk since the viewer started; restart the viewer") from None
        except RecordMismatch as exc:
            raise HTTPException(
                status_code=409,
                detail=f"saved parameters do not match this image ({exc.record_name})") from None

    return app
