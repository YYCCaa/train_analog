"""FastAPI application: serves the static frontend, the sample dials and the tip-prediction API.

    Browser                                   this module                          inference.py
   ┌──────────────────────┐  GET /           ┌──────────────────────────────┐
   │ index.html + app.js  │◀─────────────────│ static/index.html            │
   │                      │  GET /static/*   │ static/*.css|js              │
   │ header badge ◀───────┼── GET /api/model │ card JSON + file size +      │◀── TipModel.describe()
   │ Samples menu ◀───────┼── GET /api/samples  samples.json + urls         │
   │ <img> ◀──────────────┼── GET /samples/* │ samples/*.png                │
   │                      │                  │                              │
   │ square crop (PNG) ───┼─ POST ──────────▶│ /api/predict                 │──▶ predict(model, bytes, box)
   │ FormData{file, box}  │  /api/predict    │   400 / 413 / 422 {"detail"} │◀── Prediction | PredictError
   │ cues ◀── JSON ───────┼──────────────────│   200 Prediction.to_json()   │
   └──────────────────────┘                  └──────────────────────────────┘

The TFLite interpreter is created once per process (lazily, or eagerly by `configure()` from
`python -m tip_webapp --model PATH`) and shared by every request behind a threading.Lock.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from . import inference

# STATIC_DIR: Path, folder holding index.html / styles.css / app.js
STATIC_DIR = inference.PACKAGE_DIR / "static"
# SAMPLES_DIR: Path, folder holding v9_val_*.png and samples.json
SAMPLES_DIR = inference.PACKAGE_DIR / "samples"
# MAX_UPLOAD_BYTES: int, byte limit per upload (module-level so tests can monkeypatch it small)
MAX_UPLOAD_BYTES = inference.MAX_BYTES
# MAX_PIXELS: int, pixel limit per image (module-level so tests can monkeypatch it small)
MAX_PIXELS = inference.MAX_PIXELS
# MODEL_PATH: Path, the .tflite file served; replaced by configure() for `--model PATH`
MODEL_PATH: Path = inference.DEFAULT_MODEL_PATH

# _STATUS_FOR: dict[type, int], inference exception class -> HTTP status code
_STATUS_FOR = {
    inference.InvalidImageError: 400,
    inference.ImageTooLargeError: 413,
    inference.InvalidBoxError: 422,
}

# _model: TipModel | None, the process-wide interpreter (created on first use)
_model: inference.TipModel | None = None
# _model_lock: threading.Lock, guards the lazy creation of _model
_model_lock = threading.Lock()

# app: FastAPI, the ASGI application (run with `python -m tip_webapp`)
app = FastAPI(title="Tip finder", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/samples", StaticFiles(directory=SAMPLES_DIR), name="samples")


def configure(model_path: Path | str | None = None) -> inference.TipModel:
    """Select the model file (None = the shipped one) and load it now, so a bad path fails at launch.

    Returns the loaded TipModel. Raises FileNotFoundError / ValueError for a missing or wrong model.
    """
    global MODEL_PATH, _model
    with _model_lock:
        MODEL_PATH = Path(model_path) if model_path else inference.DEFAULT_MODEL_PATH
        _model = inference.TipModel(MODEL_PATH)
        return _model


def get_model() -> inference.TipModel:
    """The shared TipModel for MODEL_PATH, created on first call (thread-safe)."""
    global _model
    with _model_lock:
        if _model is None or _model.path != Path(MODEL_PATH):
            _model = inference.TipModel(MODEL_PATH)
        return _model


@app.exception_handler(RequestValidationError)
async def _validation_as_400(_request: Request, exc: RequestValidationError) -> JSONResponse:
    """FastAPI's own form-parsing errors (e.g. `file` sent as text) -> 400 {"detail": str}, like every other error."""
    first = exc.errors()[0] if exc.errors() else {}
    where = ".".join(str(p) for p in first.get("loc", ()) if p != "body") or "request"
    return JSONResponse(status_code=400, content={"detail": f"Malformed request: {where}: {first.get('msg', 'invalid')}."})


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Serve the single-page frontend."""
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")


@app.get("/api/health")
def health() -> dict[str, str]:
    """Liveness probe -> {"status": "ok"}."""
    return {"status": "ok"}


@app.get("/api/model")
def model_info() -> dict[str, object]:
    """The model card plus what the server actually loaded.

    -> {...card fields..., "served_file": str, "file_bytes": int,
        "interpreter": {"runtime", "num_threads", "input_shape", "input_dtype", "output_shape", "output_dtype"}}
    """
    model = get_model()
    card = inference.load_card(model.path)
    return {**card, "served_file": model.path.name, "file_bytes": model.file_bytes, "interpreter": model.describe()}


@app.get("/api/samples")
def samples() -> dict[str, object]:
    """samples.json with a "url" added to every sample -> {"coords": str, "samples": [{file, url, label, note, ...}]}."""
    data = json.loads((SAMPLES_DIR / "samples.json").read_text(encoding="utf-8"))
    for i, sample in enumerate(data.get("samples", []), start=1):
        sample["id"] = i
        sample["url"] = f"/samples/{sample['file']}"
    return data


@app.post("/api/predict")
async def predict(
    file: UploadFile | None = File(None),
    box: str | None = Form(None),
) -> JSONResponse:
    """Predict the pointer tip in one uploaded image (optionally inside a square box).

    file: the image (Pillow-readable); box: "left,top,size" in the uploaded image's px, or omitted.
    -> 200 Prediction.to_json(), or {"detail": str} with 400 / 413 / 422.
    """
    t_start = time.perf_counter()
    if file is None:
        raise HTTPException(status_code=400, detail="No file was uploaded. Attach an image in the 'file' field.")
    # data: bytes, at most MAX_UPLOAD_BYTES + 1 so an oversize file is detected without reading it all
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    model = get_model()
    try:
        pred = await run_in_threadpool(
            inference.predict, model, data, box, max_bytes=MAX_UPLOAD_BYTES, max_pixels=MAX_PIXELS,
        )
    except inference.PredictError as exc:
        raise HTTPException(status_code=_STATUS_FOR.get(type(exc), 400), detail=str(exc)) from exc
    body = await run_in_threadpool(pred.to_json)
    # total: whole request in ms, from the moment the body was parsed to the finished JSON
    body["timing_ms"]["total"] = round((time.perf_counter() - t_start) * 1000.0, 2)
    return JSONResponse(content=body, headers={"Cache-Control": "no-store"})
