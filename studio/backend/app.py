"""FastAPI backend for the v7 studio: schema/config endpoints, preview pool,
per-tile bundle download, job."""
import base64
import datetime
import io
import json
import os
import re
import zipfile
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, StrictBool
from starlette.middleware.trustedhost import TrustedHostMiddleware

import studio.backend  # noqa: F401  (puts repo root on sys.path)
import v7_config
import generate_synthetic_v7_mp as gen
from studio.backend.jobs import JobRunner
from studio.backend.preview_worker import render_tile

# Host header names (list[str], port stripped) the app answers; any other Host gets
# 400, so a DNS-rebinding page (Host: evil.example -> 127.0.0.1) cannot use the API
ALLOWED_HOSTS = ["127.0.0.1", "localhost"]
# Origin header values (compiled regex) accepted as the studio itself: loopback, any port
LOOPBACK_ORIGIN = re.compile(r"^http://(127\.0\.0\.1|localhost)(:\d{1,5})?$")
# upper bound of num_images per generate request (int)
MAX_NUM_IMAGES = 1_000_000


class ConfigBody(BaseModel):
    """Body with a cfg dict (section -> key -> value)."""
    config: dict


class TextBody(BaseModel):
    """Body with YAML text (str)."""
    text: str


class PreviewBody(BaseModel):
    """cfg plus 1..16 preview image seeds (list[int])."""
    config: dict
    seeds: list[int]


class BundleBody(BaseModel):
    """cfg plus the one image seed (int) to bundle as PNG + params + config."""
    config: dict
    seed: int


class GenerateBody(BaseModel):
    """Generation request; output_dir must be absolute."""
    config: dict
    output_dir: str
    num_images: int
    seed: int = 0
    workers: int | None = None
    overwrite: bool = False
    save_params: StrictBool = False  # also write params.jsonl (bool; "yes" / 1 are 422)


def _errors(errs, status=422):
    """JSON error response {"errors": list[str]}."""
    return JSONResponse({"errors": list(errs)}, status_code=status)


def _new_pool():
    """Fresh preview ProcessPoolExecutor, max_workers=min(8, cpu) (int)."""
    return ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 1))


def _auto_workers():
    """Workers (int >= 1) for a job whose request left workers blank: every core
    but one, so the preview pool stays responsive. The CLI default is unchanged."""
    return max(1, (os.cpu_count() or 1) - 1)


def create_app(frontend_dist=None, allowed_hosts=None):
    """Build the app; frontend_dist (str|None) is mounted at / when given;
    allowed_hosts (list[str]|None) replaces ALLOWED_HOSTS when given."""

    @asynccontextmanager
    async def lifespan(app):
        """App lifetime: create the preview ProcessPoolExecutor (app.state.pool) on
        startup; on shutdown stop it, cancel the job and wait <= 30 s for it."""
        # preview pool, separate from the job's own multiprocessing.Pool
        app.state.pool = _new_pool()
        try:
            yield
        finally:
            app.state.pool.shutdown(wait=False, cancel_futures=True)
            app.state.jobs.cancel()
            app.state.jobs.join(30)

    app = FastAPI(lifespan=lifespan)
    app.state.jobs = JobRunner()
    jobs = app.state.jobs  # the single JobRunner

    @app.middleware("http")
    async def _cross_site_guard(request: Request, call_next):
        """Refuse what another website could send from the user's browser:
        a non-loopback Origin -> 403; a non-GET /api request without a JSON
        Content-Type (a CORS "simple" request, sent without preflight) -> 415."""
        origin = request.headers.get("origin")  # str|None, set by browsers
        if origin is not None and not LOOPBACK_ORIGIN.match(origin):
            return _errors([f"origin {origin!r} not allowed (the studio is local only)"], 403)
        if request.url.path.startswith("/api/") and request.method not in ("GET", "HEAD"):
            # media type (str), lower case, without parameters like "; charset=utf-8"
            ctype = request.headers.get("content-type", "").split(";")[0].strip().lower()
            if ctype != "application/json":
                return _errors([f"Content-Type must be application/json, got {ctype!r}"], 415)
        return await call_next(request)

    # added last = outermost: the Host check runs before everything else
    app.add_middleware(TrustedHostMiddleware,
                       allowed_hosts=list(allowed_hosts or ALLOWED_HOSTS))

    @app.exception_handler(RequestValidationError)
    async def _bad_body(request, exc):
        """Malformed body -> 422 {"errors": ["<loc>: <msg>", ...]} (list[str])."""
        errs = [".".join(str(p) for p in e["loc"] if p != "body") + ": " + e["msg"]
                for e in exc.errors()]
        return _errors(errs or ["request body: invalid"])

    @app.get("/api/schema")
    def schema():
        """-> schema dict: section -> key -> leaf {type, min, max, step, help}."""
        return v7_config.load_schema()

    @app.get("/api/config/default")
    def config_default():
        """-> {"config": default cfg dict (section -> key -> value)}."""
        return {"config": v7_config.default_config()}

    @app.post("/api/config/parse")
    def config_parse(body: TextBody):
        """YAML text -> {"config": migrated, validated cfg, "warnings": list[str]
        (adjustments made to an old YAML, then the too-tight pointer gray range
        warnings of gen.pointer_gray_warnings; [] when none)}; bad YAML syntax -> 422."""
        try:
            cfg, warns = v7_config.parse_config_text_with_warnings(body.text)
            return {"config": cfg, "warnings": warns + gen.pointer_gray_warnings(cfg)}
        except v7_config.ConfigError as e:
            return _errors(e.errors)

    @app.post("/api/config/check")
    def config_check(body: ConfigBody):
        """cfg -> {"warnings": list[str]} of gen.pointer_gray_warnings ("section.key:
        message" when the pointer gray ranges are too tight, [] when fine); invalid
        cfg -> 422."""
        try:
            v7_config.validate_config(body.config)
        except v7_config.ConfigError as e:
            return _errors(e.errors)
        return {"warnings": gen.pointer_gray_warnings(body.config)}

    @app.post("/api/config/dump")
    def config_dump(body: ConfigBody):
        """cfg -> {"text": YAML str}; invalid cfg -> 422."""
        try:
            v7_config.validate_config(body.config)
            return {"text": v7_config.dump_config(body.config)}
        except v7_config.ConfigError as e:
            return _errors(e.errors)

    def _render_tiles(cfg, seeds):
        """Render seeds (list[int]) of a validated cfg on the preview pool ->
        list[dict] tiles (render_tile shape) in seed order, or a 500 JSONResponse;
        a dead pool is replaced and the render retried once."""
        for attempt in (0, 1):
            try:
                futs = [app.state.pool.submit(render_tile, cfg, s) for s in seeds]
                return [f.result() for f in futs]
            except BrokenProcessPool:
                # a worker died: replace the pool and retry once
                try:
                    app.state.pool.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    pass
                app.state.pool = _new_pool()
                if attempt == 1:
                    return _errors(["preview render failed: BrokenProcessPool"], 500)
            except Exception as e:
                return _errors([f"preview render failed: {type(e).__name__}: {e}"], 500)

    @app.post("/api/preview")
    def preview(body: PreviewBody):
        """cfg + seeds -> {"tiles": [{seed, png_b64, tip_x, tip_y, angle_deg, params}, ...]}
        in seed order; a dead pool is replaced and the render retried once."""
        if not 1 <= len(body.seeds) <= 16:
            return _errors(["seeds: need 1..16 seeds"])
        try:
            v7_config.validate_config(body.config)
        except v7_config.ConfigError as e:
            return _errors(e.errors)
        tiles = _render_tiles(body.config, body.seeds)  # list[dict] | JSONResponse
        return tiles if isinstance(tiles, JSONResponse) else {"tiles": tiles}

    @app.post("/api/bundle")
    def bundle(body: BundleBody):
        """cfg + seed -> application/zip download v7_seed<seed>.zip holding
        v7_seed<seed>.png (the preview tile's exact PNG bytes), params.json
        (seed, label, sampled params, full config) and config.yaml (loadable
        by the studio). Invalid cfg -> 422, render failure -> 500."""
        try:
            v7_config.validate_config(body.config)
        except v7_config.ConfigError as e:
            return _errors(e.errors)
        tiles = _render_tiles(body.config, [body.seed])  # list[dict] | JSONResponse
        if isinstance(tiles, JSONResponse):
            return tiles
        tile = tiles[0]  # dict, the one rendered tile
        name = f"v7_seed{body.seed}"  # str, file stem of the zip and the PNG
        params = {  # params.json content (dict), JSON-plain
            "generator": "generate_synthetic_v7_mp",
            "seed": body.seed,
            "angle_deg": tile["angle_deg"],
            "tip_x": tile["tip_x"],
            "tip_y": tile["tip_y"],
            "img_size": int(body.config["canvas"]["img_size"]),
            "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "sampled": tile["params"],
            "config": body.config,
        }
        buf = io.BytesIO()  # in-memory zip archive
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"{name}.png", base64.b64decode(tile["png_b64"]))
            zf.writestr("params.json", json.dumps(params, indent=2))
            zf.writestr("config.yaml", v7_config.dump_config(body.config))
        return Response(buf.getvalue(), media_type="application/zip",
                        headers={"Content-Disposition": f'attachment; filename="{name}.zip"'})

    @app.post("/api/generate", status_code=202)
    def generate(body: GenerateBody):
        """Start the generation job -> 202 {"state": "running"}. 400 bad path,
        422 out-of-bounds count/workers or bad cfg, 409 job running / old data."""
        out = body.output_dir  # requested output folder (str)
        # Windows: "\foo" has no drive, so it would land on the cwd's drive
        driveless = os.name == "nt" and os.path.splitdrive(out)[0] == ""  # bool
        if not os.path.isabs(out) or driveless:
            return _errors(["output_dir: must be an absolute path"
                            + (" with a drive letter or UNC prefix" if os.name == "nt" else "")],
                           400)
        if not 1 <= body.num_images <= MAX_NUM_IMAGES:
            return _errors([f"num_images: must be 1..{MAX_NUM_IMAGES}"])
        cpu = os.cpu_count() or 1  # logical cores (int)
        if body.workers is not None and not 1 <= body.workers <= cpu:
            return _errors([f"workers: must be 1..{cpu}, or blank for auto"])
        try:
            v7_config.validate_config(body.config)
        except v7_config.ConfigError as e:
            return _errors(e.errors)
        # worker processes for this job (int >= 1); blank -> leave one core free
        workers = body.workers if body.workers is not None else _auto_workers()
        try:
            # kwarg only when set, so a start() stand-in with the old signature still fits
            extra = {"save_params": True} if body.save_params else {}  # dict[str, bool]
            jobs.start(body.config, out, body.num_images, body.seed, workers, body.overwrite,
                       **extra)
        except gen.OutputDirNotEmpty:  # subclasses RuntimeError, so it must come first
            return _errors([f"'{out}' already contains dataset files "
                            "(dial_*.png / labels.csv / params.jsonl / config.yaml); "
                            "tick the overwrite box to replace them"], 409)
        except RuntimeError as e:
            return _errors([str(e)], 409)
        return {"state": "running"}

    @app.get("/api/jobs/current")
    def job_status():
        """-> job status {state, done, total, output_dir, error}."""
        return jobs.status()

    @app.post("/api/jobs/current/cancel")
    def job_cancel():
        """Request cancel of the running job -> job status at that moment."""
        return jobs.cancel()

    if frontend_dist:
        app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
    return app
