"""FastAPI backend for the v7 studio: schema/config endpoints, preview pool, job."""
import os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import studio.backend  # noqa: F401  (puts repo root on sys.path)
import v7_config
import generate_synthetic_v7_mp as gen
from studio.backend.jobs import JobRunner
from studio.backend.preview_worker import render_tile


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


class GenerateBody(BaseModel):
    """Generation request; output_dir must be absolute."""
    config: dict
    output_dir: str
    num_images: int
    seed: int = 0
    workers: int | None = None
    overwrite: bool = False


def _errors(errs, status=422):
    """JSON error response {"errors": list[str]}."""
    return JSONResponse({"errors": list(errs)}, status_code=status)


def _new_pool():
    """Fresh preview ProcessPoolExecutor, max_workers=min(8, cpu) (int)."""
    return ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 1))


def create_app(frontend_dist=None):
    """Build the app; frontend_dist (str|None) is mounted at / when given."""

    @asynccontextmanager
    async def lifespan(app):
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

    @app.exception_handler(RequestValidationError)
    async def _bad_body(request, exc):
        """Malformed body -> 422 {"errors": ["<loc>: <msg>", ...]} (list[str])."""
        errs = [".".join(str(p) for p in e["loc"] if p != "body") + ": " + e["msg"]
                for e in exc.errors()]
        return _errors(errs or ["request body: invalid"])

    @app.get("/api/schema")
    def schema():
        return v7_config.load_schema()

    @app.get("/api/config/default")
    def config_default():
        return {"config": v7_config.default_config()}

    @app.post("/api/config/parse")
    def config_parse(body: TextBody):
        try:
            return {"config": v7_config.parse_config_text(body.text)}
        except v7_config.ConfigError as e:
            return _errors(e.errors)

    @app.post("/api/config/dump")
    def config_dump(body: ConfigBody):
        try:
            v7_config.validate_config(body.config)
            return {"text": v7_config.dump_config(body.config)}
        except v7_config.ConfigError as e:
            return _errors(e.errors)

    @app.post("/api/preview")
    def preview(body: PreviewBody):
        if not 1 <= len(body.seeds) <= 16:
            return _errors(["seeds: need 1..16 seeds"])
        try:
            v7_config.validate_config(body.config)
        except v7_config.ConfigError as e:
            return _errors(e.errors)
        for attempt in (0, 1):
            try:
                futs = [app.state.pool.submit(render_tile, body.config, s)
                        for s in body.seeds]
                return {"tiles": [f.result() for f in futs]}
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

    @app.post("/api/generate", status_code=202)
    def generate(body: GenerateBody):
        if not os.path.isabs(body.output_dir):
            return _errors(["output_dir: must be an absolute path"], 400)
        if body.num_images < 1:
            return _errors(["num_images: must be >= 1"])
        try:
            v7_config.validate_config(body.config)
        except v7_config.ConfigError as e:
            return _errors(e.errors)
        try:
            jobs.start(body.config, body.output_dir, body.num_images, body.seed,
                       body.workers, body.overwrite)
        except RuntimeError as e:
            return _errors([str(e)], 409)
        except gen.OutputDirNotEmpty as e:
            return _errors([f"{e} (set overwrite to replace)"], 409)
        return {"state": "running"}

    @app.get("/api/jobs/current")
    def job_status():
        return jobs.status()

    @app.post("/api/jobs/current/cancel")
    def job_cancel():
        return jobs.cancel()

    if frontend_dist:
        app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
    return app
