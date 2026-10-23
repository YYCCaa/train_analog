"""Launch the v7 studio: build the React frontend if needed, then serve the
FastAPI backend + built frontend on 127.0.0.1 with uvicorn.

    python studio/run_studio.py [--port 8765] [--no-browser]

Works from any working directory: the repo root is made the cwd and
sys.path[0], so `studio.backend.app` and the spawned preview workers import.
"""
import argparse
import os
import subprocess
import sys
import threading
import webbrowser

# repo root (str, absolute): parent of this file's folder
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# frontend project folder and its Vite build output (str, absolute)
FRONTEND_DIR = os.path.join(REPO_ROOT, "studio", "frontend")
DIST_DIR = os.path.join(FRONTEND_DIR, "dist")
# timeout for each npm step (s, int)
NPM_TIMEOUT = 600


def _newest_source_mtime():
    """Newest mtime (float) among frontend sources (src/, index.html, configs)."""
    paths = [os.path.join(FRONTEND_DIR, n) for n in
             ("index.html", "package.json", "package-lock.json", "vite.config.ts", "tsconfig.json")]
    for root, _dirs, files in os.walk(os.path.join(FRONTEND_DIR, "src")):
        paths += [os.path.join(root, f) for f in files]
    return max((os.path.getmtime(p) for p in paths if os.path.exists(p)), default=0.0)


def needs_build():
    """True if dist/index.html is missing or older than any frontend source."""
    index = os.path.join(DIST_DIR, "index.html")
    return not os.path.exists(index) or os.path.getmtime(index) < _newest_source_mtime()


def _npm(*args):
    """Run `npm <args>` in FRONTEND_DIR; raise SystemExit on failure/timeout."""
    cmd = "npm " + " ".join(args)
    print(f"[run_studio] {cmd}", flush=True)
    try:
        subprocess.run(cmd, cwd=FRONTEND_DIR, shell=True, check=True, timeout=NPM_TIMEOUT)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        raise SystemExit(f"[run_studio] `{cmd}` failed: {e}") from e


def build_frontend():
    """npm ci (if node_modules is missing) then npm run build."""
    if not os.path.isdir(os.path.join(FRONTEND_DIR, "node_modules")):
        _npm("ci")
    _npm("run", "build")


def main():
    ap = argparse.ArgumentParser(description="v7 dial studio (localhost only)")
    ap.add_argument("--port", type=int, default=8765, help="port on 127.0.0.1 (default 8765)")
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    args = ap.parse_args()

    os.chdir(REPO_ROOT)
    if sys.path[0] != REPO_ROOT:
        sys.path.insert(0, REPO_ROOT)
    if needs_build():
        build_frontend()

    import uvicorn
    from studio.backend.app import create_app

    app = create_app(frontend_dist=DIST_DIR)
    url = f"http://127.0.0.1:{args.port}/"  # studio page URL (str)
    print(f"[run_studio] serving {url}", flush=True)
    if not args.no_browser:
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    uvicorn.run(app, host="127.0.0.1", port=args.port, workers=1, log_level="warning")


if __name__ == "__main__":  # required: preview pool workers re-import this file on spawn
    main()
