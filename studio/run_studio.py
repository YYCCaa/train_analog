"""Launch the v7 studio: build the React frontend if needed, then serve the
FastAPI backend + built frontend on 127.0.0.1 with uvicorn.

    python studio/run_studio.py [--port 8765] [--no-browser]

Works from any working directory: the repo root is made the cwd and
sys.path[0], so `studio.backend.app` and the spawned preview workers import.
"""
import argparse
import os
import socket
import subprocess
import sys
import threading
import time
import webbrowser

# repo root (str, absolute): parent of this file's folder
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# frontend project folder and its Vite build output (str, absolute)
FRONTEND_DIR = os.path.join(REPO_ROOT, "studio", "frontend")
DIST_DIR = os.path.join(FRONTEND_DIR, "dist")
# timeout for each npm step (s, int)
NPM_TIMEOUT = 600
# how long to wait for the server port before giving up on opening a browser (s, float)
BROWSER_WAIT = 30.0


def _newest_source_mtime():
    """Newest mtime (float) among frontend sources (src/, index.html, configs)."""
    # candidate source files, list[str] of absolute paths
    paths = [os.path.join(FRONTEND_DIR, n) for n in
             ("index.html", "package.json", "package-lock.json", "vite.config.ts", "tsconfig.json")]
    for root, _dirs, files in os.walk(os.path.join(FRONTEND_DIR, "src")):
        paths += [os.path.join(root, f) for f in files]
    return max((os.path.getmtime(p) for p in paths if os.path.exists(p)), default=0.0)


def needs_build():
    """True if dist/index.html is missing or older than any frontend source."""
    index = os.path.join(DIST_DIR, "index.html")  # built page (str, absolute)
    return not os.path.exists(index) or os.path.getmtime(index) < _newest_source_mtime()


def needs_install():
    """True if node_modules is missing or older than package-lock.json (deps changed).

    npm writes node_modules/.package-lock.json on every install; compare to it.
    """
    lock = os.path.join(FRONTEND_DIR, "package-lock.json")  # committed lockfile (str)
    stamp = os.path.join(FRONTEND_DIR, "node_modules", ".package-lock.json")  # install stamp (str)
    if not os.path.exists(stamp):
        return True
    return os.path.exists(lock) and os.path.getmtime(lock) > os.path.getmtime(stamp)


def _npm(*args):
    """Run `npm <args>` (str words) in FRONTEND_DIR; raise SystemExit on failure/timeout."""
    cmd = "npm " + " ".join(args)  # shell command line (str); constant words only
    print(f"[run_studio] {cmd}", flush=True)
    try:
        subprocess.run(cmd, cwd=FRONTEND_DIR, shell=True, check=True, timeout=NPM_TIMEOUT)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        raise SystemExit(f"[run_studio] `{cmd}` failed: {e}") from e


def build_frontend():
    """npm ci (if dependencies are missing or stale) then npm run build."""
    if needs_install():
        _npm("ci")
    _npm("run", "build")


def _open_browser_when_ready(url, port):
    """Poll 127.0.0.1:port (int) until it accepts connections, then open url (str).

    Gives up silently after BROWSER_WAIT seconds.
    """
    deadline = time.time() + BROWSER_WAIT  # give-up time (float, epoch s)
    while time.time() < deadline:
        with socket.socket() as s:
            s.settimeout(0.5)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                webbrowser.open(url)
                return
        time.sleep(0.2)
    print(f"[run_studio] server not answering after {BROWSER_WAIT:.0f} s; open {url} manually",
          flush=True)


def main():
    # command-line parser (argparse.ArgumentParser)
    ap = argparse.ArgumentParser(description="v7 dial studio (localhost only)")
    ap.add_argument("--port", type=int, default=8765, help="port on 127.0.0.1 (default 8765)")
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    args = ap.parse_args()  # parsed flags: port (int), no_browser (bool)

    os.chdir(REPO_ROOT)
    if sys.path[0] != REPO_ROOT:
        sys.path.insert(0, REPO_ROOT)
    if needs_build():
        build_frontend()

    import uvicorn
    from studio.backend.app import create_app

    app = create_app(frontend_dist=DIST_DIR)  # FastAPI app serving /api + dist/ at /
    url = f"http://127.0.0.1:{args.port}/"  # studio page URL (str)
    print(f"[run_studio] serving {url}", flush=True)
    if not args.no_browser:
        threading.Thread(target=_open_browser_when_ready, args=(url, args.port),
                         daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=args.port, workers=1, log_level="warning")


if __name__ == "__main__":  # required: preview pool workers re-import this file on spawn
    main()
