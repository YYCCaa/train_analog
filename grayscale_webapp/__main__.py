"""Entry point: `python -m grayscale_webapp [--host 127.0.0.1] [--port 8000]` runs the app under uvicorn."""

from __future__ import annotations

import argparse

import uvicorn


def main(argv: list[str] | None = None) -> None:
    """Parse CLI flags (argv: list[str] | None, defaults to sys.argv[1:]) and serve until interrupted."""
    parser = argparse.ArgumentParser(prog="python -m grayscale_webapp", description="Image to grayscale web app.")
    parser.add_argument("--host", default="127.0.0.1", help="interface to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="TCP port (default: 8000)")
    args = parser.parse_args(argv)

    from grayscale_webapp.app.main import app  # imported late so --help stays fast

    print(f"Grayscale web app on http://{args.host}:{args.port}/  (Ctrl+C to stop)", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
