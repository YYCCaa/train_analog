"""Entry point: `python -m tip_webapp [--host 127.0.0.1] [--port 8010] [--model PATH]` runs the app under uvicorn."""

from __future__ import annotations

import argparse

import uvicorn


def main(argv: list[str] | None = None) -> None:
    """Parse CLI flags (argv: list[str] | None, defaults to sys.argv[1:]), load the model, serve until interrupted."""
    parser = argparse.ArgumentParser(
        prog="python -m tip_webapp",
        description="Find the pointer tip of an analog dial with the TipNet v9 TFLite model.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="interface to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8010, help="TCP port (default: 8010)")
    parser.add_argument("--model", default=None, help="path to a .tflite model (default: tip_webapp/model/tipnet_v9.tflite)")
    args = parser.parse_args(argv)

    from tip_webapp.app import main as web  # imported late so --help stays fast

    try:
        model = web.configure(args.model)  # load now: a bad --model fails here, not on the first request
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(f"Model {model.path} ({model.file_bytes / 1024:.0f} KB, {model.runtime}, {model.num_threads} threads)", flush=True)
    print(f"Tip finder on http://{args.host}:{args.port}/  (Ctrl+C to stop)", flush=True)
    uvicorn.run(web.app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
