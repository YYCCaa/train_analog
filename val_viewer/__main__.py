"""Entry point: `python -m val_viewer [--data-dir D] [--run R] ...` ranks the val split, then serves the page."""

from __future__ import annotations

import argparse
import os

import train_v9_tip as tv


def main(argv: list[str] | None = None) -> None:
    """Parse flags (argv: list[str] | None, default sys.argv[1:]), load + rank, serve until interrupted.

    A missing data dir / labels.csv / run dir / weights exits with code 2 and one `error: ...` line."""
    parser = argparse.ArgumentParser(prog="python -m val_viewer",
                                     description="Best / worst n TipNet v9 validation images with heatmaps.")
    parser.add_argument("--data-dir", default=tv.DEFAULT_DATA_DIR, help="dataset folder (labels.csv + PNGs)")
    parser.add_argument("--run", default=tv.DEFAULT_RUNS_DIR + "/v9_tipnet_a",
                        help="run folder holding tip.pth or best_p99.pth")
    parser.add_argument("--val-split", type=float, default=0.15, help="validation fraction (default 0.15)")
    parser.add_argument("--cache-dir", default=None, help="ranking cache folder (default val_viewer/.cache)")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8020)
    args = parser.parse_args(argv)

    from val_viewer.app.data import ValData  # late: keeps --help fast
    from val_viewer.app.main import create_app
    import uvicorn

    try:
        data = ValData(args.data_dir, args.run, val_split=args.val_split,
                       cache_dir=args.cache_dir, device=args.device)
    except FileNotFoundError as exc:
        parser.exit(2, "error: " + str(exc).replace(os.linesep, " ").replace("\n", " ") + "\n")
    print(f"Val viewer on http://{args.host}:{args.port}/", flush=True)
    uvicorn.run(create_app(data), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
