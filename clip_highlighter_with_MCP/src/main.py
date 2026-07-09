from __future__ import annotations

import argparse
import asyncio
import sys

from src.pipeline.orchestrator import run_pipeline
from src.utils.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI clip highlighter")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_cmd = subparsers.add_parser("run", help="Run full clip-highlighting pipeline")
    run_cmd.add_argument("--url", required=True, help="YouTube video URL")

    return parser


async def _main() -> None:
    """Parse CLI args and run selected commands."""
    configure_logging()
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        result = await run_pipeline(url=args.url)
        print("Generated clips:")
        for clip in result.final_clips:
            print(f"- {clip}")


def main() -> None:
    """CLI entrypoint with user-friendly error output."""
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        print("Run cancelled by user.")
        raise SystemExit(130)
    except Exception as exc:  # noqa: BLE001
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        print(
            "Check that FFmpeg is installed, DEEPSEEK_API_KEY is set, and the YouTube URL is valid.",
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
