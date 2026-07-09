from __future__ import annotations

import asyncio
from pathlib import Path


class ReframeError(RuntimeError):
    """Raised when vertical reframing fails."""

    pass


def _build_reframed_name(path: Path) -> str:
    """Build deterministic output filename for reframed clips."""
    return f"{path.stem}_vertical.mp4"


async def _run_ffmpeg_reframe(
    input_path: Path,
    output_path: Path,
    width: int,
    height: int,
) -> None:
    """Run ffmpeg to create a vertical center-cropped output clip."""
    # First scale while preserving aspect ratio, then center-crop to exact target dimensions.
    filter_chain = f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}"
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-vf",
        filter_chain,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _stdout, stderr = await process.communicate()

    if process.returncode != 0:
        message = stderr.decode("utf-8", errors="replace").strip() or "ffmpeg reframe failed"
        raise ReframeError(f"Failed to reframe '{input_path.name}': {message}")


async def reframe_vertical(
    clips: list[Path],
    output_dir: Path,
    width: int = 1080,
    height: int = 1920,
) -> list[Path]:
    """Reframe clips to a vertical aspect ratio using center crop."""
    if not clips:
        raise ReframeError("No clips provided for reframing")
    if width <= 0 or height <= 0:
        raise ReframeError("Reframe width/height must be positive integers")

    output_dir.mkdir(parents=True, exist_ok=True)
    reframed_paths: list[Path] = []

    for clip_path in clips:
        if not clip_path.exists():
            raise ReframeError(f"Clip not found for reframing: {clip_path}")

        output_path = output_dir / _build_reframed_name(clip_path)
        await _run_ffmpeg_reframe(
            input_path=clip_path,
            output_path=output_path,
            width=width,
            height=height,
        )

        if not output_path.exists():
            raise ReframeError(f"Reframed clip was not created: {output_path}")

        reframed_paths.append(output_path)

    return reframed_paths
