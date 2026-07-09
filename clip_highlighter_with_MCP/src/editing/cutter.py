from __future__ import annotations

import asyncio
from pathlib import Path

from src.models.schemas import ClipAsset, HighlightCandidate


class ClipCutError(RuntimeError):
    """Raised when clip cutting fails or clip windows are invalid."""

    pass


def _format_seconds_for_filename(value: float) -> str:
    """Convert seconds to a zero-padded millisecond token for filenames."""

    millis = int(round(value * 1000))
    return f"{millis:08d}"


def _build_clip_filename(index: int, start_seconds: float, end_seconds: float) -> str:
    """Build deterministic clip filename using index and timestamp bounds."""

    start_token = _format_seconds_for_filename(start_seconds)
    end_token = _format_seconds_for_filename(end_seconds)
    return f"clip_{index:02d}_{start_token}_{end_token}.mp4"


async def _run_ffmpeg_cut(
    source_video: Path,
    output_path: Path,
    start_seconds: float,
    end_seconds: float,
) -> None:
    """Execute ffmpeg trim command for a single clip window."""

    duration = end_seconds - start_seconds
    # Re-encode to keep output compatibility predictable across source formats.
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_seconds:.3f}",
        "-i",
        str(source_video),
        "-t",
        f"{duration:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-c:a",
        "aac",
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
        error_message = stderr.decode("utf-8", errors="replace").strip() or "ffmpeg failed"
        raise ClipCutError(f"ffmpeg failed for clip '{output_path.name}': {error_message}")


async def cut_clips(
    source_video: Path,
    highlights: list[HighlightCandidate],
    output_dir: Path,
) -> list[ClipAsset]:
    """Cut short clips from a source video using ffmpeg."""
    if not source_video.exists():
        raise ClipCutError(f"Source video does not exist: {source_video}")

    output_dir.mkdir(parents=True, exist_ok=True)
    clips: list[ClipAsset] = []

    for index, segment in enumerate(highlights, start=1):
        if segment.end_seconds <= segment.start_seconds:
            raise ClipCutError(
                f"Invalid highlight window at index {index}: end must be greater than start",
            )

        clip_path = output_dir / _build_clip_filename(
            index=index,
            start_seconds=segment.start_seconds,
            end_seconds=segment.end_seconds,
        )

        await _run_ffmpeg_cut(
            source_video=source_video,
            output_path=clip_path,
            start_seconds=segment.start_seconds,
            end_seconds=segment.end_seconds,
        )

        if not clip_path.exists():
            raise ClipCutError(f"ffmpeg finished but output clip was not created: {clip_path}")

        clips.append(
            ClipAsset(
                clip_index=index,
                source_video=source_video,
                output_path=clip_path,
                start_seconds=segment.start_seconds,
                end_seconds=segment.end_seconds,
            )
        )

    return clips
