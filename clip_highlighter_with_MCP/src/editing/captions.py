from __future__ import annotations

import asyncio
import re
import textwrap
from pathlib import Path

from src.config.settings import settings
from src.models.schemas import ClipAsset, TranscriptSegment


class CaptionError(RuntimeError):
    """Raised when caption generation or burn-in fails."""

    pass


def _format_srt_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    total_milliseconds = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def _sanitize_caption_text(text: str) -> str:
    """Normalize caption text to avoid noisy spacing and empty lines."""
    return re.sub(r"\s+", " ", text).strip()


def _wrap_caption_text(text: str) -> str:
    """Wrap caption text to a narrow width suitable for vertical video."""
    wrapped_lines = textwrap.wrap(
        text,
        width=settings.caption_max_line_chars,
        break_long_words=False,
        break_on_hyphens=False,
    )
    return "\n".join(wrapped_lines) if wrapped_lines else text


def _build_force_style() -> str:
    """Build ffmpeg subtitle styling tuned for vertical short-form clips."""
    return (
        f"FontName=Arial,FontSize={settings.caption_font_size},PrimaryColour=&H00FFFFFF,"
        f"OutlineColour=&H00000000,BorderStyle=3,Outline=1,Shadow=0,Alignment=2,"
        f"MarginV={settings.caption_margin_v}"
    )


def _segments_for_clip(
    clip: ClipAsset,
    transcript_segments: list[TranscriptSegment],
) -> list[TranscriptSegment]:
    """Return transcript pieces that overlap the clip window in clip-relative time."""
    clip_segments: list[TranscriptSegment] = []

    for segment in transcript_segments:
        overlap_start = max(segment.start_seconds, clip.start_seconds)
        overlap_end = min(segment.end_seconds, clip.end_seconds)

        if overlap_end <= overlap_start:
            continue

        relative_start = overlap_start - clip.start_seconds
        relative_end = overlap_end - clip.start_seconds
        cleaned_text = _sanitize_caption_text(segment.text)
        if not cleaned_text:
            continue

        wrapped_text = _wrap_caption_text(cleaned_text)

        clip_segments.append(
            TranscriptSegment(
                start_seconds=relative_start,
                end_seconds=relative_end,
                text=wrapped_text,
            )
        )

    return clip_segments


def _build_srt_content(segments: list[TranscriptSegment]) -> str:
    """Build SRT text from clip-relative transcript segments."""
    blocks: list[str] = []
    for index, segment in enumerate(segments, start=1):
        start_timestamp = _format_srt_timestamp(segment.start_seconds)
        end_timestamp = _format_srt_timestamp(segment.end_seconds)
        blocks.append(f"{index}\n{start_timestamp} --> {end_timestamp}\n{segment.text}")

    return "\n\n".join(blocks) + ("\n" if blocks else "")


def _escape_ffmpeg_subtitle_path(path: Path) -> str:
    """Escape subtitle file path for ffmpeg subtitles filter usage."""
    value = str(path.resolve())
    value = value.replace("\\", "/")
    value = value.replace(":", "\\:")
    value = value.replace("'", "\\'")
    return value


async def _burn_subtitles(
    clip_path: Path,
    srt_path: Path,
    output_path: Path,
) -> None:
    """Burn SRT subtitles into a clip using ffmpeg."""
    escaped_srt = _escape_ffmpeg_subtitle_path(srt_path)

    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(clip_path),
        "-vf",
        (
            "subtitles='"
            f"{escaped_srt}"
            f"':force_style='{_build_force_style()}'"
        ),
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
        message = stderr.decode("utf-8", errors="replace").strip() or "ffmpeg subtitles failed"
        raise CaptionError(f"Failed to burn captions for '{clip_path.name}': {message}")


async def add_captions(
    clips: list[ClipAsset],
    transcript_segments: list[TranscriptSegment],
    output_dir: Path,
    srt_output_dir: Path | None = None,
) -> list[Path]:
    """Generate SRT captions per clip and burn them into output video files."""
    if not clips:
        raise CaptionError("No clips provided for captioning")
    if not transcript_segments:
        raise CaptionError("No transcript segments provided for captioning")

    output_dir.mkdir(parents=True, exist_ok=True)
    srt_dir = srt_output_dir or output_dir
    srt_dir.mkdir(parents=True, exist_ok=True)
    captioned_paths: list[Path] = []

    for clip in clips:
        if not clip.output_path.exists():
            raise CaptionError(f"Clip not found for captioning: {clip.output_path}")

        segments = _segments_for_clip(clip, transcript_segments)
        if not segments:
            raise CaptionError(
                f"No transcript overlap found for clip window {clip.start_seconds}-{clip.end_seconds}",
            )

        srt_path = srt_dir / f"{clip.output_path.stem}.srt"
        srt_content = _build_srt_content(segments)
        srt_path.write_text(srt_content, encoding="utf-8")

        output_path = output_dir / f"{clip.output_path.stem}_captioned.mp4"
        await _burn_subtitles(clip.output_path, srt_path, output_path)

        if not output_path.exists():
            raise CaptionError(f"Captioned clip was not created: {output_path}")

        captioned_paths.append(output_path)

    return captioned_paths
