from __future__ import annotations

import asyncio
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from src.models.schemas import VideoAsset


class YouTubeIngestionError(RuntimeError):
    """Base error for YouTube ingestion failures."""

    pass


class InvalidYouTubeUrlError(YouTubeIngestionError):
    """Raised when a provided URL is not a supported YouTube video URL."""

    pass


class YouTubeMetadataError(YouTubeIngestionError):
    """Raised when video metadata cannot be fetched or parsed."""

    pass


class YouTubeDownloadError(YouTubeIngestionError):
    """Raised when yt-dlp download fails or output cannot be found."""

    pass


def _validate_youtube_url(url: str) -> None:
    """Validate that the URL points to a supported YouTube video format."""

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise InvalidYouTubeUrlError("URL must start with http:// or https://")

    host = (parsed.netloc or "").lower()
    path = parsed.path or ""

    if host.endswith("youtube.com"):
        has_watch_query = bool(parse_qs(parsed.query).get("v"))
        has_short_path = path.startswith("/shorts/") or path.startswith("/live/")
        if not (has_watch_query or has_short_path):
            raise InvalidYouTubeUrlError(
                "youtube.com URL must contain a valid video reference (v=... or /shorts/...)",
            )
        return

    if host.endswith("youtu.be") and path.strip("/"):
        return

    raise InvalidYouTubeUrlError("Only YouTube URLs are supported")


async def _run_command(command: list[str]) -> tuple[str, str]:
    """Run a subprocess command and return decoded stdout/stderr."""

    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout_bytes, stderr_bytes = await process.communicate()
    stdout = stdout_bytes.decode("utf-8", errors="replace").strip()
    stderr = stderr_bytes.decode("utf-8", errors="replace").strip()

    if process.returncode != 0:
        raise YouTubeIngestionError(stderr or "yt-dlp command failed")

    return stdout, stderr


async def _fetch_metadata(url: str) -> dict:
    """Fetch metadata for a YouTube URL without downloading media."""

    command = [
        "yt-dlp",
        "--no-playlist",
        "--dump-single-json",
        "--skip-download",
        url,
    ]

    try:
        stdout, _ = await _run_command(command)
    except YouTubeIngestionError as exc:
        raise YouTubeMetadataError(f"Failed to fetch video metadata: {exc}") from exc

    if not stdout:
        raise YouTubeMetadataError("yt-dlp did not return metadata")

    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise YouTubeMetadataError("yt-dlp returned invalid JSON metadata") from exc

    if not isinstance(payload, dict):
        raise YouTubeMetadataError("Unexpected metadata format from yt-dlp")

    return payload


async def ingest_youtube_url(url: str, output_dir: Path) -> VideoAsset:
    """Download a YouTube video and return its local asset metadata."""

    _validate_youtube_url(url)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(output_dir / "%(id)s.%(ext)s")
    metadata = await _fetch_metadata(url)
    video_id = metadata.get("id")
    title = metadata.get("title")

    if not video_id:
        raise YouTubeMetadataError("Video metadata missing required id field")

    command = [
        "yt-dlp",
        "--no-playlist",
        "-f",
        "bv*+ba/b",
        "--merge-output-format",
        "mp4",
        "-o",
        output_template,
        url,
    ]

    try:
        await _run_command(command)
    except YouTubeIngestionError as exc:
        raise YouTubeDownloadError(f"Failed to download video: {exc}") from exc

    # Prefer merged MP4 output; fallback to any extension yt-dlp produced.
    preferred_path = output_dir / f"{video_id}.mp4"
    if preferred_path.exists():
        local_path = preferred_path
    else:
        candidates = sorted(output_dir.glob(f"{video_id}.*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise YouTubeDownloadError("Download completed but output file was not found")
        local_path = candidates[0]

    return VideoAsset(source_url=url, local_path=local_path, video_id=video_id, title=title)
