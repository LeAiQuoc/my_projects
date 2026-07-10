from __future__ import annotations

import logging
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from src.config.settings import settings
from src.editing.captions import add_captions
from src.editing.cutter import cut_clips
from src.editing.reframer import reframe_vertical
from src.highlights.llm_highlighter import detect_highlights
from src.ingest.youtube_ingestor import ingest_youtube_url
from src.mcp.contracts import ClipInput, HighlightInput, TranscriptSegmentInput
from src.mcp.errors import fail, ok
from src.models.schemas import ClipAsset, HighlightCandidate, TranscriptSegment
from src.pipeline.orchestrator import _cleanup_temp_video_files, run_pipeline
from src.transcription.whisper_service import transcribe_video

# Keep noisy third-party progress/log lines away from MCP stdout transport.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
logging.getLogger("httpx").setLevel(logging.WARNING)

mcp = FastMCP("clip-highlighter")


async def _run_guarded(coro):
    """Run coroutine while redirecting stdout to stderr to protect MCP stdio protocol."""
    with redirect_stdout(sys.stderr):
        return await coro


@mcp.tool()
async def health_check() -> dict[str, Any]:
    """Fast connectivity check for MCP clients and stdio transport."""
    try:
        return ok(
            {
                "status": "ok",
                "python_executable": sys.executable,
                "python_version": sys.version.split()[0],
                "cwd": str(Path.cwd()),
            }
        )
    except Exception as exc:  # noqa: BLE001
        return fail("INTERNAL_ERROR", str(exc))


def _serialize_transcript(segments: list[TranscriptSegment]) -> list[dict[str, Any]]:
    return [
        {
            "start_seconds": segment.start_seconds,
            "end_seconds": segment.end_seconds,
            "text": segment.text,
        }
        for segment in segments
    ]


def _serialize_highlights(highlights: list[HighlightCandidate]) -> list[dict[str, Any]]:
    return [
        {
            "start_seconds": item.start_seconds,
            "end_seconds": item.end_seconds,
            "hook_title": item.hook_title,
            "rationale": item.rationale,
            "confidence": item.confidence,
        }
        for item in highlights
    ]


def _serialize_clips(clips: list[ClipAsset]) -> list[dict[str, Any]]:
    return [
        {
            "clip_index": clip.clip_index,
            "source_video": str(clip.source_video),
            "output_path": str(clip.output_path),
            "start_seconds": clip.start_seconds,
            "end_seconds": clip.end_seconds,
        }
        for clip in clips
    ]


@mcp.tool()
async def ingest_youtube(url: str, output_dir: str | None = None) -> dict[str, Any]:
    """Download a YouTube video and return local metadata."""
    try:
        video = await _run_guarded(
            ingest_youtube_url(url=url, output_dir=Path(output_dir) if output_dir else settings.input_dir),
        )
        return ok(
            {
                "source_url": video.source_url,
                "local_path": str(video.local_path),
                "video_id": video.video_id,
                "title": video.title,
            }
        )
    except Exception as exc:  # noqa: BLE001
        return fail("INGEST_ERROR", str(exc))


@mcp.tool()
async def transcribe_media(video_path: str, language: str | None = None) -> dict[str, Any]:
    """Transcribe media into timestamped text segments."""
    try:
        transcript = await _run_guarded(transcribe_video(video_path=Path(video_path), language=language))
        return ok({"segments": _serialize_transcript(transcript), "segment_count": len(transcript)})
    except Exception as exc:  # noqa: BLE001
        return fail("TRANSCRIPTION_ERROR", str(exc))


@mcp.tool()
async def select_highlights(
    transcript_segments: list[dict[str, Any]],
    top_k: int = 5,
    min_seconds: int = 30,
    max_seconds: int = 60,
) -> dict[str, Any]:
    """Select highlight windows from transcript segments using DeepSeek."""
    try:
        transcript = [TranscriptSegment(**TranscriptSegmentInput.model_validate(item).model_dump()) for item in transcript_segments]
        highlights = await _run_guarded(
            detect_highlights(
                transcript_segments=transcript,
                top_k=top_k,
                min_seconds=min_seconds,
                max_seconds=max_seconds,
            ),
        )
        return ok({"highlights": _serialize_highlights(highlights), "count": len(highlights)})
    except Exception as exc:  # noqa: BLE001
        return fail("HIGHLIGHT_ERROR", str(exc))


@mcp.tool()
async def cut_video_clips(
    source_video: str,
    highlights: list[dict[str, Any]],
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Cut raw clips from a source video using highlight windows."""
    try:
        highlight_models = [HighlightCandidate(**HighlightInput.model_validate(item).model_dump()) for item in highlights]
        clips = await _run_guarded(
            cut_clips(
                source_video=Path(source_video),
                highlights=highlight_models,
                output_dir=Path(output_dir) if output_dir else settings.temp_dir,
            ),
        )
        return ok({"clips": _serialize_clips(clips), "count": len(clips)})
    except Exception as exc:  # noqa: BLE001
        return fail("CUT_ERROR", str(exc))


@mcp.tool()
async def caption_video_clips(
    clips: list[dict[str, Any]],
    transcript_segments: list[dict[str, Any]],
    output_dir: str | None = None,
    srt_output_dir: str | None = None,
) -> dict[str, Any]:
    """Burn captions into clips using transcript overlap and ffmpeg subtitles."""
    try:
        clip_models = [
            ClipAsset(
                clip_index=item.clip_index,
                source_video=Path(item.source_video),
                output_path=Path(item.output_path),
                start_seconds=item.start_seconds,
                end_seconds=item.end_seconds,
            )
            for item in (ClipInput.model_validate(raw) for raw in clips)
        ]
        transcript = [TranscriptSegment(**TranscriptSegmentInput.model_validate(item).model_dump()) for item in transcript_segments]
        captioned = await _run_guarded(
            add_captions(
                clips=clip_models,
                transcript_segments=transcript,
                output_dir=Path(output_dir) if output_dir else settings.temp_dir,
                srt_output_dir=Path(srt_output_dir) if srt_output_dir else settings.temp_dir,
            ),
        )
        return ok({"captioned_paths": [str(path) for path in captioned], "count": len(captioned)})
    except Exception as exc:  # noqa: BLE001
        return fail("CAPTION_ERROR", str(exc))


@mcp.tool()
async def reframe_video_clips(
    clips: list[str],
    output_dir: str | None = None,
    width: int | None = None,
    height: int | None = None,
) -> dict[str, Any]:
    """Convert clips to vertical format with center crop."""
    try:
        reframed = await _run_guarded(
            reframe_vertical(
                clips=[Path(item) for item in clips],
                output_dir=Path(output_dir) if output_dir else settings.output_dir,
                width=width or settings.vertical_output_width,
                height=height or settings.vertical_output_height,
            ),
        )
        return ok({"reframed_paths": [str(path) for path in reframed], "count": len(reframed)})
    except Exception as exc:  # noqa: BLE001
        return fail("REFRAME_ERROR", str(exc))


@mcp.tool()
async def cleanup_temp(temp_dir: str | None = None) -> dict[str, Any]:
    """Remove temporary artifacts from the configured temp directory."""
    try:
        target = Path(temp_dir) if temp_dir else settings.temp_dir
        with redirect_stdout(sys.stderr):
            _cleanup_temp_video_files(target)
        return ok({"cleaned_temp_dir": str(target)})
    except Exception as exc:  # noqa: BLE001
        return fail("INTERNAL_ERROR", str(exc))


@mcp.tool()
async def run_full_pipeline(url: str) -> dict[str, Any]:
    """Run the full clip-highlighter pipeline and return final artifacts."""
    try:
        result = await _run_guarded(run_pipeline(url))
        return ok(
            {
                "source": {
                    "source_url": result.source.source_url,
                    "local_path": str(result.source.local_path),
                    "video_id": result.source.video_id,
                    "title": result.source.title,
                },
                "transcript_segments": _serialize_transcript(result.transcript_segments),
                "highlights": _serialize_highlights(result.highlights),
                "final_clips": [str(path) for path in result.final_clips],
            }
        )
    except Exception as exc:  # noqa: BLE001
        return fail("INTERNAL_ERROR", str(exc))


if __name__ == "__main__":
    mcp.run(transport="stdio")
