from __future__ import annotations

from pathlib import Path

from src.config.settings import settings
from src.editing.captions import add_captions
from src.editing.cutter import cut_clips
from src.editing.reframer import reframe_vertical
from src.highlights.llm_highlighter import detect_highlights
from src.ingest.youtube_ingestor import ingest_youtube_url
from src.models.schemas import PipelineResult
from src.transcription.whisper_service import transcribe_video
from src.utils.logging import get_logger

logger = get_logger(__name__)


_TEMP_VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".srt"}


def _cleanup_temp_video_files(temp_dir: Path) -> None:
    """Remove temporary video artifacts while leaving non-video intermediates in place."""

    if not temp_dir.exists():
        return

    for path in temp_dir.iterdir():
        if path.is_file() and path.suffix.lower() in _TEMP_VIDEO_SUFFIXES:
            try:
                path.unlink()
            except OSError as exc:
                logger.warning("temp_cleanup.failed", path=str(path), error=str(exc))


async def run_pipeline(url: str) -> PipelineResult:
    """Run the full clip-highlighter pipeline with clip-level fault tolerance."""

    logger.info("pipeline.start", url=url)

    logger.info("phase.start", phase="ingest")
    source = await ingest_youtube_url(url=url, output_dir=settings.input_dir)
    logger.info("phase.complete", phase="ingest", video_path=str(source.local_path), video_id=source.video_id)

    logger.info("phase.start", phase="transcription")
    transcript = await transcribe_video(source.local_path, language=settings.default_language)
    logger.info("phase.complete", phase="transcription", segments=len(transcript))

    logger.info("phase.start", phase="highlight_selection")
    highlights = await detect_highlights(
        transcript_segments=transcript,
        top_k=settings.default_top_clips,
        min_seconds=settings.default_clip_min_seconds,
        max_seconds=settings.default_clip_max_seconds,
    )
    logger.info("phase.complete", phase="highlight_selection", highlights=len(highlights))

    logger.info("phase.start", phase="clip_cutting")
    raw_clips = await cut_clips(
        source_video=source.local_path,
        highlights=highlights,
        output_dir=settings.temp_dir,
    )
    logger.info("phase.complete", phase="clip_cutting", clips=len(raw_clips))

    logger.info("phase.start", phase="post_process")
    final_paths = []
    failed_clip_indices: list[int] = []

    for clip in raw_clips:
        logger.info("clip.start", clip_index=clip.clip_index, operation="caption_and_reframe")
        try:
            captioned_paths = await add_captions(
                clips=[clip],
                transcript_segments=transcript,
                output_dir=settings.temp_dir,
            )

            reframed_paths = await reframe_vertical(
                clips=captioned_paths,
                output_dir=settings.output_dir,
                width=settings.vertical_output_width,
                height=settings.vertical_output_height,
            )
            final_paths.extend(reframed_paths)
            logger.info("clip.complete", clip_index=clip.clip_index, outputs=len(reframed_paths))
        except Exception as exc:  # noqa: BLE001
            # Continue processing remaining clips even if one clip fails post-processing.
            failed_clip_indices.append(clip.clip_index)
            logger.warning(
                "clip.failed",
                clip_index=clip.clip_index,
                operation="caption_and_reframe",
                error=str(exc),
            )

    logger.info(
        "phase.complete",
        phase="post_process",
        succeeded=len(final_paths),
        failed=len(failed_clip_indices),
    )

    if not final_paths:
        logger.warning("pipeline.no_outputs", failed_clips=failed_clip_indices)

    logger.info(
        "pipeline.complete",
        clips=len(final_paths),
        failed_clips=len(failed_clip_indices),
        output_dir=str(settings.output_dir),
    )

    _cleanup_temp_video_files(settings.temp_dir)

    return PipelineResult(
        source=source,
        transcript_segments=transcript,
        highlights=highlights,
        final_clips=final_paths,
    )
